"""Composite home-run projection model.

The model is intentionally transparent: every raw input is normalized to a 0-100
"sub-score" against fixed league reference ranges (so scores are comparable across
dates, not just within a single slate), and the composite scores are explicit
weighted blends of those sub-scores plus environment/matchup multipliers.

Pipeline per hitter row:
  1. Resolve the effective batting side vs the probable pitcher and platoon edge.
  2. Normalize Statcast quality, season HR rate, and recent form to sub-scores.
  3. Build matchup and environment (park + weather) multipliers / sub-scores.
  4. Combine into four product scores (HR Score, Longshot, Consistency, Sneaky)
     and a per-game HR probability.

See README.md ("Methodology") for the full rationale and the weighting tables,
which are defined here as constants so the docs and the code never drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .parks import (
    get_park,
    humidity_hr_multiplier,
    park_hr_multiplier,
    temp_hr_multiplier,
    wind_hr_multiplier,
)
from .ulx import hr_environment, power_checks

# League average HR per plate appearance (modern run environment).
LEAGUE_HR_PER_PA = 0.034
# Expected plate appearances for a starter in a 9-inning game.
DEFAULT_PA = 4.1
# Soft ceiling on the per-PA HR rate (keeps top game-HR odds realistic).
P_HR_CEIL = 0.085

# --- Reference ranges (≈5th–95th percentile of qualified hitters) for 0-100 scaling.
REF = {
    "barrel_pct": (3.0, 18.0),
    "hard_hit_pct": (30.0, 55.0),
    "avg_ev": (86.0, 93.0),
    "max_ev": (103.0, 117.0),
    "xwoba": (0.290, 0.400),
    "hr_per_pa": (0.010, 0.070),
    "recent_hr_rate": (0.010, 0.080),
    "k_pct": (15.0, 32.0),
    "whiff_pct": (15.0, 35.0),
    "chase_pct": (20.0, 38.0),
    "zone_contact_pct": (78.0, 93.0),
    "fb_pct": (25.0, 48.0),
    "gb_pct": (35.0, 55.0),
    "ld_pct": (15.0, 27.0),
    "pull_pct": (32.0, 48.0),
    "hr_fb": (6.0, 22.0),
    "xiso": (0.090, 0.280),
    "xslg": (0.330, 0.560),
    "brl_pa": (2.0, 12.0),
    "vs_pitch_woba": (0.290, 0.400),
}

# League-average fly-ball rate (FanGraphs batted-ball FB%); fly balls are the
# raw material of home runs, so above-average FB% earns a real HR-rate boost.
LEAGUE_FB_PCT = 35.0
# League-average HR/FB (home runs per fly ball) — the fly-ball -> HR conversion
# rate, a direct measure of game power applied to balls in the air.
LEAGUE_HR_FB = 12.5

# --- Composite HR Score weights (must sum to 1.0). ---
HR_SCORE_WEIGHTS = {
    "power_quality": 0.34,   # Statcast batted-ball quality
    "season_hr": 0.16,       # season-long HR/PA
    "recent_form": 0.16,     # last 7/15/30-day HR rate
    "matchup": 0.16,         # opposing pitcher + platoon edge
    "environment": 0.18,     # park + weather
}

# Sub-weights inside the Statcast "power quality" score (sum to 1.0).
POWER_QUALITY_WEIGHTS = {
    "barrel_pct": 0.30,
    "xiso": 0.15,        # expected isolated power (xSLG - xBA)
    "hard_hit_pct": 0.18,
    "xwoba": 0.17,
    "max_ev": 0.12,
    "avg_ev": 0.08,
}

# Recent-form blend (sum to 1.0): the 7-day window is the loudest signal.
RECENT_FORM_WEIGHTS = {"hr_rate_7": 0.50, "hr_rate_15": 0.30, "hr_rate_30": 0.20}


def scale(value: float, lo: float, hi: float) -> float:
    """Min-max scale a value into [0, 100], clipped at the reference bounds."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    return float(np.clip((value - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def effective_bat_side(bats: str, pitcher_throws: str) -> str:
    """Resolve the side a hitter swings from vs this pitcher.

    Switch hitters bat opposite the pitcher's throwing hand (the platoon-favored
    side), so a switch hitter facing a RHP bats lefty.
    """
    bats = (bats or "R").upper()[0]
    throws = (pitcher_throws or "R").upper()[0]
    if bats == "S":
        return "L" if throws == "R" else "R"
    return bats


def has_platoon_advantage(bats: str, pitcher_throws: str) -> bool:
    """True when the matchup is platoon-favorable for the hitter."""
    bats = (bats or "R").upper()[0]
    throws = (pitcher_throws or "R").upper()[0]
    if bats == "S":
        return True
    return bats != throws


def matchup_multiplier(row: pd.Series) -> tuple[float, float]:
    """Return (matchup_multiplier, matchup_score 0-100) from the pitcher matchup.

    Drivers: pitcher HR/9 allowed, barrel% allowed, fly-ball lean, and the
    hitter's platoon edge. A homer-prone flyball pitcher with a platoon
    disadvantage is the juiciest matchup.
    """
    hr9 = row.get("pitcher_hr9", 1.2)
    barrel_allowed = row.get("pitcher_barrel_pct_allowed", 8.0)
    lean = str(row.get("pitcher_lean", "NEU")).upper()
    platoon = has_platoon_advantage(row.get("bats", "R"), row.get("pitcher_throws", "R"))

    # HR/9 maps ~ [0.6, 1.7] -> multiplier [0.86, 1.18].
    hr9_mult = np.clip(0.86 + (hr9 - 0.6) * (0.32 / 1.1), 0.80, 1.22)
    # Barrels allowed: league ~8%. +/-1.5% per point, gentle.
    barrel_mult = np.clip(1.0 + (barrel_allowed - 8.0) * 0.012, 0.90, 1.12)
    lean_mult = {"GB": 0.93, "NEU": 1.0, "FB": 1.08}.get(lean, 1.0)
    platoon_mult = 1.06 if platoon else 0.95

    mult = float(hr9_mult * barrel_mult * lean_mult * platoon_mult)

    # A 0-100 sub-score: center the multiplier (~0.75–1.45 plausible) onto 0-100.
    score = scale(mult, 0.80, 1.30)
    return mult, score


def environment_components(row: pd.Series) -> dict:
    """Compute park + weather multipliers and a combined environment sub-score."""
    home_team = row.get("home_team", row.get("team"))
    park = get_park(home_team)
    eff_side = effective_bat_side(row.get("bats", "R"), row.get("pitcher_throws", "R"))

    park_mult = park_hr_multiplier(home_team, eff_side)
    wind_mult = wind_hr_multiplier(
        park, row.get("wind_mph"), row.get("wind_dir_deg"), eff_side
    )
    temp_mult = temp_hr_multiplier(row.get("temp_f"))
    humid_mult = humidity_hr_multiplier(
        row.get("humidity_pct"), park.get("altitude_ft", 0) if park else 0
    )

    env_mult = float(park_mult * wind_mult * temp_mult * humid_mult)
    # Combined environment plausibly ranges ~0.78–1.30; map to 0-100.
    env_score = scale(env_mult, 0.85, 1.20)
    return {
        "park_factor": round(park_mult * 100, 1) if park else 100.0,
        "park_mult": park_mult,
        "wind_mult": round(wind_mult, 3),
        "temp_mult": round(temp_mult, 3),
        "humidity_mult": round(humid_mult, 3),
        "env_mult": env_mult,
        "env_score": env_score,
    }


def _power_quality_score(row: pd.Series) -> dict:
    subs = {
        "barrel_pct": scale(row.get("barrel_pct"), *REF["barrel_pct"]),
        "xiso": scale(row.get("xiso"), *REF["xiso"]),
        "hard_hit_pct": scale(row.get("hard_hit_pct"), *REF["hard_hit_pct"]),
        "xwoba": scale(row.get("xwoba"), *REF["xwoba"]),
        "max_ev": scale(row.get("max_ev"), *REF["max_ev"]),
        "avg_ev": scale(row.get("avg_ev"), *REF["avg_ev"]),
    }
    pq = sum(subs[k] * w for k, w in POWER_QUALITY_WEIGHTS.items())
    return {"power_quality_score": pq, "_pq_subs": subs}


def _recent_form_rate(row: pd.Series) -> float:
    return sum(row.get(k, 0.0) * w for k, w in RECENT_FORM_WEIGHTS.items())


def expected_hr(row: pd.Series) -> tuple[float, float, float]:
    """Season expected HR from batted-ball quality. Returns (xhr/PA, xHR, HR-xHR).

    Barrels per PA (brl_pa) is the cleanest input — roughly ~62% of barrels leave
    the yard — with a small contact baseline; we fall back to barrel% when brl_pa
    is unavailable. The HR - xHR gap flags over/under-performers (regression).
    Note: sprint speed is *not* used — it has no measurable effect on HR power, so
    xHR is deliberately not "sprint-adjusted"; sprint speed is shown as context.
    """
    pa = row.get("pa")
    brl_pa = row.get("brl_pa")
    barrel = row.get("barrel_pct")
    fb = row.get("fb_pct")
    if brl_pa is not None:
        xhr_per_pa = (brl_pa / 100.0) * 0.62 + 0.004
    elif barrel is not None:
        xhr_per_pa = 0.0034 * barrel + (0.0002 * (fb - 35.0) if fb is not None else 0.0)
    else:
        return (LEAGUE_HR_PER_PA, float("nan"), float("nan"))
    xhr_per_pa = float(np.clip(xhr_per_pa, 0.004, 0.095))
    if pa is None or (isinstance(pa, float) and np.isnan(pa)):
        return (xhr_per_pa, float("nan"), float("nan"))
    xhr_season = xhr_per_pa * pa
    hr_minus_xhr = (row.get("season_hr", xhr_season) or xhr_season) - xhr_season
    return (xhr_per_pa, xhr_season, hr_minus_xhr)


def pitch_type_matchup(row: pd.Series) -> tuple[float, float]:
    """Pitch-mix-weighted hitter performance vs the probable pitcher's arsenal.

    Combines the hitter's wOBA-like marks vs fastballs / breaking / offspeed with
    the pitcher's pitch mix. Returns (multiplier, 0-100 score). Modeled today; the
    real version would pull per-batter run value by pitch type from Statcast.
    """
    vs = (row.get("vs_fb"), row.get("vs_br"), row.get("vs_os"))
    mix = (row.get("pitcher_mix_fb"), row.get("pitcher_mix_br"), row.get("pitcher_mix_os"))
    if any(v is None for v in vs) or any(m is None for m in mix):
        return 1.0, 50.0
    total = sum(mix) or 1.0
    woba = sum(m * v for m, v in zip(mix, vs)) / total
    score = scale(woba, *REF["vs_pitch_woba"])
    mult = float(np.clip(0.92 + (woba - 0.320) / 0.080 * 0.10, 0.90, 1.10))
    return mult, score


def score_row(row: pd.Series) -> dict:
    """Score a single hitter-vs-game row. Returns a dict of derived fields."""
    out: dict = {}

    eff_side = effective_bat_side(row.get("bats", "R"), row.get("pitcher_throws", "R"))
    platoon = has_platoon_advantage(row.get("bats", "R"), row.get("pitcher_throws", "R"))
    out["effective_bats"] = eff_side
    out["platoon_adv"] = platoon

    # --- Sub-scores ---
    pq = _power_quality_score(row)
    power_quality = pq["power_quality_score"]
    season_hr_score = scale(row.get("hr_per_pa"), *REF["hr_per_pa"])
    recent_rate = _recent_form_rate(row)
    recent_form_score = scale(recent_rate, *REF["recent_hr_rate"])
    k_score = scale(row.get("k_pct"), *REF["k_pct"])  # high = strikeout-prone
    # Swing-and-miss (whiff) rate: high = more boom-or-bust, lower contact floor.
    # Fall back to the K% signal when whiff isn't available.
    whiff_raw = row.get("whiff_pct")
    whiff_score = scale(whiff_raw, *REF["whiff_pct"]) if whiff_raw is not None else k_score
    # Combined swing-and-miss signal (whiff weighted over K%).
    swing_miss_score = 0.6 * whiff_score + 0.4 * k_score

    # Plate discipline + batted-ball profile.
    chase_score = scale(row.get("chase_pct"), *REF["chase_pct"])          # high = chases more
    zone_contact_score = scale(row.get("zone_contact_pct"), *REF["zone_contact_pct"])  # high = better floor
    fb_pct = row.get("fb_pct")
    fb_score = scale(fb_pct, *REF["fb_pct"])
    # Fly-ball multiplier on HR rate: balls hit in the air vs on the ground.
    fb_mult = (float(np.clip(1.0 + (fb_pct - LEAGUE_FB_PCT) / LEAGUE_FB_PCT * 0.5, 0.85, 1.18))
               if fb_pct is not None else 1.0)
    # Batted-ball distribution + pull + HR/FB conversion (real, FanGraphs).
    gb_score = scale(row.get("gb_pct"), *REF["gb_pct"])      # high = grounders (bad for HR)
    ld_score = scale(row.get("ld_pct"), *REF["ld_pct"])
    pull_score = scale(row.get("pull_pct"), *REF["pull_pct"])  # pulled air balls leave the yard
    hr_fb = row.get("hr_fb")
    hr_fb_score = scale(hr_fb, *REF["hr_fb"])
    # HR/FB multiplier on HR rate: how often this bat's fly balls clear the wall.
    hr_fb_mult = (float(np.clip(1.0 + (hr_fb - LEAGUE_HR_FB) / LEAGUE_HR_FB * 0.35, 0.88, 1.15))
                  if hr_fb is not None else 1.0)

    brl_pa_score = scale(row.get("brl_pa"), *REF["brl_pa"])

    # Pitcher matchup, then fold in the pitch-type (arsenal) edge.
    matchup_mult, matchup_score = matchup_multiplier(row)
    pitch_mult, pitch_score = pitch_type_matchup(row)
    matchup_mult *= pitch_mult
    matchup_score = 0.75 * matchup_score + 0.25 * pitch_score
    env = environment_components(row)

    # Expected HR (season) and the over/under-performance gap (regression signal).
    xhr_per_pa, xhr_season, hr_minus_xhr = expected_hr(row)

    out.update(env)
    out["power_quality_score"] = round(power_quality, 1)
    out["season_hr_score"] = round(season_hr_score, 1)
    out["recent_form_score"] = round(recent_form_score, 1)
    out["matchup_mult"] = round(matchup_mult, 3)
    out["matchup_score"] = round(matchup_score, 1)
    out["pitch_matchup_score"] = round(pitch_score, 1)
    out["brl_pa_score"] = round(brl_pa_score, 1)
    out["xhr_season"] = round(xhr_season, 1) if xhr_season == xhr_season else None
    out["hr_minus_xhr"] = round(hr_minus_xhr, 1) if hr_minus_xhr == hr_minus_xhr else None
    out["barrel_score"] = round(pq["_pq_subs"]["barrel_pct"], 1)
    out["max_ev_score"] = round(pq["_pq_subs"]["max_ev"], 1)
    out["hard_hit_score"] = round(pq["_pq_subs"]["hard_hit_pct"], 1)
    out["whiff_score"] = round(whiff_score, 1)
    out["chase_score"] = round(chase_score, 1)
    out["zone_contact_score"] = round(zone_contact_score, 1)
    out["fb_score"] = round(fb_score, 1)
    out["fb_mult"] = round(fb_mult, 3)
    out["pull_score"] = round(pull_score, 1)
    out["hr_fb_score"] = round(hr_fb_score, 1)
    out["hr_fb_mult"] = round(hr_fb_mult, 3)
    out["gb_score"] = round(gb_score, 1)
    out["ld_score"] = round(ld_score, 1)

    # --- ULX power checklist + HR environment (instilled thresholds) ---
    ulx = power_checks(row)
    out["ulx_checks"] = ulx["ulx_checks"]
    out["ulx_total"] = ulx["ulx_total"]
    out["ulx_score"] = ulx["ulx_score"]
    out["ulx_grade"] = ulx["ulx_grade"]
    out["same_handed_smasher"] = ulx["same_handed_smasher"]
    hr_env = hr_environment({
        "wind_mult": out["wind_mult"], "temp_f": row.get("temp_f"),
        "park_factor": out["park_factor"], "pitcher_hr9": row.get("pitcher_hr9"),
        "pitcher_lean": row.get("pitcher_lean"),
    })
    out["hr_env_score"] = hr_env["hr_env_score"]
    out["hr_env_count"] = hr_env["hr_env_count"]
    out["hr_hunting"] = hr_env["hr_hunting"]

    # --- Composite HR Score (0-100) ---
    hr_score = (
        HR_SCORE_WEIGHTS["power_quality"] * power_quality
        + HR_SCORE_WEIGHTS["season_hr"] * season_hr_score
        + HR_SCORE_WEIGHTS["recent_form"] * recent_form_score
        + HR_SCORE_WEIGHTS["matchup"] * matchup_score
        + HR_SCORE_WEIGHTS["environment"] * env["env_score"]
    )
    out["hr_score"] = round(hr_score, 1)

    # --- Per-game HR probability (>=1 HR) ---
    # Blend season rate, recent form, and a quality-implied rate, then apply the
    # matchup and environment multipliers.
    quality_implied = LEAGUE_HR_PER_PA * (0.5 + power_quality / 100.0)  # 0.5x–1.5x league
    # Cap the recent-form rate so a small-sample hot streak can't blow up the
    # estimate beyond what's physically plausible.
    recent_capped = min(recent_rate, 0.095)
    base_rate = (
        0.55 * row.get("hr_per_pa", LEAGUE_HR_PER_PA)
        + 0.25 * recent_capped
        + 0.20 * quality_implied
    )
    raw_rate = base_rate * matchup_mult * env["env_mult"] * fb_mult * hr_fb_mult
    # Soft ceiling instead of a hard clip: probabilities saturate smoothly toward
    # P_CEIL so elite bats keep their ordering (no bunching at the cap) and the
    # implied odds stay in a realistic HR-prop range (top spots ~ +280..+360).
    p_adj = float(max(0.002, P_HR_CEIL * (1.0 - np.exp(-raw_rate / P_HR_CEIL))))
    # Expected PA depends on the batting-order spot (top of order bats more).
    from .lineup import expected_pa
    pa = expected_pa(row.get("lineup_spot"))
    out["expected_pa"] = round(pa, 2)
    p_game = 1.0 - (1.0 - p_adj) ** pa
    out["hr_prob_pa"] = round(p_adj, 4)
    out["hr_prob_game"] = round(p_game, 4)
    out["xhr"] = round(p_adj * pa, 3)  # expected HR in the game
    # Fair American odds for the >=1 HR prop (no vig), handy for +EV checks.
    out["fair_odds"] = _prob_to_american(p_game)

    # --- Longshot Score (boom-or-bust ceiling) ---
    # ULX: "longshots don't win parlays, PROFILES do" — the power checklist is the
    # backbone, alongside air-ball power (fly-ball rate, HR/FB, pull) and the spot's
    # HR environment.
    longshot = (
        0.24 * ulx["ulx_score"]
        + 0.22 * out["max_ev_score"]
        + 0.14 * out["barrel_score"]
        + 0.10 * fb_score
        + 0.08 * hr_fb_score
        + 0.06 * pull_score
        + 0.10 * env["env_score"]
        + 0.06 * matchup_score
    )
    # Reward variance (more swing-and-miss & more chasing = more boom-or-bust) and
    # slightly de-emphasize players who are already chalk (not a "longshot").
    variance_signal = 0.7 * swing_miss_score + 0.3 * chase_score
    variance_bonus = 1.0 + (variance_signal - 50.0) / 500.0   # ±0.10
    chalk_penalty = 1.0 - max(0.0, (out["hr_prob_game"] - 0.12)) * 0.5
    out["longshot_score"] = round(float(np.clip(longshot * variance_bonus * chalk_penalty, 0, 100)), 1)

    # --- Consistency Score (high floor) ---
    # Floor rewards bat-to-ball skill: low swing-and-miss and strong in-zone
    # contact (Z-Contact%) — the cleanest repeatable-contact signal.
    contact_floor = 0.6 * (100.0 - swing_miss_score) + 0.4 * zone_contact_score
    confidence = float(np.clip(row.get("pa", 200) / 450.0, 0.6, 1.0))  # sample-size trust
    consistency = (
        0.28 * out["hard_hit_score"]
        + 0.22 * contact_floor
        + 0.20 * season_hr_score
        + 0.15 * scale(row.get("avg_ev"), *REF["avg_ev"])
        + 0.15 * scale(row.get("xwoba"), *REF["xwoba"])
    ) * confidence
    out["consistency_score"] = round(consistency, 1)

    # --- Sneaky Score (under-the-radar value) ---
    form_gap = recent_rate - row.get("hr_per_pa", LEAGUE_HR_PER_PA)  # heating up?
    form_gap_score = scale(form_gap, -0.02, 0.04)
    # Under-radar: lower season HR profile but still real batted-ball pop.
    under_radar = (100.0 - season_hr_score) * (power_quality / 100.0)
    # Regression: hitting FEWER HR than expected (negative gap) = positive-
    # regression candidate, a classic sneaky play. Neutral when xHR is unknown.
    regression_score = (scale(-hr_minus_xhr, -4.0, 8.0)
                        if hr_minus_xhr == hr_minus_xhr else 50.0)
    sneaky = (
        0.26 * matchup_score
        + 0.22 * env["env_score"]
        + 0.22 * form_gap_score
        + 0.16 * under_radar
        + 0.14 * regression_score
    )
    out["sneaky_score"] = round(sneaky, 1)
    out["regression_score"] = round(regression_score, 1)
    out["form_gap"] = round(form_gap, 4)

    out["rationale"] = _build_rationale(row, out)
    out["sneaky_reasons"] = _build_sneaky_reasons(row, out)
    return out


def _prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.001, 0.999))
    if p >= 0.5:
        return int(round(-100 * p / (1 - p)))
    return int(round(100 * (1 - p) / p))


def _build_rationale(row: pd.Series, out: dict) -> str:
    bits = []
    if out["barrel_score"] >= 70:
        bits.append(f"elite barrel rate ({row.get('barrel_pct')}%)")
    elif out["barrel_score"] >= 50:
        bits.append(f"solid barrel rate ({row.get('barrel_pct')}%)")
    if out["max_ev_score"] >= 70:
        bits.append(f"big raw power ({row.get('max_ev')} mph max EV)")
    if row.get("fb_pct") is not None and out.get("fb_score", 0) >= 65:
        bits.append(f"fly-ball hitter ({row.get('fb_pct')}% FB)")
    if out["recent_form_score"] >= 65:
        bits.append("hot recent form")
    if out["park_factor"] >= 106:
        bits.append(f"HR-friendly park ({int(out['park_factor'])} factor)")
    elif out["park_factor"] <= 94:
        bits.append(f"pitcher-friendly park ({int(out['park_factor'])} factor)")
    if out["wind_mult"] >= 1.05:
        bits.append("wind blowing out")
    elif out["wind_mult"] <= 0.95:
        bits.append("wind blowing in")
    if out["platoon_adv"]:
        bits.append(f"platoon edge vs {row.get('pitcher_throws')}HP")
    if row.get("pitcher_hr9", 1.2) >= 1.4:
        bits.append(f"facing HR-prone arm ({row.get('pitcher_hr9')} HR/9)")
    if not bits:
        bits.append("balanced profile")
    return "; ".join(bits[:5]).capitalize()


def _build_sneaky_reasons(row: pd.Series, out: dict) -> str:
    reasons = []
    if out["form_gap"] > 0.012:
        reasons.append("heating up beyond season line")
    if str(row.get("pitcher_lean")).upper() == "FB" and out["park_factor"] >= 102:
        reasons.append("flyball pitcher in a homer park")
    if row.get("pitcher_hr9", 1.2) >= 1.4 and row.get("power_tier", 3) <= 3:
        reasons.append("low-profile bat vs a hittable arm")
    if out["wind_mult"] >= 1.06:
        reasons.append("tailwind carry")
    if out["platoon_adv"] and out["power_quality_score"] >= 55:
        reasons.append("quiet platoon power edge")
    if out.get("hr_minus_xhr") is not None and out["hr_minus_xhr"] <= -2.0:
        reasons.append(f"under xHR by {abs(out['hr_minus_xhr']):.0f} (due to regress up)")
    if out.get("pitch_matchup_score", 50) >= 65:
        reasons.append("strong vs this arm's pitch mix")
    if not reasons and out["sneaky_score"] >= 55:
        reasons.append("favorable matchup/park combo")
    return "; ".join(reasons[:3]).capitalize() if reasons else ""


def score_slate(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the model to every row and return the enriched DataFrame."""
    if df is None or df.empty:
        return df
    scored = df.apply(score_row, axis=1, result_type="expand")
    result = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    return result
