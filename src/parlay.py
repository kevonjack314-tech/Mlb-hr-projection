"""HR parlay generator — driven by the model and the graded record.

Legs are chosen on measured signal, not a checklist: the calibrated model HR
probability, the model-vs-price edge, and the per-role reliability factors
LEARNED from real graded outcomes (src/tuning.py).

Roles are just the tier bands the record is kept in (they map to the season-HR
tiers in trends.py) — ⚓ Anchor = star bats, 💰 Value = mid-tier, 🚀 Longshot =
under-the-radar. They exist so results can be tracked per band, not as a
prescription for how a ticket must be built.

Default composition comes from THE RECORD, not a playbook. Across the graded
log, Value legs have out-performed their predicted rate while Longshot legs
have badly under-performed, so the default ticket is Anchor + Value(s) and
skips longshots. Ask for `strategy="boom"` if you want the lottery ticket.

Diversification: no two legs from the same game (one weather/park/pitcher
environment shouldn't decide the whole ticket) and prefer different power
archetypes.

Each ticket gets combined odds, the model's true win probability (per-leg
probabilities scaled by the learned role reliability), EV vs the actual price,
and a data-driven quality checklist with a green/yellow/red light.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .lineup import spot_role_fit
from .trends import TIER_MID, TIER_STAR, tier_of
from .tuning import role_prob_factor
from .odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    format_american,
)

# Probability bands -> role (these mirror the season-HR tier bands).
ANCHOR_MIN_PROB = 0.15      # ~ +560 or shorter
VALUE_MIN_PROB = 0.10       # ~ +900 .. +560
LONGSHOT_MIN_PROB = 0.045   # ~ +2000 .. +900

# Record-driven composition: Value legs have beaten their predicted hit rate in
# the graded log while Longshot legs have badly missed theirs, so the default
# ticket leans Anchor + Value and does NOT force a longshot in.
COMPOSITIONS = {
    1: ["Anchor"],
    2: ["Anchor", "Value"],
    3: ["Anchor", "Value", "Value"],
    4: ["Anchor", "Anchor", "Value", "Value"],
    5: ["Anchor", "Anchor", "Value", "Value", "Value"],
}

ROLE_EMOJI = {"Anchor": "⚓", "Value": "💰", "Longshot": "🚀"}


def assign_role(prob: float, season_hr=None) -> str | None:
    """Parlay role for a bat. Tier-first when the season HR total is known:
    ⭐ stars (18+ HR) are the Anchors, 🔷 mid bats (8-17) the Value plays, and
    🎯 under-the-radar bats (≤7) the Longshots — matching how books actually
    price them. Bats below the longshot probability floor stay off tickets.
    Falls back to pure probability bands when season HRs are missing."""
    if prob < LONGSHOT_MIN_PROB:
        return None
    if season_hr is not None and season_hr == season_hr:  # known, non-NaN
        tier = tier_of(season_hr)
        if tier == TIER_STAR:
            return "Anchor" if prob >= VALUE_MIN_PROB else "Value"
        if tier == TIER_MID:
            return "Value" if prob >= LONGSHOT_MIN_PROB * 1.5 else "Longshot"
        return "Longshot"
    if prob >= ANCHOR_MIN_PROB:
        return "Anchor"
    if prob >= VALUE_MIN_PROB:
        return "Value"
    return "Longshot"


def archetype(row: pd.Series) -> str:
    """Coarse power archetype, used to diversify a ticket."""
    if row.get("pull_score", 50) >= 60 and row.get("fb_score", 50) >= 55:
        return "Pull/Loft"
    if row.get("max_ev_score", 50) >= 72:
        return "Raw Power"
    if row.get("barrel_score", 50) >= 65:
        return "Barrel"
    if row.get("whiff_score", 50) <= 40:
        return "Contact"
    return "Balanced"


def role_fit(row: pd.Series, role: str) -> float:
    """How good a player is *for a given role*.

    Blends the role's headline score with a lineup-spot bonus (Anchor wants 3-5,
    Value 6-7, Longshot 7-9) and a recurring-history bonus
    for bats that have actually homered from the spot they're hitting today.
    """
    spot = row.get("lineup_spot")
    spot_bonus = spot_role_fit(spot, role)
    # Recurring HR-by-spot signal: HR/game from this exact spot, scaled to ~0-8.
    rate = row.get("spot_hr_rate")
    hist_bonus = float(min(8.0, (rate or 0.0) * 40.0)) if pd.notna(rate) else 0.0
    # Self-calibration: if history says this rating homers MORE than the model
    # credits (positive cal_edge), lean into it — improves picks over time.
    cal = row.get("cal_edge_pct")
    cal_bonus = float(min(6.0, max(0.0, cal))) if pd.notna(cal) else 0.0
    hist_bonus += cal_bonus
    # Opposing-starter matchup: HRs that pitcher allowed to THIS lineup spot over
    # their last 10 games — a juicy-spot signal. ~2.3 pts per HR, capped at 7.
    sp = row.get("sp_hr_at_spot")
    if pd.notna(sp):
        hist_bonus += float(min(7.0, float(sp) * 2.3))
    # Live Trends Lab signals: bats riding a HR streak (back-to-back pattern),
    # tiers favored by yesterday's rotation, spots hot on this weekday.
    hist_bonus += 2.5 * float(row.get("hot_streak") or 0)
    hist_bonus += 1.5 * float(row.get("tier_lean") or 0)
    heat = row.get("dow_spot_heat")
    if pd.notna(heat):
        hist_bonus += float(heat)
    if role == "Anchor":
        return float(row.get("hr_score", 0)) + spot_bonus + hist_bonus
    if role == "Value":
        return (float(row.get("sneaky_score", 0)) + 0.5 * float(row.get("edge_pct", 0))
                + spot_bonus + hist_bonus)
    return float(row.get("longshot_score", 0)) + spot_bonus + hist_bonus  # Longshot


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["role"] = df.apply(
        lambda r: assign_role(r["hr_prob_game"], r.get("season_hr")), axis=1)
    df["archetype"] = df.apply(archetype, axis=1)
    return df


def _pick(pool: pd.DataFrame, used_games: set, used_arch: set,
          max_per_game: int, diversify_arch: bool, rng=None, topk: int = 4):
    """Pick a strong-fit row honoring game/archetype diversification, relaxing the
    archetype rule (then the game rule) only if nothing else qualifies.

    With `rng` set (shuffle mode), pick at random among the top-`topk` qualifying
    candidates instead of always the single best — for variety without quality loss.
    """
    if pool.empty:
        return None
    for relax_arch in (False, True) if diversify_arch else (True,):
        for relax_game in (False, True):
            cands = []
            for _, row in pool.iterrows():
                if not relax_game and row["game"] in used_games:
                    continue
                if not relax_arch and row["archetype"] in used_arch:
                    continue
                cands.append(row)
                if rng is None or len(cands) >= topk:
                    break
            if cands:
                return cands[rng.randrange(len(cands))] if rng is not None else cands[0]
    return None


def generate_parlay(df: pd.DataFrame, n_legs: int = 3, strategy: str = "model",
                    max_per_game: int = 1, diversify_arch: bool = True,
                    seed: int | None = None) -> dict:
    """Build a parlay. Returns {legs: DataFrame, summary: dict, checklist: list}.

    Pass `seed` (shuffle mode) to re-roll among the top candidates per role and
    produce a different valid ticket each time.
    """
    n_legs = int(np.clip(n_legs, 1, 5))
    rng = random.Random(seed) if seed is not None else None
    df = enrich(df)

    legs: list[pd.Series] = []
    used_games: set = set()
    used_arch: set = set()
    used_players: set = set()

    def take(row, role):
        d = row.to_dict()
        d["role"] = role
        legs.append(d)
        used_games.add(row["game"])
        used_arch.add(row["archetype"])
        used_players.add(row["player"])

    if strategy == "model":
        comp = COMPOSITIONS[n_legs]
        for role in comp:
            pool = df[(df["role"] == role) & (~df["player"].isin(used_players))]
            if not pool.empty:
                pool = pool.assign(_fit=pool.apply(lambda r: role_fit(r, role), axis=1)) \
                           .sort_values("_fit", ascending=False)
            row = _pick(pool, used_games, used_arch, max_per_game, diversify_arch, rng)
            if row is None:  # fall back to any remaining role-eligible bat
                pool2 = df[~df["player"].isin(used_players)].assign(
                    _fit=df[~df["player"].isin(used_players)].apply(
                        lambda r: role_fit(r, role), axis=1)).sort_values("_fit", ascending=False)
                row = _pick(pool2, used_games, used_arch, max_per_game, diversify_arch, rng)
            if row is not None:
                take(row, row.get("role") or role)
    else:
        # Ranked strategies: order the board, then fill with diversification.
        if strategy == "safe":            # chalk: highest HR probability
            ranked = df.sort_values("hr_prob_game", ascending=False)
        elif strategy == "value":         # best edge vs the book (shines on live odds)
            ranked = df.sort_values(["edge_pct", "sneaky_score"], ascending=False)
        elif strategy == "boom":          # ceiling among genuine longer-odds bats
            elig = df[df["role"].isin(["Value", "Longshot"])]
            ranked = (elig if len(elig) >= n_legs else df).sort_values(
                "longshot_score", ascending=False)
        else:
            ranked = df.sort_values("hr_score", ascending=False)
        while len(legs) < n_legs:
            pool = ranked[~ranked["player"].isin(used_players)]
            row = _pick(pool, used_games, used_arch, max_per_game, diversify_arch, rng)
            if row is None:
                break
            rl = row.get("role")
            take(row, rl if isinstance(rl, str) else "Leg")

    legs_df = pd.DataFrame(legs)
    summary, checklist = _summarize(legs_df, n_legs)
    return {"legs": legs_df, "summary": summary, "checklist": checklist}


def summarize_selection(df: pd.DataFrame, players: list[str]) -> dict:
    """Build & grade a custom parlay from a hand-picked list of players."""
    df = enrich(df)
    legs = df[df["player"].isin(players)].copy()
    if legs.empty:
        return {"legs": legs, "summary": {"n_legs": 0}, "checklist": []}
    legs["role"] = legs.apply(
        lambda r: assign_role(r["hr_prob_game"], r.get("season_hr")), axis=1).fillna("Leg")
    summary, checklist = _summarize(legs, len(legs))
    return {"legs": legs, "summary": summary, "checklist": checklist}


def _summarize(legs: pd.DataFrame, n_legs: int) -> tuple[dict, list]:
    if legs.empty:
        return {"n_legs": 0}, []

    dec = legs["book_odds"].map(american_to_decimal)
    combined_dec = float(dec.prod())
    combined_american = decimal_to_american(combined_dec)
    implied = 1.0 / combined_dec
    # Ticket win%: independence assumption, with each leg's probability scaled
    # by its role's REAL track-record reliability (learned daily; 1.0 until
    # enough logged legs exist — see src/tuning.py).
    leg_probs = [
        float(np.clip(p * role_prob_factor(str(r)), 0.002, 0.6))
        for p, r in zip(legs["hr_prob_game"], legs.get("role", [""] * len(legs)))
    ]
    model_prob = float(np.prod(leg_probs))
    ev_pct = model_prob * combined_dec - 1.0          # per $1 stake

    summary = {
        "n_legs": len(legs),
        "combined_decimal": round(combined_dec, 2),
        "combined_american": combined_american,
        "combined_american_str": format_american(combined_american),
        "implied_prob": round(implied * 100, 1),
        "model_prob": round(model_prob * 100, 1),
        "ev_pct": round(ev_pct * 100, 1),
        "any_live": bool(legs.get("odds_is_live", pd.Series([False])).any()),
        "payout_per_10": round(10 * (combined_dec - 1), 2),
    }

    archs = set(legs["archetype"])
    games = set(legs["game"])
    spots = set(legs["lineup_spot"].dropna()) if "lineup_spot" in legs else set()
    n = len(legs)
    probs = pd.to_numeric(legs["hr_prob_game"], errors="coerce")
    _mean = lambda col, d: (pd.to_numeric(legs[col], errors="coerce").mean()
                            if col in legs else d)
    # Data-driven quality checks: measured signal and the graded record, not a
    # checklist of profile minimums.
    checks = [
        # Every leg clears the record's realistic floor — no dead legs.
        ("No dead legs (every leg ≥ 8% model HR)", bool((probs >= 0.08).all())),
        # At least one genuinely strong bat carries the ticket.
        ("Carried by a strong bat (top leg ≥ 20%)", bool(probs.max() >= 0.20)),
        # Longshots have badly under-performed in the graded record.
        ("No record-fading longshots", not (legs["role"] == "Longshot").any() or n < 3),
        ("Different games (uncorrelated)", len(games) == n),
        ("Different power archetypes", len(archs) >= min(n, 2)),
        ("Different lineup spots", len(spots) >= min(n, 3) or len(spots) == n),
        ("Real batted-ball power (barrels)", _mean("barrel_score", 50) >= 55),
        ("Contact quality holding or rising", _mean("barrel_trend", 0) >= -1.0),
        ("Matchup edge (starter + bullpen)", _mean("matchup_score", 50) >= 52),
        ("Favorable park / weather", _mean("env_score", 50) >= 50),
        ("Priced sanely (no lottery legs)", bool((legs["book_odds"] <= 2200).all())),
        ("Ticket win% clears the bar", model_prob >= (0.03 if n >= 4 else 0.06)),
    ]
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    ratio = passed / total
    summary["checks_passed"] = passed
    summary["checks_total"] = total
    summary["light"] = "🟢 GREEN" if ratio >= 0.7 else ("🟡 YELLOW" if ratio >= 0.5 else "🔴 RED")
    return summary, checks
