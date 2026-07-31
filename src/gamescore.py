"""Projected final scores for every game on the slate.

Everything here is built from data the app already pulls for the HR model —
posted lineups with real Statcast quality per hitter, the opposing starter's
peripherals, the opposing bullpen, the park, and the weather. Nothing new is
fetched.

The chain, per side of a game:

    posted lineup -> PA-weighted xwOBA -> runs above/below league average
      x opposing starter (weighted by how deep he goes)
      x opposing bullpen (the innings he doesn't cover)
      x park RUN factor (not the HR factor — see parks.park_run_multiplier)
      x weather                                      = expected runs

Expected runs then go through a negative-binomial run distribution (baseball
scoring is overdispersed — Poisson badly understates blowouts and shutouts),
which is convolved to give win probability, the full total-runs distribution,
and an over/under read on any line. Extra innings break ties in proportion to
each side's scoring rate.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .parks import (
    humidity_hr_multiplier, park_run_multiplier, temp_hr_multiplier,
    wind_hr_multiplier, get_park,
)

# --------------------------------------------------------------------------- #
# League baselines (2024-26 scoring environment)
# --------------------------------------------------------------------------- #
LEAGUE_RPG = 4.45           # runs per team per game
LEAGUE_XWOBA = 0.318        # league-average xwOBA
WOBA_SCALE = 1.25           # wRAA denominator
PA_PER_GAME = 37.8          # team plate appearances in a 9-inning game
HOME_FIELD_RUNS = 0.12      # modern home edge, in runs — small and shrinking

LEAGUE_SP_HR9 = 1.25
LEAGUE_SP_BARREL = 8.0      # barrel% allowed
LEAGUE_SP_K_PCT = 22.5      # strikeout rate — the biggest run-suppression lever
LEAGUE_SP_BB_PCT = 8.0
LEAGUE_SP_FIP = 4.15
LEAGUE_PEN_HR9 = 1.15
LEAGUE_PEN_ERA = 4.05
LEAGUE_PEN_K_PCT = 23.5
SP_INNINGS_SHARE = 0.58     # fallback: a modern starter covers ~5.2 of 9

# Runs are overdispersed relative to Poisson. Real MLB team-game scoring is
# mean ~4.45 with variance ~9.7 — a ratio of ~2.2, because innings are
# correlated (rallies) rather than independent. Poisson would put a 4.45-run
# offense at 1.1% shutouts; NB at 2.2 gives 5.4% against a real ~7%, and lands
# the variance (9.72 vs 9.7) and the P(≤2 runs) shape (0.30 vs 0.29) — which is
# what actually drives win probability and totals.
RUN_DISPERSION = 2.2
MAX_RUNS = 22               # distribution support (P(23+) is ~0)

# Share of a team's plate appearances by lineup spot in a 9-inning game. The
# leadoff hitter gets ~22% more trips than the 9-hole, which is exactly why
# lineup spot matters as much as it does.
SPOT_PA_SHARE = np.array([4.65, 4.55, 4.45, 4.35, 4.24, 4.12, 4.00, 3.88, 3.76])


def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def _med(df: pd.DataFrame, col: str):
    """Median of a column that may not exist at all (feeds can be partial)."""
    if df is None or col not in df.columns:
        return None
    return _num(pd.to_numeric(df[col], errors="coerce").median())


# --------------------------------------------------------------------------- #
# Offense
# --------------------------------------------------------------------------- #
def lineup_offense(side: pd.DataFrame) -> dict:
    """PA-weighted lineup quality for one team's posted batters.

    Weighting by lineup spot is the whole point: a .380-xwOBA bat hitting
    leadoff is worth meaningfully more than the same bat hitting 7th, because
    he gets an extra trip roughly every fourth game.
    """
    if side is None or side.empty:
        return {"lineup_xwoba": LEAGUE_XWOBA, "batters": 0, "lineup_iso": None}
    s = side.copy()
    spot = pd.to_numeric(s.get("lineup_spot"), errors="coerce")
    s["_spot"] = spot.fillna(5.0).clip(1, 9)
    s = s.sort_values("_spot")
    xw = pd.to_numeric(s.get("xwoba"), errors="coerce")
    if xw.notna().sum() == 0:
        return {"lineup_xwoba": LEAGUE_XWOBA, "batters": len(s), "lineup_iso": None}
    w = SPOT_PA_SHARE[(s["_spot"].to_numpy(dtype=int) - 1).clip(0, 8)]
    m = xw.notna().to_numpy()
    lineup_xwoba = float(np.average(xw.to_numpy()[m], weights=w[m]))
    iso = _med(s, "iso")
    return {
        "lineup_xwoba": round(lineup_xwoba, 4),
        "lineup_iso": round(iso, 4) if iso is not None else None,
        "batters": int(m.sum()),
    }


def _runs_from_xwoba(lineup_xwoba: float) -> float:
    """League runs/game shifted by the lineup's wRAA per PA."""
    wraa_per_pa = (float(lineup_xwoba) - LEAGUE_XWOBA) / WOBA_SCALE
    return LEAGUE_RPG + wraa_per_pa * PA_PER_GAME


# --------------------------------------------------------------------------- #
# Pitching
# --------------------------------------------------------------------------- #
def pitching_multiplier(opp_side: pd.DataFrame) -> dict:
    """How much the opposing staff suppresses (or feeds) run scoring.

    The starter's numbers come off the rows of the hitters facing him, so this
    reads the SAME peripherals the HR model uses — HR/9 and barrel% allowed —
    plus the opposing bullpen's HR/9 for the innings the starter won't cover.
    """
    out = {"sp_mult": 1.0, "pen_mult": 1.0, "staff_mult": 1.0,
           "sp_hr9": None, "pen_hr9": None, "sp_name": None}
    if opp_side is None or opp_side.empty:
        return out
    hr9 = _med(opp_side, "pitcher_hr9")
    brl = _med(opp_side, "pitcher_barrel_pct_allowed")
    k_pct = _med(opp_side, "pitcher_k_pct")
    bb_pct = _med(opp_side, "pitcher_bb_pct")
    fip = _med(opp_side, "pitcher_fip")
    pen = _med(opp_side, "bullpen_hr9")
    pen_k = _med(opp_side, "bullpen_k_pct")
    pen_era = _med(opp_side, "bullpen_era")

    sp = 1.0
    # Strikeouts are the primary run-suppression mechanism: a punched-out hitter
    # cannot advance a runner, reach on an error, or find a hole. This is the
    # largest single term here and the model ran without it entirely.
    if k_pct is not None:
        sp += 0.55 * (LEAGUE_SP_K_PCT - k_pct) / LEAGUE_SP_K_PCT
    if bb_pct is not None:
        sp += 0.14 * (bb_pct - LEAGUE_SP_BB_PCT) / LEAGUE_SP_BB_PCT
    # FIP over ERA: it strips the defense and sequencing luck that make a
    # starter's ERA a poor guide to how he'll pitch tonight.
    if fip is not None:
        sp += 0.20 * (fip / LEAGUE_SP_FIP - 1.0)
    elif hr9 is not None:
        sp += 0.16 * (hr9 / LEAGUE_SP_HR9 - 1.0)
    if brl is not None:
        sp += 0.10 * (brl / LEAGUE_SP_BARREL - 1.0)
    # A starter who has lost velo off his own baseline is getting hit harder
    # than his season line says — the season line hasn't caught up yet.
    dv = _med(opp_side, "sp_velo_delta")
    if dv is not None:
        sp += 0.010 * (-dv)          # -1.0 mph => +1% runs
    sp = float(np.clip(sp, 0.62, 1.45))

    p = 1.0
    if pen_era is not None:
        p += 0.26 * (pen_era / LEAGUE_PEN_ERA - 1.0)
    elif pen is not None:
        p += 0.20 * (pen / LEAGUE_PEN_HR9 - 1.0)
    if pen_k is not None:
        p += 0.30 * (LEAGUE_PEN_K_PCT - pen_k) / LEAGUE_PEN_K_PCT
    p = float(np.clip(p, 0.75, 1.32))

    # How much of the game he actually covers. A 6.5-IP arm exposes the lineup
    # to the pen for 2.5 innings; a 4-IP arm for 5, which is a different game.
    share = sp_innings_share(_med(opp_side, "sp_ip_per_start"))

    out.update({
        "sp_mult": round(sp, 4), "pen_mult": round(p, 4),
        "staff_mult": round(share * sp + (1 - share) * p, 4),
        "sp_hr9": hr9, "pen_hr9": pen, "sp_k_pct": k_pct, "sp_fip": fip,
        "sp_innings_share": round(share, 3),
        "sp_name": (opp_side.get("pitcher_name").iloc[0]
                    if "pitcher_name" in opp_side.columns and len(opp_side) else None),
    })
    return out


def sp_innings_share(ip_per_start) -> float:
    """Fraction of the game's innings the starter covers, from his real IP/GS."""
    ip = _num(ip_per_start)
    if ip is None:
        return SP_INNINGS_SHARE
    return float(np.clip(ip / 9.0, 0.15, 0.85))


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def environment_multiplier(row: pd.Series, home_team: str) -> dict:
    """Park RUN factor x weather, as a runs multiplier.

    Weather is reused from the HR model but damped: a wind that adds 12% to
    home runs adds nowhere near 12% to runs, because most runs don't leave
    the yard.
    """
    park_mult = park_run_multiplier(str(home_team))
    park = get_park(str(home_team))
    wind = wind_hr_multiplier(park, _num(row.get("wind_mph"), 0.0) or 0.0,
                              _num(row.get("wind_dir_deg")), )
    temp = temp_hr_multiplier(_num(row.get("temp_f")))
    alt = float((park or {}).get("altitude_ft", 0) or 0)
    hum = humidity_hr_multiplier(_num(row.get("humidity_pct")), alt)
    weather_hr = wind * temp * hum
    weather_runs = 1.0 + 0.45 * (weather_hr - 1.0)      # damped to runs
    return {
        "park_run_mult": round(park_mult, 4),
        "weather_run_mult": round(weather_runs, 4),
        "env_run_mult": round(park_mult * weather_runs, 4),
        "wind_mult": round(wind, 3), "temp_f": _num(row.get("temp_f")),
    }


# --------------------------------------------------------------------------- #
# Run distribution
# --------------------------------------------------------------------------- #
def run_pmf(mean_runs: float, dispersion: float = RUN_DISPERSION,
            max_runs: int = MAX_RUNS) -> np.ndarray:
    """Negative-binomial P(team scores exactly k runs), k = 0..max_runs.

    Poisson would say a 4.5-run offense is shut out 1.1% of the time; the real
    number is ~7%. Baseball innings are correlated (rallies), so the tails are
    fat on both ends. Variance = dispersion x mean reproduces that.
    """
    mu = max(float(mean_runs), 0.05)
    if dispersion <= 1.0:                                # Poisson limit
        k = np.arange(max_runs + 1)
        logp = -mu + k * np.log(mu) - np.array([math.lgamma(i + 1) for i in k])
        pmf = np.exp(logp)
        return pmf / pmf.sum()
    r = mu / (dispersion - 1.0)                          # NB size
    p = r / (r + mu)                                     # NB prob of success
    k = np.arange(max_runs + 1)
    logp = (np.array([math.lgamma(i + r) - math.lgamma(r) - math.lgamma(i + 1)
                      for i in k]) + r * math.log(p) + k * math.log1p(-p))
    pmf = np.exp(logp)
    return pmf / pmf.sum()


def matchup_probabilities(home_runs: float, away_runs: float) -> dict:
    """Win probability + total-runs distribution from the two run means.

    Regulation ties are broken by extra innings, which we split in proportion
    to each side's scoring rate rather than 50/50 — the better offense wins
    more extra-inning games — with the home team's last-at-bat edge folded in.
    """
    h, a = run_pmf(home_runs), run_pmf(away_runs)
    joint = np.outer(h, a)
    idx = np.arange(len(h))
    home_reg = float(joint[idx[:, None] > idx[None, :]].sum())
    away_reg = float(joint[idx[:, None] < idx[None, :]].sum())
    tie = float(np.trace(joint))
    share = (home_runs + 0.30) / max(home_runs + away_runs + 0.60, 1e-6)
    p_home = home_reg + tie * share
    totals = np.convolve(h, a)
    return {
        "p_home": round(float(np.clip(p_home, 0.02, 0.98)), 4),
        "p_away": round(float(np.clip(1.0 - p_home, 0.02, 0.98)), 4),
        "p_tie_reg": round(tie, 4),
        "totals": totals,
        "p_home_shutout_win": round(float(joint[1:, 0].sum()), 4),
    }


def over_under(totals: np.ndarray, line: float) -> dict:
    """P(over) / P(under) for a half-run or whole-number total."""
    k = np.arange(len(totals))
    p_over = float(totals[k > line].sum())
    p_push = float(totals[k == line].sum()) if float(line).is_integer() else 0.0
    p_under = float(totals[k < line].sum())
    return {"p_over": round(p_over, 4), "p_under": round(p_under, 4),
            "p_push": round(p_push, 4)}


# --------------------------------------------------------------------------- #
# The slate
# --------------------------------------------------------------------------- #
def _american(p: float) -> int:
    """Fair American odds for a probability (no vig)."""
    p = float(np.clip(p, 0.01, 0.99))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _league_shift(slate: pd.DataFrame) -> float:
    """How far this slate's lineups sit above/below a true league-average bat.

    Coverage decides the level. If the feed only resolves the bats it has
    Statcast rows for, every posted lineup reads .340 and every game projects
    as a 12-run track meet. Twenty teams are not all above average on the same
    night — that is a measurement artifact, so the slate is re-centered on the
    league and the model runs on RELATIVE lineup strength, which is what the
    projection actually depends on.
    """
    if slate is None or slate.empty or "team" not in slate.columns:
        return 0.0
    per_team = [lineup_offense(g)["lineup_xwoba"]
                for _, g in slate.groupby("team", sort=False)]
    per_team = [x for x in per_team if x is not None and x == x]
    if len(per_team) < 4:
        return 0.0
    return float(np.mean(per_team)) - LEAGUE_XWOBA


def predict_games(slate: pd.DataFrame, default_total_line: float = 8.5,
                  center_to_league: bool = True) -> pd.DataFrame:
    """One projected final score per game on the slate.

    Expects the scored slate the app already builds (a row per hitter, with
    `game`, `team`, `is_home`, `home_team`, and the usual Statcast columns).
    """
    if slate is None or slate.empty or "game" not in slate.columns:
        return pd.DataFrame()
    shift = _league_shift(slate) if center_to_league else 0.0
    rows = []
    for game, g in slate.groupby("game", sort=False):
        home_team = str(g["home_team"].iloc[0]) if "home_team" in g.columns else None
        is_home = g.get("is_home")
        if is_home is None or home_team is None:
            continue
        home = g[is_home.astype(bool)]
        away = g[~is_home.astype(bool)]
        if home.empty or away.empty:
            continue
        away_team = str(away["team"].iloc[0])

        h_off, a_off = lineup_offense(home), lineup_offense(away)
        h_pit = pitching_multiplier(home)     # the staff the HOME hitters face
        a_pit = pitching_multiplier(away)
        env = environment_multiplier(g.iloc[0], home_team)

        h_runs = (_runs_from_xwoba(h_off["lineup_xwoba"] - shift) * h_pit["staff_mult"]
                  * env["env_run_mult"] + HOME_FIELD_RUNS)
        a_runs = (_runs_from_xwoba(a_off["lineup_xwoba"] - shift) * a_pit["staff_mult"]
                  * env["env_run_mult"])
        h_runs, a_runs = float(np.clip(h_runs, 1.4, 11.0)), float(np.clip(a_runs, 1.4, 11.0))

        probs = matchup_probabilities(h_runs, a_runs)
        ou = over_under(probs["totals"], default_total_line)
        totals = probs["totals"]
        fav_home = probs["p_home"] >= 0.5

        rows.append({
            "game": game, "home_team": home_team, "away_team": away_team,
            "home_runs_exp": round(h_runs, 2), "away_runs_exp": round(a_runs, 2),
            "score_line": f"{away_team} {round(a_runs):.0f} – {home_team} {round(h_runs):.0f}",
            "total_exp": round(h_runs + a_runs, 2),
            "most_likely_total": int(np.argmax(totals)),
            "margin_exp": round(h_runs - a_runs, 2),
            "p_home": probs["p_home"], "p_away": probs["p_away"],
            "winner": home_team if fav_home else away_team,
            "win_prob": probs["p_home"] if fav_home else probs["p_away"],
            "fair_odds": _american(probs["p_home"] if fav_home else probs["p_away"]),
            "total_line": default_total_line,
            "p_over": ou["p_over"], "p_under": ou["p_under"],
            "total_lean": "Over" if ou["p_over"] > ou["p_under"] else "Under",
            "total_lean_prob": max(ou["p_over"], ou["p_under"]),
            "home_lineup_xwoba": h_off["lineup_xwoba"],
            "away_lineup_xwoba": a_off["lineup_xwoba"],
            "home_sp": a_pit["sp_name"], "away_sp": h_pit["sp_name"],
            "home_faces_hr9": h_pit["sp_hr9"], "away_faces_hr9": a_pit["sp_hr9"],
            "home_staff_mult": h_pit["staff_mult"], "away_staff_mult": a_pit["staff_mult"],
            "park_run_mult": env["park_run_mult"],
            "weather_run_mult": env["weather_run_mult"],
            "batters_home": h_off["batters"], "batters_away": a_off["batters"],
            "temp_f": env["temp_f"], "wind_mult": env["wind_mult"],
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = attach_confidence(out)
    return out.sort_values("confidence", ascending=False).reset_index(drop=True)


def attach_confidence(games: pd.DataFrame) -> pd.DataFrame:
    """0-100 confidence per game: how much edge, on how much real data.

    A big projected margin built on two half-posted lineups is not a strong
    play — the confidence has to know the difference.
    """
    g = games.copy()
    edge = (pd.to_numeric(g["win_prob"], errors="coerce") - 0.5).abs() * 2.0   # 0..1
    tot_edge = (pd.to_numeric(g["total_lean_prob"], errors="coerce") - 0.5).abs() * 2.0
    filled = ((pd.to_numeric(g["batters_home"], errors="coerce").clip(0, 9)
               + pd.to_numeric(g["batters_away"], errors="coerce").clip(0, 9)) / 18.0)
    have_sp = (g["home_faces_hr9"].notna().astype(float)
               + g["away_faces_hr9"].notna().astype(float)) / 2.0
    quality = 0.65 * filled + 0.35 * have_sp
    g["confidence"] = (100.0 * (0.60 * edge + 0.20 * tot_edge + 0.20 * 0.0)
                       * (0.55 + 0.45 * quality)).round(1).clip(0, 100)
    g["data_quality_pct"] = (100.0 * quality).round(0)
    return g


# --------------------------------------------------------------------------- #
# "Who I like"
# --------------------------------------------------------------------------- #
def game_rationale(row: pd.Series) -> str:
    """Plain-English why, drawn from whichever inputs actually moved this game."""
    bits = []
    home, away = row["home_team"], row["away_team"]
    winner = row["winner"]
    loser = away if winner == home else home
    hx, ax = _num(row.get("home_lineup_xwoba")), _num(row.get("away_lineup_xwoba"))
    if hx is not None and ax is not None and abs(hx - ax) >= 0.008:
        better = home if hx > ax else away
        bits.append(f"{better} has the better lineup by xwOBA "
                    f"({max(hx, ax):.3f} vs {min(hx, ax):.3f})")
    hs, as_ = _num(row.get("home_staff_mult")), _num(row.get("away_staff_mult"))
    if hs is not None and as_ is not None:
        # staff_mult > 1 = the staff THIS side faces gives up more.
        soft = home if hs > as_ else away
        if abs(hs - as_) >= 0.03:
            bits.append(f"{soft} draws the softer staff "
                        f"({max(hs, as_):.2f}x vs {min(hs, as_):.2f}x run environment)")
    pm = _num(row.get("park_run_mult"))
    if pm is not None and abs(pm - 1.0) >= 0.03:
        bits.append(f"{'run-friendly' if pm > 1 else 'run-suppressing'} park "
                    f"({pm:.2f}x runs)")
    wm = _num(row.get("weather_run_mult"))
    if wm is not None and abs(wm - 1.0) >= 0.02:
        bits.append(f"weather {'helps' if wm > 1 else 'hurts'} scoring ({wm:.2f}x)")
    if winner == home:
        bits.append("home team gets last at-bat")
    lean, lp = row.get("total_lean"), _num(row.get("total_lean_prob"), 0.5)
    if lp and lp >= 0.55:
        bits.append(f"{lean} {row.get('total_line')} at {lp*100:.0f}%")
    head = f"{winner} over {loser} ({float(row['win_prob'])*100:.0f}%)"
    return head + (" — " + "; ".join(bits) if bits else "")


def best_bets(games: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    """The games I actually like, ranked by confidence, with the why attached."""
    if games is None or games.empty:
        return pd.DataFrame()
    g = games.sort_values("confidence", ascending=False).head(n).copy()
    g["rationale"] = g.apply(game_rationale, axis=1)
    g["pick"] = g["winner"] + " ML (" + g["fair_odds"].map(
        lambda v: f"{v:+d}") + " fair)"
    g["total_pick"] = (g["total_lean"] + " " + g["total_line"].astype(str)
                       + " (" + (g["total_lean_prob"] * 100).round(0).astype(int).astype(str)
                       + "%)")
    return g.reset_index(drop=True)


def game_of_the_day(games: pd.DataFrame) -> pd.Series | None:
    """The single game I like most — my favorite play on the board.

    Confidence alone would keep handing this to whichever game happened to have
    the fullest lineups. The pick has to be an actual EDGE, so this scores the
    side edge and the total edge together and demands the projection be built
    on real data before it can win.
    """
    if games is None or games.empty:
        return None
    g = games.copy()
    side_edge = (pd.to_numeric(g["win_prob"], errors="coerce") - 0.5).abs() * 2.0
    total_edge = (pd.to_numeric(g["total_lean_prob"], errors="coerce") - 0.5).abs() * 2.0
    quality = pd.to_numeric(g.get("data_quality_pct", 100), errors="coerce").fillna(100) / 100.0
    g["_gotd"] = (0.55 * side_edge + 0.45 * total_edge) * (0.5 + 0.5 * quality)
    best = g.sort_values("_gotd", ascending=False).iloc[0].copy()
    lean_is_side = (float(best["win_prob"]) - 0.5) * 2 >= (float(best["total_lean_prob"]) - 0.5) * 2
    best["gotd_bet"] = (
        f"{best['winner']} ML ({int(best['fair_odds']):+d} fair)" if lean_is_side
        else f"{best['total_lean']} {best['total_line']} "
             f"({float(best['total_lean_prob'])*100:.0f}%)")
    best["gotd_angle"] = "side" if lean_is_side else "total"
    best["rationale"] = game_rationale(best)
    return best


def attach_top_bat(games: pd.DataFrame, slate: pd.DataFrame) -> pd.DataFrame:
    """Tie each game's score projection to the HR model's favorite bat in it."""
    if games is None or games.empty or slate is None or slate.empty:
        return games
    col = "hr_prob_game" if "hr_prob_game" in slate.columns else "hr_score"
    if col not in slate.columns or "game" not in slate.columns:
        return games
    s = slate.copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    top = s.sort_values(col, ascending=False).groupby("game", sort=False).head(1)
    m = top.set_index("game")
    g = games.copy()
    g["top_bat"] = g["game"].map(m["player"]) if "player" in m.columns else None
    g["top_bat_team"] = g["game"].map(m["team"]) if "team" in m.columns else None
    if "hr_prob_game" in m.columns:
        g["top_bat_hr_pct"] = (g["game"].map(m["hr_prob_game"]).astype(float) * 100).round(1)
    return g
