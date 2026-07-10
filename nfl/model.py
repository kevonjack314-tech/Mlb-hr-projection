"""NFL matchup analytics — who does what this week, and WHY.

No betting framework — this projects production from:
  1. previous-season per-game baselines (yards, TDs, usage),
  2. opponent defense strength vs the position,
  3. the DEFENSIVE SCHEME the opponent runs × the player's archetype
     (the "Chase vs press-man → alpha receivers feast" engine),
  4. the player's own history vs this team, and
  5. game environment (implied points, home/dome/wind, game script).

Outputs per player: projected rush/rec/pass yards, TD likelihood,
🎯 "TD favorite" and 💯 "100-yd watch" flags, and a plain-language matchup
insight that shows the work.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .data import ARCHETYPE_LABEL, TEAMS, build_week_slate, live_week_slate
from .schemes import TEAM_OFF, def_scheme_of, scheme_boost, scheme_label

# Yardage volatility (sd as fraction of projection) for P(100)/P(275).
YDS_CV = {"rush": 0.45, "rec": 0.50, "pass": 0.22}
TD_FAVORITE_MIN = 0.35
WATCH_100_MIN = 0.28


def _def_mult(opp_def: int) -> float:
    return {1: 1.14, 2: 1.07, 3: 1.0, 4: 0.93, 5: 0.86}.get(int(opp_def), 1.0)


def _p_over(exp: float, line: float, cv: float) -> float:
    if exp <= 0:
        return 0.0
    sd = max(8.0, exp * cv)
    return float(np.clip(
        1.0 - 0.5 * (1 + math.erf((line - exp) / (sd * math.sqrt(2)))), 0.0, 0.95))


def score_row(row: pd.Series) -> dict:
    out: dict = {}
    pos, arch = row["pos"], row["archetype"]
    scheme = def_scheme_of(row["opponent"])
    yds_mult, td_mult, reason = scheme_boost(arch, scheme)
    dmult = _def_mult(row["opp_def"])
    implied = float(row["team_implied"])
    env = 1.0 + (implied - 23.0) * 0.012          # good offenses lift everyone
    if row.get("is_home"):
        env *= 1.02
    # vs-team history nudge (only with a real sample).
    vs_mult = 1.0
    if int(row.get("vs_games", 0)) >= 2:
        vs_mult = 1.0 + (float(row["vs_ypg_mult"]) - 1.0) * 0.35   # damped

    # ---- Projected yards ----
    rush = float(row["rush_ypg"]) * dmult * (yds_mult if pos in ("RB", "QB") else 1.0) \
        * env * vs_mult
    rec = float(row["rec_ypg"]) * dmult * (yds_mult if pos in ("WR", "TE", "RB") else 1.0) \
        * env * vs_mult
    pas = float(row["pass_ypg"]) * dmult * yds_mult * env * vs_mult
    if not row.get("dome") and float(row.get("wind_mph", 0)) >= 15:
        pas *= 0.90
        rec *= 0.94
    out["proj_rush_yds"] = round(rush, 1)
    out["proj_rec_yds"] = round(rec, 1)
    out["proj_pass_yds"] = round(pas, 1)
    out["proj_scrimmage"] = round(rush + rec, 1)

    # ---- 100-yd watch (275 for QBs) ----
    if pos == "QB":
        p100 = _p_over(pas, 275.0, YDS_CV["pass"])
        watch_label = "275-yd watch"
    elif pos == "RB":
        p100 = _p_over(rush + rec, 100.0, YDS_CV["rush"])
        watch_label = "100-yd watch"
    else:
        p100 = _p_over(rec, 100.0, YDS_CV["rec"])
        watch_label = "100-yd watch"
    out["p_100"] = round(p100, 3)
    out["watch_100"] = bool(p100 >= WATCH_100_MIN)
    out["watch_label"] = watch_label

    # ---- TD likelihood ----
    team_tds = max(0.6, implied / 7.0 * 0.85)
    exp_tds = team_tds * float(row["rz_share"]) * dmult * td_mult
    # Previous-season TD rate is half the story; blend it in.
    prev_rate = float(row["prev_tds"]) / max(1, int(row["prev_games"]))
    exp_tds = 0.65 * exp_tds + 0.35 * prev_rate * td_mult
    if int(row.get("vs_games", 0)) >= 2 and int(row.get("vs_tds", 0)) >= 2:
        exp_tds *= 1.08                                # scores on these guys
    p_td = 1.0 - math.exp(-exp_tds)
    out["exp_tds"] = round(exp_tds, 3)
    out["td_prob"] = round(float(np.clip(p_td, 0.02, 0.72)), 4)
    out["td_favorite"] = bool(out["td_prob"] >= TD_FAVORITE_MIN)

    # ---- Matchup score (0-100) for ranking boards ----
    out["matchup_score"] = round(float(np.clip(
        40 + (yds_mult - 1) * 120 + (td_mult - 1) * 80 + (dmult - 1) * 100
        + (implied - 23) * 1.2 + (vs_mult - 1) * 60, 0, 100)), 1)

    # ---- The insight (show the work, Chase-style) ----
    opp = row["opponent"]
    bits = [f"{row['player']} ({ARCHETYPE_LABEL.get(arch, arch)}) vs {opp} — "
            f"they run {scheme_label(scheme)}"]
    if reason:
        bits[0] += f": {reason}"
    prod = {"QB": f"{row['pass_ypg']:.0f} pass yds/g",
            "RB": f"{row['rush_ypg']:.0f} rush yds/g"}.get(
                pos, f"{row['rec_ypg']:.0f} rec yds/g")
    bits.append(f"{prod} and {int(row['prev_tds'])} TDs last season")
    if int(row.get("vs_games", 0)) >= 2:
        bits.append(f"{row['vs_ypg_mult']:.0%} of his usual output with "
                    f"{int(row['vs_tds'])} TDs in {int(row['vs_games'])} career "
                    f"games vs them")
    flags = []
    if out["td_favorite"]:
        flags.append("🎯 TD favorite")
    if out["watch_100"]:
        flags.append(f"💯 {watch_label}")
    out["insight"] = ". ".join(bits) + (". → " + " & ".join(flags) if flags else ".")
    out["def_scheme"] = scheme_label(scheme)
    out["off_system"] = TEAM_OFF.get(row["team"], "")
    return out


def score_slate(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    scored = df.apply(score_row, axis=1, result_type="expand")
    return pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)


def get_week_slate(week: int, season: int = 2026, prefer_live: bool = True):
    """(scored_slate, source_label). Live via nfl_data_py once the season starts."""
    df = live_week_slate(week, season) if prefer_live else None
    if df is not None:
        return score_slate(df), "LIVE (nflverse)"
    return score_slate(build_week_slate(week, season)), "MODELED (prev-season baselines)"
