"""3+ made threes — basketball's version of the home-run model.

The chain is the same as baseball's and football's, over a different scarce
resource:

    minutes  ->  three-point attempts  ->  makes  ->  P(3 or more)

Each link is a separate failure mode, and minutes is by far the biggest. A
40%% shooter who plays 18 minutes is a worse bet than a 34%% shooter who plays
36, and most of the error in public three-point props is a minutes error rather
than a shooting error. So minutes is modelled explicitly and carries its own
confidence, instead of being folded silently into a season average.

The probability itself is exact rather than approximated. If attempts are
Poisson and each attempt goes in independently with probability p, then MAKES
are Poisson with mean (attempts x p) -- Poisson thinning. That gives a closed
form for P(3+) with no simulation and no binomial fudge over a fractional
number of attempts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .teams import (
    LEAGUE_3PA_PER_GAME, LEAGUE_3P_PCT, game_pace, pace_multiplier,
    rest_multiplier,
)

MADE_THREES_LINE = 3            # the threshold the prop pays on
LEAGUE_MINUTES_STARTER = 32.0

REF = {
    "minutes": (12.0, 38.0),
    "fg3a_per36": (1.5, 11.0),
    "fg3_pct": (0.28, 0.44),
    "pace": (94.0, 105.0),
    "opp_3pa_allowed": (30.0, 42.0),
}

SCORE_WEIGHTS = {
    "volume": 0.44,      # attempts is the engine — you cannot make what you don't shoot
    "minutes": 0.24,     # the biggest source of error in public projections
    "accuracy": 0.18,
    "matchup": 0.08,     # how freely the opponent lets threes go up
    "pace": 0.06,
}


def scale(value, lo: float, hi: float) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def p_at_least(lam: float, k: int = MADE_THREES_LINE) -> float:
    """P(X >= k) for X ~ Poisson(lam) — the makes distribution, exactly.

    Poisson thinning: Poisson attempts, each converted independently, gives
    Poisson makes. No simulation, and it handles a fractional attempt
    projection honestly instead of rounding it to an integer first.
    """
    lam = max(float(lam), 0.0)
    if lam <= 0:
        return 0.0
    cdf = sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(k))
    return float(np.clip(1.0 - cdf, 0.0005, 0.97))


def project_minutes(row: pd.Series) -> dict:
    """Minutes tonight, and how much to trust the number.

    Starts from the player's recent average rather than his season average —
    rotations change — then applies the things that are known in advance:
    a back-to-back, and teammates being out.
    """
    base = _num(row.get("minutes_l5"))
    if base is None:
        base = _num(row.get("minutes_season"), LEAGUE_MINUTES_STARTER)
    rest = rest_multiplier(row.get("days_rest"), bool(row.get("back_to_back")))
    mins = base * rest["rest_minutes_mult"]
    # A starter out ahead of him is the single most actionable minutes signal
    # there is, and it is knowable hours before tip.
    bump = _num(row.get("teammate_out_minutes_bump"), 0.0) or 0.0
    mins += bump
    mins = float(np.clip(mins, 0.0, 44.0))
    spread = _num(row.get("minutes_stdev_l5"))
    confidence = 1.0 if spread is None else float(np.clip(1.0 - spread / 12.0, 0.3, 1.0))
    return {"proj_minutes": round(mins, 1), "minutes_confidence": round(confidence, 2),
            "rest_note": rest["rest_note"],
            "rest_shooting_mult": rest["rest_shooting_mult"]}


def score_row(row: pd.Series) -> dict:
    out: dict = {}
    mn = project_minutes(row)
    minutes = mn["proj_minutes"]

    pace = game_pace(_num(row.get("team_pace")), _num(row.get("opp_pace")),
                     row.get("home_team"))
    pace_mult = pace_multiplier(pace)

    fg3a36 = _num(row.get("fg3a_per36"), 4.0) or 4.0
    attempts = fg3a36 * (minutes / 36.0) * pace_mult

    # The opponent cannot do much about whether a shooter is open, but it can
    # do a lot about how many threes get attempted at all.
    opp_3pa = _num(row.get("opp_3pa_allowed"))
    if opp_3pa is not None:
        attempts *= float(np.clip(opp_3pa / LEAGUE_3PA_PER_GAME, 0.86, 1.16))

    pct = _num(row.get("fg3_pct"), LEAGUE_3P_PCT) or LEAGUE_3P_PCT
    # Shrink toward league average on a thin sample — a 48% shooter on 40
    # attempts is a 38% shooter who has had a good month.
    att_sample = _num(row.get("fg3a_season"), 150.0) or 150.0
    w = float(np.clip(att_sample / (att_sample + 120.0), 0.2, 0.95))
    pct = w * pct + (1 - w) * LEAGUE_3P_PCT
    pct *= mn["rest_shooting_mult"]
    opp_d = _num(row.get("opp_3p_pct_allowed"))
    if opp_d is not None:
        pct *= float(np.clip(1.0 + (opp_d - LEAGUE_3P_PCT) * 0.8, 0.95, 1.05))
    if row.get("is_home"):
        pct *= 1.008          # small, real, and consistent across seasons
    pct = float(np.clip(pct, 0.22, 0.48))

    lam = max(attempts * pct, 0.0)
    p3 = p_at_least(lam, MADE_THREES_LINE)

    score = (SCORE_WEIGHTS["volume"] * scale(fg3a36, *REF["fg3a_per36"])
             + SCORE_WEIGHTS["minutes"] * scale(minutes, *REF["minutes"])
             + SCORE_WEIGHTS["accuracy"] * scale(pct, *REF["fg3_pct"])
             + SCORE_WEIGHTS["matchup"] * scale(opp_3pa if opp_3pa is not None
                                                else LEAGUE_3PA_PER_GAME,
                                                *REF["opp_3pa_allowed"])
             + SCORE_WEIGHTS["pace"] * scale(pace, *REF["pace"]))

    out.update({
        "proj_minutes": minutes,
        "minutes_confidence": mn["minutes_confidence"],
        "game_pace": round(pace, 1),
        "proj_3pa": round(attempts, 2),
        "proj_3p_pct": round(pct, 4),
        "proj_3pm": round(lam, 2),
        "threes_prob_game": round(p3, 4),
        "threes_score": round(float(np.clip(score, 0, 100)), 1),
        "volume_score": round(scale(fg3a36, *REF["fg3a_per36"]), 1),
        "minutes_score": round(scale(minutes, *REF["minutes"]), 1),
        "accuracy_score": round(scale(pct, *REF["fg3_pct"]), 1),
        "pace_score": round(scale(pace, *REF["pace"]), 1),
        "rest_note": mn["rest_note"],
        "fair_odds": _prob_to_american(p3),
    })
    out["rationale"] = _rationale(row, out)
    return out


def _prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.005, 0.98))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _rationale(row: pd.Series, out: dict) -> str:
    bits = []
    fg3a36 = _num(row.get("fg3a_per36"))
    if fg3a36 is not None and fg3a36 >= 8.0:
        bits.append(f"high-volume shooter ({fg3a36:.1f} 3PA/36)")
    if out["proj_minutes"] >= 34:
        bits.append(f"heavy minutes ({out['proj_minutes']:.0f})")
    elif out["proj_minutes"] <= 20:
        bits.append(f"limited minutes ({out['proj_minutes']:.0f})")
    bump = _num(row.get("teammate_out_minutes_bump"), 0.0) or 0.0
    if bump >= 3:
        bits.append(f"+{bump:.0f} min with a starter out")
    if out["proj_3p_pct"] >= 0.390:
        bits.append(f"knocking them down ({out['proj_3p_pct']*100:.1f}%)")
    if out["game_pace"] >= 102:
        bits.append(f"fast game ({out['game_pace']:.0f} pace)")
    elif out["game_pace"] <= 96:
        bits.append(f"slow game ({out['game_pace']:.0f} pace)")
    opp = _num(row.get("opp_3pa_allowed"))
    if opp is not None and opp >= 39:
        bits.append("defense concedes threes")
    if out.get("rest_note"):
        bits.append(out["rest_note"])
    if out["minutes_confidence"] <= 0.6:
        bits.append("volatile minutes — lower confidence")
    return "; ".join(bits[:5]).capitalize() if bits else ""


LISTED_3PA_CAP = 0.92


def conserve_team_3pa(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a team's projected three-point attempts inside what it actually takes.

    Same failure the football model had: every player's attempts are estimated
    from his own rate, nothing makes the nine of them agree, and the sum drifts
    above the team's real volume. Threes are a shared resource too — there are
    only so many possessions — so the listed rotation is normalised against the
    team's attempt total, less a reserve for players not on the board.
    """
    if df is None or df.empty or "proj_3pa" not in df.columns:
        return df
    if not {"game", "team"} <= set(df.columns):
        return df
    out = df.copy()
    out["threes_conserve_mult"] = 1.0
    for _, idx in out.groupby(["game", "team"], sort=False).groups.items():
        rows = out.loc[idx]
        team_3pa = pd.to_numeric(rows.get("team_3pa_per_game"), errors="coerce")
        supply = float(team_3pa.iloc[0]) if team_3pa.notna().any() else LEAGUE_3PA_PER_GAME
        pace_adj = float(pd.to_numeric(rows["game_pace"], errors="coerce").iloc[0]) / 99.0
        cap = supply * pace_adj * LISTED_3PA_CAP
        projected = float(pd.to_numeric(rows["proj_3pa"], errors="coerce").sum())
        if projected <= cap or projected <= 0:
            continue
        k = cap / projected
        out.loc[idx, "threes_conserve_mult"] = round(k, 4)
        att = pd.to_numeric(out.loc[idx, "proj_3pa"], errors="coerce") * k
        lam = att * pd.to_numeric(out.loc[idx, "proj_3p_pct"], errors="coerce")
        # Not rounded, for the same reason the football model does not round
        # its scaled touchdowns: rounding nine players puts the team back over
        # the supply this pass exists to respect. Display rounds instead.
        out.loc[idx, "proj_3pa"] = att
        out.loc[idx, "proj_3pm"] = lam
        p = [p_at_least(v) for v in lam]
        out.loc[idx, "threes_prob_game"] = [round(v, 4) for v in p]
        out.loc[idx, "fair_odds"] = [_prob_to_american(v) for v in p]
    return out


def score_slate(df: pd.DataFrame) -> pd.DataFrame:
    """Score every rotation player, then make each team's attempts add up."""
    if df is None or df.empty:
        return df
    scored = df.apply(lambda r: pd.Series(score_row(r)), axis=1)
    out = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    return conserve_team_3pa(out)


def shooter_of_the_day(df: pd.DataFrame):
    if df is None or df.empty or "threes_prob_game" not in df.columns:
        return None
    return df.sort_values(["threes_prob_game", "threes_score"], ascending=False).iloc[0]
