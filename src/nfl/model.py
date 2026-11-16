"""Anytime-touchdown projection — football's home-run model.

Same shape as the baseball model, different physics. The baseball version
asks: how often does this bat turn a plate appearance into a home run, and how
many plate appearances does he get? The football version asks the same two
questions about a scarcer resource:

    expected team touchdowns  x  this player's share of them

A team scores roughly 2.3 offensive touchdowns a game, so unlike home runs the
supply is genuinely fixed and shared. That makes ROLE the dominant term, far
more than talent: a backup running back who gets the carries inside the five
is a better touchdown bet than a star receiver who does his work between the
twenties. This is the single thing casual touchdown betting gets wrong, and
the model is built around it.

Touchdown supply is converted to a probability the same way the HR model turns
a per-PA rate into a per-game one — Poisson on the expected count, so a player
with 0.5 expected touchdowns is 39% rather than 50%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .teams import pass_weather_multiplier

# A team scores about this many OFFENSIVE touchdowns per game (defensive and
# special-teams scores are not available to these props and are excluded).
LEAGUE_TEAM_TDS = 2.30
# An implied team total of this many points corresponds to that league average.
LEAGUE_TEAM_TOTAL = 22.5
# Points per offensive touchdown drive once extra points are folded in; the
# rest of a team's points come from field goals.
TD_SHARE_OF_POINTS = 0.71

# Reference ranges for 0-100 sub-scores, by position where it matters.
REF = {
    "rz_tgt_share": (0.05, 0.30),    # red-zone target share
    "inside5_share": (0.05, 0.65),   # share of carries inside the 5
    "snap_share": (0.35, 0.95),
    "tgt_share": (0.05, 0.30),
    "rush_share": (0.05, 0.70),
    "ypc": (3.2, 5.6),
    "yprr": (0.9, 2.6),              # yards per route run
    "td_rate_prior": (0.05, 0.55),   # season anytime-TD hit rate
}

# How much of a team's touchdowns each position group takes, league-wide. The
# model starts here and moves off it on the player's actual role.
POSITION_PRIOR = {"RB": 0.34, "WR": 0.40, "TE": 0.16, "QB": 0.08, "FB": 0.02}

SCORE_WEIGHTS = {
    "role": 0.46,          # goal-line and red-zone usage — the dominant term
    "volume": 0.22,        # snaps, targets, carries
    "efficiency": 0.14,    # how much he does with them
    "matchup": 0.10,       # opponent TDs allowed to the position
    "team_offense": 0.08,  # how many touchdowns there are to share
}


def scale(value, lo: float, hi: float) -> float:
    """Min-max a value into 0-100 against fixed league references, so scores
    compare across weeks rather than only within one slate."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 50.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0) * 100.0)


def _num(v, default=None):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return default if f != f else f


def expected_team_tds(row: pd.Series) -> float:
    """Offensive touchdowns the player's team is expected to score.

    Uses the implied team total when the market gives one — that is the single
    best public estimate of how much scoring there is to share, and it already
    contains everything the market knows about injuries and matchup. Falls back
    to the league average when there is no line.
    """
    total = _num(row.get("implied_team_total"))
    if total is None:
        return LEAGUE_TEAM_TDS
    tds = (total * TD_SHARE_OF_POINTS) / 7.0
    return float(np.clip(tds, 0.6, 5.5))


def td_share(row: pd.Series) -> dict:
    """This player's share of his team's touchdowns, split rush vs receiving.

    Red-zone and inside-the-5 usage carry the weight because that is where
    touchdowns actually happen. A player's overall yardage share is a much
    weaker signal — plenty of productive players rarely see the goal line.
    """
    pos = str(row.get("position", "WR")).upper()
    prior = POSITION_PRIOR.get(pos, 0.15)

    rush_share = _num(row.get("rush_share"), 0.0) or 0.0
    inside5 = _num(row.get("inside5_share"))
    rz_tgt = _num(row.get("rz_tgt_share"))
    tgt_share = _num(row.get("tgt_share"), 0.0) or 0.0
    snap = _num(row.get("snap_share"), 0.6) or 0.6

    # Rushing touchdowns: inside-the-5 carries dominate. When that split isn't
    # available, overall rush share is a weak stand-in and is treated as such.
    if inside5 is not None:
        rush_td_share = 0.78 * inside5 + 0.22 * rush_share
    else:
        rush_td_share = 0.55 * rush_share
    # Receiving touchdowns: red-zone targets dominate for the same reason.
    if rz_tgt is not None:
        rec_td_share = 0.72 * rz_tgt + 0.28 * tgt_share
    else:
        rec_td_share = 0.60 * tgt_share

    # Split the team's touchdowns between the run and the pass. Teams score
    # roughly 45% of offensive touchdowns on the ground.
    rush_pool, pass_pool = 0.45, 0.55
    wx = pass_weather_multiplier(row.get("home_team"), row.get("wind_mph"),
                                 row.get("temp_f"))
    pass_pool *= wx["pass_mult"]
    rush_pool *= wx["rush_mult"]
    s = rush_pool + pass_pool
    rush_pool, pass_pool = rush_pool / s, pass_pool / s

    share = rush_pool * rush_td_share + pass_pool * rec_td_share
    # Shrink toward the positional prior when usage data is thin — a player we
    # know nothing about should land near his position's average, not at zero.
    have = sum(x is not None for x in (inside5, rz_tgt)) + (rush_share > 0) + (tgt_share > 0)
    trust = min(1.0, have / 3.0)
    share = trust * share + (1.0 - trust) * prior * snap

    return {"td_share": float(np.clip(share, 0.0, 0.75)),
            "rush_pool": round(rush_pool, 3), "pass_pool": round(pass_pool, 3),
            "weather_note": wx["weather_note"]}


def matchup_multiplier(row: pd.Series) -> tuple[float, float]:
    """Opponent adjustment: TDs allowed to the position, and red-zone defense."""
    pos = str(row.get("position", "WR")).upper()
    allowed = _num(row.get("opp_td_allowed_pos"))      # per game, to this position
    rz_rate = _num(row.get("opp_rz_td_rate"))          # opp red-zone TD% allowed
    base = {"RB": 0.78, "WR": 0.92, "TE": 0.38, "QB": 0.18}.get(pos, 0.6)
    mult = 1.0
    if allowed is not None and base > 0:
        mult *= float(np.clip(allowed / base, 0.82, 1.20))
    if rz_rate is not None:
        mult *= float(np.clip(1.0 + (rz_rate - 0.57) * 0.55, 0.88, 1.14))
    mult = float(np.clip(mult, 0.80, 1.22))
    return mult, scale(mult, 0.82, 1.20)


def score_row(row: pd.Series) -> dict:
    """Score one player. Returns the projection plus every sub-score behind it."""
    out: dict = {}
    pos = str(row.get("position", "WR")).upper()

    team_tds = expected_team_tds(row)
    sh = td_share(row)
    mult, matchup_score = matchup_multiplier(row)

    expected_tds = team_tds * sh["td_share"] * mult
    # Poisson: a player with 0.5 expected touchdowns scores at least one 39% of
    # the time, not 50%. Multi-TD games are exactly where the difference lives.
    p_td = 1.0 - float(np.exp(-max(expected_tds, 0.0)))
    p_td = float(np.clip(p_td, 0.004, 0.92))

    role_score = max(
        scale(row.get("inside5_share"), *REF["inside5_share"]),
        scale(row.get("rz_tgt_share"), *REF["rz_tgt_share"]),
    ) if (row.get("inside5_share") is not None or row.get("rz_tgt_share") is not None) \
        else 50.0
    volume_score = (0.5 * scale(row.get("snap_share"), *REF["snap_share"])
                    + 0.5 * max(scale(row.get("tgt_share"), *REF["tgt_share"]),
                                scale(row.get("rush_share"), *REF["rush_share"])))
    eff_score = (scale(row.get("ypc"), *REF["ypc"]) if pos in ("RB", "FB")
                 else scale(row.get("yprr"), *REF["yprr"]))
    team_score = scale(team_tds, 1.2, 3.6)

    td_score = (SCORE_WEIGHTS["role"] * role_score
                + SCORE_WEIGHTS["volume"] * volume_score
                + SCORE_WEIGHTS["efficiency"] * eff_score
                + SCORE_WEIGHTS["matchup"] * matchup_score
                + SCORE_WEIGHTS["team_offense"] * team_score)

    out.update({
        "expected_team_tds": round(team_tds, 2),
        "td_share": round(sh["td_share"], 4),
        "expected_tds": round(expected_tds, 3),
        "td_prob_game": round(p_td, 4),
        "td_score": round(float(np.clip(td_score, 0, 100)), 1),
        "role_score": round(role_score, 1),
        "volume_score": round(volume_score, 1),
        "efficiency_score": round(eff_score, 1),
        "matchup_score": round(matchup_score, 1),
        "matchup_mult": round(mult, 3),
        "team_offense_score": round(team_score, 1),
        "rush_pool": sh["rush_pool"], "pass_pool": sh["pass_pool"],
        "weather_note": sh["weather_note"],
        "fair_odds": _prob_to_american(p_td),
    })
    out["rationale"] = _rationale(row, out)
    return out


def _prob_to_american(p: float) -> int:
    p = float(np.clip(p, 0.005, 0.98))
    return int(round(-100 * p / (1 - p))) if p >= 0.5 else int(round(100 * (1 - p) / p))


def _rationale(row: pd.Series, out: dict) -> str:
    """Plain-English why, drawn only from inputs that actually moved this player."""
    bits = []
    i5 = _num(row.get("inside5_share"))
    if i5 is not None and i5 >= 0.35:
        bits.append(f"owns the goal line ({i5*100:.0f}% of carries inside the 5)")
    rz = _num(row.get("rz_tgt_share"))
    if rz is not None and rz >= 0.18:
        bits.append(f"red-zone target magnet ({rz*100:.0f}% share)")
    if out["expected_team_tds"] >= 3.0:
        bits.append(f"team projected for {out['expected_team_tds']:.1f} TDs")
    elif out["expected_team_tds"] <= 1.6:
        bits.append(f"low-scoring projection ({out['expected_team_tds']:.1f} team TDs)")
    if out["matchup_mult"] >= 1.12:
        bits.append("defense leaks TDs to the position")
    elif out["matchup_mult"] <= 0.88:
        bits.append("tough matchup for the position")
    if out.get("weather_note"):
        bits.append(out["weather_note"])
    snap = _num(row.get("snap_share"))
    if snap is not None and snap >= 0.85:
        bits.append(f"on the field for {snap*100:.0f}% of snaps")
    return "; ".join(bits[:5]).capitalize() if bits else ""


# Touchdowns scored by players NOT on the board — a third receiver, a fullback,
# a lineman on a tackle-eligible — plus the handful that never belong to an
# offensive skill player at all. The listed players share the rest.
LISTED_TD_CAP = 0.90


def conserve_team_tds(df: pd.DataFrame) -> pd.DataFrame:
    """Stop a team handing out more touchdowns than it is projected to score.

    Each player's share is estimated from his own usage, independently of his
    teammates — red-zone target share from one source, goal-line carries from
    another. Nothing makes those agree, and in practice they do not: before
    this pass the demo slate's nine listed players were splitting 1.1x to 1.6x
    of their own team's touchdown supply between them, which quietly inflated
    every projection on the board.

    Touchdowns are a fixed, shared resource, so they are normalised within a
    team: if the listed players' expected touchdowns exceed the team's supply
    (less a reserve for players who are not on the board), every one of them is
    scaled down by the same factor and the probability is recomputed. Players
    keep their ORDER — this changes the level, not the ranking.
    """
    if df is None or df.empty or "expected_tds" not in df.columns:
        return df
    if not {"game", "team", "expected_team_tds"} <= set(df.columns):
        return df
    out = df.copy()
    out["td_conserve_mult"] = 1.0
    for (_, _), idx in out.groupby(["game", "team"], sort=False).groups.items():
        rows = out.loc[idx]
        supply = float(pd.to_numeric(rows["expected_team_tds"], errors="coerce").iloc[0])
        projected = float(pd.to_numeric(rows["expected_tds"], errors="coerce").sum())
        cap = supply * LISTED_TD_CAP
        if projected <= cap or projected <= 0:
            continue
        k = cap / projected
        out.loc[idx, "td_conserve_mult"] = round(k, 4)
        scaled = pd.to_numeric(out.loc[idx, "expected_tds"], errors="coerce") * k
        # Deliberately NOT rounded: rounding nine players to 3dp puts the team
        # back over its own supply, which is the invariant this pass exists to
        # hold. Display rounds; the model keeps full precision.
        out.loc[idx, "expected_tds"] = scaled
        out.loc[idx, "td_share"] = (pd.to_numeric(out.loc[idx, "td_share"],
                                                  errors="coerce") * k)
        p = (1.0 - np.exp(-scaled.clip(lower=0.0))).clip(0.004, 0.92)
        out.loc[idx, "td_prob_game"] = p.round(4)
        out.loc[idx, "fair_odds"] = [_prob_to_american(v) for v in p]
    return out


def score_slate(df: pd.DataFrame) -> pd.DataFrame:
    """Score every player on the slate. Returns the frame with projections added.

    Row scoring is independent per player; the team conservation pass that
    follows is what makes the board add up. Use this rather than `score_row`
    directly unless you specifically want one player's unconstrained estimate.
    """
    if df is None or df.empty:
        return df
    scored = df.apply(lambda r: pd.Series(score_row(r)), axis=1)
    out = pd.concat([df.reset_index(drop=True), scored.reset_index(drop=True)], axis=1)
    return conserve_team_tds(out)


def td_of_the_day(df: pd.DataFrame):
    """The single touchdown pick the model likes most."""
    if df is None or df.empty or "td_prob_game" not in df.columns:
        return None
    return df.sort_values(["td_prob_game", "td_score"], ascending=False).iloc[0]
