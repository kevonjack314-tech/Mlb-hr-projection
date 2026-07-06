"""ULX prop ladder — bet-type suitability, cash probabilities, and decision tree.

Instilled from the ULX syllabus / betting-syllabus infographics:

THE BETTING PYRAMID ("don't get stuck on HRs"): a smart card is ~10% home runs,
~20% extra-base hits, ~25% total bases, ~25% stolen bases, ~20% volume
(runs + RBIs + hits). "Volume is king" at the base; HRs are the high-risk top.

WHAT ACTUALLY CASHES (approx hit rates): Hits 70-80% · Total Bases 55-65% ·
Runs 50-60% · RBIs 45-55% · Stolen Bases 30-40% · Doubles 25-35% · HRs 8-15%.

BET-TYPE CHEAT SHEET (what drives each prop):
  HR  — barrel, hard-hit, launch angle, pull%, HR/FB, xSLG (weather/park/pitcher)
  2B  — gap power, line drives, speed+power, pull/oppo spray
  TB  — hits + power, extra-base upside, quality contact, lineup position
  SB  — sprint speed, on-base ability, lineup spot
  R   — top of lineup, high OBP, power behind him
  RBI — middle of order (RISP traffic), power, high slug

THE DECISION TREE: elite HR profile? → HR. Hits doubles? → 2B. On base often?
→ Runs. Has speed? → SB. High contact / hard hit? → Hits/TB. Else → PASS.

LINEUP-SPOT ROLES (what managers look for): 1 leadoff (OBP+speed), 2 table
setter, 3 run producer (best all-around), 4 power, 5 secondary power,
6 flex, 7 lower-order threat (underrated pop), 8 defense, 9 table re-setter.
Runs live at the top of the order; RBIs in the 3-5 traffic; the ULX "hidden
value zone" is 7-9.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .odds import prob_to_american

BET_TYPES = ["HR", "TB", "H", "R", "RBI", "2B", "SB"]

BET_LABEL = {
    "HR": "💣 Home Run", "TB": "🟩 Total Bases (2+)", "H": "🟦 Hit (1+)",
    "R": "🔷 Run Scored", "RBI": "🟪 RBI", "2B": "🟧 Double", "SB": "🏃 Stolen Base",
}

# Approx league cash rates from the ULX pyramid (midpoints).
BASE_CASH_RATE = {"H": 0.72, "TB": 0.58, "R": 0.55, "RBI": 0.50,
                  "SB": 0.35, "2B": 0.30}
# Ceilings so estimates stay honest.
PROB_CAP = {"H": 0.85, "TB": 0.76, "R": 0.72, "RBI": 0.66, "SB": 0.55, "2B": 0.45}

# Lineup-spot fit per prop (manager-roles infographic): Runs at the top,
# RBIs in the 3-5 traffic, SB for table-setters, volume everywhere up top.
_SPOT_FIT = {
    "R":   {1: 1.0, 2: 0.95, 3: 0.75, 4: 0.6, 5: 0.5, 6: 0.45, 7: 0.4, 8: 0.35, 9: 0.5},
    "RBI": {1: 0.35, 2: 0.5, 3: 0.9, 4: 1.0, 5: 0.9, 6: 0.6, 7: 0.5, 8: 0.4, 9: 0.3},
    "SB":  {1: 1.0, 2: 0.85, 3: 0.5, 4: 0.3, 5: 0.3, 6: 0.5, 7: 0.6, 8: 0.55, 9: 0.7},
    "TB":  {1: 0.85, 2: 0.9, 3: 1.0, 4: 1.0, 5: 0.9, 6: 0.75, 7: 0.65, 8: 0.55, 9: 0.5},
    "H":   {1: 1.0, 2: 1.0, 3: 0.95, 4: 0.9, 5: 0.85, 6: 0.8, 7: 0.75, 8: 0.7, 9: 0.7},
    "2B":  {1: 0.8, 2: 0.9, 3: 0.95, 4: 0.9, 5: 0.9, 6: 0.8, 7: 0.75, 8: 0.65, 9: 0.6},
}


def _norm(row, key, lo, hi, default=0.5):
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return default
    if np.isnan(v):
        return default
    return float(np.clip((v - lo) / (hi - lo), 0.0, 1.0))


def _spot_fit(row, bet):
    try:
        s = int(row.get("lineup_spot"))
    except (TypeError, ValueError):
        return 0.6
    return _SPOT_FIT.get(bet, {}).get(s, 0.6)


def suitability(row: pd.Series) -> dict:
    """0-100 fit for each bet type, from the ULX cheat-sheet drivers."""
    out = {}
    # Hits: high contact + high OBP-ish + hard hit.
    out["H"] = 100 * (0.32 * _norm(row, "contact_pct", 68, 88)
                      + 0.22 * _norm(row, "hard_hit_pct", 30, 55)
                      + 0.20 * _norm(row, "zone_contact_pct", 78, 93)
                      + 0.16 * _norm(row, "xwoba", 0.290, 0.400)
                      + 0.10 * _spot_fit(row, "H"))
    # Total bases: hits + power, extra-base upside, lineup position.
    out["TB"] = 100 * (0.28 * _norm(row, "xslg", 0.330, 0.560)
                       + 0.22 * _norm(row, "barrel_pct", 3, 18)
                       + 0.16 * _norm(row, "xwoba", 0.290, 0.400)
                       + 0.14 * _norm(row, "ld_pct", 15, 27)
                       + 0.20 * _spot_fit(row, "TB"))
    # Doubles: gap power (ISO w/o pure HR loft), line drives, speed.
    out["2B"] = 100 * (0.30 * _norm(row, "ld_pct", 15, 27)
                       + 0.25 * _norm(row, "iso", 0.090, 0.280)
                       + 0.20 * _norm(row, "sprint_speed", 25.5, 29.5)
                       + 0.15 * _norm(row, "hard_hit_pct", 30, 55)
                       + 0.10 * _spot_fit(row, "2B"))
    # Runs: top of order + gets on base + speed (power behind him ≈ spot fit).
    out["R"] = 100 * (0.40 * _spot_fit(row, "R")
                      + 0.35 * _norm(row, "xwoba", 0.290, 0.400)
                      + 0.25 * _norm(row, "sprint_speed", 25.5, 29.5))
    # RBIs: middle-order traffic + slug.
    out["RBI"] = 100 * (0.45 * _spot_fit(row, "RBI")
                        + 0.35 * _norm(row, "xslg", 0.330, 0.560)
                        + 0.20 * _norm(row, "hr_fb", 6, 22))
    # Stolen bases: sprint speed gates everything.
    speed = _norm(row, "sprint_speed", 26.5, 30.0, default=0.3)
    sb = 100 * (0.60 * speed + 0.25 * _norm(row, "xwoba", 0.290, 0.400)
                + 0.15 * _spot_fit(row, "SB"))
    if speed < 0.25:            # not a runner — SB prop is dead
        sb *= 0.3
    out["SB"] = sb
    # HR: the ULX power checklist already grades this (ulx_score 0-100).
    v = row.get("ulx_score")
    out["HR"] = float(v) if pd.notna(v) else 50.0
    return {k: round(float(np.clip(v, 0, 100)), 1) for k, v in out.items()}


def cash_prob(bet: str, suit: float, row: pd.Series) -> float:
    """Estimated cash probability: ULX baseline scaled by suitability."""
    if bet == "HR":
        p = row.get("hr_prob_game")
        return float(p) if pd.notna(p) else 0.10
    base = BASE_CASH_RATE[bet]
    p = base * (0.62 + 0.62 * suit / 100.0)
    return float(np.clip(p, 0.02, PROB_CAP[bet]))


def best_bet(row: pd.Series, suits: dict) -> tuple[str, str]:
    """The ULX decision tree: HR → 2B → Runs → SB → Hits/TB → Pass."""
    checks = row.get("ulx_checks")
    if pd.notna(checks) and checks >= 7:
        return "HR", f"Elite HR profile ({int(checks)}/9 ULX checks)"
    if suits["2B"] >= 62:
        return "2B", "Gap power + line drives — doubles machine"
    spot_ok = _spot_fit(row, "R") >= 0.75
    if spot_ok and _norm(row, "xwoba", 0.290, 0.400) >= 0.55:
        return "R", "Top of the order and gets on base"
    if _norm(row, "sprint_speed", 26.5, 30.0, default=0.0) >= 0.65 and suits["SB"] >= 55:
        return "SB", "Real speed — stolen-base threat"
    if suits["H"] >= 60 or suits["TB"] >= 62:
        return "TB" if suits["TB"] >= suits["H"] else "H", "High contact / hard hit — volume cashes"
    return "PASS", "No clear edge — pass the player"


def attach_props(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-bet-type suitability, est. cash prob/odds, and the ULX best bet."""
    if df is None or df.empty:
        return df
    df = df.copy()
    suit_cols = {b: [] for b in BET_TYPES}
    prob_cols = {b: [] for b in BET_TYPES}
    best, why = [], []
    for _, row in df.iterrows():
        s = suitability(row)
        for b in BET_TYPES:
            suit_cols[b].append(s[b])
            prob_cols[b].append(round(cash_prob(b, s[b], row), 3))
        bb, reason = best_bet(row, s)
        best.append(bb)
        why.append(reason)
    for b in BET_TYPES:
        df[f"suit_{b}"] = suit_cols[b]
        df[f"prob_{b}"] = prob_cols[b]
        df[f"odds_{b}"] = [prob_to_american(p) for p in prob_cols[b]]
    df["best_bet"] = best
    df["best_bet_reason"] = why
    return df


# The ULX pyramid mix for a 5-leg "ladder" ticket (top → base).
LADDER_COMPOSITION = [
    ("HR", "suit_HR"), ("2B", "suit_2B"), ("TB", "suit_TB"),
    ("SB", "suit_SB"), ("R", "suit_R"),
]


def build_ladder_parlay(df: pd.DataFrame, n_legs: int = 5) -> dict:
    """A mixed-prop ticket per the ULX pyramid — one leg per bet type, different
    games ("never bet the same thing in every game / don't chase just homers")."""
    if df is None or df.empty:
        return {"legs": pd.DataFrame(), "summary": {}}
    comp = LADDER_COMPOSITION[:max(2, min(n_legs, len(LADDER_COMPOSITION)))]
    used_games, used_players, legs = set(), set(), []
    for bet, col in comp:
        pool = df[(~df["player"].isin(used_players)) & (~df["game"].isin(used_games))]
        if pool.empty:
            pool = df[~df["player"].isin(used_players)]
        if pool.empty:
            continue
        pick = pool.sort_values(col, ascending=False).iloc[0]
        d = pick.to_dict()
        d["bet"] = bet
        d["bet_prob"] = float(pick[f"prob_{bet}"])
        d["bet_odds"] = int(pick[f"odds_{bet}"])
        d["bet_suit"] = float(pick[col])
        legs.append(d)
        used_games.add(pick["game"])
        used_players.add(pick["player"])
    legs_df = pd.DataFrame(legs)
    if legs_df.empty:
        return {"legs": legs_df, "summary": {}}
    combined_prob = float(legs_df["bet_prob"].prod())
    dec = float((1.0 / legs_df["bet_prob"]).prod())
    summary = {
        "n_legs": len(legs_df),
        "combined_prob": round(combined_prob * 100, 1),
        "combined_decimal": round(dec, 2),
        "combined_american": prob_to_american(combined_prob),
        "payout_per_10": round(10 * (dec - 1), 2),
    }
    return {"legs": legs_df, "summary": summary}
