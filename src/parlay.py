"""HR parlay generator — the ULX role-based formula.

Build parlays with ROLES, not names (per the ULX playbook):

  • ANCHOR        — highest-confidence HR bat (best HR Score among realistic-odds
                    plays). Usually ~ +250 to +600.
  • VALUE BAT     — good profile / underpriced (high Sneaky + edge, mid odds).
                    Usually ~ +450 to +950.
  • DEEP LONGSHOT — overlooked high-ceiling bat (best Longshot score, long odds).
                    Usually ~ +850 to +2000.

Composition by number of legs (1-5):
  1: Anchor
  2: Anchor + Value
  3: Anchor + Value + Longshot          (the canonical ULX ticket)
  4: Anchor + Value + Value + Longshot
  5: Anchor + Value + Value + Longshot + Longshot

Diversification rules (straight from the infographic's "what to avoid"):
  - no two legs from the same game (avoid stacking one environment),
  - prefer different archetypes (not all pull/loft, not all raw-power),
  - don't stack three chalk bombs or three pure longshots — the role mix does this.

Each ticket gets a 10-point checklist and a green/yellow/red light, plus combined
odds, model true probability, and EV computed from the book odds in odds.py.
"""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

from .lineup import spot_role_fit
from .tuning import role_prob_factor
from .odds import (
    american_to_decimal,
    american_to_prob,
    decimal_to_american,
    format_american,
)

# Probability bands -> role (aligned to the infographic's odds ranges).
ANCHOR_MIN_PROB = 0.15      # ~ +560 or shorter
VALUE_MIN_PROB = 0.10       # ~ +900 .. +560
LONGSHOT_MIN_PROB = 0.045   # ~ +2000 .. +900

COMPOSITIONS = {
    1: ["Anchor"],
    2: ["Anchor", "Value"],
    3: ["Anchor", "Value", "Longshot"],
    4: ["Anchor", "Value", "Value", "Longshot"],
    5: ["Anchor", "Value", "Value", "Longshot", "Longshot"],
}

ROLE_EMOJI = {"Anchor": "⚓", "Value": "💰", "Longshot": "🚀"}


def assign_role(prob: float) -> str | None:
    if prob >= ANCHOR_MIN_PROB:
        return "Anchor"
    if prob >= VALUE_MIN_PROB:
        return "Value"
    if prob >= LONGSHOT_MIN_PROB:
        return "Longshot"
    return None


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
    Value 6-7, Longshot 7-9 — per the ULX playbook) and a recurring-history bonus
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
    # ULX power checklist: "profiles win parlays" — reward green checks (0-9),
    # weighted most for Value/Longshot legs where the profile is the whole case.
    checks = row.get("ulx_checks")
    if pd.notna(checks):
        weight = 1.0 if role == "Anchor" else 1.6
        hist_bonus += float(checks) * weight
    if role == "Anchor":
        return float(row.get("hr_score", 0)) + spot_bonus + hist_bonus
    if role == "Value":
        return (float(row.get("sneaky_score", 0)) + 0.5 * float(row.get("edge_pct", 0))
                + spot_bonus + hist_bonus)
    return float(row.get("longshot_score", 0)) + spot_bonus + hist_bonus  # Longshot


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["role"] = df["hr_prob_game"].map(assign_role)
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


def generate_parlay(df: pd.DataFrame, n_legs: int = 3, strategy: str = "ulx",
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

    if strategy == "ulx":
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
    legs["role"] = legs["hr_prob_game"].map(assign_role).fillna("Leg")
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

    roles = set(legs["role"])
    archs = set(legs["archetype"])
    games = set(legs["game"])
    spots = set(legs["lineup_spot"].dropna()) if "lineup_spot" in legs else set()
    n = len(legs)
    checks = [
        ("Anchor identified", "Anchor" in roles),
        ("Value bat found", "Value" in roles or n == 1),
        ("Longshot selected", "Longshot" in roles or n < 3),
        ("Different archetypes", len(archs) >= min(n, 2)),
        ("Different games (no stacking)", len(games) == n),
        ("Different lineup spots", len(spots) >= min(n, 3) or len(spots) == n),
        ("Good matchups", legs["matchup_score"].mean() >= 52),
        ("Favorable HR environment", legs["env_score"].mean() >= 50),
        ("Reasonable odds (no crazy legs)", bool((legs["book_odds"] <= 2200).all())),
        ("Form / hard contact", legs.get("recent_form_score", pd.Series([50])).mean() >= 45
         or legs.get("hard_hit_score", pd.Series([50])).mean() >= 55),
        ("Makes sense (live model prob)", model_prob >= (0.03 if n >= 4 else 0.06)),
    ]
    passed = sum(1 for _, ok in checks if ok)
    total = len(checks)
    ratio = passed / total
    summary["checks_passed"] = passed
    summary["checks_total"] = total
    summary["light"] = "🟢 GREEN" if ratio >= 0.7 else ("🟡 YELLOW" if ratio >= 0.5 else "🔴 RED")
    return summary, checks
