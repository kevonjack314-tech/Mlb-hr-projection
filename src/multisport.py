"""Run the shared parlay machinery over any sport's board.

`src/parlay.py` is 330 lines of logic that took a season of graded outcomes to
get right: role assignment, game and archetype diversification, the ULX
checklist, the learned per-role reliability factors, EV against the book. None
of it is actually about baseball — it is about picking uncorrelated legs from a
board of priced probabilities. Only the COLUMN NAMES are baseball.

So rather than forking it twice and letting three copies drift apart (the
leakage guard had to be written once; it should not have to be written three
times), each sport's board is translated into the canonical column names, run
through the real builder, and translated back. A fix to the parlay logic lands
in all three sports at once.
"""

from __future__ import annotations

import pandas as pd

from .sports import get_sport

# Per sport: the probability column, the headline score, and the column that
# stands in for "how good is this player in general", which drives role tiers.
COLUMN_MAP = {
    "mlb": {"prob": "hr_prob_game", "score": "hr_score", "tier_stat": "season_hr"},
    "nfl": {"prob": "td_prob_game", "score": "td_score", "tier_stat": "season_tds"},
    "nba": {"prob": "threes_prob_game", "score": "threes_score",
            "tier_stat": "fg3a_season"},
}

# What counts as a distinct archetype for diversification, per sport. Taking
# three bellcow running backs is the football version of taking three dead-pull
# lefties: one game script going wrong takes the whole ticket.
ARCHETYPE_COL = {"nfl": "archetype", "nba": "archetype"}

# Tier thresholds on the sport's tier_stat, mapping to Anchor / Value / Longshot.
TIER_THRESHOLDS = {
    "mlb": (18, 8),            # season HR
    "nfl": (6, 2),             # season TDs
    "nba": (300, 140),         # season 3PA — volume is the tier that matters
}


def role_bands(sport: str) -> dict:
    """Anchor / Value / Longshot probability cutoffs, scaled to the sport.

    The baseball bands (Anchor at 15%, Longshot floor at 4.5%) are tuned to an
    event that happens about 12% of the time. An anytime touchdown happens
    about twice as often and a 3+ threes hit nearly three times as often, so
    reused unchanged they call every player an Anchor — and worse, when the
    tier stat breaks the tie, they label the highest-probability bat on the
    board a Longshot. Scaling by the base rate keeps the roles meaning the
    same thing in each sport: Anchor is the safe leg, Longshot is the swing.
    """
    from .parlay import ANCHOR_MIN_PROB, LONGSHOT_MIN_PROB, VALUE_MIN_PROB
    spec = get_sport(sport)
    base = get_sport("mlb").base_rate or 0.12
    k = (spec.base_rate or base) / base
    return {"anchor": ANCHOR_MIN_PROB * k, "value": VALUE_MIN_PROB * k,
            "longshot": LONGSHOT_MIN_PROB * k}


def assign_roles(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Label each player Anchor / Value / Longshot on this sport's own bands."""
    if df is None or df.empty:
        return df
    b = role_bands(sport)
    p = pd.to_numeric(df[prob_col(sport)], errors="coerce")

    def _role(v):
        if v != v or v < b["longshot"]:
            return None
        if v >= b["anchor"]:
            return "Anchor"
        if v >= b["value"]:
            return "Value"
        return "Longshot"

    out = df.copy()
    out["role"] = [_role(v) for v in p]
    return out


def to_canonical(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Rename a sport's board into the columns the shared builder expects."""
    spec = COLUMN_MAP.get(sport, COLUMN_MAP["mlb"])
    out = df.copy()
    if sport == "mlb":
        return out
    ren = {}
    if spec["prob"] in out.columns:
        ren[spec["prob"]] = "hr_prob_game"
    if spec["score"] in out.columns:
        ren[spec["score"]] = "hr_score"
    out = out.rename(columns=ren)
    # The builder reads season_hr only to pick a tier band; feed it the
    # sport's own volume stat rescaled onto the same thresholds.
    stat = spec["tier_stat"]
    hi, lo = TIER_THRESHOLDS.get(sport, (18, 8))
    if stat in out.columns:
        v = pd.to_numeric(out[stat], errors="coerce")
        out["season_hr"] = v.map(
            lambda x: 20.0 if x == x and x >= hi else (
                10.0 if x == x and x >= lo else (3.0 if x == x else float("nan"))))
    # Scores the builder blends for the Value and Longshot roles. Without them
    # every non-Anchor leg would score 0 and the roles would pick at random.
    if "sneaky_score" not in out.columns:
        out["sneaky_score"] = out.get("hr_score", 50.0)
    if "longshot_score" not in out.columns:
        out["longshot_score"] = out.get("hr_score", 50.0)
    if "edge_pct" not in out.columns:
        out["edge_pct"] = 0.0
    # Diversification needs a lineup-spot-like field; use the depth-chart slot
    # where the sport has one so "different lineup spots" still means something.
    if "lineup_spot" not in out.columns:
        out["lineup_spot"] = float("nan")
    return out


def from_canonical(legs: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Put the sport's own column names back on the chosen legs."""
    if legs is None or legs.empty or sport == "mlb":
        return legs
    spec = COLUMN_MAP.get(sport, COLUMN_MAP["mlb"])
    ren = {"hr_prob_game": spec["prob"], "hr_score": spec["score"]}
    return legs.rename(columns={k: v for k, v in ren.items() if k in legs.columns})


# Typical hold a book takes on a player prop. Model-implied prices are shaded
# by this so a ticket built on them does not read as free money: without it
# every leg would price at its own fair odds and EV would come out at exactly
# zero, which is worse than useless — it looks like a real result.
PROP_HOLD = 0.06


def attach_model_odds(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Give a board model-implied prices when no live odds feed is available.

    Clearly labelled as model-implied, because the one thing that must never
    happen is a fallback being read as a market. Prices are the model's own
    probability shaded by a typical prop hold and clamped into the sport's
    realistic odds band, so they look like prices a book would actually hang.
    """
    from .odds import format_american  # noqa: F401  (kept for symmetry)
    if df is None or df.empty:
        return df
    spec = get_sport(sport)
    out = df.copy()
    p = pd.to_numeric(out[prob_col(sport)], errors="coerce").clip(0.005, 0.95)
    shaded = (p * (1.0 + PROP_HOLD)).clip(0.006, 0.96)
    dec = 1.0 / shaded

    def _amer(d: float) -> int:
        return int(round((d - 1) * 100)) if d >= 2 else -int(round(100 / (d - 1)))

    odds = [_amer(d) for d in dec]
    lo = min(b[0] for b in spec.tier_bands.values()) if spec.tier_bands else -400
    hi = max(b[1] for b in spec.tier_bands.values()) if spec.tier_bands else 2500
    out["book_odds"] = [int(min(max(o, lo), hi)) for o in odds]
    out["odds_source"] = "model-implied"
    out["odds_is_live"] = False
    book_dec = pd.Series([(1 + o / 100) if o > 0 else (1 + 100 / abs(o))
                          for o in out["book_odds"]], index=out.index)
    out["edge_pct"] = ((p * book_dec - 1.0) * 100).round(1)
    out["fair_odds_model"] = [_amer(1.0 / max(v, 1e-6)) for v in p]
    return out


def generate_parlay(df: pd.DataFrame, sport: str, n_legs: int = 3,
                    strategy: str = "ulx", **kwargs) -> dict:
    """Build a ticket for any sport using the shared, graded-record-tuned builder."""
    from .parlay import generate_parlay as _build
    if "book_odds" not in df.columns:
        df = attach_model_odds(df, sport)
    if sport != "mlb":
        df = assign_roles(df, sport)
    canon = to_canonical(df, sport)
    res = _build(canon, n_legs=n_legs, strategy=strategy, **kwargs)
    res["legs"] = from_canonical(res["legs"], sport)
    res["sport"] = sport
    return res


def prob_col(sport: str) -> str:
    return COLUMN_MAP.get(sport, COLUMN_MAP["mlb"])["prob"]


def score_col(sport: str) -> str:
    return COLUMN_MAP.get(sport, COLUMN_MAP["mlb"])["score"]


def build_slate(sport: str, date_iso: str, prefer_live: bool = False,
                with_notes: bool = False):
    """One entry point for every sport's board. Returns (scored_df, source).

    Every sport keeps the same guard: a synthetic slate is labelled DEMO in the
    source string, so modelled numbers can never be mistaken for real ones.
    """
    spec = get_sport(sport)
    if spec.key == "mlb":
        import datetime as dt

        from .model import score_slate
        from .sources import get_slate
        df, source, notes = get_slate(dt.date.fromisoformat(date_iso),
                                      prefer_live=prefer_live)
        scored = score_slate(df)
    elif spec.key == "nfl":
        from .nfl.model import score_slate
        from .nfl.sources import get_slate
        df, source, notes = get_slate(date_iso, prefer_live=prefer_live)
        scored = score_slate(df)
    else:
        from .nba.model import score_slate
        from .nba.sources import get_slate
        df, source, notes = get_slate(date_iso, prefer_live=prefer_live)
        scored = score_slate(df)
    return (scored, source, notes) if with_notes else (scored, source)
