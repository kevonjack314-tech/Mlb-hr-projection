"""Lineup-spot HR data — expected PA by spot, league context, and a recurring log.

Why lineup spot matters for HR:
  • **Plate appearances.** Top-of-order bats get more PAs per game (more swings =
    more HR chances). We turn the batting-order spot into an expected-PA figure
    that feeds the per-game HR probability directly.
  • **Role fit (ULX).** The infographic ties roles to spots — Anchor = middle
    order (3-5), Value = 6-7, Deep-Space Longshot = bottom (7-9). The parlay
    builder uses each bat's spot to grade role fit and diversify "different
    lineup spots".
  • **Player history by spot.** We track how many HRs each hitter has hit *from
    the spot they're batting today* (before the current date), as a contextual
    signal — surfaced in the tools and nudging parlay selection.

Recurring log: every time history is built we append one row per hitter-day
(date, player, team, lineup_spot, hr) to `data/lineup_hr_log.csv`, de-duped on
(date, player). The log therefore **accumulates across runs and as the date
advances** — run `scripts/update_lineup_log.py` (or the app) daily to keep it
current. Per-player and league HR-by-spot rates are aggregated from this log.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "lineup_hr_log.csv")

# Expected plate appearances by batting-order spot (top of order bats more).
EXPECTED_PA_BY_SPOT = {1: 4.6, 2: 4.5, 3: 4.4, 4: 4.3, 5: 4.1,
                       6: 4.0, 7: 3.9, 8: 3.8, 9: 3.7}
DEFAULT_PA = 4.1

# Relative HR-per-PA by spot (league context; ~1.0 average). The spot's own
# effect on HR *rate* is mild — most of the lineup-spot HR signal comes from PA
# volume and hitter quality — so this stays gentle and is used for display only.
LEAGUE_HR_REL_BY_SPOT = {1: 0.96, 2: 1.05, 3: 1.10, 4: 1.12, 5: 1.04,
                         6: 0.99, 7: 0.95, 8: 0.92, 9: 0.88}

# ULX role each spot fits best (used as a small parlay role-fit nudge).
SPOT_ROLE = {1: "Value", 2: "Anchor", 3: "Anchor", 4: "Anchor", 5: "Anchor",
             6: "Value", 7: "Value", 8: "Longshot", 9: "Longshot"}

# Maps a demo roster index (sluggers listed first) to a realistic batting-order
# spot: top bats hit 3-4, then 2-5-1, then 6-9. The 10th+ bat is a bench player
# (None) so it doesn't start. Real slates use the posted lineup instead.
_DEMO_SPOT_ORDER = [3, 4, 2, 5, 1, 6, 7, 8, 9]


def demo_spot_for_index(idx: int):
    """Realistic lineup spot for a demo roster index, or None for bench bats."""
    return _DEMO_SPOT_ORDER[idx] if 0 <= idx < len(_DEMO_SPOT_ORDER) else None


def expected_pa(spot) -> float:
    try:
        return EXPECTED_PA_BY_SPOT.get(int(spot), DEFAULT_PA)
    except (TypeError, ValueError):
        return DEFAULT_PA


def spot_role_fit(spot, role: str) -> float:
    """Small additive bonus (0-6) when a bat's spot suits the parlay role."""
    try:
        s = int(spot)
    except (TypeError, ValueError):
        return 0.0
    if role == "Anchor":
        return 6.0 if 3 <= s <= 5 else (3.0 if s == 2 else 0.0)
    if role == "Value":
        return 6.0 if s in (1, 6, 7) else 2.0
    if role == "Longshot":
        return 6.0 if s >= 8 else (3.0 if s == 7 else 0.0)
    return 0.0


# --------------------------------------------------------------------------- #
# Recurring HR-by-lineup-spot log
# --------------------------------------------------------------------------- #
def load_log() -> pd.DataFrame:
    if not os.path.exists(_LOG_PATH):
        return pd.DataFrame(columns=["date", "player", "team", "lineup_spot", "hr"])
    try:
        return pd.read_csv(_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=["date", "player", "team", "lineup_spot", "hr"])


def update_log_from_history(slate_hist: pd.DataFrame) -> int:
    """Append one row per hitter-day from a scored history slate. De-dupes on
    (date, player). Returns the number of new rows written. Recurring: call this
    each day to grow the log."""
    if slate_hist is None or slate_hist.empty:
        return 0
    need = {"date", "player", "lineup_spot"}
    if not need.issubset(slate_hist.columns):
        return 0
    hr_col = "hr_count" if "hr_count" in slate_hist.columns else (
        "hit_hr" if "hit_hr" in slate_hist.columns else None)
    new = slate_hist[["date", "player", "team", "lineup_spot"]].copy()
    new["hr"] = (slate_hist[hr_col].astype(float) if hr_col else 0.0)
    # Store the model's pre-game rating too, so the system can learn over time
    # which ratings actually homer (see learn.py).
    if "hr_score" in slate_hist.columns:
        new["hr_score"] = slate_hist["hr_score"].astype(float)
    existing = load_log()
    combined = pd.concat([existing, new], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "player"], keep="last")
    try:
        os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
        combined.to_csv(_LOG_PATH, index=False)
    except Exception:
        return 0
    return len(combined) - len(existing)


def _spot_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate HR and PA (games) by lineup spot from a log/history frame."""
    g = df.groupby("lineup_spot")
    out = pd.DataFrame({"games": g.size(), "hr": g["hr"].sum()})
    out["hr_per_game"] = (out["hr"] / out["games"]).replace([np.inf, -np.inf], 0).fillna(0)
    return out.reset_index()


def league_spot_table(slate_hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """League HR-by-spot, preferring the recurring log, falling back to history."""
    log = load_log()
    src = log if not log.empty else (slate_hist if slate_hist is not None else pd.DataFrame())
    if src is None or src.empty or "lineup_spot" not in src.columns:
        return pd.DataFrame()
    if "hr" not in src.columns:
        src = src.assign(hr=src.get("hr_count", src.get("hit_hr", 0)).astype(float))
    return _spot_table(src.dropna(subset=["lineup_spot"]))


def player_spot_hr(slate_hist: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per-(player, spot) HR totals from the recurring log (or history fallback).

    Columns: player, lineup_spot, games, hr, hr_per_game.
    """
    log = load_log()
    src = log if not log.empty else (slate_hist if slate_hist is not None else pd.DataFrame())
    if src is None or src.empty or {"player", "lineup_spot"}.issubset(src.columns) is False:
        return pd.DataFrame(columns=["player", "lineup_spot", "games", "hr", "hr_per_game"])
    if "hr" not in src.columns:
        src = src.assign(hr=src.get("hr_count", src.get("hit_hr", 0)).astype(float))
    g = src.dropna(subset=["lineup_spot"]).groupby(["player", "lineup_spot"])
    out = pd.DataFrame({"games": g.size(), "hr": g["hr"].sum()}).reset_index()
    out["hr_per_game"] = (out["hr"] / out["games"]).replace([np.inf, -np.inf], 0).fillna(0)
    return out


def attach_spot_signal(slate: pd.DataFrame, player_spot: pd.DataFrame) -> pd.DataFrame:
    """Add `spot_hr_at_current` (HRs the bat has hit from today's spot) and
    `spot_hr_rate` (per game) to the projection slate."""
    slate = slate.copy()
    if player_spot is None or player_spot.empty or "lineup_spot" not in slate.columns:
        slate["spot_hr_at_current"] = 0.0
        slate["spot_hr_rate"] = np.nan
        return slate
    key = player_spot.set_index(["player", "lineup_spot"])
    hrs, rates = [], []
    for _, row in slate.iterrows():
        try:
            rec = key.loc[(row["player"], int(row["lineup_spot"]))]
            hrs.append(float(rec["hr"]))
            rates.append(float(rec["hr_per_game"]))
        except (KeyError, TypeError, ValueError):
            hrs.append(0.0)
            rates.append(np.nan)
    slate["spot_hr_at_current"] = hrs
    slate["spot_hr_rate"] = rates
    return slate
