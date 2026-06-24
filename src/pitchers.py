"""Probable-pitcher HR-allowed profile by batting-order spot (last N games).

For each probable starter we look at the home runs they've *allowed* over their
last ~5 games and break them down by the **lineup spot** of the batter who took
them deep. That tells you which order positions tend to punish this pitcher — a
direct, actionable matchup signal (target those spots in parlays).

  • LIVE: `statcast_pitcher` for the trailing window → pick the pitcher's last N
    game_pks → HR-allowed events → the batter's lineup spot from that game's box
    score (sources.fetch_batting_order_map) → counts per spot.
  • OFFLINE: a deterministic per-pitcher distribution weighted toward the middle
    of the order, so the view works without network.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from functools import lru_cache

import numpy as np

# League-ish shape of HR-allowed by spot (middle order does the most damage).
_SPOT_HR_WEIGHTS = {1: 0.09, 2: 0.11, 3: 0.14, 4: 0.15, 5: 0.13,
                    6: 0.11, 7: 0.10, 8: 0.09, 9: 0.08}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


def _demo_hr_by_spot(name: str, end_date_iso: str, n_games: int) -> tuple[dict, int, int]:
    rng = np.random.default_rng(_seed(name or "P", end_date_iso, "pithr"))
    total = int(rng.integers(2, 9))            # HRs allowed over the window
    spots = list(range(1, 10))
    w = np.array([_SPOT_HR_WEIGHTS[s] for s in spots], dtype=float)
    counts = rng.multinomial(total, w / w.sum())
    return {s: int(c) for s, c in zip(spots, counts)}, n_games, total


@lru_cache(maxsize=256)
def _live_hr_by_spot(pitcher_id: int, end_date_iso: str, n_games: int):
    try:
        from . import sources as src_mod
        from . import statcast as sc_mod
        if not sc_mod.is_available() or not pitcher_id:
            return None
        import pandas as pd
        import pybaseball as pyb

        end = dt.date.fromisoformat(end_date_iso)
        start = end - dt.timedelta(days=45)
        df = pyb.statcast_pitcher(start.isoformat(), end.isoformat(), pitcher_id)
        if df is None or df.empty or "game_pk" not in df.columns:
            return None
        df = df.copy()
        df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce").dt.date
        # The pitcher's last N games (by date) in the window.
        gd = df.dropna(subset=["game_pk"]).groupby("game_pk")["game_date"].max().sort_values()
        last_games = list(gd.index[-n_games:])
        sub = df[df["game_pk"].isin(last_games)]
        hr = sub[sub["events"] == "home_run"]
        counts = {s: 0 for s in range(1, 10)}
        for _, r in hr.iterrows():
            order = dict(src_mod.fetch_batting_order_map(r.get("game_pk")))
            spot = order.get(r.get("batter"))
            if spot:
                counts[spot] += 1
        return counts, len(last_games), int(len(hr))
    except Exception:
        return None


def pitcher_recent_hr_by_spot(pitcher_id, name: str, end_date_iso: str,
                              n_games: int = 5, prefer_live: bool = True):
    """Return (counts_by_spot: dict, games: int, total_hr: int, source: str)."""
    if prefer_live and pitcher_id:
        live = _live_hr_by_spot(int(pitcher_id), end_date_iso, n_games)
        if live is not None:
            return live[0], live[1], live[2], "LIVE"
    counts, n, total = _demo_hr_by_spot(name, end_date_iso, n_games)
    return counts, n, total, "modeled"


def hottest_spots(counts: dict, top: int = 2) -> list[int]:
    """The lineup spots that take this pitcher deep most (for a one-line read)."""
    if not counts or sum(counts.values()) == 0:
        return []
    return [s for s, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top]
            if counts[s] > 0]
