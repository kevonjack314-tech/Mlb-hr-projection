"""HR prop odds: live (The Odds API) with a model-implied fallback.

Two odds concepts per hitter:
  • **Fair odds** — the vig-free price implied by the model's game HR probability
    (already computed in model.py as `fair_odds`).
  • **Book odds** — what you'd actually bet. LIVE from a sportsbook feed when an
    `ODDS_API_KEY` is configured and reachable; otherwise a *model-implied market
    price* = the fair price shaded by a typical HR-prop hold, so the app always
    shows realistic, sortable odds.

`edge_pct` = model HR% − book-implied HR%. With model-implied odds it sits around
−hold (the vig you'd pay); with real book odds it can go positive — that's a +EV
spot the parlay tools surface.

Live source: The Odds API (`api.the-odds-api.com`), market `batter_home_runs`.
Set the key via the `ODDS_API_KEY` environment variable and allowlist the host.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

from .statcast import normalize_name

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
DEFAULT_HOLD = 0.10  # typical HR-prop hold used for model-implied book odds
TIMEOUT = 12


# --------------------------------------------------------------------------- #
# Odds math
# --------------------------------------------------------------------------- #
def american_to_decimal(a: float) -> float:
    a = float(a)
    return 1.0 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(d: float) -> int:
    d = float(d)
    if d <= 1.0:
        return -100000
    return int(round((d - 1.0) * 100.0)) if d >= 2.0 else int(round(-100.0 / (d - 1.0)))


def american_to_prob(a: float) -> float:
    a = float(a)
    return 100.0 / (a + 100.0) if a > 0 else (-a) / (-a + 100.0)


def prob_to_american(p: float) -> int:
    p = float(np.clip(p, 1e-4, 0.9999))
    return int(round(-100.0 * p / (1.0 - p))) if p >= 0.5 else int(round(100.0 * (1.0 - p) / p))


def model_market_odds(prob: float, hold: float = DEFAULT_HOLD) -> int:
    """Shade a true probability into a realistic book price (worse than fair)."""
    market_prob = float(np.clip(prob * (1.0 + hold), 0.01, 0.97))
    return prob_to_american(market_prob)


def format_american(a) -> str:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "—"
    a = int(a)
    return f"+{a}" if a > 0 else str(a)


# --------------------------------------------------------------------------- #
# Live odds (The Odds API)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=8)
def fetch_live_hr_odds(date_iso: str) -> dict:
    """{normalized_player_name: {'odds': american, 'book': name}} or {} on failure.

    Keeps the best (longest) available price per player across books. Requires an
    ODDS_API_KEY env var; returns {} if missing, blocked, or unparled.
    """
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return {}
    try:
        ev = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events",
            params={"apiKey": key, "dateFormat": "iso"}, timeout=TIMEOUT,
        )
        ev.raise_for_status()
        events = ev.json()
    except Exception:
        return {}

    result: dict = {}
    for e in events:
        if not str(e.get("commence_time", "")).startswith(date_iso):
            continue
        try:
            o = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{e['id']}/odds",
                params={"apiKey": key, "regions": "us",
                        "markets": "batter_home_runs", "oddsFormat": "american"},
                timeout=TIMEOUT,
            )
            if o.status_code != 200:
                continue
            data = o.json()
        except Exception:
            continue
        for bm in data.get("bookmakers", []):
            for mk in bm.get("markets", []):
                if mk.get("key") != "batter_home_runs":
                    continue
                for out in mk.get("outcomes", []):
                    # "Over"/"Yes" is the to-hit-a-HR side; player is in description.
                    side = str(out.get("name", "")).lower()
                    if side in ("under", "no"):
                        continue
                    player = out.get("description") or out.get("name")
                    price = out.get("price")
                    if player is None or price is None:
                        continue
                    nk = normalize_name(player)
                    prev = result.get(nk)
                    if prev is None or price > prev["odds"]:
                        result[nk] = {"odds": int(price), "book": bm.get("title", "book")}
    return result


def attach_odds(df: pd.DataFrame, date_iso: str, use_live: bool = True) -> pd.DataFrame:
    """Add book_odds / odds_source / implied_prob / edge_pct to a scored slate."""
    df = df.copy()
    live = fetch_live_hr_odds(date_iso) if use_live else {}

    book, source = [], []
    for _, row in df.iterrows():
        nk = normalize_name(row["player"])
        hit = live.get(nk)
        if hit:
            book.append(hit["odds"])
            source.append(f"LIVE · {hit['book']}")
        else:
            book.append(model_market_odds(row["hr_prob_game"]))
            source.append("model")
    df["book_odds"] = book
    df["odds_source"] = source
    df["implied_prob"] = df["book_odds"].map(american_to_prob)
    df["edge_pct"] = ((df["hr_prob_game"] - df["implied_prob"]) * 100.0).round(1)
    df["odds_is_live"] = df["odds_source"].str.startswith("LIVE")
    return df
