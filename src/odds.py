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
from .trends import TIER_ODDS_BAND, tier_of

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


def tier_banded_market_odds(prob: float, season_hr=None, hold: float = DEFAULT_HOLD) -> int:
    """Model-implied book price, clamped into the player's tier band.

    Books don't price HR props off a pure model — stars sit ~+200 to +450,
    mid-tier bats (8–17 HR) ~+500 to +700, and under-the-radar bats +700 up.
    The model's shaded price is the starting point; the tier band is the
    reality check so an offline price never looks like something no book
    would ever hang.
    """
    price = model_market_odds(prob, hold)
    lo, hi = TIER_ODDS_BAND[tier_of(season_hr)]
    return int(min(max(price, lo), hi))


def format_american(a) -> str:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "—"
    a = int(a)
    return f"+{a}" if a > 0 else str(a)


# --------------------------------------------------------------------------- #
# Live odds (The Odds API)
# --------------------------------------------------------------------------- #
# prop -> (The Odds API market key, the standard Over line for our prop def).
# HR = Over 0.5 HRs · TB = Over 1.5 total bases (2+ TB) · H = Over 0.5 hits (1+).
PROP_MARKETS = {
    "HR": ("batter_home_runs", 0.5),
    "TB": ("batter_total_bases", 1.5),
    "H": ("batter_hits", 0.5),
}
_MARKET_TO_PROP = {mk: p for p, (mk, _pt) in PROP_MARKETS.items()}
_ALL_MARKETS = ",".join(mk for mk, _pt in PROP_MARKETS.values())


def parse_event_prop_odds(data: dict, result: dict) -> None:
    """Fold one event's bookmaker odds into result {prop: {name_key: {...}}}.

    Keeps the best (longest) Over/Yes price per player per prop, only at the
    standard line for that prop (e.g. TB Over exactly 1.5).
    """
    for bm in data.get("bookmakers", []):
        for mk in bm.get("markets", []):
            prop = _MARKET_TO_PROP.get(mk.get("key"))
            if prop is None:
                continue
            target = PROP_MARKETS[prop][1]
            for out in mk.get("outcomes", []):
                side = str(out.get("name", "")).lower()
                if side in ("under", "no"):
                    continue
                point = out.get("point")
                if point is not None and abs(float(point) - target) > 0.01:
                    continue          # alt line (e.g. TB 2.5) — skip
                player = out.get("description") or out.get("name")
                price = out.get("price")
                if player is None or price is None:
                    continue
                nk = normalize_name(player)
                prev = result[prop].get(nk)
                if prev is None or price > prev["odds"]:
                    result[prop][nk] = {"odds": int(price), "book": bm.get("title", "book")}


@lru_cache(maxsize=16)
def fetch_live_prop_odds(date_iso: str, markets: str | None = None) -> dict:
    """{prop: {name_key: {'odds', 'book'}}} for the requested markets.

    `markets` is a comma-separated Odds API market string; None = all three
    (HR + TB + Hits). Player-prop credits scale with markets requested, which is
    why the TB/Hits fetch is **opt-in per tab** — the everyday HR feed requests
    only `batter_home_runs`. Requires ODDS_API_KEY; empty maps on any failure.
    """
    req_markets = markets or _ALL_MARKETS
    result: dict = {p: {} for p in PROP_MARKETS}
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        return result
    try:
        ev = requests.get(
            f"{ODDS_API_BASE}/sports/baseball_mlb/events",
            params={"apiKey": key, "dateFormat": "iso"}, timeout=TIMEOUT,
        )
        ev.raise_for_status()
        events = ev.json()
    except Exception:
        return result

    for e in events:
        if not str(e.get("commence_time", "")).startswith(date_iso):
            continue
        try:
            o = requests.get(
                f"{ODDS_API_BASE}/sports/baseball_mlb/events/{e['id']}/odds",
                params={"apiKey": key, "regions": "us",
                        "markets": req_markets, "oddsFormat": "american"},
                timeout=TIMEOUT,
            )
            if o.status_code != 200:
                continue
            parse_event_prop_odds(o.json(), result)
        except Exception:
            continue
    return result


def fetch_live_hr_odds(date_iso: str) -> dict:
    """HR-only fetch (single market — the cheapest call; used by attach_odds)."""
    return fetch_live_prop_odds(date_iso, PROP_MARKETS["HR"][0]).get("HR", {})


def attach_prop_lines(df: pd.DataFrame, date_iso: str, use_live: bool = True) -> pd.DataFrame:
    """Overlay real TB / Hits prop lines onto the slate's estimated odds.

    Where a live line exists: odds_TB / odds_H are replaced with the real book
    price, odds_src_* records the book, and edge_*_pct = model cash prob − the
    book's implied prob (positive = +EV). Otherwise the modeled estimates stand.
    """
    df = df.copy()
    live = fetch_live_prop_odds(date_iso) if use_live else {p: {} for p in PROP_MARKETS}
    for prop in ("TB", "H"):
        if f"odds_{prop}" not in df.columns:
            continue
        odds_col, src_col, edge_col = [], [], []
        for _, row in df.iterrows():
            hit = live[prop].get(normalize_name(row["player"]))
            if hit:
                odds_col.append(hit["odds"])
                src_col.append(f"LIVE · {hit['book']}")
                edge_col.append(round(
                    (float(row.get(f"prob_{prop}", 0.0))
                     - american_to_prob(hit["odds"])) * 100, 1))
            else:
                odds_col.append(int(row[f"odds_{prop}"]))
                src_col.append("est")
                edge_col.append(np.nan)
        df[f"odds_{prop}"] = odds_col
        df[f"odds_src_{prop}"] = src_col
        df[f"edge_{prop}_pct"] = edge_col
    return df


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
            book.append(tier_banded_market_odds(row["hr_prob_game"], row.get("season_hr")))
            source.append("model")
    df["book_odds"] = book
    df["odds_source"] = source
    df["implied_prob"] = df["book_odds"].map(american_to_prob)
    df["edge_pct"] = ((df["hr_prob_game"] - df["implied_prob"]) * 100.0).round(1)
    df["odds_is_live"] = df["odds_source"].str.startswith("LIVE")
    return df
