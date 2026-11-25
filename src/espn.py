"""Shared ESPN public-API client for the football and basketball boards.

Baseball has MLB StatsAPI and Baseball Savant. Football and basketball have no
equivalent free, rich, keyless source — but ESPN's public site API carries the
three things the models actually need:

  * the slate, with **real DraftKings lines** (spread + total). For football
    the implied team total is the single biggest input to a touchdown
    projection, and it already contains everything the market knows about
    injuries and matchup.
  * venue, including an `indoor` flag, and the game's weather.
  * per-player season rates, which is what the role and volume terms run on.

Every call is best-effort. A failure returns None and the caller falls back to
the deterministic synthetic slate, exactly as the baseball pipeline does, so a
blocked network degrades the board instead of taking the page down.
"""

from __future__ import annotations

import json
import time as _time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

SITE = "https://site.api.espn.com/apis/site/v2/sports"
WEB = "https://site.web.api.espn.com/apis/common/v3/sports"
TIMEOUT = 25
MAX_WORKERS = 8

LEAGUE_PATH = {"nfl": "football/nfl", "nba": "basketball/nba"}

_FAIL_TTL = 300.0
_DIAG: dict = {}


def note_diag(source: str, msg) -> None:
    _DIAG[source] = str(msg)[:200]


def get_diagnostics() -> dict:
    return dict(_DIAG)


def _cache_ok(fn):
    """Memoize, but never remember a failure for long.

    Same guard the Statcast loaders use: one transient 5xx at boot must not
    pin an empty board for the life of the process.
    """
    store: dict = {}

    def wrapped(*args):
        hit = store.get(args)
        now = _time.time()
        if hit is not None:
            val, ts, ok = hit
            if ok or (now - ts) < _FAIL_TTL:
                return val
        val = fn(*args)
        ok = bool(val)
        store[args] = (val, now, ok)
        if ok:
            _DIAG.pop(fn.__name__, None)
        return val

    wrapped.cache_clear = store.clear
    wrapped.__name__ = fn.__name__
    return wrapped


def fetch_json(url: str, params: dict | None = None):
    """GET and parse, or None. Never raises — the caller degrades instead."""
    if params:
        url = f"{url}?{urllib.parse.urlencode(params)}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))
    except Exception as exc:
        note_diag(url.split("?")[0].rsplit("/", 2)[-1], exc)
        return None


def fetch_many(urls: list[str]) -> list:
    """Fetch in parallel. Per-player season stats are one call each, and a full
    slate is a few hundred — serially that is minutes, threaded it is seconds."""
    if not urls:
        return []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        return list(ex.map(fetch_json, urls))


def _num(v, default=None):
    """ESPN ships numbers as strings, with commas, dashes and '--' for empty."""
    if v is None:
        return default
    if isinstance(v, (int, float)):
        return default if v != v else float(v)
    s = str(v).strip().replace(",", "")
    if s in ("", "--", "-", "None"):
        return default
    try:
        return float(s)
    except ValueError:
        return default


@_cache_ok
def scoreboard(league: str, date_iso: str) -> tuple:
    """Games for a date, with lines, venue and weather. Tuple of dicts."""
    path = LEAGUE_PATH.get(league)
    if not path:
        return ()
    yyyymmdd = date_iso.replace("-", "")
    data = fetch_json(f"{SITE}/{path}/scoreboard", {"dates": yyyymmdd, "limit": 100})
    if not data or not data.get("events"):
        return ()
    out = []
    for ev in data["events"]:
        comps = ev.get("competitions") or []
        if not comps:
            continue
        c = comps[0]
        teams = {}
        for comp in c.get("competitors", []):
            t = comp.get("team") or {}
            teams[comp.get("homeAway")] = {
                "abbr": t.get("abbreviation"), "id": t.get("id"),
                "name": t.get("displayName"),
            }
        if "home" not in teams or "away" not in teams:
            continue
        odds = (c.get("odds") or [{}])[0]
        total = _num(odds.get("overUnder"))
        spread = _num(odds.get("spread"))        # home-relative; negative = home favored
        venue = c.get("venue") or {}
        weather = ev.get("weather") or {}
        status = ((c.get("status") or {}).get("type") or {}).get("name", "")
        out.append({
            "event_id": ev.get("id"),
            "game": f"{teams['away']['abbr']} @ {teams['home']['abbr']}",
            "start": ev.get("date"),
            "status": status,
            "home": teams["home"], "away": teams["away"],
            "game_total": total, "spread": spread,
            # The market's own split of the total. For football this is the
            # biggest single input into a touchdown projection.
            "home_implied": (round(total / 2 - spread / 2, 2)
                             if total is not None and spread is not None else None),
            "away_implied": (round(total / 2 + spread / 2, 2)
                             if total is not None and spread is not None else None),
            "venue": venue.get("fullName"),
            "indoor": bool(venue.get("indoor")),
            "temp_f": _num(weather.get("temperature")),
            "weather": weather.get("displayValue"),
        })
    return tuple(out)


@_cache_ok
def roster(league: str, team_id: str) -> tuple:
    """Players on a team, flattened out of ESPN's position groups.

    Anyone listed as out, injured-reserve or suspended is dropped here rather
    than downstream — a board that projects a touchdown for a player who is
    not going to dress is worse than one that omits him.
    """
    path = LEAGUE_PATH.get(league)
    if not path or not team_id:
        return ()
    data = fetch_json(f"{SITE}/{path}/teams/{team_id}/roster")
    if not data:
        return ()
    groups = data.get("athletes") or []
    out = []
    for grp in groups:
        label = str(grp.get("position", "")).lower() if isinstance(grp, dict) else ""
        if label in ("injuredreserveorout", "suspended", "practicesquad"):
            continue
        items = grp.get("items", []) if isinstance(grp, dict) and "items" in grp else [grp]
        for a in items:
            if not isinstance(a, dict):
                continue
            pos = ((a.get("position") or {}).get("abbreviation") or "").upper()
            status = ((a.get("status") or {}).get("type") or "").lower()
            injuries = a.get("injuries") or []
            hurt = any(str((i or {}).get("status", "")).lower()
                       in ("out", "injured reserve", "doubtful") for i in injuries)
            if status in ("inactive", "injured") or hurt:
                continue
            out.append({"id": a.get("id"), "name": a.get("displayName"),
                        "position": pos, "jersey": a.get("jersey")})
    return tuple(out)


def athlete_stats_urls(league: str, athlete_ids) -> list[str]:
    path = LEAGUE_PATH.get(league)
    return [f"{WEB}/{path}/athletes/{i}/stats" for i in athlete_ids] if path else []


def parse_stat_categories(payload) -> dict:
    """Flatten ESPN's {names: [...], statistics: [{stats: [...]}]} into a dict.

    Keys come back prefixed by category ("rushing.rushingAttempts") because the
    same short name appears in more than one category — `gamesPlayed` is in all
    of them, and reading the wrong one silently rescales every per-game rate.
    """
    out: dict = {}
    if not isinstance(payload, dict):
        return out
    for cat in payload.get("categories") or []:
        cname = cat.get("name") or ""
        names = cat.get("names") or []
        blocks = cat.get("statistics") or []
        stats = (blocks[0].get("stats") if blocks else None) or []
        for n, v in zip(names, stats):
            out[f"{cname}.{n}"] = v
    return out
