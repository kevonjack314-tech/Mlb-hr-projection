"""Live data acquisition with graceful offline fallback.

Order of preference for a given date:
  1. LIVE: MLB StatsAPI (schedule, probable pitchers, active rosters) +
     Open-Meteo (weather) + optional Statcast leaderboards via pybaseball.
  2. DEMO: fully synthetic slate from `demo.py` (always works, no network).

Every public function is defensive: any network/parse failure downgrades the
affected piece (or the whole slate) to the synthetic path and records a note,
so the UI can always render something and tell the user where the data came
from. Statcast season metrics for hitters are the heaviest/most fragile pull;
when unavailable we attach deterministic synthetic profiles to the *real* games
so matchups/park/weather stay authentic while batted-ball quality is modeled.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache

import pandas as pd
import requests

from . import demo
from .parks import get_park, load_park_factors

STATSAPI = "https://statsapi.mlb.com/api/v1"
SCHEDULE_URL = STATSAPI + "/schedule"
TIMEOUT = 12

# MLB StatsAPI team id -> our park/team abbreviation.
_TEAM_ID_TO_ABBR = {
    109: "ARI", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC", 145: "CWS",
    113: "CIN", 114: "CLE", 115: "COL", 116: "DET", 117: "HOU", 118: "KC",
    108: "LAA", 119: "LAD", 146: "MIA", 158: "MIL", 142: "MIN", 121: "NYM",
    147: "NYY", 133: "ATH", 143: "PHI", 134: "PIT", 135: "SD", 137: "SF",
    136: "SEA", 138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}


def _get_json(url: str, params: dict | None = None) -> dict | None:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


@lru_cache(maxsize=64)
def fetch_schedule(date_iso: str) -> tuple:
    """Return a tuple of game dicts for the date, or () on failure.

    Each game: home/away abbr, probable pitcher ids+names+hands, game datetime,
    venue id. Cached by date string.
    """
    data = _get_json(
        SCHEDULE_URL,
        {
            "sportId": 1,
            "date": date_iso,
            "hydrate": "probablePitcher,team,venue",
        },
    )
    if not data or not data.get("dates"):
        return ()
    games = []
    for d in data["dates"]:
        for g in d.get("games", []):
            home = g["teams"]["home"]["team"]
            away = g["teams"]["away"]["team"]
            home_abbr = _TEAM_ID_TO_ABBR.get(home.get("id"))
            away_abbr = _TEAM_ID_TO_ABBR.get(away.get("id"))
            if not home_abbr or not away_abbr:
                continue
            hp = g["teams"]["home"].get("probablePitcher", {}) or {}
            ap = g["teams"]["away"].get("probablePitcher", {}) or {}
            games.append(
                {
                    "game_pk": g.get("gamePk"),
                    "game_datetime": g.get("gameDate"),
                    "home": home_abbr,
                    "away": away_abbr,
                    "home_id": home.get("id"),
                    "away_id": away.get("id"),
                    "home_pitcher_id": hp.get("id"),
                    "home_pitcher_name": hp.get("fullName"),
                    "away_pitcher_id": ap.get("id"),
                    "away_pitcher_name": ap.get("fullName"),
                }
            )
    return tuple(games)


@lru_cache(maxsize=512)
def fetch_pitcher_hand(player_id: int) -> str:
    if not player_id:
        return "R"
    data = _get_json(f"{STATSAPI}/people/{player_id}")
    try:
        return data["people"][0]["pitchHand"]["code"]
    except Exception:
        return "R"


@lru_cache(maxsize=64)
def fetch_lineup_or_roster(team_id: int, game_pk: int | None) -> tuple:
    """Return (player_id, name, bats, position) tuples.

    Prefers the posted lineup from the live feed (close to game time); otherwise
    falls back to the team's active position-player roster.
    """
    # Try the live boxscore lineup first.
    if game_pk:
        live = _get_json(f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live")
        try:
            box = live["liveData"]["boxscore"]["teams"]
            for side in ("home", "away"):
                t = box[side]
                if t["team"]["id"] != team_id:
                    continue
                batting_order = t.get("battingOrder") or []
                players = []
                for pid in batting_order[:9]:
                    p = t["players"].get(f"ID{pid}", {})
                    person = p.get("person", {})
                    bats = (p.get("batSide", {}) or {}).get("code", "R")
                    pos = (p.get("position", {}) or {}).get("abbreviation", "DH")
                    players.append((person.get("id"), person.get("fullName"), bats, pos))
                if players:
                    return tuple(players)
        except Exception:
            pass

    # Fall back to the active roster (hitters only-ish).
    data = _get_json(f"{STATSAPI}/teams/{team_id}/roster", {"rosterType": "active"})
    players = []
    try:
        for entry in data["roster"]:
            pos = entry.get("position", {}).get("abbreviation", "")
            if pos in ("P", "SP", "RP"):
                continue
            person = entry.get("person", {})
            pid = person.get("id")
            bats = "R"
            details = _get_json(f"{STATSAPI}/people/{pid}")
            try:
                bats = details["people"][0]["batSide"]["code"]
            except Exception:
                bats = "R"
            players.append((pid, person.get("fullName"), bats, pos))
    except Exception:
        return ()
    return tuple(players[:13])


@lru_cache(maxsize=64)
def fetch_weather(home_abbr: str, date_iso: str, hour: int = 19) -> dict:
    """Open-Meteo hourly forecast at the park for the given local hour.

    Returns neutral values on any failure or for roofed parks.
    """
    park = get_park(home_abbr)
    if park is None:
        return {"temp_f": None, "wind_mph": None, "wind_dir_deg": None, "humidity_pct": None}
    roof = str(park.get("roof", "open")).lower()
    data = _get_json(
        "https://api.open-meteo.com/v1/forecast",
        {
            "latitude": park["lat"],
            "longitude": park["lon"],
            "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "start_date": date_iso,
            "end_date": date_iso,
            "timezone": "auto",
        },
    )
    try:
        h = data["hourly"]
        idx = min(hour, len(h["time"]) - 1)
        temp = h["temperature_2m"][idx]
        humid = h["relative_humidity_2m"][idx]
        wind = h["wind_speed_10m"][idx]
        wdir = h["wind_direction_10m"][idx]
    except Exception:
        return {"temp_f": None, "wind_mph": None, "wind_dir_deg": None, "humidity_pct": None}

    # Roofed/closed games neutralize wind.
    if roof in ("dome", "closed"):
        wind, wdir = 0.0, None
    return {"temp_f": temp, "wind_mph": wind, "wind_dir_deg": wdir, "humidity_pct": humid}


def _hitter_metrics(name: str, bats: str, slate_seed: str) -> dict:
    """Attach hitter Statcast metrics.

    Hook point for real Statcast leaderboards (pybaseball). Currently uses the
    deterministic synthetic profile keyed by name; replace `_statcast_lookup`
    to wire live batted-ball data without touching the rest of the pipeline.
    """
    real = _statcast_lookup(name)
    if real is not None:
        return real
    # Tier inferred from the demo pool if the player is known, else neutral (3).
    tier = _known_tier(name)
    return demo._hitter_profile(name, bats, tier, slate_seed)


@lru_cache(maxsize=1)
def _name_to_tier() -> dict:
    mapping = {}
    for team in demo.TEAMS.values():
        for name, _bats, tier, _pos in team["hitters"]:
            mapping[name] = tier
    return mapping


def _known_tier(name: str) -> int:
    return _name_to_tier().get(name, 3)


def _statcast_lookup(name: str):
    """Return a real Statcast profile dict for `name`, or None if unavailable.

    Left as a stub that returns None by default (keeps the tool dependency-light
    and offline-safe). Implement with pybaseball's
    `statcast_batter_exitvelo_barrels` season leaderboard to go fully live.
    """
    return None


def _pitcher_metrics(name: str, throws: str, team_abbr: str, slate_seed: str) -> dict:
    prof = demo._pitcher_profile(team_abbr, slate_seed)
    if name:
        prof["pitcher_name"] = name
    if throws:
        prof["pitcher_throws"] = throws
    return prof


def build_live_slate(game_date: dt.date) -> tuple[pd.DataFrame | None, list[str]]:
    """Build a per-hitter slate from real MLB schedule data. Returns (df, notes)."""
    notes: list[str] = []
    date_iso = game_date.isoformat()
    games = fetch_schedule(date_iso)
    if not games:
        return None, ["No live schedule available for this date."]

    slate_seed = date_iso
    rows = []
    for g in games:
        home, away = g["home"], g["away"]
        home_hand = fetch_pitcher_hand(g.get("home_pitcher_id"))
        away_hand = fetch_pitcher_hand(g.get("away_pitcher_id"))
        weather = fetch_weather(home, date_iso)

        home_pitcher = _pitcher_metrics(g.get("home_pitcher_name"), home_hand, home, slate_seed)
        away_pitcher = _pitcher_metrics(g.get("away_pitcher_name"), away_hand, away, slate_seed)

        for side, team, team_id, opp, opp_pitcher in (
            ("away", away, g["away_id"], home, home_pitcher),
            ("home", home, g["home_id"], away, away_pitcher),
        ):
            roster = fetch_lineup_or_roster(team_id, g.get("game_pk"))
            if not roster:
                # Fall back to the demo roster for this team.
                tinfo = demo.TEAMS.get(team)
                if not tinfo:
                    continue
                roster = [(None, n, b, p) for n, b, _t, p in tinfo["hitters"]]
            park = get_park(home)
            team_name = park["team_name"] if (park and team == home) else demo.TEAMS.get(team, {}).get("name", team)
            for pid, name, bats, pos in roster:
                if not name:
                    continue
                metrics = _hitter_metrics(name, bats, slate_seed)
                row = {
                    "player": name,
                    "team": team,
                    "team_name": demo.TEAMS.get(team, {}).get("name", team),
                    "bats": bats or "R",
                    "position": pos or "DH",
                    "opponent": opp,
                    "home_team": home,
                    "is_home": side == "home",
                    "game": f"{away} @ {home}",
                }
                row.update(weather)
                row.update(metrics)
                row.update(opp_pitcher)
                rows.append(row)

    if not rows:
        return None, notes + ["Live games found but no hitters resolved."]
    notes.append(f"Live schedule: {len(games)} games from MLB StatsAPI.")
    notes.append("Batted-ball metrics are modeled (synthetic) unless a Statcast feed is wired in.")
    return pd.DataFrame(rows), notes


def get_slate(game_date: dt.date, prefer_live: bool = True) -> tuple[pd.DataFrame, str, list[str]]:
    """Return (slate_df, source_label, notes).

    source_label is one of: 'LIVE (modeled metrics)', 'DEMO (synthetic)'.
    """
    notes: list[str] = []
    if prefer_live:
        try:
            df, live_notes = build_live_slate(game_date)
            notes.extend(live_notes)
            if df is not None and not df.empty:
                return df, "LIVE (modeled metrics)", notes
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"Live fetch failed: {exc}")
    df = demo.build_demo_slate(game_date)
    notes.append("Using synthetic demo slate (deterministic by date).")
    return df, "DEMO (synthetic)", notes
