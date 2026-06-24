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
                for spot, pid in enumerate(batting_order[:9], start=1):
                    p = t["players"].get(f"ID{pid}", {})
                    person = p.get("person", {})
                    bats = (p.get("batSide", {}) or {}).get("code", "R")
                    pos = (p.get("position", {}) or {}).get("abbreviation", "DH")
                    players.append((person.get("id"), person.get("fullName"), bats, pos, spot))
                if players:
                    return tuple(players)
        except Exception:
            pass

    # Fall back to the active roster (hitters only-ish); spot is approximate.
    data = _get_json(f"{STATSAPI}/teams/{team_id}/roster", {"rosterType": "active"})
    players = []
    try:
        for i, entry in enumerate(data["roster"]):
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
            players.append((pid, person.get("fullName"), bats, pos, min(len(players) + 1, 9)))
    except Exception:
        return ()
    return tuple(players[:13])


@lru_cache(maxsize=1024)
def fetch_batting_order_map(game_pk) -> tuple:
    """Return ((player_id, spot), …) from a game's box score.

    MLB StatsAPI encodes batting order as e.g. "500" (5th spot) or "501" (a sub
    who took the 5th spot); the lineup spot is the hundreds digit. This lets us
    recover the *actual* spot each player batted in for a completed game.
    """
    if not game_pk:
        return ()
    data = _get_json(f"{STATSAPI}/game/{game_pk}/boxscore")
    out = []
    try:
        for side in ("home", "away"):
            for _key, p in data["teams"][side]["players"].items():
                bo = p.get("battingOrder")
                if not bo:
                    continue
                spot = int(bo) // 100
                pid = p.get("person", {}).get("id")
                if pid and 1 <= spot <= 9:
                    out.append((pid, spot))
    except Exception:
        return ()
    return tuple(out)


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


def season_year_for(game_date: dt.date) -> int:
    """The MLB stats season relevant to a date (offseason -> prior season)."""
    return game_date.year if game_date.month >= 4 else game_date.year - 1


def _hitter_metrics(name: str, bats: str, slate_seed: str, mlbam_id, year: int,
                    end_date_iso: str) -> tuple[dict, bool]:
    """Attach hitter Statcast metrics. Returns (profile, used_real_data).

    Strategy: start from the deterministic modeled profile (guarantees every
    field), then overlay any REAL season metrics (barrel%, EV, xwOBA, HR/PA…)
    and REAL recent-form HR rates that we can resolve from Statcast/FanGraphs.
    So the result is real wherever live data exists and modeled only to fill
    gaps — and it never crashes if the live feeds are blocked.
    """
    tier = _known_tier(name)
    profile = dict(demo._hitter_profile(name, bats, tier, slate_seed))
    used_real = False

    real = _statcast_lookup(name, mlbam_id, year)
    if real:
        profile.update(real)
        used_real = True

    recent = _recent_form_lookup(end_date_iso, mlbam_id)
    if recent:
        profile.update(recent)
        used_real = True

    splits = _batter_splits_lookup(end_date_iso, mlbam_id)
    if splits:
        profile.update(splits)
        used_real = True

    return profile, used_real


@lru_cache(maxsize=1)
def _name_to_tier() -> dict:
    mapping = {}
    for team in demo.TEAMS.values():
        for name, _bats, tier, _pos in team["hitters"]:
            mapping[name] = tier
    return mapping


def _known_tier(name: str) -> int:
    return _name_to_tier().get(name, 3)


def _statcast_lookup(name: str, mlbam_id=None, year: int | None = None):
    """Return a real season Statcast/FanGraphs profile dict, or None."""
    try:
        from . import statcast
        if year is None:
            year = season_year_for(dt.date.today())
        return statcast.lookup_season(year, name, mlbam_id)
    except Exception:
        return None


def _recent_form_lookup(end_date_iso: str, mlbam_id=None):
    """Return real {hr_rate_7,15,30} for a batter, or None."""
    try:
        from . import statcast
        return statcast.lookup_recent_form(end_date_iso, mlbam_id)
    except Exception:
        return None


def _batter_splits_lookup(end_date_iso: str, mlbam_id=None):
    """Return real {vs_fb, vs_br, vs_os} wOBA splits for a batter, or None."""
    try:
        from . import statcast
        return statcast.lookup_batter_splits(end_date_iso, mlbam_id)
    except Exception:
        return None


def _lean_from_gb(gb_pct: float | None) -> str:
    if gb_pct is None:
        return "NEU"
    if gb_pct >= 46:
        return "GB"
    if gb_pct <= 38:
        return "FB"
    return "NEU"


def _pitcher_metrics(name: str, throws: str, team_abbr: str, slate_seed: str,
                     pitcher_id=None, year: int | None = None,
                     end_date_iso: str | None = None) -> tuple[dict, bool]:
    """Build pitcher metrics. Returns (profile, used_real_data).

    Starts from the modeled profile, then overlays REAL FanGraphs peripherals
    (HR/9, GB%, FB%, barrels allowed) by name and the REAL Statcast pitch mix by
    id; the fly-ball/ground-ball lean is recomputed from real GB% when available.
    """
    prof = dict(demo._pitcher_profile(team_abbr, slate_seed))
    if name:
        prof["pitcher_name"] = name
    if throws:
        prof["pitcher_throws"] = throws
    prof["pitcher_id"] = pitcher_id
    used_real = False

    try:
        from . import statcast
        peri = statcast.lookup_pitching(year, name) if year else None
        if peri:
            prof.update(peri)
            if "pitcher_gb_pct" in peri:
                prof["pitcher_lean"] = _lean_from_gb(peri.get("pitcher_gb_pct"))
            used_real = True
        if end_date_iso:
            mix = statcast.lookup_pitch_mix(end_date_iso, pitcher_id)
            if mix:
                prof.update(mix)
                used_real = True
    except Exception:
        pass
    return prof, used_real


def build_live_slate(game_date: dt.date) -> tuple[pd.DataFrame | None, list[str]]:
    """Build a per-hitter slate from real MLB schedule data. Returns (df, notes)."""
    notes: list[str] = []
    date_iso = game_date.isoformat()
    games = fetch_schedule(date_iso)
    if not games:
        return None, ["No live schedule available for this date."]

    slate_seed = date_iso
    year = season_year_for(game_date)
    real_hitters = 0
    total_hitters = 0
    real_pitchers = 0
    rows = []
    for g in games:
        home, away = g["home"], g["away"]
        home_hand = fetch_pitcher_hand(g.get("home_pitcher_id"))
        away_hand = fetch_pitcher_hand(g.get("away_pitcher_id"))
        weather = fetch_weather(home, date_iso)

        home_pitcher, hp_real = _pitcher_metrics(
            g.get("home_pitcher_name"), home_hand, home, slate_seed,
            g.get("home_pitcher_id"), year, date_iso)
        away_pitcher, ap_real = _pitcher_metrics(
            g.get("away_pitcher_name"), away_hand, away, slate_seed,
            g.get("away_pitcher_id"), year, date_iso)
        real_pitchers += int(hp_real) + int(ap_real)

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
                roster = [(None, n, b, p, demo.demo_spot_for_index(i) or 9)
                          for i, (n, b, _t, p) in enumerate(tinfo["hitters"])]
            park = get_park(home)
            team_name = park["team_name"] if (park and team == home) else demo.TEAMS.get(team, {}).get("name", team)
            for pid, name, bats, pos, spot in roster:
                if not name:
                    continue
                metrics, used_real = _hitter_metrics(
                    name, bats, slate_seed, pid, year, date_iso
                )
                total_hitters += 1
                real_hitters += int(used_real)
                row = {
                    "player": name,
                    "team": team,
                    "team_name": demo.TEAMS.get(team, {}).get("name", team),
                    "bats": bats or "R",
                    "position": pos or "DH",
                    "lineup_spot": spot,
                    "opponent": opp,
                    "home_team": home,
                    "is_home": side == "home",
                    "game": f"{away} @ {home}",
                    "data_quality": "real" if used_real else "modeled",
                }
                row.update(weather)
                row.update(metrics)
                row.update(opp_pitcher)
                rows.append(row)

    if not rows:
        return None, notes + ["Live games found but no hitters resolved."]
    notes.append(f"Live schedule: {len(games)} games from MLB StatsAPI.")
    if real_hitters:
        pct = 100 * real_hitters / max(1, total_hitters)
        notes.append(
            f"Real Statcast/FanGraphs metrics resolved for {real_hitters}/"
            f"{total_hitters} hitters ({pct:.0f}%); remainder modeled."
        )
    else:
        notes.append(
            "Statcast/FanGraphs feed unavailable (host not allowlisted or "
            "pybaseball missing) — batted-ball metrics modeled. See README."
        )
    if real_pitchers:
        notes.append(
            f"Real pitcher peripherals + pitch mix resolved for {real_pitchers}/"
            f"{2 * len(games)} probable starters; remainder modeled."
        )
    return pd.DataFrame(rows), notes


def get_slate(game_date: dt.date, prefer_live: bool = True) -> tuple[pd.DataFrame, str, list[str]]:
    """Return (slate_df, source_label, notes).

    source_label is one of: 'LIVE (real Statcast)', 'LIVE (modeled metrics)',
    'DEMO (synthetic)'.
    """
    notes: list[str] = []
    if prefer_live:
        try:
            df, live_notes = build_live_slate(game_date)
            notes.extend(live_notes)
            if df is not None and not df.empty:
                has_real = "data_quality" in df.columns and (df["data_quality"] == "real").any()
                label = "LIVE (real Statcast)" if has_real else "LIVE (modeled metrics)"
                return df, label, notes
        except Exception as exc:  # pragma: no cover - defensive
            notes.append(f"Live fetch failed: {exc}")
    df = demo.build_demo_slate(game_date)
    notes.append("Using synthetic demo slate (deterministic by date).")
    return df, "DEMO (synthetic)", notes
