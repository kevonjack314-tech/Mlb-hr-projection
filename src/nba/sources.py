"""Live NBA slate from ESPN's public API, with a synthetic fallback.

The three-point model runs on minutes, attempt rate and accuracy, and ESPN's
per-athlete averages carry all three directly — no derivation needed, unlike
football where goal-line role has to be inferred from observed touchdowns.

What it does NOT carry is tonight's availability beyond the roster's own
injury flags, or a minutes projection. Season averages are used as the minutes
baseline and the board says so; a starter ruled out an hour before tip is the
single biggest thing this feed cannot see.
"""

from __future__ import annotations

import pandas as pd

from .. import espn
from ..espn import _num, athlete_stats_urls, fetch_many, parse_stat_categories
from .demo import build_demo_slate
from .teams import LEAGUE_3PA_PER_GAME, TEAMS

# Below this a player is not a three-point bet and is only noise on the board.
MIN_MINUTES = 8.0


def _split_made_attempted(value):
    """ESPN packs makes and attempts into one field: '2.4-6.1'."""
    if not value:
        return None, None
    parts = str(value).split("-")
    if len(parts) != 2:
        return None, None
    return _num(parts[0]), _num(parts[1])


def _player_rates(stats: dict) -> dict | None:
    mins = _num(stats.get("averages.avgMinutes"))
    made, att = _split_made_attempted(
        stats.get("averages.avgThreePointFieldGoalsMade-avgThreePointFieldGoalsAttempted"))
    pct = _num(stats.get("averages.threePointFieldGoalPct"))
    if mins is None or att is None:
        return None
    if pct is not None and pct > 1.5:          # ESPN ships it as a percentage
        pct /= 100.0
    return {
        "minutes_season": round(mins, 1),
        "minutes_l5": round(mins, 1),          # no recent split on this feed
        "fg3a_per_game": round(att, 2),
        "fg3a_per36": round(att / mins * 36.0, 2) if mins >= 1 else 0.0,
        "fg3m_per_game": round(made, 2) if made is not None else None,
        "fg3_pct": round(pct, 4) if pct else None,
        "games_played": _num(stats.get("averages.gamesPlayed")),
    }


def _team_players(team: dict) -> list[dict]:
    people = espn.roster("nba", team.get("id"))
    if not people:
        return []
    payloads = fetch_many(athlete_stats_urls("nba", [p["id"] for p in people]))
    out = []
    for p, payload in zip(people, payloads):
        rates = _player_rates(parse_stat_categories(payload))
        if not rates or (rates["minutes_season"] or 0) < MIN_MINUTES:
            continue
        gp = rates.get("games_played") or 0
        out.append({**p, **rates,
                    # Season attempts, used to shrink a hot percentage.
                    "fg3a_season": round((rates["fg3a_per_game"] or 0) * gp, 0)})
    return out


def build_live_slate(date_iso: str):
    notes: list[str] = []
    games = espn.scoreboard("nba", date_iso)
    if not games:
        return None, ["No ESPN schedule for this date."]
    playable = [g for g in games if g["status"] in ("STATUS_SCHEDULED", "")]
    if not playable:
        notes.append(f"{len(games)} games found, none still scheduled.")
        return None, notes

    rows, with_line = [], 0
    for g in playable:
        if g["game_total"] is not None:
            with_line += 1
        for side in ("home", "away"):
            team, opp = g[side], g["away" if side == "home" else "home"]
            abbr = team["abbr"]
            players = _team_players(team)
            if not players:
                notes.append(f"No usable roster for {abbr}.")
                continue
            team_3pa = sum(p["fg3a_per_game"] or 0 for p in players) or LEAGUE_3PA_PER_GAME
            for p in players:
                rows.append({
                    "player": p["name"], "espn_id": p["id"],
                    "team": abbr, "opponent": opp["abbr"],
                    "team_name": TEAMS.get(abbr, team["name"]),
                    "game": g["game"], "home_team": g["home"]["abbr"],
                    "is_home": side == "home",
                    "position": p.get("position"),
                    "game_total": g["game_total"], "spread": g["spread"],
                    "team_3pa_per_game": round(team_3pa, 1),
                    # ESPN's public feed has no pace or opponent-defense
                    # splits; the model falls back to league average for both
                    # and the board is flagged accordingly.
                    "team_pace": None, "opp_pace": None,
                    "opp_3pa_allowed": None, "opp_3p_pct_allowed": None,
                    "back_to_back": False, "days_rest": None,
                    "minutes_l5": p["minutes_l5"],
                    "minutes_season": p["minutes_season"],
                    "minutes_stdev_l5": None,
                    "fg3a_per36": p["fg3a_per36"],
                    "fg3_pct": p["fg3_pct"],
                    "fg3a_season": p["fg3a_season"],
                    "games_played": p["games_played"],
                    "data_quality": "live",
                })
    if not rows:
        return None, notes + ["ESPN returned games but no usable players."]
    notes.append(f"{len(playable)} games, {with_line} with a posted total.")
    notes.append("Minutes are season averages — this feed has no last-5 split, "
                 "no pace and no opponent perimeter defence.")
    return pd.DataFrame(rows), notes


def get_slate(date_iso: str, prefer_live: bool = True):
    if prefer_live:
        try:
            df, notes = build_live_slate(date_iso)
        except Exception as exc:
            df, notes = None, [f"Live NBA slate failed: {exc}"]
        if df is not None and not df.empty:
            return df, "LIVE (ESPN + DraftKings lines)", notes
        return (build_demo_slate(date_iso), "DEMO (synthetic)",
                (notes or []) + ["Fell back to the synthetic slate."])
    return build_demo_slate(date_iso), "DEMO (synthetic)", []
