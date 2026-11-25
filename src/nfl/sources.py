"""Live NFL slate from ESPN's public API, with a synthetic fallback.

What the model wants is inside-the-5 carries and red-zone target share. No
free keyless feed carries those. What ESPN does carry is better than a guess:

  * the real DraftKings spread and total, which split into implied team
    totals — the biggest single input to a touchdown projection;
  * each player's season rushing attempts, targets, and touchdowns split by
    rush and receive, which gives his OBSERVED share of the touchdowns his
    team has actually scored. That is the goal-line role, revealed.

Same contract as the baseball loader: any failure degrades to the
deterministic synthetic slate and says so in the source label, so a modelled
board can never be mistaken for a real one.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from .. import espn
from ..espn import _num, athlete_stats_urls, fetch_many, parse_stat_categories
from .demo import build_demo_slate
from .teams import TEAMS, is_indoor

SKILL = ("QB", "RB", "WR", "TE", "FB")
# Per team, how many of each position can realistically score. Beyond this the
# players are not bettable and every extra one is an athlete request.
POSITION_CAP = {"QB": 1, "RB": 5, "WR": 7, "TE": 4, "FB": 1}


def _player_rates(stats: dict) -> dict:
    """Season rates from one athlete's ESPN stat payload."""
    g = max(_num(stats.get("rushing.gamesPlayed"))
            or _num(stats.get("receiving.gamesPlayed")) or 0.0, 1.0)
    att = _num(stats.get("rushing.rushingAttempts"), 0.0) or 0.0
    tgt = _num(stats.get("receiving.receivingTargets"), 0.0) or 0.0
    rec_yds = _num(stats.get("receiving.receivingYards"), 0.0) or 0.0
    return {
        "games": g,
        "rush_att": att, "targets": tgt,
        "rush_td": _num(stats.get("rushing.rushingTouchdowns"), 0.0) or 0.0,
        "rec_td": _num(stats.get("receiving.receivingTouchdowns"), 0.0) or 0.0,
        # scoring.totalTouchdowns excludes passing TDs, which is exactly the
        # scope of an anytime-touchdown prop.
        "season_tds": _num(stats.get("scoring.totalTouchdowns"), 0.0) or 0.0,
        "ypc": _num(stats.get("rushing.yardsPerRushAttempt")),
        # Yards per target stands in for yards per route run; ESPN has no
        # route data, and this is the closest honest proxy.
        "yprr": round(rec_yds / tgt, 2) if tgt >= 5 else None,
    }


def _team_players(league_team: dict, date_iso: str) -> list[dict]:
    """Roster -> capped skill players -> season rates, in one threaded batch."""
    team_id = league_team.get("id")
    people = [p for p in espn.roster("nfl", team_id) if p["position"] in SKILL]
    kept, seen = [], {}
    for p in people:
        n = seen.get(p["position"], 0)
        if n >= POSITION_CAP.get(p["position"], 2):
            continue
        seen[p["position"]] = n + 1
        kept.append(p)
    if not kept:
        return []
    payloads = fetch_many(athlete_stats_urls("nfl", [p["id"] for p in kept]))
    out = []
    for p, payload in zip(kept, payloads):
        rates = _player_rates(parse_stat_categories(payload))
        out.append({**p, **rates})
    return out


def build_live_slate(date_iso: str):
    """Returns (DataFrame, notes). None when the slate can't be built."""
    notes: list[str] = []
    games = espn.scoreboard("nfl", date_iso)
    if not games:
        return None, ["No ESPN schedule for this date."]
    playable = [g for g in games if g["status"] in ("STATUS_SCHEDULED", "")]
    if not playable:
        notes.append(f"{len(games)} games found, none still scheduled "
                     "(already started or final).")
        return None, notes

    rows, with_line = [], 0
    for g in playable:
        if g["home_implied"] is not None:
            with_line += 1
        indoor = g["indoor"] or is_indoor(g["home"]["abbr"])
        for side in ("home", "away"):
            team = g[side]
            abbr = team["abbr"]
            implied = g[f"{side}_implied"]
            players = _team_players(team, date_iso)
            if not players:
                notes.append(f"No usable roster for {abbr}.")
                continue
            # Team totals, so each player's share is measured against his own
            # team rather than a league constant.
            t_att = sum(p["rush_att"] for p in players) or 1.0
            t_tgt = sum(p["targets"] for p in players) or 1.0
            t_rush_td = sum(p["rush_td"] for p in players)
            t_rec_td = sum(p["rec_td"] for p in players)
            t_tds = t_rush_td + t_rec_td
            opp = g["away" if side == "home" else "home"]["abbr"]
            for p in players:
                rows.append({
                    "player": p["name"], "espn_id": p["id"],
                    "team": abbr, "opponent": opp,
                    "team_name": TEAMS.get(abbr, (team["name"],))[0],
                    "game": g["game"], "home_team": g["home"]["abbr"],
                    "is_home": side == "home",
                    "position": p["position"],
                    "game_total": g["game_total"], "spread": g["spread"],
                    "implied_team_total": implied,
                    "roof": "dome" if indoor else "open",
                    "wind_mph": 0.0 if indoor else None,
                    "temp_f": 70.0 if indoor else g["temp_f"],
                    "weather_desc": "indoors" if indoor else g["weather"],
                    # Usage shares, measured within the team.
                    "rush_share": round(p["rush_att"] / t_att, 4),
                    "tgt_share": round(p["targets"] / t_tgt, 4),
                    # Observed goal-line role: his share of the touchdowns the
                    # team has already scored, split so the weather pool shift
                    # still lands on the right component.
                    "season_rush_td_share": (round(p["rush_td"] / t_rush_td, 4)
                                             if t_rush_td else None),
                    "season_rec_td_share": (round(p["rec_td"] / t_rec_td, 4)
                                            if t_rec_td else None),
                    "team_season_tds": t_tds,
                    # Split out, so the model can weigh each observed share
                    # against the events that actually produced it.
                    "team_season_rush_tds": t_rush_td,
                    "team_season_rec_tds": t_rec_td,
                    "season_tds": p["season_tds"],
                    "games_played": p["games"],
                    "ypc": p["ypc"], "yprr": p["yprr"],
                    # Snap share is not public; approximate it from how much of
                    # the team's touches the player actually gets.
                    "snap_share": round(min(0.98, 0.25 + 0.75 * max(
                        p["rush_att"] / t_att, p["targets"] / t_tgt)), 3),
                    "data_quality": "live",
                })
    if not rows:
        return None, notes + ["ESPN returned games but no usable players."]
    notes.append(f"{len(playable)} games, {with_line} with a posted line.")
    return pd.DataFrame(rows), notes


def get_slate(date_iso: str, prefer_live: bool = True):
    """(DataFrame, source_label, notes) — live when possible, synthetic otherwise."""
    if prefer_live:
        try:
            df, notes = build_live_slate(date_iso)
        except Exception as exc:                     # never take the board down
            df, notes = None, [f"Live NFL slate failed: {exc}"]
        if df is not None and not df.empty:
            return df, "LIVE (ESPN + DraftKings lines)", notes
        return (build_demo_slate(date_iso), "DEMO (synthetic)",
                (notes or []) + ["Fell back to the synthetic slate."])
    return build_demo_slate(date_iso), "DEMO (synthetic)", []
