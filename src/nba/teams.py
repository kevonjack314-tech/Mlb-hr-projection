"""NBA team context.

Basketball has almost no venue effect worth modelling — no park factors, no
wind. What takes their place is PACE and the opponent's willingness to let
threes go up: a possession is the scarce resource, and the number of them is
set by the two teams together.

Denver's altitude is the one real building effect, and it shows up in pace
(visiting teams tire and the game speeds up) rather than in shooting.
"""

from __future__ import annotations

TEAMS = {
    "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
    "LAC": "LA Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
    "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans", "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder", "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns", "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs", "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz", "WAS": "Washington Wizards",
}

LEAGUE_PACE = 99.0          # possessions per 48 minutes
LEAGUE_3PA_PER_GAME = 35.5  # team three-point attempts
LEAGUE_3P_PCT = 0.365
ALTITUDE_TEAMS = {"DEN": 5280, "UTA": 4226}


def game_pace(home_pace=None, away_pace=None, home_team=None) -> float:
    """Possessions this game is likely to run at.

    Pace is a property of the MATCHUP, not of one team: a fast team playing a
    slow one lands between them. Taking the average rather than either side's
    own number is the difference between projecting a track meet and projecting
    what actually happens.
    """
    vals = [v for v in (home_pace, away_pace) if v is not None and v == v]
    pace = sum(vals) / len(vals) if vals else LEAGUE_PACE
    if str(home_team).upper() in ALTITUDE_TEAMS:
        pace *= 1.015        # visitors tire; the game speeds up
    return float(min(max(pace, 90.0), 108.0))


def pace_multiplier(pace: float) -> float:
    """Extra possessions mean extra shots, proportionally."""
    return float(pace) / LEAGUE_PACE


def rest_multiplier(days_rest=None, is_back_to_back=False) -> dict:
    """Legs matter for three-point shooting more than for anything else.

    A tired shooter's release flattens and his minutes often get managed, so
    the back-to-back hit lands on both terms. This is the closest thing the
    sport has to the hitter-fatigue signal in the baseball model.
    """
    out = {"rest_shooting_mult": 1.0, "rest_minutes_mult": 1.0, "rest_note": ""}
    if is_back_to_back:
        out.update(rest_shooting_mult=0.972, rest_minutes_mult=0.965,
                   rest_note="back-to-back")
        return out
    try:
        d = float(days_rest)
        if d >= 3:
            out.update(rest_shooting_mult=1.012, rest_note=f"{int(d)} days rest")
    except (TypeError, ValueError):
        pass
    return out
