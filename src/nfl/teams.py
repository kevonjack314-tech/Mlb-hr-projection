"""NFL venues and team context.

The baseball model leans hard on park and weather because a home run is a
batted ball fighting air. Football's equivalent is narrower but real: wind and
cold suppress the PASSING game, which pushes touchdowns from receivers toward
running backs rather than removing them. A dome removes that entirely.

So venue here is not a scoring multiplier so much as a MIX shifter, and the
model uses it that way.
"""

from __future__ import annotations

# team -> (full name, venue, roof, surface, altitude_ft)
# roof: "dome" (fixed), "retractable", "open"
TEAMS = {
    "ARI": ("Arizona Cardinals", "State Farm Stadium", "retractable", 1070),
    "ATL": ("Atlanta Falcons", "Mercedes-Benz Stadium", "retractable", 1050),
    "BAL": ("Baltimore Ravens", "M&T Bank Stadium", "open", 30),
    "BUF": ("Buffalo Bills", "Highmark Stadium", "open", 600),
    "CAR": ("Carolina Panthers", "Bank of America Stadium", "open", 751),
    "CHI": ("Chicago Bears", "Soldier Field", "open", 597),
    "CIN": ("Cincinnati Bengals", "Paycor Stadium", "open", 490),
    "CLE": ("Cleveland Browns", "Huntington Bank Field", "open", 571),
    "DAL": ("Dallas Cowboys", "AT&T Stadium", "retractable", 600),
    "DEN": ("Denver Broncos", "Empower Field at Mile High", "open", 5280),
    "DET": ("Detroit Lions", "Ford Field", "dome", 600),
    "GB": ("Green Bay Packers", "Lambeau Field", "open", 640),
    "HOU": ("Houston Texans", "NRG Stadium", "retractable", 50),
    "IND": ("Indianapolis Colts", "Lucas Oil Stadium", "retractable", 715),
    "JAX": ("Jacksonville Jaguars", "EverBank Stadium", "open", 16),
    "KC": ("Kansas City Chiefs", "GEHA Field at Arrowhead", "open", 750),
    "LV": ("Las Vegas Raiders", "Allegiant Stadium", "dome", 2030),
    "LAC": ("Los Angeles Chargers", "SoFi Stadium", "dome", 125),
    "LAR": ("Los Angeles Rams", "SoFi Stadium", "dome", 125),
    "MIA": ("Miami Dolphins", "Hard Rock Stadium", "open", 10),
    "MIN": ("Minnesota Vikings", "U.S. Bank Stadium", "dome", 830),
    "NE": ("New England Patriots", "Gillette Stadium", "open", 289),
    "NO": ("New Orleans Saints", "Caesars Superdome", "dome", 3),
    "NYG": ("New York Giants", "MetLife Stadium", "open", 7),
    "NYJ": ("New York Jets", "MetLife Stadium", "open", 7),
    "PHI": ("Philadelphia Eagles", "Lincoln Financial Field", "open", 39),
    "PIT": ("Pittsburgh Steelers", "Acrisure Stadium", "open", 730),
    "SF": ("San Francisco 49ers", "Levi's Stadium", "open", 10),
    "SEA": ("Seattle Seahawks", "Lumen Field", "open", 20),
    "TB": ("Tampa Bay Buccaneers", "Raymond James Stadium", "open", 48),
    "TEN": ("Tennessee Titans", "Nissan Stadium", "open", 400),
    "WAS": ("Washington Commanders", "Northwest Stadium", "open", 200),
}

INDOOR = {t for t, v in TEAMS.items() if v[2] in ("dome", "retractable")}


def venue(team: str) -> dict | None:
    v = TEAMS.get(str(team).upper())
    if not v:
        return None
    name, stadium, roof, alt = v
    return {"team_name": name, "stadium": stadium, "roof": roof, "altitude_ft": alt}


def is_indoor(team: str) -> bool:
    """Retractable roofs are treated as closed — they usually are in bad weather,
    which is exactly when the distinction would have mattered."""
    return str(team).upper() in INDOOR


def pass_weather_multiplier(home_team: str, wind_mph=None, temp_f=None) -> dict:
    """How much tonight's weather suppresses the PASSING game.

    Returns {pass_mult, rush_mult, note}. The two move opposite ways on
    purpose: bad weather does not delete touchdowns, it moves them from the
    air to the ground. A model that only applied a penalty would quietly make
    every player on a windy slate a worse bet, when the goal-line back is
    actually a BETTER one.
    """
    out = {"pass_mult": 1.0, "rush_mult": 1.0, "weather_note": ""}
    if is_indoor(home_team):
        out["weather_note"] = "indoors — weather neutral"
        return out
    notes = []
    pm = 1.0
    try:
        w = float(wind_mph)
        if w == w and w >= 10:
            # Passing efficiency falls off hard past ~15 mph; below 10 it is noise.
            pm *= max(0.80, 1.0 - (w - 10.0) * 0.012)
            if w >= 15:
                notes.append(f"{w:.0f} mph wind")
    except (TypeError, ValueError):
        pass
    try:
        t = float(temp_f)
        if t == t and t <= 32:
            pm *= max(0.90, 1.0 - (32.0 - t) * 0.004)
            notes.append(f"{t:.0f}°F")
    except (TypeError, ValueError):
        pass
    out["pass_mult"] = round(pm, 4)
    # Touchdowns lost in the air come back on the ground, damped.
    out["rush_mult"] = round(1.0 + (1.0 - pm) * 0.55, 4)
    out["weather_note"] = ("; ".join(notes) + " — leans run" if notes else "")
    return out
