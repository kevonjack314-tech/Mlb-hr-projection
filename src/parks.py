"""Ballpark factors and geometry.

Park factors are 3-year regressed HR park factors on a scale where 100 is league
average. A value of 110 means the park yields ~10% more home runs than average for
an identical batted-ball profile; 90 means ~10% fewer. Handedness splits matter a
lot (e.g. Yankee Stadium's short right-porch inflates LHB HR, Fenway's Monster
suppresses RHB HR while its short left distance is offset by the wall height).

Sources / calibration: Statcast park factors (baseballsavant.mlb.com/leaderboard/
statcast-park-factors), ESPN park factors, and public batted-ball park research.
Values are bundled so the tool works fully offline; refresh annually.
"""

from __future__ import annotations

import math
import os
from functools import lru_cache

import pandas as pd

_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "park_factors.csv")


@lru_cache(maxsize=1)
def load_park_factors() -> pd.DataFrame:
    """Load the bundled park-factor table indexed by team abbreviation."""
    df = pd.read_csv(_DATA_PATH)
    df = df.set_index("team_abbr", drop=False)
    return df


def get_park(team_abbr: str) -> dict | None:
    """Return the park record for a home team abbreviation, or None if unknown."""
    df = load_park_factors()
    if team_abbr in df.index:
        return df.loc[team_abbr].to_dict()
    return None


def park_hr_multiplier(team_abbr: str, bat_side: str = "R") -> float:
    """Handedness-aware park HR multiplier centered at 1.0 (league average).

    bat_side: 'L' or 'R'. Switch hitters should be evaluated from the side they
    will bat against the listed pitcher (handled upstream).
    """
    park = get_park(team_abbr)
    if park is None:
        return 1.0
    if bat_side.upper().startswith("L"):
        factor = park.get("hr_factor_lhb", park["hr_factor"])
    else:
        factor = park.get("hr_factor_rhb", park["hr_factor"])
    return float(factor) / 100.0


def wind_hr_multiplier(park: dict | None, wind_speed_mph: float, wind_dir_deg: float | None,
                       bat_side: str = "R") -> float:
    """Estimate the wind contribution to HR rate.

    The model resolves the wind vector against the line from home plate toward
    center field (the park's `orientation_deg`, the compass bearing the batter
    faces). A tailwind blowing *out* toward the outfield lifts fly balls and adds
    carry; an *in* wind knocks them down. Magnitude scales ~1.5% per mph of the
    out/in component, capped to keep extreme readings sane.

    Roofed/domed games (passed as wind_speed_mph<=0) are neutral.
    """
    if park is None or wind_speed_mph is None or wind_speed_mph <= 0:
        return 1.0
    roof = str(park.get("roof", "open")).lower()
    if roof in ("dome", "retractable_closed", "closed"):
        return 1.0
    if wind_dir_deg is None:
        return 1.0

    field_bearing = float(park.get("orientation_deg", 0.0))
    # Meteorological wind_dir_deg is the direction the wind blows FROM.
    # Convert to the direction it blows TOWARD.
    blow_toward = (wind_dir_deg + 180.0) % 360.0
    # Component of the wind along the home->center axis. +1 = straight out to CF.
    angle = math.radians(blow_toward - field_bearing)
    out_component = math.cos(angle) * wind_speed_mph

    # Pull-side nuance: a cross wind toward the batter's pull field helps a touch.
    # Approximate by giving 30% credit to the cross component on the pull side.
    cross_component = math.sin(angle) * wind_speed_mph
    pull_sign = 1.0 if bat_side.upper().startswith("R") else -1.0
    pull_help = max(0.0, cross_component * pull_sign) * 0.30

    effective = out_component + pull_help
    mult = 1.0 + (effective * 0.015)
    return max(0.80, min(1.25, mult))


def temp_hr_multiplier(temp_f: float | None) -> float:
    """Warmer air is thinner and the ball carries farther.

    Empirically HR rate moves ~1% per ~3.5 degrees F around a 70F baseline.
    Clamped to a sensible band.
    """
    if temp_f is None:
        return 1.0
    mult = 1.0 + (temp_f - 70.0) * (0.01 / 3.5)
    return max(0.90, min(1.12, mult))


def humidity_hr_multiplier(humidity_pct: float | None, altitude_ft: float = 0.0) -> float:
    """Humidity has a small, second-order effect on carry.

    Humid air is slightly *less* dense (water vapor is lighter than dry air), so
    very humid conditions add a touch of carry. The effect is minor and we keep
    it small; high-altitude humidor parks (Coors, Chase) are handled by their
    park factors already.
    """
    if humidity_pct is None:
        return 1.0
    mult = 1.0 + (humidity_pct - 50.0) * 0.0004
    return max(0.98, min(1.02, mult))
