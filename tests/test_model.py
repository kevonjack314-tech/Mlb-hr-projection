"""Smoke tests for the HR projection model. Run with: python -m pytest tests/ -q"""

import datetime as dt

from src.demo import build_demo_slate
from src.model import (
    effective_bat_side,
    has_platoon_advantage,
    score_row,
    score_slate,
)
from src.parks import park_hr_multiplier, wind_hr_multiplier, get_park


def _slate():
    return score_slate(build_demo_slate(dt.date(2026, 6, 18)))


def test_slate_builds_and_scores():
    df = _slate()
    assert len(df) > 100
    for col in ("hr_score", "hr_prob_game", "longshot_score",
                "consistency_score", "sneaky_score", "fair_odds"):
        assert col in df.columns


def test_score_ranges_are_sane():
    df = _slate()
    assert df.hr_score.between(0, 100).all()
    assert df.longshot_score.between(0, 100).all()
    assert df.consistency_score.between(0, 100).all()
    assert df.sneaky_score.between(0, 100).all()
    # Game HR probability should stay in a realistic band.
    assert df.hr_prob_game.between(0.0, 0.35).all()
    assert df.hr_prob_game.max() > 0.15  # at least some strong spots


def test_deterministic_by_date():
    a = build_demo_slate(dt.date(2026, 6, 18))
    b = build_demo_slate(dt.date(2026, 6, 18))
    assert a.equals(b)


def test_switch_hitter_and_platoon_logic():
    assert effective_bat_side("S", "R") == "L"
    assert effective_bat_side("S", "L") == "R"
    assert has_platoon_advantage("L", "R") is True
    assert has_platoon_advantage("R", "R") is False
    assert has_platoon_advantage("S", "L") is True


def test_park_and_wind_multipliers():
    # Coors should boost; Oracle/Oakland should suppress.
    assert park_hr_multiplier("COL", "R") > 1.0
    assert park_hr_multiplier("OAK", "L") < 1.0
    park = get_park("CHC")
    # Wind blowing straight out (toward CF bearing) should help; in should hurt.
    out = wind_hr_multiplier(park, 15, (park["orientation_deg"] + 180) % 360, "R")
    inw = wind_hr_multiplier(park, 15, park["orientation_deg"], "R")
    assert out > 1.0 > inw


def test_rationale_present():
    df = _slate()
    assert df["rationale"].astype(str).str.len().gt(0).all()
