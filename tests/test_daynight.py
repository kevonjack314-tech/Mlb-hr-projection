"""Day/night park splits: start-time HR modifier."""

import datetime as dt

from src.demo import build_demo_slate
from src.model import score_slate
from src.parks import daynight_hr_multiplier
from src.sources import _game_is_night


def test_marine_layer_parks_suppress_at_night():
    # SF day > neutral > SF night.
    assert daynight_hr_multiplier("SF", False) > 1.0
    assert daynight_hr_multiplier("SF", True) < 1.0
    assert daynight_hr_multiplier("SD", True) < daynight_hr_multiplier("SD", False)


def test_roofed_parks_are_neutral():
    for park in ("HOU", "ARI", "TB", "TOR", "MIA", "MIL", "TEX"):
        assert daynight_hr_multiplier(park, True) == 1.0
        assert daynight_hr_multiplier(park, False) == 1.0


def test_missing_starttime_is_neutral():
    assert daynight_hr_multiplier("SF", None) == 1.0
    assert daynight_hr_multiplier("SF", float("nan")) == 1.0


def test_game_is_night_from_utc():
    park = {"lon": -122.39}   # San Francisco
    # 02:15 UTC = ~6:07pm local -> night.
    assert _game_is_night("2026-07-15T02:15:00Z", park) is True
    # 20:05 UTC = ~12:07pm local -> day.
    assert _game_is_night("2026-07-15T20:05:00Z", park) is False
    assert _game_is_night(None, park) is None
    assert _game_is_night("2026-07-15T02:15:00Z", None) is None


def test_scored_slate_carries_daynight():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "daynight_mult" in df.columns
    assert df["daynight_mult"].between(0.90, 1.10).all()
