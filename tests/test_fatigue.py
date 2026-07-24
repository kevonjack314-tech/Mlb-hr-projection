"""Hitter fatigue: consecutive games & day-after-night."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate


def _base():
    return {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}


def test_long_stretch_penalizes():
    fresh = pd.Series({**_base(), "bat_games_in_row": 3})
    tired = pd.Series({**_base(), "bat_games_in_row": 13})
    m_fresh, _ = matchup_multiplier(fresh)
    m_tired, _ = matchup_multiplier(tired)
    assert m_tired < m_fresh


def test_short_stretch_is_neutral():
    # Under the 8-game threshold there's no penalty.
    a = matchup_multiplier(pd.Series({**_base(), "bat_games_in_row": 3}))[0]
    b = matchup_multiplier(pd.Series({**_base(), "bat_games_in_row": 7}))[0]
    assert a == b


def test_day_after_night_penalizes():
    normal = pd.Series({**_base(), "day_after_night": False})
    dan = pd.Series({**_base(), "day_after_night": True})
    assert matchup_multiplier(dan)[0] < matchup_multiplier(normal)[0]


def test_missing_fatigue_is_neutral():
    none = matchup_multiplier(pd.Series(_base()))[0]
    zero = matchup_multiplier(pd.Series({**_base(), "bat_games_in_row": 5,
                                         "day_after_night": False}))[0]
    assert none == zero


def test_team_fatigue_table_parsing(monkeypatch):
    from src import sources

    def fake_json(url, params=None):
        # Team 147 plays 3 straight days ending on the 15th; a night game on
        # the 14th and a day game on the 15th -> day_after_night.
        return {"dates": [
            {"date": "2026-07-13", "games": [{"dayNight": "night",
                "teams": {"home": {"team": {"id": 147}},
                          "away": {"team": {"id": 111}}}}]},
            {"date": "2026-07-14", "games": [{"dayNight": "night",
                "teams": {"home": {"team": {"id": 147}},
                          "away": {"team": {"id": 111}}}}]},
            {"date": "2026-07-15", "games": [{"dayNight": "day",
                "teams": {"home": {"team": {"id": 147}},
                          "away": {"team": {"id": 111}}}}]},
        ]}

    monkeypatch.setattr(sources, "_get_json", fake_json)
    monkeypatch.setattr(sources, "_TEAM_ID_TO_ABBR", {147: "NYY", 111: "BOS"})
    sources.team_fatigue_table.cache_clear()
    tbl = {a: (g, d) for a, g, d in sources.team_fatigue_table("2026-07-15")}
    assert tbl["NYY"][0] == 3        # 3 games in a row
    assert tbl["NYY"][1] is True     # day after night
    sources.team_fatigue_table.cache_clear()


def test_scored_slate_carries_fatigue():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "bat_games_in_row" in df.columns
    assert df["bat_games_in_row"].between(3, 14).all()
