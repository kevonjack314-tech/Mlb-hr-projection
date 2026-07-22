"""Within-series familiarity: hitters improve as they re-see a staff."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate


def test_later_series_games_raise_matchup():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    g1 = pd.Series({**base, "series_game": 1})
    g2 = pd.Series({**base, "series_game": 2})
    g3 = pd.Series({**base, "series_game": 3})
    m1, _ = matchup_multiplier(g1)
    m2, _ = matchup_multiplier(g2)
    m3, _ = matchup_multiplier(g3)
    assert m3 > m2 > m1


def test_missing_series_game_is_neutral():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    none = pd.Series(base)
    g1 = pd.Series({**base, "series_game": 1})
    m_none, _ = matchup_multiplier(none)
    m_g1, _ = matchup_multiplier(g1)
    assert m_none == m_g1     # game 1 == no-info == neutral


def test_scored_slate_carries_series_game():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "series_game" in df.columns
    assert df["series_game"].between(1, 4).all()
    assert df["series_game"].nunique() >= 2
