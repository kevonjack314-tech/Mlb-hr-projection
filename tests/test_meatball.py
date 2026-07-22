"""Meatball (middle-middle) supply: a mistake-rate HR signal."""

import datetime as dt

import numpy as np
import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate
from src.statcast import meatball_rates


def test_meatball_rates_counts_zone5():
    # Pitcher 1 grooves 3/10 down the middle; pitcher 2 only 1/10.
    pitches = pd.DataFrame({
        "pitcher": [1] * 10 + [2] * 10,
        "zone": [5, 5, 5, 1, 2, 3, 4, 6, 7, 8] + [5, 1, 2, 3, 4, 6, 7, 8, 9, 11],
    })
    out = meatball_rates(pitches, min_pitches=10).set_index("pitcher_id")
    assert out.loc[1, "sp_meatball_pct"] == 30.0
    assert out.loc[2, "sp_meatball_pct"] == 10.0


def test_meatball_respects_min_pitches():
    pitches = pd.DataFrame({"pitcher": [1] * 5, "zone": [5, 5, 1, 2, 3]})
    assert meatball_rates(pitches, min_pitches=100).empty


def test_meatball_empty_safe():
    assert meatball_rates(pd.DataFrame()).empty
    assert meatball_rates(pd.DataFrame({"pitcher": [1]})).empty  # no zone col


def test_meatball_raises_matchup_multiplier():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    groover = pd.Series({**base, "sp_meatball_pct": 7.5})
    painter = pd.Series({**base, "sp_meatball_pct": 3.5})
    neutral = pd.Series(base)
    m_groove, _ = matchup_multiplier(groover)
    m_paint, _ = matchup_multiplier(painter)
    m_none, _ = matchup_multiplier(neutral)
    assert m_groove > m_none > m_paint


def test_scored_slate_carries_meatball():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "sp_meatball_pct" in df.columns
    assert df["sp_meatball_pct"].between(2, 9).all()
    assert df["sp_meatball_pct"].nunique() > 3
