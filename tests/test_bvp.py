"""Batter-vs-pitcher career HR history ('owns this guy')."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate
from src.statcast import bvp_counts


def test_bvp_counts_hr_and_pa():
    # Batter 1 vs this pitcher: 3 PA-ending events, 2 of them home runs.
    pitches = pd.DataFrame({
        "batter": [1, 1, 1, 1, 2, 2],
        "events": [None, "home_run", "strikeout", "home_run", "single", "home_run"],
    })
    out = bvp_counts(pitches).set_index("mlbam_id")
    assert out.loc[1, "bvp_pa"] == 3 and out.loc[1, "bvp_hr"] == 2
    assert out.loc[2, "bvp_pa"] == 2 and out.loc[2, "bvp_hr"] == 1


def test_bvp_counts_empty_safe():
    assert bvp_counts(pd.DataFrame()).empty
    assert bvp_counts(pd.DataFrame({"batter": [1]})).empty   # no events col


def test_bvp_history_raises_matchup():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    owns = pd.Series({**base, "bvp_hr": 3, "bvp_pa": 20})
    faced = pd.Series({**base, "bvp_hr": 0, "bvp_pa": 20})
    none = pd.Series(base)
    m_owns, _ = matchup_multiplier(owns)
    m_faced, _ = matchup_multiplier(faced)
    m_none, _ = matchup_multiplier(none)
    assert m_owns > m_faced == m_none        # 0 HR == no history == neutral


def test_bvp_small_sample_gate_and_cap():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    # 1 HR but only 3 career PA -> below the 5-PA gate, no boost.
    tiny = matchup_multiplier(pd.Series({**base, "bvp_hr": 1, "bvp_pa": 3}))[0]
    none = matchup_multiplier(pd.Series(base))[0]
    assert tiny == none
    # A huge HR count is capped at +10%.
    huge = matchup_multiplier(pd.Series({**base, "bvp_hr": 20, "bvp_pa": 80}))[0]
    assert huge <= none * 1.10 + 1e-9


def test_scored_slate_carries_bvp():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "bvp_hr" in df.columns and "bvp_pa" in df.columns
    assert (df["bvp_hr"] >= 1).any()          # some hitters own their matchup
