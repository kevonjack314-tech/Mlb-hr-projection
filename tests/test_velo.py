"""Fastball velocity decline: last start vs season baseline."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate
from src.statcast import velo_deltas


def _fb(pid, day, speed, n):
    return pd.DataFrame({"pitcher": [pid] * n, "game_date": [day] * n,
                         "release_speed": [speed] * n, "pitch_family": ["fb"] * n})


def test_velo_delta_last_vs_baseline():
    d1, d2, d3 = dt.date(2026, 7, 1), dt.date(2026, 7, 8), dt.date(2026, 7, 15)
    df = pd.concat([_fb(1, d1, 95.0, 20), _fb(1, d2, 95.0, 20), _fb(1, d3, 92.5, 20)])
    out = velo_deltas(df, min_fb=15).set_index("pitcher_id")
    assert out.loc[1, "sp_velo_last"] == 92.5
    assert out.loc[1, "sp_velo_base"] == 95.0
    assert out.loc[1, "sp_velo_delta"] == -2.5


def test_velo_ignores_nonfastballs_and_thin_samples():
    d1, d2 = dt.date(2026, 7, 1), dt.date(2026, 7, 8)
    # Only 5 FB last start -> below min_fb, excluded.
    df = pd.concat([_fb(1, d1, 94.0, 20), _fb(1, d2, 90.0, 5)])
    assert velo_deltas(df, min_fb=15).empty
    # Breaking balls never count toward FB velo.
    br = _fb(2, d1, 84.0, 20).assign(pitch_family="br")
    assert velo_deltas(pd.concat([br, _fb(2, d2, 94.0, 20)]), min_fb=15).empty


def test_velo_empty_safe():
    assert velo_deltas(pd.DataFrame()).empty
    assert velo_deltas(pd.DataFrame({"pitcher": [1]})).empty


def test_velo_drop_raises_matchup():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU"}
    dead_arm = pd.Series({**base, "sp_velo_delta": -2.0})
    fresh = pd.Series({**base, "sp_velo_delta": 0.0})
    m_dead, _ = matchup_multiplier(dead_arm)
    m_fresh, _ = matchup_multiplier(fresh)
    assert m_dead > m_fresh


def test_scored_slate_carries_velo():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    for c in ("sp_velo_delta", "sp_velo_last", "sp_velo_base"):
        assert c in df.columns
    # Some starters should be flagged with a velo dip.
    assert (df["sp_velo_delta"] <= -1.0).any()
