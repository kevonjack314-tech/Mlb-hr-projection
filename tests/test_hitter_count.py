"""Predictability in hitter's counts: auto-fastball × the batter's FB damage."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import pitch_type_matchup, score_slate
from src.statcast import hitter_count_fb


def _p(pid, balls, strikes, fam, n):
    return pd.DataFrame({"pitcher": [pid] * n, "balls": [balls] * n,
                         "strikes": [strikes] * n, "pitch_family": [fam] * n})


def test_hitter_count_fb_rate():
    # In 2-0 counts: 30 FB, 10 breaking -> 75% auto-fastball. Other counts ignored.
    df = pd.concat([_p(1, 2, 0, "fb", 30), _p(1, 2, 0, "br", 10),
                    _p(1, 0, 2, "fb", 50)])  # pitcher's count -> excluded
    out = hitter_count_fb(df, min_pitches=40).set_index("pitcher_id")
    assert out.loc[1, "sp_hitter_count_fb"] == 75.0


def test_hitter_count_min_pitches():
    df = _p(1, 3, 1, "fb", 10)
    assert hitter_count_fb(df, min_pitches=40).empty


def test_hitter_count_empty_safe():
    assert hitter_count_fb(pd.DataFrame()).empty
    assert hitter_count_fb(pd.DataFrame({"pitcher": [1]})).empty


def test_predictable_fb_helps_fb_masher():
    base = {"vs_fb": 0.420, "vs_br": 0.300, "vs_os": 0.300,
            "pitcher_mix_fb": 55, "pitcher_mix_br": 30, "pitcher_mix_os": 15}
    predictable = pd.Series({**base, "sp_hitter_count_fb": 70})
    sneaky = pd.Series({**base, "sp_hitter_count_fb": 45})
    m_pred, _ = pitch_type_matchup(predictable)
    m_sneak, _ = pitch_type_matchup(sneaky)
    assert m_pred > m_sneak
    # A weak FB hitter gets no gift from predictability.
    weak = pd.Series({**base, "vs_fb": 0.260, "sp_hitter_count_fb": 70})
    m_weak, _ = pitch_type_matchup(weak)
    m_weak_base, _ = pitch_type_matchup(pd.Series({**base, "vs_fb": 0.260,
                                                   "sp_hitter_count_fb": 45}))
    assert m_weak <= m_weak_base + 1e-9


def test_scored_slate_carries_hitter_count_fb():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "sp_hitter_count_fb" in df.columns
    assert df["sp_hitter_count_fb"].between(40, 75).all()
