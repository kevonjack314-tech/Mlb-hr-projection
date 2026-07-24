"""Rolling contact-quality trend: a stabler 'heating up' signal than HR-rate-7."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import score_row, score_slate


def _base():
    return {"barrel_pct": 9.0, "xiso": 0.170, "hard_hit_pct": 42.0, "xwoba": 0.330,
            "max_ev": 108.0, "avg_ev": 89.0, "hr_per_pa": 0.035, "lineup_spot": 4,
            "bats": "R", "pitcher_throws": "L", "hr_rate_7": 0.03, "hr_rate_15": 0.03,
            "hr_rate_30": 0.03}


def test_positive_trend_lifts_recent_form():
    rising = score_row(pd.Series({**_base(), "barrel_trend": 6.0, "xwoba_trend": 0.05}))
    flat = score_row(pd.Series({**_base(), "barrel_trend": 0.0, "xwoba_trend": 0.0}))
    cooling = score_row(pd.Series({**_base(), "barrel_trend": -6.0, "xwoba_trend": -0.05}))
    assert rising["recent_form_score"] > flat["recent_form_score"] > cooling["recent_form_score"]


def test_trend_nudge_is_bounded():
    # An absurd trend can't blow up the form score beyond the +/-12 clamp.
    huge = score_row(pd.Series({**_base(), "barrel_trend": 50.0, "xwoba_trend": 1.0}))
    flat = score_row(pd.Series({**_base(), "barrel_trend": 0.0, "xwoba_trend": 0.0}))
    assert huge["recent_form_score"] - flat["recent_form_score"] <= 12.0 + 1e-6


def test_missing_trend_is_neutral():
    base = _base()
    with_none = score_row(pd.Series(base))
    with_zero = score_row(pd.Series({**base, "barrel_trend": 0.0, "xwoba_trend": 0.0}))
    assert abs(with_none["recent_form_score"] - with_zero["recent_form_score"]) < 1e-6


def test_scored_slate_carries_trend_cols():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    for c in ("barrel_pct_14", "barrel_trend", "xwoba_trend"):
        assert c in df.columns
    assert df["barrel_trend"].nunique() > 5
    # Hot bats (positive trend) should exist and lift the form score.
    assert (df["barrel_trend"] > 0).any() and (df["barrel_trend"] < 0).any()
