"""Ballpark fence geometry × pull side (porch fit)."""

import datetime as dt

from src.demo import build_demo_slate
from src.model import score_slate
from src.parks import porch_fit


def test_pull_lefty_loves_the_yankee_porch():
    # NYY right field is ~314 ft with a standard wall.
    pull_lhb = porch_fit("NYY", "L", pull_pct=48)
    spray_lhb = porch_fit("NYY", "L", pull_pct=40)
    oppo_lhb = porch_fit("NYY", "L", pull_pct=30)
    assert pull_lhb["park_fit_mult"] > spray_lhb["park_fit_mult"] > oppo_lhb["park_fit_mult"]
    assert pull_lhb["park_fit_mult"] > 1.02
    assert "porch" in pull_lhb["park_fit_note"]


def test_monster_neutralizes_fenways_short_left():
    # BOS LF is 310 ft but behind a 37-ft wall -> effective ~327: no porch gift.
    rhb = porch_fit("BOS", "R", pull_pct=48)
    assert rhb["park_porch_ft"] >= 320
    assert rhb["park_fit_mult"] < porch_fit("NYY", "L", pull_pct=48)["park_fit_mult"]


def test_average_pull_rate_stays_near_neutral():
    # An average-pull bat is already priced by the handedness park factor.
    fit = porch_fit("NYY", "L", pull_pct=40)
    assert 0.96 <= fit["park_fit_mult"] <= 1.04


def test_missing_park_or_pull_is_safe():
    assert porch_fit("???", "L", 45)["park_fit_mult"] == 1.0
    fit = porch_fit("NYY", "L", None)     # no pull data -> CF-only tweak
    assert 0.95 <= fit["park_fit_mult"] <= 1.05


def test_scored_slate_carries_porch_columns():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "park_fit_mult" in df.columns and "park_porch_ft" in df.columns
    assert df["park_fit_mult"].between(0.90, 1.12).all()
    # The signal must actually vary across the slate (not a constant).
    assert df["park_fit_mult"].nunique() > 5
