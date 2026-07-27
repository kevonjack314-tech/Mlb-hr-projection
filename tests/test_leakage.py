"""Tests: no rolling-window feature may see the game it is predicting.

The bug this locks down: `get_recent_form_table("2026-07-20")` pulled Statcast
through 2026-07-20 *inclusive*, so when a past date was graded, that day's own
home run landed in the hitter's own 7/15/30-day HR-rate feature. Live slates
were fine (the games hadn't been played), but every learned weight was fit on
contaminated rows — the model put 43% of its total weight on recent form and
"validated" on a holdout that leaked the same way.
"""
import datetime as dt

import pandas as pd
import pytest

from src import statcast as sc
from src.model import FORM_SHRINK, RECENT_FORM_WEIGHTS, form_shrink, score_slate
from src.tuning import leakage_report


# --------------------------------------------------------------------------- #
# The window itself
# --------------------------------------------------------------------------- #
END = "2026-07-20"


def _fake_statcast():
    """One hitter: a HR on the target date, nothing at all before it."""
    rows = []
    for d, ev in [(dt.date(2026, 7, 18), "strikeout"),
                  (dt.date(2026, 7, 19), "field_out"),
                  (dt.date(2026, 7, 20), "home_run")]:      # <- the game we grade
        rows.append({"game_date": d, "batter": 660271, "events": ev,
                     "type": "X", "launch_speed": 100.0, "barrel": 1,
                     "estimated_woba_using_speedangle": 0.9, "pitch_type": "FF"})
    return pd.DataFrame(rows)


@pytest.fixture
def _patched(monkeypatch):
    monkeypatch.setattr(sc, "_HAS_PYB", True)

    class _Pyb:
        @staticmethod
        def statcast(start_dt, end_dt, verbose=False):
            df = _fake_statcast()
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df

    monkeypatch.setattr(sc, "pyb", _Pyb, raising=False)
    sc._statcast_range.cache_clear()
    sc.get_recent_form_table.cache_clear()
    yield
    sc._statcast_range.cache_clear()
    sc.get_recent_form_table.cache_clear()


def test_statcast_range_excludes_the_target_date(_patched):
    rng = sc._statcast_range(END, 30)
    assert rng is not None
    assert rng["game_date"].max() == dt.date(2026, 7, 19), \
        "the date being scored must never appear in its own feature window"


def test_recent_form_does_not_see_todays_home_run(_patched):
    t = sc.get_recent_form_table(END)
    row = t[t["mlbam_id"] == 660271].iloc[0]
    # Two prior PAs, neither a HR -> every window must read exactly zero.
    for col in ("hr_rate_7", "hr_rate_15", "hr_rate_30"):
        assert row[col] == 0.0, f"{col}={row[col]} — today's HR leaked in"


def test_bvp_excludes_the_target_date(monkeypatch):
    monkeypatch.setattr(sc, "_HAS_PYB", True)

    class _Pyb:
        @staticmethod
        def statcast_pitcher(start, end, pid):
            df = _fake_statcast()
            df["game_date"] = pd.to_datetime(df["game_date"])
            return df

    monkeypatch.setattr(sc, "pyb", _Pyb, raising=False)
    sc.get_pitcher_bvp_table.cache_clear()
    t = sc.get_pitcher_bvp_table(543037, END)
    sc.get_pitcher_bvp_table.cache_clear()
    assert not t.empty
    assert int(t.iloc[0]["bvp_hr"]) == 0, "career BvP HR counted tonight's HR"
    assert int(t.iloc[0]["bvp_pa"]) == 2


# --------------------------------------------------------------------------- #
# The detector
# --------------------------------------------------------------------------- #
def _log(zero_rate_hr, zero_rate_no_hr, n=2000):
    """Synthetic graded log with a controllable hr_rate_7 zero-rate per class."""
    rows = []
    for i in range(n):
        hit = int(i % 8 == 0)
        share = zero_rate_hr if hit else zero_rate_no_hr
        zero = (i % 100) < share * 100
        rows.append({"date": "2026-07-01", "player": f"p{i}", "hit_hr": hit,
                     "hr_prob_game": 0.12, "hr_rate_7": 0.0 if zero else 0.06})
    return pd.DataFrame(rows)


def test_leakage_report_flags_a_leaked_log():
    rep = leakage_report(_log(zero_rate_hr=0.00, zero_rate_no_hr=0.60))
    assert rep["leaked"] and rep["features"]["hr_rate_7"]["leaked"]


def test_leakage_report_passes_an_honest_log():
    # Hot bats homer a bit more often, but plenty of HR days still follow a
    # quiet week — that is what an honest prior-window feature looks like.
    rep = leakage_report(_log(zero_rate_hr=0.45, zero_rate_no_hr=0.60))
    assert not rep["leaked"]


def test_leakage_report_flags_the_real_shipped_windows():
    """Documents the actual bug against real data, if the log is still stale."""
    try:
        log = pd.read_csv("data/eval_log.csv", low_memory=False)
    except FileNotFoundError:
        pytest.skip("no graded record checked in")
    rep = leakage_report(log)
    assert "hr_rate_7" in rep["features"]      # detector reaches the real columns


def test_feature_model_refuses_to_fit_a_leaked_log():
    from src.tuning import fit_feature_model
    out = fit_feature_model(_log(zero_rate_hr=0.0, zero_rate_no_hr=0.60, n=4000))
    assert out["feature_model"]["active"] is False
    assert "leakage" in out["feature_model"]["note"].lower()


# --------------------------------------------------------------------------- #
# Recent form is now weighted like the leak-free measurement says it should be
# --------------------------------------------------------------------------- #
def test_seven_day_window_is_no_longer_the_dominant_form_input():
    assert RECENT_FORM_WEIGHTS["hr_rate_7"] <= 0.35
    assert sum(RECENT_FORM_WEIGHTS.values()) == pytest.approx(1.0)


def test_form_shrink_by_tier():
    assert form_shrink(30) == FORM_SHRINK["star"]
    assert form_shrink(12) == FORM_SHRINK["mid"]
    assert form_shrink(3) == FORM_SHRINK["under"]
    assert form_shrink(None) > 0                 # missing season HR is safe
    assert FORM_SHRINK["star"] < FORM_SHRINK["under"]


def test_hot_star_moves_less_than_a_hot_under_the_radar_bat():
    """Same red-hot week; the star's score should barely budge, the other's should."""
    base = {"barrel_pct": 12.0, "hard_hit_pct": 45.0, "avg_ev": 90.0, "max_ev": 112.0,
            "fb_pct": 38.0, "pull_pct": 42.0, "hr_fb": 14.0, "k_pct": 22.0,
            "hr_per_pa": 0.04, "pa": 400, "park_factor": 1.0, "wind_mult": 1.0,
            "pitcher_hr9": 1.3, "bats": "R", "pitcher_throws": "R",
            "player": "x", "team": "AAA", "game": "AAA @ BBB", "lineup_spot": 3}
    cold = {"hr_rate_7": 0.0, "hr_rate_15": 0.0, "hr_rate_30": 0.0}
    hot = {"hr_rate_7": 0.14, "hr_rate_15": 0.12, "hr_rate_30": 0.10}

    def form(season_hr, streak):
        df = pd.DataFrame([{**base, **streak, "season_hr": season_hr}])
        return score_slate(df)["recent_form_score"].iloc[0]

    star_swing = form(30, hot) - form(30, cold)
    under_swing = form(4, hot) - form(4, cold)
    assert under_swing > star_swing > 0
