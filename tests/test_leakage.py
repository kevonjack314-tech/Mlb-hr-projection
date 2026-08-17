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


# --------------------------------------------------------------------------- #
# Per-date cleaning — a record gets fixed one day at a time
# --------------------------------------------------------------------------- #
def _mixed_log(clean_dates=("2026-07-26",), leaked_dates_=("2026-07-01",), n=400):
    """A log where some dates were graded before the window fix and some after."""
    rows = []
    for d in list(clean_dates) + list(leaked_dates_):
        leaked = d in leaked_dates_
        for i in range(n):
            hit = int(i % 8 == 0)
            # Leaked days: an HR day essentially never shows a zero window.
            zero = (i % 100) < (0 if (leaked and hit) else 60)
            rows.append({"date": d, "player": f"{d}-p{i}", "hit_hr": hit,
                         "hr_prob_game": 0.12, "hr_score": 60.0,
                         "hr_rate_7": 0.0 if zero else 0.06})
    return pd.DataFrame(rows)


def test_leaked_dates_are_identified_individually():
    from src.tuning import leaked_dates
    bad = leaked_dates(_mixed_log())
    assert bad == {"2026-07-01"}


def test_clean_log_keeps_the_good_days_and_drops_the_bad():
    from src.tuning import clean_log
    kept, note = clean_log(_mixed_log())
    assert set(kept["date"]) == {"2026-07-26"}
    assert note and "leaked date" in note
    # A fully clean log is passed straight through, untouched.
    ok = _mixed_log(clean_dates=("2026-07-26", "2026-07-27"), leaked_dates_=())
    kept2, note2 = clean_log(ok)
    assert len(kept2) == len(ok) and note2 is None


def test_calibration_refuses_to_train_on_leaked_dates():
    """The calibration curve touches every live probability — it must be clean."""
    from src.tuning import fit_calibration
    out = fit_calibration(_mixed_log(n=600))
    assert "excluded" in out
    # Only the clean date's rows may reach the fit.
    assert out["n"] <= 600


def test_a_date_with_too_few_home_runs_is_not_judged():
    """Too small to tell, too small to matter — keep it rather than guess."""
    from src.tuning import leaked_dates
    # 3 home runs on the date — below the 5 needed to call it either way, even
    # though none of them shows a zero window and the non-HR games do.
    tiny = pd.DataFrame({"date": ["2026-07-02"] * 20,
                         "hit_hr": [1, 1, 1] + [0] * 17,
                         "hr_rate_7": [0.06] * 3 + [0.0] * 12 + [0.06] * 5})
    assert leaked_dates(tiny) == set()
    # Same shape with 8 home runs IS judged, and flagged.
    big = pd.DataFrame({"date": ["2026-07-02"] * 40,
                        "hit_hr": [1] * 8 + [0] * 32,
                        "hr_rate_7": [0.06] * 8 + [0.0] * 24 + [0.06] * 8})
    assert leaked_dates(big) == {"2026-07-02"}


def test_a_feed_with_no_zeros_at_all_is_never_judged_leaked():
    """The worst failure mode would be silently deleting the whole training set.

    If a future feed change stopped emitting exact zeros, every date would look
    leaked and every fit would train on nothing. A date is only judged when its
    non-HR games show zeros to compare against.
    """
    from src.tuning import clean_log, leaked_dates
    smooth = pd.DataFrame({"date": ["2026-07-02"] * 200,
                           "hit_hr": [1] * 25 + [0] * 175,
                           "hr_rate_7": [0.03 + i * 1e-4 for i in range(200)]})
    assert leaked_dates(smooth) == set()
    kept, note = clean_log(smooth)
    assert len(kept) == 200 and note is None


def test_a_failed_feed_day_is_kept_not_called_clean_or_leaked():
    """All-NaN recent form means the pull failed — it cannot have leaked.

    Those rows still carry a valid prediction and outcome, so calibration
    should keep them; only the FEATURE model (which needs the columns) filters
    them out, via its coverage threshold.
    """
    from src.tuning import clean_log, fit_calibration, leaked_dates
    dead = pd.DataFrame({"date": ["2026-07-08"] * 400,
                         "player": [f"p{i}" for i in range(400)],
                         "hit_hr": [1 if i % 8 == 0 else 0 for i in range(400)],
                         "hr_prob_game": 0.12,
                         "hr_rate_7": [float("nan")] * 400})
    assert leaked_dates(dead) == set()
    kept, note = clean_log(dead)
    assert len(kept) == 400 and note is None
    assert fit_calibration(dead)["n"] == 400      # usable for calibration
