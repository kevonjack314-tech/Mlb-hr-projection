"""Tests: the score predictor's feedback loop.

Grading is kept as a pure join (`grade_projections`) so the whole record —
run error, winner accuracy, calibration, park bias — is testable offline
without touching the network.
"""
import numpy as np
import pandas as pd
import pytest

from src.gamegrade import (
    GAME_EVAL_COLS, append_game_rows, game_record, grade_projections,
    total_bias_by_park, win_prob_calibration,
)


def _projections():
    return pd.DataFrame([
        # projected home win, projected Over  -> home wins 7-3, total 10 (Over)
        {"game": "AAA @ BBB", "home_team": "BBB", "away_team": "AAA",
         "home_runs_exp": 5.4, "away_runs_exp": 3.9, "total_exp": 9.3,
         "most_likely_total": 8, "winner": "BBB", "win_prob": 0.62, "p_home": 0.62,
         "total_line": 8.5, "p_over": 0.61, "total_lean": "Over",
         "total_lean_prob": 0.61, "confidence": 40.0, "data_quality_pct": 100.0,
         "gotd": 1},
        # projected away win, projected Under -> away loses 2-6, total 8 (Under)
        {"game": "CCC @ DDD", "home_team": "DDD", "away_team": "CCC",
         "home_runs_exp": 3.8, "away_runs_exp": 4.6, "total_exp": 8.4,
         "most_likely_total": 7, "winner": "CCC", "win_prob": 0.55, "p_home": 0.45,
         "total_line": 8.5, "p_over": 0.44, "total_lean": "Under",
         "total_lean_prob": 0.56, "confidence": 25.0, "data_quality_pct": 90.0,
         "gotd": 0},
    ])


def _finals():
    return [
        {"game": "AAA @ BBB", "home_team": "BBB", "away_team": "AAA",
         "home_runs": 7, "away_runs": 3, "innings": 9},
        {"game": "CCC @ DDD", "home_team": "DDD", "away_team": "CCC",
         "home_runs": 6, "away_runs": 2, "innings": 9},
    ]


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def test_grade_joins_projections_to_real_finals():
    g = grade_projections(_projections(), _finals(), "2026-07-20")
    assert len(g) == 2
    assert set(g.columns) <= set(GAME_EVAL_COLS)
    r = g.set_index("game")
    assert r.loc["AAA @ BBB", "winner_actual"] == "BBB"
    assert r.loc["AAA @ BBB", "winner_correct"] == 1
    assert r.loc["CCC @ DDD", "winner_correct"] == 0     # picked CCC, DDD won


def test_run_errors_are_absolute_and_per_side():
    g = grade_projections(_projections(), _finals(), "2026-07-20").set_index("game")
    row = g.loc["AAA @ BBB"]
    assert row["abs_err_home"] == pytest.approx(abs(7 - 5.4), abs=1e-6)
    assert row["abs_err_away"] == pytest.approx(abs(3 - 3.9), abs=1e-6)
    assert row["abs_err_total"] == pytest.approx(abs(10 - 9.3), abs=1e-6)


def test_total_lean_is_graded_against_the_line():
    g = grade_projections(_projections(), _finals(), "2026-07-20").set_index("game")
    # 10 runs > 8.5 and the model said Over -> correct.
    assert g.loc["AAA @ BBB", "total_over_actual"] == 1
    assert g.loc["AAA @ BBB", "total_lean_correct"] == 1
    # 8 runs < 8.5 and the model said Under -> correct.
    assert g.loc["CCC @ DDD", "total_over_actual"] == 0
    assert g.loc["CCC @ DDD", "total_lean_correct"] == 1


def test_a_push_is_not_scored_as_a_loss():
    """A whole-number line landing exactly on the total pushes — grade it -1."""
    proj = _projections().head(1).assign(total_line=10.0)
    g = grade_projections(proj, _finals(), "2026-07-20").iloc[0]
    assert g["total_over_actual"] == -1
    assert g["total_lean_correct"] == -1


def test_games_without_a_final_are_dropped_not_half_graded():
    g = grade_projections(_projections(), _finals()[:1], "2026-07-20")
    assert list(g["game"]) == ["AAA @ BBB"]
    assert grade_projections(_projections(), [], "2026-07-20").empty
    assert grade_projections(pd.DataFrame(), _finals(), "2026-07-20").empty


# --------------------------------------------------------------------------- #
# The record + baselines
# --------------------------------------------------------------------------- #
def _log(n=60, seed=7):
    """Synthetic graded log: home wins 55% and the model is right 65% of the time."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        home_won = rng.random() < 0.55
        model_right = rng.random() < 0.65
        picked_home = home_won if model_right else not home_won
        ha = 5 if home_won else 3
        aa = 3 if home_won else 5
        rows.append({
            "date": f"2026-07-{(i % 20) + 1:02d}", "game": f"A{i} @ B{i}",
            "home_team": "BBB", "away_team": "AAA",
            "home_runs_exp": 4.5, "away_runs_exp": 4.2, "total_exp": 8.7,
            "winner": "BBB" if picked_home else "AAA",
            "win_prob": 0.60, "p_home": 0.60 if picked_home else 0.40,
            "total_line": 8.5, "total_lean": "Under", "total_lean_prob": 0.55,
            "confidence": 30.0, "gotd": int(i % 20 == 0),
            "home_runs_actual": ha, "away_runs_actual": aa,
            "total_actual": ha + aa, "innings": 9,
            "winner_actual": "BBB" if home_won else "AAA",
            "winner_correct": int(picked_home == home_won),
            "total_over_actual": 0, "total_lean_correct": 1,
            "abs_err_home": abs(ha - 4.5), "abs_err_away": abs(aa - 4.2),
            "abs_err_total": abs(ha + aa - 8.7),
        })
    return pd.DataFrame(rows)


def test_record_reports_accuracy_and_baselines():
    rec = game_record(_log())
    assert rec["n"] == 60 and rec["days"] == 20
    assert 0 < rec["mae_total"] < 5
    assert 50 <= rec["winner_pct"] <= 80
    # The baseline is the whole point — a bare accuracy number means nothing.
    assert "home_baseline_pct" in rec and "ml_brier_baseline" in rec
    assert rec["beats_home_baseline"] is True


def test_record_calls_out_a_model_that_loses_to_its_baseline():
    """An always-wrong model must be reported as NOT beating the baseline."""
    bad = _log()
    bad["winner_correct"] = 0
    bad["p_home"] = 0.95
    bad["home_runs_actual"], bad["away_runs_actual"] = 2, 6   # away always wins
    rec = game_record(bad)
    assert rec["beats_home_baseline"] is False
    assert rec["beats_brier_baseline"] is False


def test_total_bias_sign_is_readable():
    """+bias must mean the model projects too MANY runs."""
    hot = _log()
    hot["total_exp"] = hot["total_actual"] + 2.0
    rec = game_record(hot)
    assert rec["total_bias"] == pytest.approx(2.0, abs=0.01)
    cold = _log()
    cold["total_exp"] = cold["total_actual"] - 1.5
    assert game_record(cold)["total_bias"] == pytest.approx(-1.5, abs=0.01)


def test_empty_record_is_safe():
    rec = game_record(pd.DataFrame())
    assert rec == {"n": 0, "days": 0}
    assert win_prob_calibration(pd.DataFrame()).empty
    assert total_bias_by_park(pd.DataFrame()).empty


# --------------------------------------------------------------------------- #
# Calibration + park bias
# --------------------------------------------------------------------------- #
def test_win_prob_calibration_buckets():
    cal = win_prob_calibration(_log(n=200, seed=3), bins=4)
    assert not cal.empty
    assert {"games", "predicted", "actual", "gap"} <= set(cal.columns)
    assert (cal["predicted"].between(0, 100)).all()
    assert (cal["actual"].between(0, 100)).all()


def test_park_bias_flags_a_stale_run_factor():
    """A park the model is always too high on must surface with a + bias."""
    log = _log(n=80, seed=11)
    log.loc[log.index[:20], "home_team"] = "COL"
    log.loc[log.index[:20], "total_exp"] = log.loc[log.index[:20], "total_actual"] + 3.0
    t = total_bias_by_park(log)
    assert not t.empty
    top = t.iloc[0]
    assert top["home_team"] == "COL" and top["bias"] > 2.0


# --------------------------------------------------------------------------- #
# Log I/O
# --------------------------------------------------------------------------- #
def test_append_is_deduped_by_date_and_game(tmp_path, monkeypatch):
    from src import gamegrade as gg
    monkeypatch.setattr(gg, "GAME_EVAL_LOG_PATH", str(tmp_path / "g.csv"))
    rows = grade_projections(_projections(), _finals(), "2026-07-20")
    assert append_game_rows(rows) == 2
    assert append_game_rows(rows) == 0          # same day+games -> no growth
    assert len(gg.load_game_log()) == 2
    again = grade_projections(_projections(), _finals(), "2026-07-21")
    assert append_game_rows(again) == 2
