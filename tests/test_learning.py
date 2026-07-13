"""Tests for the learned feature model + feature logging (src/tuning.py)."""

import numpy as np
import pandas as pd
import pytest

from src import tuning
from src.tuning import (
    EVAL_COLS,
    FEATURE_COLS,
    apply_feature_model,
    fit_feature_model,
)


@pytest.fixture()
def isolated_tuning(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "EVAL_LOG_PATH", str(tmp_path / "eval.csv"))
    monkeypatch.setattr(tuning, "TUNING_PATH", str(tmp_path / "tune.json"))
    tuning.reload_tuning()
    yield
    tuning.reload_tuning()


def test_eval_cols_carry_the_feature_vector():
    # The graded record must log the model's raw inputs, not just the outputs.
    for f in ("barrel_pct", "hard_hit_pct", "fb_pct", "hr_fb", "xiso",
              "park_factor", "hr_rate_7", "season_hr", "expected_pa"):
        assert f in FEATURE_COLS and f in EVAL_COLS


def _synthetic_log(n_days=12, per_day=250, seed=7):
    """Hitter-days where hit_hr truly depends on barrel_pct + fb_pct, while the
    baseline hr_prob_game is a flat (uninformative) 6%."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_days):
        date = f"2026-06-{d+1:02d}"
        barrel = rng.uniform(2, 20, per_day)
        fb = rng.uniform(20, 50, per_day)
        z = -3.2 + 0.12 * barrel + 0.02 * fb
        y = rng.random(per_day) < 1 / (1 + np.exp(-z))
        df = pd.DataFrame({
            "date": date,
            "player": [f"p{d}_{i}" for i in range(per_day)],
            "hr_prob_game": 0.06,
            "hit_hr": y.astype(int),
            "barrel_pct": barrel,
            "fb_pct": fb,
        })
        # Fill the rest of the feature vector with noise so coverage is high.
        for f in FEATURE_COLS:
            if f not in df.columns:
                df[f] = rng.uniform(0, 100, per_day)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_feature_model_warms_up_below_threshold(isolated_tuning):
    small = _synthetic_log(n_days=2, per_day=100)
    fm = fit_feature_model(small)["feature_model"]
    assert fm["active"] is False and "warming" in fm["note"]


def test_feature_model_learns_and_beats_flat_baseline(isolated_tuning):
    log = _synthetic_log()
    fm = fit_feature_model(log)["feature_model"]
    assert fm["n"] >= tuning.FEATURE_MODEL_MIN_ROWS
    assert fm["val_brier_model"] < fm["val_brier_baseline"]
    assert fm["active"] is True
    # The truly predictive feature should carry real positive weight.
    coef = dict(zip(fm["features"], fm["coef"]))
    assert coef["barrel_pct"] > abs(coef["ld_pct"])


def test_apply_feature_model_gated_and_clamped(isolated_tuning):
    log = _synthetic_log()

    # Inactive model -> exact no-op.
    tuning.save_tuning({"feature_model": {"active": False}})
    slate = log.head(50).copy()
    out = apply_feature_model(slate)
    assert out is slate

    # Active model -> probabilities move, but stay in the clamp band.
    t = fit_feature_model(log)
    tuning.save_tuning(t)
    out = apply_feature_model(slate)
    assert out is not slate
    p0 = slate["hr_prob_game"].to_numpy()
    p1 = out["hr_prob_game"].to_numpy()
    assert (p1 >= 0.6 * p0 - 1e-9).all() and (p1 <= 1.4 * p0 + 1e-9).all()
    assert (p1 != p0).any()


def test_scoring_still_works_with_feature_model_active(isolated_tuning):
    import datetime as dt
    from src.demo import build_demo_slate
    from src.model import score_slate

    tuning.save_tuning(fit_feature_model(_synthetic_log()))
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert df["hr_prob_game"].between(0.0, 0.35).all()
    assert df["fair_odds"].notna().all()
