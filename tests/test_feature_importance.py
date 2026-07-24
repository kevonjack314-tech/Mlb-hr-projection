"""Feature-importance panel: surface the learned model's weights."""

import pandas as pd
import pytest

from src import tuning
from src.tuning import feature_importance


@pytest.fixture()
def isolated_tuning(tmp_path, monkeypatch):
    monkeypatch.setattr(tuning, "TUNING_PATH", str(tmp_path / "tune.json"))
    tuning.reload_tuning()
    yield
    tuning.reload_tuning()


def test_none_until_trained(isolated_tuning):
    tuning.save_tuning({"feature_model": {"active": False, "note": "warming up"}})
    assert feature_importance() is None


def test_ranked_by_absolute_weight(isolated_tuning):
    tuning.save_tuning({"feature_model": {
        "active": True, "n": 5000, "val_days": 6,
        "val_brier_model": 0.09, "val_brier_baseline": 0.10,
        "features": ["barrel_pct", "sp_meatball_pct", "series_game"],
        "coef": [0.40, -0.20, 0.05],
    }})
    fi = feature_importance()
    assert fi["active"] is True and fi["n"] == 5000
    rows = fi["rows"]
    # Sorted by |weight|: barrel (0.40) > meatball (0.20) > series (0.05).
    assert [r["feature"] for r in rows] == ["barrel_pct", "sp_meatball_pct", "series_game"]
    # Direction sign is surfaced.
    assert rows[0]["direction"].startswith("↑")
    assert rows[1]["direction"].startswith("↓")
    # Importance is a share of total |coef| (0.40/0.65 ~ 61.5%).
    assert rows[0]["importance_pct"] == pytest.approx(61.5, abs=0.5)
    # Human-readable labels.
    assert rows[0]["label"] == "Barrel%"
    assert rows[1]["label"] == "Starter meatball rate"


def test_top_n_limit(isolated_tuning):
    feats = [f"f{i}" for i in range(30)]
    tuning.save_tuning({"feature_model": {
        "active": True, "n": 5000, "features": feats,
        "coef": [float(i) for i in range(30)]}})
    fi = feature_importance(top_n=10)
    assert len(fi["rows"]) == 10
    assert fi["rows"][0]["feature"] == "f29"     # largest coef first


def test_handles_mismatched_lengths(isolated_tuning):
    tuning.save_tuning({"feature_model": {
        "active": True, "features": ["a", "b"], "coef": [0.1]}})
    assert feature_importance() is None
