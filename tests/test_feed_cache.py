"""The data-feed cache must retry failures instead of pinning them forever.

Regression test for the deployed-app bug where one transient Savant/FanGraphs
hiccup at boot blanked every season metric (all '—') until a full restart,
because lru_cache memoized the empty table for the process lifetime.
"""

import pandas as pd

from src import statcast as sc


def test_cache_ok_retries_failures_but_pins_successes(monkeypatch):
    monkeypatch.setattr(sc, "_FAIL_TTL", 0.0)   # failed pulls retry immediately

    calls = {"n": 0}

    @sc._cache_ok
    def flaky(year):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame()               # transient failure
        return pd.DataFrame({"x": [1]})         # feed recovered

    assert flaky(2026).empty                    # first call fails...
    assert not flaky(2026).empty                # ...second call RETRIES and wins
    assert not flaky(2026).empty
    assert calls["n"] == 2                      # success is pinned (no 3rd pull)


def test_cache_ok_remembers_failures_briefly(monkeypatch):
    monkeypatch.setattr(sc, "_FAIL_TTL", 3600.0)
    calls = {"n": 0}

    @sc._cache_ok
    def always_down(year):
        calls["n"] += 1
        return None

    assert always_down(2026) is None
    assert always_down(2026) is None
    assert calls["n"] == 1     # within the fail-TTL there's no hammering


def test_diagnostics_capture_and_clear():
    sc.note_diag("test_feed", ValueError("boom"))
    assert "boom" in sc.get_diagnostics()["test_feed"]
    sc._DIAG.pop("test_feed", None)


def test_previous_hrs_metrics_come_from_eval_record():
    """The Previous-HRs card must show the metrics the model gave the player
    BEFORE that game — sourced from the graded eval record, so a live-feed
    outage can never blank them."""
    import pandas as pd
    from src.history import _eval_feature_index, eval_features_for

    idx = _eval_feature_index()
    ev = pd.read_csv("data/eval_log.csv")
    hrs = ev[(ev["hit_hr"] == 1) & ev["barrel_pct"].notna()]
    assert len(idx) > 1000 and len(hrs) > 100     # backfilled record is present

    sample = hrs.iloc[0]
    got = eval_features_for(idx, sample["date"], sample["player"])
    assert got is not None
    feats, scored = got
    for key in ("barrel_pct", "hard_hit_pct", "fb_pct", "xiso", "season_hr"):
        assert key in feats and feats[key] == feats[key]
    assert 0 < scored["hr_prob_game"] < 1 and scored["hr_score"] > 0
    # Nothing that score_slate re-derives may leak in as an input.
    assert not {"matchup_score", "env_score", "ulx_checks", "park_factor"} & set(feats)
