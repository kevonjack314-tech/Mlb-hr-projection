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
