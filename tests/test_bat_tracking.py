"""Statcast bat-tracking: swing speed & squared-up rate."""

import datetime as dt
import io

import pandas as pd

from src.demo import build_demo_slate
from src.model import _power_quality_score, score_slate


def test_bat_speed_lifts_power_quality():
    base = {"barrel_pct": 9.0, "xiso": 0.170, "hard_hit_pct": 42.0,
            "xwoba": 0.330, "max_ev": 108.0, "avg_ev": 89.0}
    fast = pd.Series({**base, "bat_speed": 78.0})
    slow = pd.Series({**base, "bat_speed": 68.0})
    none = pd.Series(base)
    pq_fast = _power_quality_score(fast)["power_quality_score"]
    pq_slow = _power_quality_score(slow)["power_quality_score"]
    pq_none = _power_quality_score(none)["power_quality_score"]
    assert pq_fast > pq_slow
    # No bat-speed data -> the outcome-based core is untouched.
    assert pq_none == _power_quality_score(pd.Series(base))["power_quality_score"]


def test_bat_tracking_parser(monkeypatch):
    import requests
    from src import statcast as sc

    csv = ("id,name,avg_bat_speed,squared_up_per_swing,fast_swing_rate\n"
           "592450,Aaron Judge,77.4,0.29,0.62\n"
           "660271,Shohei Ohtani,75.9,0.27,0.58\n")

    class FakeResp:
        status_code = 200
        text = csv
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    sc.get_bat_tracking_table.cache_clear()
    df = sc.get_bat_tracking_table(2026).set_index("mlbam_id")
    assert df.loc[592450, "bat_speed"] == 77.4
    assert df.loc[592450, "squared_up_pct"] == 29.0    # fraction -> pct
    assert df.loc[660271, "fast_swing_pct"] == 58.0
    sc.get_bat_tracking_table.cache_clear()


def test_bat_tracking_bad_response_safe(monkeypatch):
    import requests
    from src import statcast as sc

    class FakeResp:
        status_code = 200
        text = "garbage,cols\n1,2\n"
        def raise_for_status(self):
            pass

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    sc.get_bat_tracking_table.cache_clear()
    assert sc.get_bat_tracking_table(2026).empty
    sc.get_bat_tracking_table.cache_clear()


def test_scored_slate_carries_bat_tracking():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    for c in ("bat_speed", "squared_up_pct", "fast_swing_pct"):
        assert c in df.columns
    assert df["bat_speed"].between(64, 80).all()
    assert df["bat_speed"].nunique() > 5
