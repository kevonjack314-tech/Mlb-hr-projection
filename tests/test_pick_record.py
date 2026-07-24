"""Tests for the pick-record tracker (HR of the Day + top picks + roles)."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import attach_confidence, hr_of_the_day, score_slate
from src.tuning import _streak, load_eval_log, pick_record


def test_streak():
    assert _streak([1, 1, 0, 1, 1, 1]) == "W3"
    assert _streak([1, 0, 0]) == "L2"
    assert _streak([]) == "—"


def test_confidence_and_hotd_on_slate():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    conf = attach_confidence(df)
    assert conf["confidence"].between(0, 100).all()
    pick = hr_of_the_day(df)
    assert pick is not None
    assert pick["confidence"] == conf["confidence"].max()


def test_pick_record_synthetic():
    log = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2", "d3", "d3"],
        "player": ["A", "B", "A", "C", "B", "C"],
        "team": ["X"] * 6,
        "hr_prob_game": [0.2, 0.1, 0.25, 0.05, 0.15, 0.06],
        "hit_hr": [1, 0, 0, 1, 1, 0],
        "hr_of_day": [1, 0, 1, 0, 1, 0],
        "top_pick": [1, 1, 1, 1, 1, 1],
        "parlay_role": ["Anchor", "Value", "Anchor", "Longshot", "Value", "Longshot"],
    })
    rec = pick_record(log)
    h = rec["hotd"]
    assert (h["wins"], h["losses"], h["days"]) == (2, 1, 3)
    assert h["streak"] == "W1"
    t = rec["top5"]
    assert t["picks"] == 6 and t["wins"] == 3 and t["days_with_hit"] == 3
    assert rec["roles"]["Anchor"]["record" if False else "legs"] == 2
    assert rec["roles"]["Longshot"]["wins"] == 1


def test_pick_record_on_real_log():
    rec = pick_record(load_eval_log())
    h = rec["hotd"]
    assert h is not None and h["days"] >= 30       # backfilled + stamped
    assert h["wins"] + h["losses"] == h["days"]
    assert 0 <= h["hit_rate"] <= 100
    t = rec["top5"]
    # ~5 picks/day, but a short-slate day (few games) can log fewer than 5.
    assert t is not None
    assert t["days"] * 4 <= t["picks"] <= t["days"] * 5
    assert set(rec["roles"]) == {"Anchor", "Value", "Longshot"}


def test_pick_record_empty():
    rec = pick_record(pd.DataFrame())
    assert rec["hotd"] is None and rec["top5"] is None and rec["roles"] == {}
