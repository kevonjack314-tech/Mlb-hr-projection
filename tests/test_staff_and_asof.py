"""Tests: opposing-staff quality beyond home runs, and point-in-time stats.

Two gaps this closes.

1. The run model graded a starter on HR/9 and barrel% allowed and nothing else
   — it could not tell a 30% strikeout arm from a 15% one, which is the primary
   run-suppression mechanism in baseball. That is why projected totals spread
   0.56 runs against reality's 4.54.

2. `expected_pa` split every hitter's night between starter and bullpen with a
   fixed 65/35, so an opener was priced identically to a seven-inning arm.

3. Backfilled season stats were looked up "as of today", so a June graded row
   carried the hitter's end-of-July line.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src import statcast as sc
from src.gamescore import (
    LEAGUE_PEN_K_PCT, LEAGUE_SP_K_PCT, SP_INNINGS_SHARE, pitching_multiplier,
    sp_innings_share,
)
from src.lineup import expected_pa, pa_split


# --------------------------------------------------------------------------- #
# Strikeouts drive the run environment
# --------------------------------------------------------------------------- #
def _staff(**kw):
    base = {"pitcher_hr9": [1.25] * 9, "pitcher_barrel_pct_allowed": [8.0] * 9,
            "pitcher_k_pct": [LEAGUE_SP_K_PCT] * 9, "pitcher_fip": [4.15] * 9,
            "bullpen_hr9": [1.15] * 9, "bullpen_k_pct": [LEAGUE_PEN_K_PCT] * 9,
            "bullpen_era": [4.05] * 9}
    base.update({k: [v] * 9 for k, v in kw.items()})
    return pd.DataFrame(base)


def test_strikeout_rate_moves_the_run_environment():
    """The signal the model was missing entirely."""
    ace = pitching_multiplier(_staff(pitcher_k_pct=32.0))
    soft = pitching_multiplier(_staff(pitcher_k_pct=14.0))
    assert soft["sp_mult"] > 1.0 > ace["sp_mult"]
    # And it has to be a BIG effect — bigger than the barrel term it sits next to.
    brl = pitching_multiplier(_staff(pitcher_barrel_pct_allowed=12.0))
    assert abs(soft["sp_mult"] - 1.0) > abs(brl["sp_mult"] - 1.0)


def test_two_starters_with_identical_hr9_are_no_longer_identical():
    """Same home-run profile, opposite strikeout profile — different games."""
    a = pitching_multiplier(_staff(pitcher_k_pct=31.0))
    b = pitching_multiplier(_staff(pitcher_k_pct=15.0))
    assert a["sp_hr9"] == b["sp_hr9"]
    assert abs(a["staff_mult"] - b["staff_mult"]) > 0.10


def test_fip_is_preferred_over_hr9_when_present():
    good_fip = pitching_multiplier(_staff(pitcher_fip=2.90))
    bad_fip = pitching_multiplier(_staff(pitcher_fip=5.40))
    assert bad_fip["sp_mult"] > good_fip["sp_mult"]
    # With FIP absent the HR/9 fallback still works.
    no_fip = _staff().drop(columns=["pitcher_fip"])
    assert pitching_multiplier(no_fip)["sp_mult"] == pytest.approx(1.0, abs=0.02)


def test_bullpen_quality_beyond_home_runs():
    fire = pitching_multiplier(_staff(bullpen_era=5.60, bullpen_k_pct=17.0))
    lights_out = pitching_multiplier(_staff(bullpen_era=2.80, bullpen_k_pct=30.0))
    assert fire["pen_mult"] > 1.0 > lights_out["pen_mult"]


def test_missing_new_columns_are_still_safe():
    """Feeds go partial; a slate without K% must not crash or skew."""
    bare = pd.DataFrame({"pitcher_hr9": [1.25] * 9})
    out = pitching_multiplier(bare)
    assert out["sp_mult"] == pytest.approx(1.0, abs=0.02)
    assert pitching_multiplier(pd.DataFrame())["staff_mult"] == 1.0


# --------------------------------------------------------------------------- #
# How deep the starter goes
# --------------------------------------------------------------------------- #
def test_innings_share_follows_real_ip_per_start():
    assert sp_innings_share(7.0) > sp_innings_share(5.0) > sp_innings_share(2.0)
    assert sp_innings_share(None) == SP_INNINGS_SHARE
    assert sp_innings_share(float("nan")) == SP_INNINGS_SHARE
    assert 0.15 <= sp_innings_share(0.5) <= 0.85      # an opener is clamped


def test_pa_split_prices_an_opener_differently_from_a_workhorse():
    """The fixed 65/35 blend priced these the same. They are not the same."""
    horse = pa_split(3, sp_ip_per_start=6.8)
    opener = pa_split(3, sp_ip_per_start=1.5)
    assert horse["pa_vs_sp"] > opener["pa_vs_sp"]
    assert opener["pa_vs_pen"] > horse["pa_vs_pen"]
    # Total trips are set by the lineup spot and don't change.
    assert horse["pa_total"] == opener["pa_total"] == expected_pa(3)
    for s in (horse, opener):
        assert s["pa_vs_sp"] + s["pa_vs_pen"] == pytest.approx(s["pa_total"])


def test_pa_split_without_ip_falls_back_sanely():
    s = pa_split(4, None)
    assert 0.5 < s["sp_share"] < 0.75
    assert s["pa_total"] == expected_pa(4)


def test_opener_shifts_the_hr_matchup_toward_the_bullpen():
    """A homer-prone pen matters more when the starter goes two innings."""
    from src.model import matchup_multiplier
    row = {"pitcher_hr9": 0.7, "bullpen_hr9": 2.0, "lineup_spot": 3,
           "bats": "R", "pitcher_throws": "R"}
    horse = matchup_multiplier({**row, "sp_ip_per_start": 6.8})
    opener = matchup_multiplier({**row, "sp_ip_per_start": 1.5})
    assert opener > horse, "facing a bad pen all night must raise HR odds"


# --------------------------------------------------------------------------- #
# Point-in-time season stats
# --------------------------------------------------------------------------- #
def _season_pitches():
    """One hitter: quiet in April, then a huge July. As of June he is not a star."""
    rows = []
    for d, ev, speed, barrel in (
        [(dt.date(2026, 4, 5), "field_out", 85.0, 0)] * 30
        + [(dt.date(2026, 4, 6), "single", 90.0, 0)] * 10
        + [(dt.date(2026, 7, 20), "home_run", 108.0, 1)] * 40
    ):
        rows.append({"game_date": d, "batter": 660271, "events": ev, "type": "X",
                     "launch_speed": speed, "launch_angle": 22.0, "barrel": barrel,
                     "estimated_woba_using_speedangle": 0.9 if barrel else 0.25,
                     "bb_type": "fly_ball"})
    return pd.DataFrame(rows)


@pytest.fixture
def _season(monkeypatch):
    monkeypatch.setattr(sc, "_HAS_PYB", True)

    class _Pyb:
        @staticmethod
        def statcast(start_dt, end_dt, verbose=False):
            df = _season_pitches()
            lo, hi = dt.date.fromisoformat(start_dt), dt.date.fromisoformat(end_dt)
            df = df[(df["game_date"] >= lo) & (df["game_date"] <= hi)]
            out = df.copy()
            out["game_date"] = pd.to_datetime(out["game_date"])
            return out

    monkeypatch.setattr(sc, "pyb", _Pyb, raising=False)
    sc._statcast_month.cache_clear()
    sc.get_season_batter_table_as_of.cache_clear()
    yield
    sc._statcast_month.cache_clear()
    sc.get_season_batter_table_as_of.cache_clear()


def test_as_of_june_does_not_see_the_july_home_runs(_season):
    """The lookahead this closes: a June row carrying an end-of-July line."""
    june = sc.get_season_batter_table_as_of("2026-06-15")
    row = june[june["mlbam_id"] == 660271].iloc[0]
    assert row["season_hr"] == 0, "July home runs leaked into a June stat line"
    assert row["barrel_pct"] == 0.0
    assert row["avg_ev"] < 90.0


def test_as_of_august_does_see_them(_season):
    aug = sc.get_season_batter_table_as_of("2026-08-01")
    row = aug[aug["mlbam_id"] == 660271].iloc[0]
    assert row["season_hr"] == 40
    assert row["barrel_pct"] > 0
    assert row["avg_ev"] > 95.0


def test_the_window_stops_the_day_before(_season):
    """Same rule as recent form: the graded day is never in its own features."""
    day_of = sc.get_season_batter_table_as_of("2026-07-20")
    row = day_of[day_of["mlbam_id"] == 660271]
    # Only the April rows qualify, so that day's 40 HR must not be counted.
    assert row.empty or int(row.iloc[0]["season_hr"]) == 0


def test_season_to_date_pitches_are_strictly_before(_season):
    p = sc.season_to_date_pitches("2026-07-20")
    assert not p.empty
    assert p["game_date"].max() < dt.date(2026, 7, 20)


def test_month_pull_is_cached_so_backfill_stays_affordable(_season, monkeypatch):
    """Each graded date must not re-pull the whole season."""
    calls = {"n": 0}
    real = sc.pyb.statcast

    class _Counting:
        @staticmethod
        def statcast(start_dt, end_dt, verbose=False):
            calls["n"] += 1
            return real(start_dt, end_dt, verbose=verbose)

    monkeypatch.setattr(sc, "pyb", _Counting, raising=False)
    sc.get_season_batter_table_as_of("2026-06-15")
    first = calls["n"]
    assert first > 0
    sc.get_season_batter_table_as_of("2026-06-20")   # same months, all cached
    assert calls["n"] == first, "a second date re-pulled months it already had"


def test_empty_season_degrades_to_an_empty_table(monkeypatch):
    monkeypatch.setattr(sc, "_HAS_PYB", False)
    sc._statcast_month.cache_clear()
    sc.get_season_batter_table_as_of.cache_clear()
    assert sc.get_season_batter_table_as_of("2026-06-15").empty
    sc._statcast_month.cache_clear()
    sc.get_season_batter_table_as_of.cache_clear()


# --------------------------------------------------------------------------- #
# The FanGraphs payload we were under-reading
# --------------------------------------------------------------------------- #
def test_pitching_table_extracts_the_fields_it_was_ignoring(monkeypatch):
    fake = pd.DataFrame({
        "Name": ["Ace Arm", "Bulk Guy"], "Team": ["LAD", "COL"],
        "HR/9": [0.8, 1.9], "GB%": [0.46, 0.38], "FB%": [0.32, 0.44],
        "K%": [0.31, 0.15], "BB%": [0.055, 0.10], "ERA": [2.60, 5.40],
        "FIP": [2.85, 5.10], "IP": [140.0, 90.0], "GS": [22, 20],
    })
    monkeypatch.setattr(sc, "_fg_pitching_raw", lambda year: fake)
    sc.get_pitching_table.cache_clear()
    t = sc.get_pitching_table(2026).set_index("name_key")
    sc.get_pitching_table.cache_clear()
    ace = t.loc["ace arm"]
    assert ace["pitcher_k_pct"] == pytest.approx(31.0, abs=0.5)
    assert ace["pitcher_fip"] == pytest.approx(2.85)
    assert ace["sp_ip_per_start"] == pytest.approx(140.0 / 22, abs=0.01)
    assert t.loc["bulk guy"]["pitcher_k_pct"] < ace["pitcher_k_pct"]


def test_k_pct_derived_from_k_per_9_when_the_rate_column_is_absent(monkeypatch):
    fake = pd.DataFrame({"Name": ["Ace Arm"], "Team": ["LAD"], "HR/9": [0.8],
                         "K/9": [11.4], "IP": [140.0], "GS": [22]})
    monkeypatch.setattr(sc, "_fg_pitching_raw", lambda year: fake)
    sc.get_pitching_table.cache_clear()
    t = sc.get_pitching_table(2026)
    sc.get_pitching_table.cache_clear()
    assert 25.0 < float(t.iloc[0]["pitcher_k_pct"]) < 35.0


def test_bullpen_table_carries_era_and_k_pct(monkeypatch):
    fake = pd.DataFrame({
        "Name": ["r1", "r2"], "Team": ["LAD", "COL"], "GS": [0, 0],
        "IP": [200.0, 200.0], "HR": [20.0, 30.0], "ER": [80.0, 120.0],
        "SO": [240.0, 150.0], "TBF": [800.0, 850.0],
    })
    monkeypatch.setattr(sc, "_fg_pitching_raw", lambda year: fake)
    sc.get_bullpen_table.cache_clear()
    t = sc.get_bullpen_table(2026)
    sc.get_bullpen_table.cache_clear()
    assert t["LAD"]["bullpen_era"] < t["COL"]["bullpen_era"]
    assert t["LAD"]["bullpen_k_pct"] > t["COL"]["bullpen_k_pct"]
    assert t["LAD"]["bullpen_hr9"] == pytest.approx(0.9, abs=0.01)


def test_bullpen_hr9_view_still_works_without_the_new_columns(monkeypatch):
    fake = pd.DataFrame({"Name": ["r1"], "Team": ["LAD"], "GS": [0],
                         "IP": [200.0], "HR": [20.0]})
    monkeypatch.setattr(sc, "_fg_pitching_raw", lambda year: fake)
    sc.get_bullpen_hr9_table.cache_clear()
    t = sc.get_bullpen_hr9_table(2026)
    sc.get_bullpen_hr9_table.cache_clear()
    assert t["LAD"] == pytest.approx(0.9, abs=0.01)
