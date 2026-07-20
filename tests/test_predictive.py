"""Tests: predictive lineup spots, trend-forward pick signals, and the
Savant/FanGraphs feed-shape fallbacks."""

import pandas as pd

from src.lineup import typical_spots
from src.parlay import role_fit
from src.statcast import _assemble_season_table, normalize_name
from src.trends import attach_trend_signals


# --------------------------------------------------------------------------- #
# Predictive lineup spots (from the graded record)
# --------------------------------------------------------------------------- #
def test_typical_spots_from_real_record():
    spots = typical_spots()
    assert len(spots) > 100
    assert all(1 <= s <= 9 for s in spots.values())
    # Stars hit at the top of the order — the Schwarber-batting-9th bug.
    for star in ("shohei ohtani", "kyle schwarber", "aaron judge"):
        if star in spots:
            assert spots[star] <= 4, f"{star} typical spot {spots[star]}"


# --------------------------------------------------------------------------- #
# Trend signals on the slate
# --------------------------------------------------------------------------- #
def _events():
    return pd.DataFrame({
        "date": ["2026-07-10", "2026-07-11", "2026-07-11", "2026-07-11"],
        "player": ["Cold Bat", "Streak Guy", "Star A", "Star B"],
        "team": ["AAA", "BBB", "CCC", "DDD"],
        "lineup_spot": [7, 4, 3, 2],
        "hr_count": [1, 1, 1, 1],
        "season_hr": [5, 12, 25, 30],          # yesterday: 2 of 3 HRs by stars
    })


def _slate():
    return pd.DataFrame({
        "player": ["Streak Guy", "Cold Bat", "Some Star", "Some Under"],
        "lineup_spot": [4, 7, 3, 8],
        "season_hr": [12, 5, 28, 4],
    })


def test_attach_trend_signals():
    out = attach_trend_signals(_slate(), _events(), weekday_name="Saturday")
    by = out.set_index("player")
    # Back-to-back: only the bat who homered on the LAST history day is live.
    assert by.loc["Streak Guy", "hot_streak"] == 1
    assert by.loc["Cold Bat", "hot_streak"] == 0
    # Star-heavy yesterday (2/3 star HRs) -> rotation leans mid/under today.
    assert by.loc["Some Star", "tier_lean"] == -1
    assert by.loc["Some Under", "tier_lean"] == 1
    assert by.loc["Streak Guy", "tier_lean"] == 1          # mid tier favored
    assert out["dow_spot_heat"].between(0, 2).all()


def test_trend_signals_move_role_fit():
    base = pd.Series({"hr_score": 60, "sneaky_score": 55, "edge_pct": 0.0,
                      "lineup_spot": 4, "ulx_checks": 5})
    hot = pd.Series({**base, "hot_streak": 1, "tier_lean": 1, "dow_spot_heat": 2.0})
    assert role_fit(hot, "Anchor") > role_fit(base, "Anchor")
    # Reduced checklist weight: 9 checks vs 0 now moves Value fit by <= ~7.5.
    lo = pd.Series({**base, "ulx_checks": 0})
    hi = pd.Series({**base, "ulx_checks": 9})
    assert role_fit(hi, "Value") - role_fit(lo, "Value") <= 7.5


def test_empty_events_are_safe():
    out = attach_trend_signals(_slate(), pd.DataFrame(), weekday_name="Monday")
    assert (out["hot_streak"] == 0).all() and (out["tier_lean"] == 0).all()


# --------------------------------------------------------------------------- #
# Savant name-layout variants (the "'str' has no attribute 'astype'" crash)
# --------------------------------------------------------------------------- #
def _ev_base():
    return {"player_id": [660271], "brl_percent": [15.0], "avg_hit_speed": [94.0],
            "max_hit_speed": [118.0], "ev95percent": [55.0], "avg_hit_angle": [15.0]}


def test_assemble_handles_first_last_names():
    ev = pd.DataFrame({**_ev_base(), "first_name": ["Shohei"], "last_name": ["Ohtani"]})
    t = _assemble_season_table(ev, 2026)
    assert t["name_key"].iloc[0] == "shohei ohtani"


def test_assemble_handles_combined_name_column():
    ev = pd.DataFrame({**_ev_base(), "last_name, first_name": ["Ohtani, Shohei"]})
    t = _assemble_season_table(ev, 2026)
    assert t["name_key"].iloc[0] == "shohei ohtani"


def test_assemble_handles_player_name_column():
    ev = pd.DataFrame({**_ev_base(), "player_name": ["Ohtani, Shohei"]})
    t = _assemble_season_table(ev, 2026)
    assert t["name_key"].iloc[0] == "shohei ohtani"


# --------------------------------------------------------------------------- #
# FanGraphs JSON-API fallback parser
# --------------------------------------------------------------------------- #
def test_fg_api_parser(monkeypatch):
    import requests
    from src import statcast as sc

    class FakeResp:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"PlayerName": "Kyle Schwarber", "TeamNameAbb": "PHI",
                              "PA": 400, "HR": 33, "K%": 0.28, "HR/9": None}]}

    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResp())
    df = sc._fg_api_leaders(2026, "bat")
    assert list(df["Name"]) == ["Kyle Schwarber"]
    assert list(df["Team"]) == ["PHI"]
    assert normalize_name(df["Name"].iloc[0]) == "kyle schwarber"
