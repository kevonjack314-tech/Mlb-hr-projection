"""Smoke tests for the HR projection model. Run with: python -m pytest tests/ -q"""

import datetime as dt

from src.demo import build_demo_slate
from src.model import (
    effective_bat_side,
    has_platoon_advantage,
    score_row,
    score_slate,
)
from src.parks import park_hr_multiplier, wind_hr_multiplier, get_park


def _slate():
    return score_slate(build_demo_slate(dt.date(2026, 6, 18)))


def test_slate_builds_and_scores():
    df = _slate()
    assert len(df) > 100
    for col in ("hr_score", "hr_prob_game", "longshot_score",
                "consistency_score", "sneaky_score", "fair_odds"):
        assert col in df.columns


def test_score_ranges_are_sane():
    df = _slate()
    assert df.hr_score.between(0, 100).all()
    assert df.longshot_score.between(0, 100).all()
    assert df.consistency_score.between(0, 100).all()
    assert df.sneaky_score.between(0, 100).all()
    # Game HR probability should stay in a realistic band.
    assert df.hr_prob_game.between(0.0, 0.35).all()
    assert df.hr_prob_game.max() > 0.15  # at least some strong spots


def test_deterministic_by_date():
    a = build_demo_slate(dt.date(2026, 6, 18))
    b = build_demo_slate(dt.date(2026, 6, 18))
    assert a.equals(b)


def test_switch_hitter_and_platoon_logic():
    assert effective_bat_side("S", "R") == "L"
    assert effective_bat_side("S", "L") == "R"
    assert has_platoon_advantage("L", "R") is True
    assert has_platoon_advantage("R", "R") is False
    assert has_platoon_advantage("S", "L") is True


def test_park_and_wind_multipliers():
    # Coors should boost; Oracle/Oakland should suppress.
    assert park_hr_multiplier("COL", "R") > 1.0
    assert park_hr_multiplier("OAK", "L") < 1.0
    park = get_park("CHC")
    # Wind blowing straight out (toward CF bearing) should help; in should hurt.
    out = wind_hr_multiplier(park, 15, (park["orientation_deg"] + 180) % 360, "R")
    inw = wind_hr_multiplier(park, 15, park["orientation_deg"], "R")
    assert out > 1.0 > inw


def test_rationale_present():
    df = _slate()
    assert df["rationale"].astype(str).str.len().gt(0).all()


def test_hr_history_and_profile():
    from src.history import (build_hr_history, summarize_hr_profile,
                             hr_profile_centroid, add_profile_similarity,
                             calibration_table, top5_by_category)
    events, slate, source, _ = build_hr_history("2026-06-01", "2026-06-10", prefer_live=False)
    assert source == "SIMULATED"
    assert not events.empty and "barrel_pct" in events.columns
    summ = summarize_hr_profile(events, slate)
    assert summ["hr_events"] > 0 and not summ["metric_table"].empty
    # HR hitters should out-index the field on barrel rate.
    bt = summ["metric_table"].set_index("Metric")
    assert bt.loc["Barrel%", "HR hitters (avg)"] >= bt.loc["Barrel%", "All hitters (avg)"]
    # Calibration should be monotone-ish: top decile beats bottom decile.
    cal = calibration_table(slate)
    assert cal.iloc[-1]["Actual HR%"] > cal.iloc[0]["Actual HR%"]

    centroid = hr_profile_centroid(events)
    cur = add_profile_similarity(_slate(), centroid)
    assert cur["profile_match"].between(0, 100).all()
    tops = top5_by_category(cur, n=5)
    assert set(tops) == {"Overall (HR Score)", "Best Longshots",
                         "Consistent HR Hitters", "Sneaky HR Chances"}
    assert all(len(t) == 5 for t in tops.values())


def test_expected_power_and_recency_trend():
    from src.history import (build_hr_history, hr_profile_centroid, recent_trend,
                             add_profile_similarity)
    df = _slate()
    # Expected-power metrics present and sane.
    assert "xiso" in df.columns and df.xiso.between(0.05, 0.35).all()
    assert "xslg" in df.columns

    events, _, _, _ = build_hr_history("2026-05-18", "2026-06-18", prefer_live=False)
    # Recency-weighted centroid records its half-life and includes xISO.
    cen = hr_profile_centroid(events, end_date_iso="2026-06-18", half_life_days=7)
    assert cen["half_life_days"] == 7
    assert "xiso" in cen["centroid"]
    # Unweighted fallback when no date/half-life.
    flat = hr_profile_centroid(events, end_date_iso=None, half_life_days=0)
    assert flat["half_life_days"] is None
    # Profile match still valid with the recency centroid.
    assert add_profile_similarity(df, cen)["profile_match"].between(0, 100).all()
    # Trend table is non-empty and ranks by movement.
    tr = recent_trend(events, "2026-06-18", recent_days=7)
    assert not tr.empty and "Trend" in tr.columns


def test_barrel_pa_xhr_and_pitch_matchup():
    df = _slate()
    # Barrel/PA, sprint speed, vs-pitch-type splits present and sane.
    assert df["brl_pa"].between(1.5, 13).all()
    assert df["sprint_speed"].between(22, 31).all()
    for c in ("vs_fb", "vs_br", "vs_os"):
        assert df[c].between(0.2, 0.45).all()
    # Expected HR + regression gap.
    assert "xhr_season" in df.columns
    assert df["pitch_matchup_score"].between(0, 100).all()
    assert df["regression_score"].between(0, 100).all()
    # Under-xHR hitters should exist and skew the sneaky regression up.
    under = df[df["hr_minus_xhr"] <= -2.0]
    assert len(under) > 0
    assert under["regression_score"].mean() > df["regression_score"].mean()


def test_live_pitch_mix_and_splits_aggregation():
    """The live Statcast aggregation logic (pitch mix + vs-pitch wOBA) is correct."""
    import numpy as np
    import pandas as pd
    from src import statcast as sc

    fake = pd.DataFrame({
        "game_date": ["2025-06-18"] * 10,
        "batter":  [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "pitcher": [9] * 10,
        "pitch_type": ["FF", "FF", "SL", "CH", "FF", "SL", "FF", "CU", "CH", "FF"],
        "events": ["home_run", None, "single", None, "strikeout",
                   "double", None, None, "field_out", "walk"],
        "woba_value": [2.0, np.nan, 0.9, np.nan, 0.0, 1.24, np.nan, np.nan, 0.0, 0.69],
        "woba_denom": [1, np.nan, 1, np.nan, 1, 1, np.nan, np.nan, 1, 1],
    })
    fake["pitch_family"] = fake["pitch_type"].map(sc._PITCH_FAMILY)

    orig = sc._statcast_range
    sc._statcast_range = lambda end, lb=30: fake
    try:
        sc.get_pitch_mix_table.cache_clear()
        sc.get_batter_pitch_splits.cache_clear()
        mix = sc.get_pitch_mix_table("2025-06-18").set_index("pitcher_id")
        assert mix.loc[9, "pitcher_mix_fb"] == 50.0
        assert mix.loc[9, "pitcher_mix_br"] == 30.0
        assert mix.loc[9, "pitcher_mix_os"] == 20.0
        splits = sc.get_batter_pitch_splits("2025-06-18").set_index("mlbam_id")
        assert abs(splits.loc[1, "vs_fb"] - 1.0) < 1e-6   # (2.0+0.0)/(1+1)
        assert abs(splits.loc[2, "vs_br"] - 1.24) < 1e-6
    finally:
        sc._statcast_range = orig
        sc.get_pitch_mix_table.cache_clear()
        sc.get_batter_pitch_splits.cache_clear()


def test_odds_and_parlay_generator():
    from src.odds import (american_to_decimal, american_to_prob, attach_odds,
                          decimal_to_american, model_market_odds)
    from src.parlay import generate_parlay, summarize_selection

    # Odds math round-trips and book odds are worse than fair (you pay the hold).
    assert abs(american_to_decimal(+100) - 2.0) < 1e-9
    assert decimal_to_american(2.0) == 100
    assert abs(american_to_prob(-110) - 0.5238) < 1e-3
    assert model_market_odds(0.25) > 0  # underdog price
    assert american_to_prob(model_market_odds(0.25)) > 0.25  # shaded up by hold

    df = attach_odds(_slate(), "2026-06-18", use_live=False)
    for c in ("book_odds", "edge_pct", "implied_prob"):
        assert c in df.columns

    # ULX composition by leg count.
    expect = {1: ["Anchor"], 2: ["Anchor", "Value"],
              3: ["Anchor", "Value", "Longshot"],
              4: ["Anchor", "Value", "Value", "Longshot"],
              5: ["Anchor", "Value", "Value", "Longshot", "Longshot"]}
    for n in range(1, 6):
        res = generate_parlay(df, n_legs=n, strategy="ulx")
        legs = res["legs"]
        assert len(legs) == n
        assert list(legs["role"]) == expect[n]
        # No two legs from the same game (diversification).
        assert legs["game"].nunique() == n
        s = res["summary"]
        assert s["combined_decimal"] >= 1.0 and 0 <= s["model_prob"] <= 100
        assert s["checks_total"] == 11

    # Custom selection grades too.
    cust = summarize_selection(df, list(df["player"].head(3)))
    assert cust["summary"]["n_legs"] == 3


def test_lineup_spot_and_recurring_log(tmp_path, monkeypatch):
    from src import lineup
    from src.history import build_hr_history
    from src.lineup import (attach_spot_signal, expected_pa, league_spot_table,
                            player_spot_hr, spot_role_fit, update_log_from_history)

    # Expected PA monotonically falls down the order; spot role fit matches ULX.
    assert expected_pa(1) > expected_pa(5) > expected_pa(9)
    assert spot_role_fit(4, "Anchor") > spot_role_fit(8, "Anchor")
    assert spot_role_fit(9, "Longshot") > spot_role_fit(3, "Longshot")

    df = _slate()
    spots = df["lineup_spot"].dropna()
    assert spots.between(1, 9).all() and spots.nunique() == 9   # bench bats are NaN
    assert df["expected_pa"].between(3.7, 4.6).all()

    # Recurring log writes to an isolated path and accumulates (idempotent).
    monkeypatch.setattr(lineup, "_LOG_PATH", str(tmp_path / "log.csv"))
    _e, slate_hist, _s, _n = build_hr_history("2026-06-01", "2026-06-08", prefer_live=False)
    n1 = update_log_from_history(slate_hist)
    n2 = update_log_from_history(slate_hist)   # de-duped -> no growth
    assert n1 > 0 and n2 == 0
    ls = league_spot_table(slate_hist)
    assert not ls.empty and ls["hr"].sum() > 0
    ps = player_spot_hr(slate_hist)
    enriched = attach_spot_signal(df, ps)
    assert "spot_hr_at_current" in enriched.columns


def test_boxscore_batting_order_extraction(monkeypatch):
    """Real lineup spot is recovered from a game's box score (battingOrder//100)."""
    from src import sources
    fake = {"teams": {"home": {"players": {
                "ID100": {"person": {"id": 100}, "battingOrder": "500"},  # 5th
                "ID101": {"person": {"id": 101}, "battingOrder": "600"},  # 6th
                "ID102": {"person": {"id": 102}, "battingOrder": "501"},  # sub, 5th
                "ID103": {"person": {"id": 103}, "battingOrder": None},   # bench
            }}, "away": {"players": {
                "ID200": {"person": {"id": 200}, "battingOrder": "100"},  # leadoff
            }}}}
    sources.fetch_batting_order_map.cache_clear()
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: fake)
    m = dict(sources.fetch_batting_order_map(12345))
    assert m == {100: 5, 101: 6, 102: 5, 200: 1}  # bench (103) excluded


def test_pitcher_hr_by_spot_demo():
    from src.pitchers import hottest_spots, pitcher_recent_hr_by_spot
    counts, n, total, src = pitcher_recent_hr_by_spot(None, "Sonny Gray", "2026-06-20",
                                                      n_games=5, prefer_live=False)
    assert src == "modeled" and n == 5
    assert set(counts) == set(range(1, 10))
    assert sum(counts.values()) == total          # distribution sums to total HRs
    # Deterministic for the same (name, date).
    counts2, *_ = pitcher_recent_hr_by_spot(None, "Sonny Gray", "2026-06-20",
                                            n_games=5, prefer_live=False)
    assert counts2 == counts
    hot = hottest_spots(counts, 2)
    assert all(1 <= s <= 9 for s in hot)


def test_pitcher_id_handles_missing_and_nan():
    """A TBD starter (NaN/None id) must not crash the live pitcher lookup."""
    import numpy as np
    from src.pitchers import _safe_pid, pitcher_recent_hr_by_spot, sp_spot_counts_for
    assert _safe_pid(np.nan) is None and _safe_pid(None) is None
    assert _safe_pid(543037.0) == 543037 and _safe_pid("592450") == 592450
    # NaN id with prefer_live=True falls back to modeled instead of raising.
    _c, _n, _t, src = pitcher_recent_hr_by_spot(np.nan, "TBD", "2026-06-24", 5, True)
    assert src == "modeled"
    m = sp_spot_counts_for((("A @ B", "TBD", np.nan),), "2026-06-24", True)
    assert ("A @ B", "TBD") in m


def test_boxscore_hr_hitters_extraction(monkeypatch):
    """Real HR hitters (name, spot, HR count, opposing SP) come from the box score."""
    from src import sources
    fake = {"teams": {
        "home": {"team": {"id": 139}, "pitchers": [600], "players": {
            "ID600": {"person": {"id": 600, "fullName": "Home Starter"}},
            "ID700": {"person": {"id": 700, "fullName": "Junior Caminero"},
                      "battingOrder": "400", "stats": {"batting": {"homeRuns": 3}}},
            "ID701": {"person": {"id": 701, "fullName": "No HR Guy"},
                      "battingOrder": "500", "stats": {"batting": {"homeRuns": 0}}},
        }},
        "away": {"team": {"id": 147}, "pitchers": [800], "players": {
            "ID800": {"person": {"id": 800, "fullName": "Carlos Rodon"}},
            "ID900": {"person": {"id": 900, "fullName": "Aaron Judge"},
                      "battingOrder": "200", "stats": {"batting": {"homeRuns": 1}}},
        }},
    }}
    sources.fetch_game_box_hrs.cache_clear()
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: fake)
    hrs = {h["player"]: h for h in sources.fetch_game_box_hrs(777)}
    assert "No HR Guy" not in hrs                       # 0 HR excluded
    cam = hrs["Junior Caminero"]
    assert cam["hr_count"] == 3 and cam["lineup_spot"] == 4
    assert cam["team"] == "TB" and cam["pitcher_name"] == "Carlos Rodon"


def test_play_by_play_actual_pitcher(monkeypatch):
    """HRs are tagged with the ACTUAL pitcher per HR from the play-by-play feed."""
    from src import sources
    feed = {
        "gameData": {"teams": {"home": {"id": 118}, "away": {"id": 139}}},  # KC / TB
        "liveData": {
            "boxscore": {"teams": {
                "home": {"players": {}},
                "away": {"players": {"ID700": {"person": {"id": 700},
                                               "battingOrder": "400"}}},
            }},
            "plays": {"allPlays": [
                {"result": {"eventType": "home_run"}, "about": {"halfInning": "top"},
                 "matchup": {"batter": {"id": 700, "fullName": "Junior Caminero"},
                             "pitcher": {"id": 111, "fullName": "Seth Lugo"}}},
                {"result": {"eventType": "home_run"}, "about": {"halfInning": "top"},
                 "matchup": {"batter": {"id": 700, "fullName": "Junior Caminero"},
                             "pitcher": {"id": 111, "fullName": "Seth Lugo"}}},
                {"result": {"eventType": "strikeout"}, "about": {"halfInning": "bottom"},
                 "matchup": {"batter": {"id": 1}, "pitcher": {"id": 2}}},
            ]},
        }}
    sources.fetch_game_hr_details.cache_clear()
    monkeypatch.setattr(sources, "_get_json", lambda url, params=None: feed)
    hrs = sources.fetch_game_hr_details(999)
    assert len(hrs) == 2                                # 2 HR plays, K excluded
    assert all(h["player"] == "Junior Caminero" and h["lineup_spot"] == 4
               and h["pitcher_name"] == "Seth Lugo" and h["team"] == "TB" for h in hrs)


def test_sp_spot_signal_feeds_parlay():
    import pandas as pd
    from src.parlay import role_fit
    from src.pitchers import attach_sp_spot_signal

    # attach_sp_spot_signal maps (game, pitcher) + lineup spot -> HR count.
    slate = pd.DataFrame([{"game": "A @ B", "pitcher_name": "X", "lineup_spot": 4},
                          {"game": "A @ B", "pitcher_name": "X", "lineup_spot": 7}])
    out = attach_sp_spot_signal(slate, {("A @ B", "X"): {4: 2, 7: 0}})
    assert out["sp_hr_at_spot"].tolist() == [2, 0]

    # role_fit rewards a bat in a spot the opposing SP gives up HRs to.
    base = {"hr_score": 60, "sneaky_score": 55, "longshot_score": 55, "edge_pct": 0,
            "lineup_spot": 4, "spot_hr_rate": float("nan"), "cal_edge_pct": float("nan"),
            "sp_hr_at_spot": 0}
    low = pd.Series(base)
    high = pd.Series({**base, "sp_hr_at_spot": 3})
    for role in ("Anchor", "Value", "Longshot"):
        assert role_fit(high, role) > role_fit(low, role)


def test_hr_of_day_and_parlay_shuffle():
    import app
    from src.odds import attach_odds
    from src.parlay import generate_parlay

    df = attach_odds(_slate(), "2026-06-18", use_live=False)
    # HR of the Day: a single high-confidence lock.
    row = app.hr_of_the_day(df)
    assert row is not None and 0 <= row["confidence"] <= 100

    # Shuffle (seeded) produces multiple distinct valid tickets.
    combos = {tuple(generate_parlay(df, 3, "ulx", seed=s)["legs"]["player"])
              for s in range(8)}
    assert len(combos) >= 2
    # Default (no seed) is deterministic.
    a = tuple(generate_parlay(df, 3, "ulx")["legs"]["player"])
    b = tuple(generate_parlay(df, 3, "ulx")["legs"]["player"])
    assert a == b
