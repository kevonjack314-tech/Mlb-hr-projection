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
