"""NFL Matchup Lab — analytics engine smoke tests."""

import datetime as dt

from nfl.data import ARCHETYPE_LABEL, ROSTERS, TEAMS, build_week_slate
from nfl.model import get_week_slate, score_slate
from nfl.schemes import DEF_SCHEMES, TEAM_DEF, TEAM_OFF, def_scheme_of, scheme_boost


def _slate(week=1, season=2026):
    return score_slate(build_week_slate(week, season))


def test_slate_builds_and_scores():
    df = _slate()
    assert len(df) > 100
    assert df["game"].nunique() >= 10
    for col in ("proj_rush_yds", "proj_rec_yds", "proj_pass_yds", "td_prob",
               "p_100", "matchup_score", "insight", "def_scheme"):
        assert col in df.columns


def test_deterministic_by_week():
    a = build_week_slate(3, 2026)
    b = build_week_slate(3, 2026)
    assert a.equals(b)
    # Different week -> different matchups (not guaranteed identical).
    c = build_week_slate(4, 2026)
    assert not a["opponent"].equals(c["opponent"]) or not a.equals(c)


def test_probabilities_and_flags_in_range():
    df = _slate()
    assert df["td_prob"].between(0.0, 0.75).all()
    assert df["p_100"].between(0.0, 1.0).all()
    assert df["matchup_score"].between(0, 100).all()
    assert df["td_favorite"].dtype == bool
    assert df["watch_100"].dtype == bool
    # Favorites should actually be at/above the threshold.
    favs = df[df["td_favorite"]]
    assert (favs["td_prob"] >= 0.35).all()


def test_every_team_has_a_scheme_and_roster():
    for tm in TEAMS:
        assert tm in TEAM_DEF
        assert def_scheme_of(tm) in DEF_SCHEMES
        assert tm in ROSTERS and len(ROSTERS[tm]) >= 5
        assert tm in TEAM_OFF


def test_scheme_interaction_shapes_the_projection():
    """An alpha-X receiver projects worse vs two-high zone than vs press-man —
    the core 'scheme determines who eats' behavior."""
    ymult_press, tmult_press, _ = scheme_boost("alpha_x", "press_man_blitz")
    ymult_zone, tmult_zone, _ = scheme_boost("alpha_x", "two_high_zone")
    assert ymult_press > ymult_zone
    assert tmult_press > tmult_zone

    # A deep threat should be suppressed by a two-high shell more than a
    # workhorse power back is suppressed by the same shell (which actually
    # helps the power back via light boxes).
    deep_zone_y, _, _ = scheme_boost("deep_threat", "two_high_zone")
    power_zone_y, _, _ = scheme_boost("workhorse_power", "two_high_zone")
    assert deep_zone_y < 1.0 < power_zone_y


def test_insight_cites_scheme_and_history():
    df = _slate()
    row = df.iloc[0]
    assert row["player"] in row["insight"]
    assert row["def_scheme"] in row["insight"] or "vs" in row["insight"]
    # Players with >=2 games vs this opponent get a history clause.
    hist_rows = df[df["vs_games"] >= 2]
    if len(hist_rows):
        assert "career games vs them" in hist_rows.iloc[0]["insight"]


def test_get_week_slate_labels_source():
    df, source = get_week_slate(1, 2026, prefer_live=True)
    assert source.startswith("MODELED") or source.startswith("LIVE")
    assert not df.empty
