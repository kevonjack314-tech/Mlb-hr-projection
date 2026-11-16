"""Tests: the NFL anytime-touchdown model.

The thing this model has to get right, and that casual touchdown betting gets
wrong, is that touchdowns are a FIXED SHARED RESOURCE. A team scores about 2.3
of them. Role near the goal line decides who gets them; yardage between the
twenties mostly does not.
"""
import pandas as pd
import pytest

from src.nfl.demo import build_demo_slate
from src.nfl.model import (
    LEAGUE_TEAM_TDS, LISTED_TD_CAP, conserve_team_tds, expected_team_tds,
    score_row, score_slate, td_of_the_day, td_share,
)
from src.nfl.teams import is_indoor, pass_weather_multiplier, venue


def _rb(**kw):
    base = {"position": "RB", "snap_share": 0.75, "rush_share": 0.60,
            "inside5_share": 0.55, "tgt_share": 0.10, "rz_tgt_share": 0.07,
            "ypc": 4.4, "implied_team_total": 24.0, "home_team": "DET",
            "team": "DET", "game": "CHI @ DET"}
    base.update(kw)
    return pd.Series(base)


def _wr(**kw):
    base = {"position": "WR", "snap_share": 0.92, "rush_share": 0.0,
            "tgt_share": 0.27, "rz_tgt_share": 0.24, "yprr": 2.2,
            "implied_team_total": 24.0, "home_team": "DET",
            "team": "DET", "game": "CHI @ DET"}
    base.update(kw)
    return pd.Series(base)


# --------------------------------------------------------------------------- #
# Touchdown supply
# --------------------------------------------------------------------------- #
def test_team_touchdowns_come_from_the_market_total():
    """The implied team total is the best public estimate of scoring."""
    assert expected_team_tds(pd.Series({"implied_team_total": 31.0})) > \
        expected_team_tds(pd.Series({"implied_team_total": 16.0}))
    # No line -> league average, not zero.
    assert expected_team_tds(pd.Series({})) == LEAGUE_TEAM_TDS


def test_a_typical_total_implies_about_the_league_average():
    tds = expected_team_tds(pd.Series({"implied_team_total": 22.5}))
    assert 2.0 <= tds <= 2.6


# --------------------------------------------------------------------------- #
# Role beats volume — the whole point
# --------------------------------------------------------------------------- #
def test_the_goal_line_back_beats_the_between_the_twenties_star():
    """A 20%-snap back who gets the carries inside the 5 is the better TD bet
    than a every-down receiver who never sees the goal line."""
    goal_line = _rb(snap_share=0.22, rush_share=0.18, inside5_share=0.62,
                    tgt_share=0.02, rz_tgt_share=0.01)
    between = _wr(snap_share=0.95, tgt_share=0.28, rz_tgt_share=0.04)
    assert score_row(goal_line)["td_prob_game"] > score_row(between)["td_prob_game"]


def test_red_zone_target_share_moves_a_receiver_more_than_total_targets():
    heavy_rz = _wr(tgt_share=0.18, rz_tgt_share=0.28)
    heavy_vol = _wr(tgt_share=0.30, rz_tgt_share=0.06)
    assert score_row(heavy_rz)["td_prob_game"] > score_row(heavy_vol)["td_prob_game"]


def test_share_shrinks_to_the_positional_prior_when_usage_is_unknown():
    """A player we know nothing about lands near his position's average."""
    blank = pd.Series({"position": "TE", "snap_share": 0.70,
                       "implied_team_total": 22.5, "home_team": "DET"})
    sh = td_share(blank)["td_share"]
    assert 0.02 < sh < 0.20


# --------------------------------------------------------------------------- #
# Weather moves touchdowns, it does not delete them
# --------------------------------------------------------------------------- #
def test_bad_weather_shifts_touchdowns_from_the_air_to_the_ground():
    """A wind penalty that only ever subtracts would make the goal-line back a
    worse bet in a gale, when he is actually a better one."""
    calm = pass_weather_multiplier("BUF", 3, 70)
    gale = pass_weather_multiplier("BUF", 24, 18)
    assert gale["pass_mult"] < calm["pass_mult"]
    assert gale["rush_mult"] > calm["rush_mult"] == 1.0

    rb_calm = score_row(_rb(home_team="BUF", team="BUF", wind_mph=3, temp_f=70))
    rb_gale = score_row(_rb(home_team="BUF", team="BUF", wind_mph=24, temp_f=18))
    wr_calm = score_row(_wr(home_team="BUF", team="BUF", wind_mph=3, temp_f=70))
    wr_gale = score_row(_wr(home_team="BUF", team="BUF", wind_mph=24, temp_f=18))
    assert rb_gale["td_prob_game"] > rb_calm["td_prob_game"]
    assert wr_gale["td_prob_game"] < wr_calm["td_prob_game"]


def test_a_dome_ignores_the_weather_entirely():
    assert is_indoor("DET") and is_indoor("DAL")      # retractable counts
    assert not is_indoor("GB")
    w = pass_weather_multiplier("DET", 30, 5)
    assert w["pass_mult"] == w["rush_mult"] == 1.0


def test_every_team_has_a_venue():
    from src.nfl.teams import TEAMS
    assert len(TEAMS) == 32
    assert venue("DEN")["altitude_ft"] == 5280
    assert venue("ZZZ") is None


# --------------------------------------------------------------------------- #
# Conservation — the bug this model shipped with
# --------------------------------------------------------------------------- #
def test_a_team_cannot_hand_out_more_touchdowns_than_it_scores():
    """Each player's share is estimated independently, so the nine of them used
    to split 1.1x-1.6x of their own team's touchdowns between them."""
    s = score_slate(build_demo_slate("2026-11-16"))
    g = s.groupby(["game", "team"]).agg(
        projected=("expected_tds", "sum"), supply=("expected_team_tds", "first"))
    ratio = g["projected"] / g["supply"]
    assert (ratio <= LISTED_TD_CAP + 1e-6).all(), f"max {ratio.max():.3f}"


def test_conservation_scales_without_reordering():
    """It changes the level, not who the model likes."""
    raw = build_demo_slate("2026-11-16")
    scored = raw.apply(lambda r: pd.Series(score_row(r)), axis=1)
    before = pd.concat([raw.reset_index(drop=True),
                        scored.reset_index(drop=True)], axis=1)
    after = conserve_team_tds(before)
    one = before[before["team"] == before["team"].iloc[0]]
    two = after[after["team"] == after["team"].iloc[0]]
    assert list(one.sort_values("expected_tds")["player"]) == \
        list(two.sort_values("expected_tds")["player"])
    assert (after["td_prob_game"] <= before["td_prob_game"] + 1e-9).all()


def test_slate_lands_near_the_real_anytime_td_base_rate():
    s = score_slate(build_demo_slate("2026-11-16"))
    assert 0.15 <= s["td_prob_game"].mean() <= 0.27
    # Running backs lead the board; that is what the position actually does.
    by_pos = s.groupby("position")["td_prob_game"].mean()
    assert by_pos["RB"] > by_pos["WR"]


def test_probabilities_are_poisson_not_linear():
    """0.5 expected touchdowns is 39%, not 50% — multi-TD games are the gap."""
    out = score_row(_rb())
    import math
    assert out["td_prob_game"] == pytest.approx(
        1 - math.exp(-out["expected_tds"]), abs=1e-3)


# --------------------------------------------------------------------------- #
# Slate
# --------------------------------------------------------------------------- #
def test_demo_slate_is_deterministic_with_unique_players():
    a, b = build_demo_slate("2026-11-16"), build_demo_slate("2026-11-16")
    assert a.equals(b)
    assert len(a) == a["player"].nunique(), "two players sharing a name"
    assert not build_demo_slate("2026-11-23").equals(a)


def test_td_of_the_day_and_empty_slate_are_safe():
    s = score_slate(build_demo_slate("2026-11-16"))
    pick = td_of_the_day(s)
    assert pick is not None and pick["td_prob_game"] == s["td_prob_game"].max()
    assert td_of_the_day(pd.DataFrame()) is None
    assert score_slate(pd.DataFrame()).empty
    assert conserve_team_tds(pd.DataFrame()).empty
