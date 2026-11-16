"""Tests: the NBA 3+ made threes model.

Most public error in three-point props is a MINUTES error, not a shooting
error. A 40% shooter who plays 18 minutes is a worse bet than a 34% shooter
who plays 36, and the model has to say so.
"""
import math

import pandas as pd
import pytest

from src.nba.demo import build_demo_slate
from src.nba.model import (
    LISTED_3PA_CAP, MADE_THREES_LINE, conserve_team_3pa, p_at_least,
    project_minutes, score_row, score_slate, shooter_of_the_day,
)
from src.nba.teams import LEAGUE_3PA_PER_GAME, game_pace, rest_multiplier


def _shooter(**kw):
    base = {"minutes_l5": 33.0, "fg3a_per36": 8.5, "fg3_pct": 0.385,
            "fg3a_season": 320, "team_pace": 99.0, "opp_pace": 99.0,
            "opp_3pa_allowed": 35.5, "team_3pa_per_game": 35.5,
            "home_team": "BOS", "team": "BOS", "game": "NYK @ BOS",
            "is_home": True}
    base.update(kw)
    return pd.Series(base)


# --------------------------------------------------------------------------- #
# The distribution
# --------------------------------------------------------------------------- #
def test_p_at_least_matches_the_poisson_tail():
    """Poisson thinning: Poisson attempts x independent conversion = Poisson
    makes. Closed form, no simulation, no rounding a fractional attempt."""
    lam = 2.2
    expect = 1 - sum(math.exp(-lam) * lam ** i / math.factorial(i) for i in range(3))
    assert p_at_least(lam, 3) == pytest.approx(expect, abs=1e-6)


def test_more_expected_makes_is_always_more_likely():
    vals = [p_at_least(x) for x in (0.5, 1.0, 2.0, 3.0, 4.5)]
    assert vals == sorted(vals)
    assert p_at_least(0.0) == 0.0


def test_a_two_and_a_half_make_projection_prices_like_a_real_book():
    """Books hang 3+ threes around +120/+150 for a ~2.2-make projection."""
    assert 0.33 <= p_at_least(2.2) <= 0.42


# --------------------------------------------------------------------------- #
# Minutes dominate
# --------------------------------------------------------------------------- #
def test_the_bench_gunner_loses_to_the_starter_who_shoots_less():
    """Volume rate is not volume. You cannot make what you do not attempt."""
    gunner = _shooter(minutes_l5=17.0, fg3a_per36=12.0, fg3_pct=0.375)
    starter = _shooter(minutes_l5=36.0, fg3a_per36=6.5, fg3_pct=0.360)
    assert score_row(starter)["threes_prob_game"] > score_row(gunner)["threes_prob_game"]


def test_a_starter_being_out_raises_the_backup():
    base = _shooter(minutes_l5=19.0)
    bumped = _shooter(minutes_l5=19.0, teammate_out_minutes_bump=7.0)
    assert score_row(bumped)["proj_minutes"] > score_row(base)["proj_minutes"]
    assert score_row(bumped)["threes_prob_game"] > score_row(base)["threes_prob_game"]


def test_volatile_minutes_are_reported_as_lower_confidence():
    steady = score_row(_shooter(minutes_stdev_l5=1.5))
    swingy = score_row(_shooter(minutes_stdev_l5=9.0))
    assert steady["minutes_confidence"] > swingy["minutes_confidence"]
    assert "lower confidence" in swingy["rationale"]


def test_minutes_fall_back_to_season_then_to_a_starter_default():
    assert project_minutes(pd.Series({"minutes_season": 28.0}))["proj_minutes"] == 28.0
    assert project_minutes(pd.Series({}))["proj_minutes"] > 0


# --------------------------------------------------------------------------- #
# Rest, pace, matchup
# --------------------------------------------------------------------------- #
def test_a_back_to_back_costs_both_legs_and_shooting():
    r = rest_multiplier(is_back_to_back=True)
    assert r["rest_minutes_mult"] < 1.0 and r["rest_shooting_mult"] < 1.0
    fresh = score_row(_shooter(back_to_back=False, days_rest=2))
    tired = score_row(_shooter(back_to_back=True))
    assert tired["threes_prob_game"] < fresh["threes_prob_game"]
    assert tired["rest_note"] == "back-to-back"


def test_pace_is_the_matchup_average_not_one_team():
    """A fast team playing a slow one lands between them."""
    assert game_pace(104.0, 94.0) == pytest.approx(99.0)
    assert game_pace(104.0, 94.0, "DEN") > 99.0     # altitude speeds it up
    assert game_pace(None, None) > 0


def test_a_defense_that_concedes_threes_raises_attempts():
    tight = score_row(_shooter(opp_3pa_allowed=30.0))
    loose = score_row(_shooter(opp_3pa_allowed=41.0))
    assert loose["proj_3pa"] > tight["proj_3pa"]


def test_a_hot_percentage_on_a_thin_sample_is_shrunk():
    """48% on 40 attempts is a 38% shooter who has had a good month."""
    thin = score_row(_shooter(fg3_pct=0.480, fg3a_season=40))
    deep = score_row(_shooter(fg3_pct=0.480, fg3a_season=600))
    assert thin["proj_3p_pct"] < deep["proj_3p_pct"] <= 0.48


# --------------------------------------------------------------------------- #
# Conservation
# --------------------------------------------------------------------------- #
def test_a_team_cannot_attempt_more_threes_than_it_takes():
    s = score_slate(build_demo_slate("2026-11-16"))
    g = s.groupby(["game", "team"]).agg(
        att=("proj_3pa", "sum"), team=("team_3pa_per_game", "first"),
        pace=("game_pace", "first"))
    cap = g["team"] * (g["pace"] / 99.0) * LISTED_3PA_CAP
    assert (g["att"] <= cap + 1e-6).all()


def test_rim_protectors_sit_near_zero_and_snipers_near_the_top():
    s = score_slate(build_demo_slate("2026-11-16"))
    by = s.groupby("archetype")["threes_prob_game"].mean()
    assert by["rimbig"] < 0.05
    assert by["sniper"] > by["wing3d"] > by["slasher"]


def test_slate_shape_is_believable():
    s = score_slate(build_demo_slate("2026-11-16"))
    assert s["threes_prob_game"].between(0, 0.97).all()
    assert s["proj_minutes"].between(0, 44).all()
    assert 0.10 <= s["threes_prob_game"].mean() <= 0.40


# --------------------------------------------------------------------------- #
# Slate
# --------------------------------------------------------------------------- #
def test_demo_slate_is_deterministic_with_unique_players():
    a, b = build_demo_slate("2026-11-16"), build_demo_slate("2026-11-16")
    assert a.equals(b)
    assert len(a) == a["player"].nunique(), "two players sharing a name"


def test_shooter_of_the_day_and_empty_slate_are_safe():
    s = score_slate(build_demo_slate("2026-11-16"))
    pick = shooter_of_the_day(s)
    assert pick["threes_prob_game"] == s["threes_prob_game"].max()
    assert shooter_of_the_day(pd.DataFrame()) is None
    assert score_slate(pd.DataFrame()).empty
    assert conserve_team_3pa(pd.DataFrame()).empty
    assert MADE_THREES_LINE == 3
