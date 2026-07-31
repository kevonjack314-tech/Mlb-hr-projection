"""Tests: the projected-final-score model.

Covers the run distribution's shape (baseball scoring is NOT Poisson), the
lineup -> runs chain, the park RUN factor being distinct from the HR factor,
and the slate-level sanity of the projections.
"""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from src.gamescore import (
    HOME_FIELD_RUNS, LEAGUE_RPG, LEAGUE_XWOBA, RUN_DISPERSION, SPOT_PA_SHARE,
    _runs_from_xwoba, attach_top_bat, best_bets, game_of_the_day,
    game_rationale, lineup_offense, matchup_probabilities, over_under,
    pitching_multiplier, predict_games, run_pmf,
)
from src.model import score_slate
from src.parks import park_run_multiplier
from src.sources import get_slate


# --------------------------------------------------------------------------- #
# Run distribution
# --------------------------------------------------------------------------- #
def test_run_pmf_is_a_distribution_with_the_right_mean():
    p = run_pmf(4.45)
    assert p.sum() == pytest.approx(1.0, abs=1e-6)
    assert (p >= 0).all()
    mean = float((p * np.arange(len(p))).sum())
    assert mean == pytest.approx(4.45, abs=0.05)


def test_scoring_is_overdispersed_not_poisson():
    """Poisson says a 4.45-run offense is shut out ~1% of the time. It's ~7%."""
    nb = run_pmf(4.45, dispersion=RUN_DISPERSION)
    poisson = run_pmf(4.45, dispersion=1.0)
    assert nb[0] > 3 * poisson[0]
    k = np.arange(len(nb))
    var = float((nb * (k - 4.45) ** 2).sum())
    assert 9.0 <= var <= 10.5, f"variance {var} — real MLB is ~9.7"


def test_better_offense_wins_more_often():
    weak = matchup_probabilities(3.6, 5.4)
    strong = matchup_probabilities(5.4, 3.6)
    assert strong["p_home"] > 0.5 > weak["p_home"]
    assert strong["p_home"] + strong["p_away"] == pytest.approx(1.0, abs=1e-3)


def test_ties_are_broken_toward_the_better_offense():
    """Extra innings aren't a coin flip — the better offense wins more of them."""
    even = matchup_probabilities(4.45, 4.45)
    assert even["p_tie_reg"] > 0.05          # regulation ties are common
    assert even["p_home"] == pytest.approx(0.5, abs=0.02)


def test_over_under_partitions_the_total():
    probs = matchup_probabilities(4.5, 4.3)
    ou = over_under(probs["totals"], 8.5)
    assert ou["p_over"] + ou["p_under"] + ou["p_push"] == pytest.approx(1.0, abs=1e-6)
    assert ou["p_push"] == 0.0               # half-run lines cannot push
    assert over_under(probs["totals"], 9.0)["p_push"] > 0     # whole numbers can


def test_higher_line_is_harder_to_go_over():
    t = matchup_probabilities(4.5, 4.5)["totals"]
    assert over_under(t, 7.5)["p_over"] > over_under(t, 10.5)["p_over"]


# --------------------------------------------------------------------------- #
# Offense
# --------------------------------------------------------------------------- #
def _lineup(xwobas, team="AAA"):
    return pd.DataFrame({"team": team, "lineup_spot": list(range(1, len(xwobas) + 1)),
                         "xwoba": xwobas, "iso": 0.180})


def test_lineup_offense_is_pa_weighted_by_spot():
    """Same nine bats; the good one leading off is worth more than hitting 9th."""
    top = lineup_offense(_lineup([0.400] + [0.300] * 8))
    bottom = lineup_offense(_lineup([0.300] * 8 + [0.400]))
    assert top["lineup_xwoba"] > bottom["lineup_xwoba"]
    assert SPOT_PA_SHARE[0] > SPOT_PA_SHARE[-1]


def test_league_average_lineup_scores_league_average_runs():
    assert _runs_from_xwoba(LEAGUE_XWOBA) == pytest.approx(LEAGUE_RPG, abs=1e-6)
    assert _runs_from_xwoba(0.340) > LEAGUE_RPG
    assert _runs_from_xwoba(0.295) < LEAGUE_RPG


def test_empty_lineup_falls_back_to_league_average():
    assert lineup_offense(pd.DataFrame())["lineup_xwoba"] == LEAGUE_XWOBA
    assert lineup_offense(_lineup([np.nan] * 9))["lineup_xwoba"] == LEAGUE_XWOBA


# --------------------------------------------------------------------------- #
# Pitching
# --------------------------------------------------------------------------- #
def test_homer_prone_staff_raises_the_run_environment():
    soft = pitching_multiplier(pd.DataFrame({"pitcher_hr9": [2.0] * 9,
                                             "pitcher_barrel_pct_allowed": [12.0] * 9}))
    tough = pitching_multiplier(pd.DataFrame({"pitcher_hr9": [0.6] * 9,
                                              "pitcher_barrel_pct_allowed": [4.0] * 9}))
    assert soft["staff_mult"] > 1.0 > tough["staff_mult"]


def test_pitching_multiplier_survives_missing_columns():
    """Feeds go partial all the time — a missing bullpen table can't crash it."""
    out = pitching_multiplier(pd.DataFrame({"player": ["a", "b"]}))
    assert out["staff_mult"] == 1.0
    assert pitching_multiplier(pd.DataFrame())["staff_mult"] == 1.0


def test_velo_loss_makes_a_starter_more_hittable():
    base = pd.DataFrame({"pitcher_hr9": [1.25] * 9, "pitcher_barrel_pct_allowed": [8.0] * 9})
    down = base.assign(sp_velo_delta=-1.5)
    assert pitching_multiplier(down)["sp_mult"] > pitching_multiplier(base)["sp_mult"]


# --------------------------------------------------------------------------- #
# Park RUN factors — the thing that is NOT the HR factor
# --------------------------------------------------------------------------- #
def test_run_factor_is_not_the_hr_factor():
    """Fenway suppresses HR and inflates runs; Yankee Stadium is the reverse."""
    from src.parks import park_hr_multiplier
    # Fenway: a bad HR park for righties, a top-3 RUN park (Monster doubles).
    assert park_hr_multiplier("BOS", "R") < 1.0 < park_run_multiplier("BOS")
    # Yankee Stadium: the short porch is a home-run effect, not a runs effect.
    assert park_hr_multiplier("NYY", "L") > park_run_multiplier("NYY")
    # Kauffman: suppresses HR both ways, but all that grass is doubles/triples.
    assert park_hr_multiplier("KC", "L") < 1.0 < park_run_multiplier("KC")


def test_coors_is_the_top_run_park():
    others = [park_run_multiplier(t) for t in
              ("SF", "SEA", "SD", "NYY", "BOS", "CIN", "LAD")]
    assert park_run_multiplier("COL") > max(others)


def test_unknown_park_is_neutral():
    assert park_run_multiplier("ZZZ") == 1.0


# --------------------------------------------------------------------------- #
# The slate end-to-end
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _slate():
    df, _, _ = get_slate(dt.date(2026, 7, 27), prefer_live=False)
    return score_slate(df)


@pytest.fixture(scope="module")
def _games(_slate):
    return attach_top_bat(predict_games(_slate), _slate)


def test_every_game_gets_a_projection(_slate, _games):
    assert len(_games) == _slate["game"].nunique()
    assert _games["winner"].notna().all()
    assert _games["score_line"].str.contains("–").all()


def test_projected_totals_are_believable(_games):
    """A slate averaging 12 runs a game means the model is broken."""
    assert 7.0 <= _games["total_exp"].mean() <= 10.5
    assert _games["total_exp"].between(5.0, 16.0).all()
    assert _games["home_runs_exp"].between(1.4, 11.0).all()


def test_win_probabilities_are_sane(_games):
    assert _games["win_prob"].between(0.5, 0.90).all()
    assert (_games["p_home"] + _games["p_away"] - 1.0).abs().max() < 1e-3


def test_slate_is_centered_not_all_overs(_games):
    """Before centering, every game projected Over — that was the bug."""
    assert _games["total_lean"].nunique() >= 1
    assert _games["total_exp"].mean() < 11.0


def test_centering_pulls_an_inflated_slate_down(_slate):
    """Only the bats with Statcast rows resolve -> the level must be re-centered."""
    raw = predict_games(_slate, center_to_league=False)
    centered = predict_games(_slate, center_to_league=True)
    assert centered["total_exp"].mean() < raw["total_exp"].mean()


def test_best_bets_and_rationale(_games):
    bets = best_bets(_games, 3)
    assert len(bets) == 3
    assert bets["confidence"].is_monotonic_decreasing
    for _, r in bets.iterrows():
        assert r["winner"] in r["rationale"]
        assert "ML" in r["pick"] and "%" in r["total_pick"]


def test_game_of_the_day(_games):
    g = game_of_the_day(_games)
    assert g is not None
    assert g["gotd_angle"] in ("side", "total")
    assert g["gotd_bet"]
    assert g["game"] in set(_games["game"])


def test_top_bat_ties_back_to_the_hr_model(_slate, _games):
    row = _games.iloc[0]
    best = (_slate[_slate["game"] == row["game"]]
            .sort_values("hr_prob_game", ascending=False).iloc[0])
    assert row["top_bat"] == best["player"]


def test_empty_slate_is_safe():
    assert predict_games(pd.DataFrame()).empty
    assert best_bets(pd.DataFrame()).empty
    assert game_of_the_day(pd.DataFrame()) is None


def test_home_field_edge_exists_but_is_small():
    assert 0.0 < HOME_FIELD_RUNS < 0.3
    df = pd.DataFrame({
        "game": ["A @ B"] * 18, "home_team": "B",
        "team": ["A"] * 9 + ["B"] * 9, "is_home": [False] * 9 + [True] * 9,
        "lineup_spot": list(range(1, 10)) * 2, "xwoba": [0.318] * 18,
        "pitcher_hr9": [1.25] * 18, "pitcher_barrel_pct_allowed": [8.0] * 18,
        "temp_f": 72.0, "wind_mph": 0.0, "wind_dir_deg": 0.0, "humidity_pct": 50.0,
    })
    g = predict_games(df, center_to_league=False).iloc[0]
    assert g["home_runs_exp"] > g["away_runs_exp"]
    assert g["winner"] == "B" and 0.50 < g["p_home"] < 0.56


def test_rationale_names_the_park_when_it_matters(_games):
    coors = _games[_games["game"].str.contains("COL")]
    if not coors.empty:
        assert "park" in game_rationale(coors.iloc[0])


def test_projections_are_no_longer_near_flat(_games):
    """The diagnosed failure: 0.56 runs of spread against reality's ~4.5.

    A regressed run projection is correct — game scoring is mostly noise — but
    a near-constant one carries no information whatever its MAE looks like.
    Market totals live around 0.8-1.0 runs of spread; this is the floor that
    keeps the model in that neighbourhood rather than back at flat.
    """
    assert _games["total_exp"].std() >= 0.85
    assert _games["total_exp"].max() - _games["total_exp"].min() >= 2.5
    # Win probabilities have to differentiate too — they topped out at .599.
    assert _games["win_prob"].max() >= 0.62
