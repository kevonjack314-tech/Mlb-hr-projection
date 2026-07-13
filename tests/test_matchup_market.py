"""Tests for accuracy gains #3-5: market blend + totals, real platoon splits,
and bullpen exposure."""

import numpy as np
import pandas as pd
import pytest

from src import odds as odds_mod
from src import statcast as sc_mod
from src.model import _platoon_multiplier, matchup_multiplier
from src.odds import MARKET_BLEND_W, attach_odds, american_to_prob


# --------------------------------------------------------------------------- #
# #4 — real platoon splits
# --------------------------------------------------------------------------- #
def test_platoon_uses_real_split_when_available():
    base = {"bats": "L", "pitcher_throws": "R"}
    crusher = pd.Series({**base, "woba_vs_r": 0.400})
    struggler = pd.Series({**base, "woba_vs_r": 0.250})
    assert _platoon_multiplier(crusher) > 1.05
    assert _platoon_multiplier(struggler) < 0.95
    # A big split moves the whole matchup multiplier.
    assert matchup_multiplier(crusher)[0] > matchup_multiplier(struggler)[0]


def test_platoon_falls_back_to_flat_prior():
    adv = pd.Series({"bats": "L", "pitcher_throws": "R"})       # platoon edge
    dis = pd.Series({"bats": "R", "pitcher_throws": "R"})
    assert _platoon_multiplier(adv) == 1.06
    assert _platoon_multiplier(dis) == 0.95
    nan_split = pd.Series({"bats": "L", "pitcher_throws": "R", "woba_vs_r": float("nan")})
    assert _platoon_multiplier(nan_split) == 1.06


# --------------------------------------------------------------------------- #
# #5 — bullpen exposure
# --------------------------------------------------------------------------- #
def test_bullpen_hr9_shifts_matchup():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.2}
    leaky = pd.Series({**base, "bullpen_hr9": 1.6})
    stingy = pd.Series({**base, "bullpen_hr9": 0.7})
    none = pd.Series(base)
    m_leaky, _ = matchup_multiplier(leaky)
    m_stingy, _ = matchup_multiplier(stingy)
    m_none, _ = matchup_multiplier(none)
    assert m_leaky > m_none > m_stingy


def test_bullpen_table_normalizes_team_codes(monkeypatch):
    fake = pd.DataFrame({
        "Team": ["TBR", "TBR", "SDP", "NYY", "NYY"],
        "GS": [0, 0, 0, 0, 12],          # the GS=12 row is a starter -> excluded
        "IP": [80.0, 40.0, 100.0, 60.0, 90.0],
        "HR": [16, 8, 6, 8, 20],
    })
    monkeypatch.setattr(sc_mod, "_fg_pitching_raw", lambda year: fake)
    sc_mod.get_bullpen_hr9_table.cache_clear()
    table = sc_mod.get_bullpen_hr9_table(1999)
    assert table["TB"] == pytest.approx(9 * 24 / 120, abs=0.01)   # FG code fixed
    assert table["SD"] == pytest.approx(0.54, abs=0.01)
    assert table["NYY"] == pytest.approx(1.2, abs=0.01)           # starter excluded
    sc_mod.get_bullpen_hr9_table.cache_clear()


# --------------------------------------------------------------------------- #
# #3 — market blend + game totals
# --------------------------------------------------------------------------- #
def _mini_slate():
    return pd.DataFrame({
        "player": ["Aaron Judge", "Some Guy"],
        "home_team": ["NYY", "NYY"],
        "season_hr": [30, 5],
        "hr_prob_game": [0.20, 0.05],
        "fair_odds": [400, 1900],
    })


def test_attach_odds_offline_is_pure_model():
    df = attach_odds(_mini_slate(), "2026-07-13", use_live=False)
    assert (df["hr_prob_game"] == df["hr_prob_model"]).all()   # untouched
    assert df["game_total"].isna().all()
    assert not df["odds_is_live"].any()


def test_market_blend_and_totals(monkeypatch):
    monkeypatch.setattr(odds_mod, "fetch_live_hr_odds",
                        lambda d: {"aaron judge": {"odds": 310, "book": "TestBook"}})
    monkeypatch.setattr(odds_mod, "fetch_game_totals", lambda d: {"NYY": 10.5})
    df = attach_odds(_mini_slate(), "2026-07-13", use_live=True)

    judge = df[df["player"] == "Aaron Judge"].iloc[0]
    other = df[df["player"] == "Some Guy"].iloc[0]

    # Totals nudge: 10.5 total > league 8.6 lifts everyone (clipped at +8%).
    tot_mult = min(1.08, 1 + (10.5 - 8.6) * 0.022)
    assert other["hr_prob_game"] == pytest.approx(0.05 * tot_mult, abs=1e-3)

    # Live row: blended toward the de-vigged market price.
    p_after_totals = 0.20 * tot_mult
    market_fair = american_to_prob(310) / 1.10
    expected = (1 - MARKET_BLEND_W) * p_after_totals + MARKET_BLEND_W * market_fair
    assert judge["hr_prob_game"] == pytest.approx(expected, abs=1e-3)
    assert judge["odds_is_live"] and judge["book_odds"] == 310
    assert judge["hr_prob_model"] == pytest.approx(0.20)   # pre-blend preserved
    assert judge["game_total"] == 10.5

    # Edge is computed from the final (blended) probability.
    assert judge["edge_pct"] == pytest.approx(
        (judge["hr_prob_game"] - american_to_prob(310)) * 100, abs=0.1)


def test_totals_mapping_handles_all_30_names():
    assert len(set(odds_mod.TEAM_FULL_TO_ABBR.values())) == 30
