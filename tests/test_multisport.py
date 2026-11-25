"""Tests: one pipeline, three sports.

The parlay builder, the role bands and the odds fallback are shared. The point
of sharing them is that a fix lands everywhere at once — the leakage guard had
to be written once and should not have to be written three times — so these
tests check the shared pieces behave per sport AND that baseball is unchanged.
"""
import os

import pandas as pd
import pytest

from src.multisport import (
    PROP_HOLD, assign_roles, attach_model_odds, build_slate, from_canonical,
    generate_parlay, prob_col, role_bands, score_col, to_canonical,
)
from src.sports import SPORTS, get_sport

_APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "app.py")

NEW_SPORTS = ("nfl", "nba")
ALL_SPORTS = ("mlb", "nfl", "nba")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
def test_every_sport_targets_a_rare_binary_event():
    for key in ALL_SPORTS:
        s = get_sport(key)
        assert s.prop_label and s.odds_api_key and s.event_short
        # The whole premise: an event rare enough to be worth parlaying.
        assert 0.05 < s.base_rate < 0.5


def test_unknown_sport_degrades_to_mlb_rather_than_raising():
    assert get_sport("nhl").key == "mlb"
    assert get_sport(None).key == "mlb"


def test_mlb_keeps_its_existing_data_paths():
    """The 63-day graded record lives at data/ and must not move."""
    assert get_sport("mlb").data_subdir == ""
    assert get_sport("nfl").data_subdir == "nfl"


# --------------------------------------------------------------------------- #
# Role bands scale to the sport
# --------------------------------------------------------------------------- #
def test_role_bands_scale_with_how_often_the_event_happens():
    """Reused unchanged, the baseball bands called a 42% touchdown player a
    Longshot — it cleared every band, and the tier stat broke the tie."""
    mlb, nfl, nba = (role_bands(s) for s in ALL_SPORTS)
    assert nfl["anchor"] > mlb["anchor"]
    assert nba["anchor"] > nfl["anchor"]
    for b in (mlb, nfl, nba):
        assert b["anchor"] > b["value"] > b["longshot"] > 0


@pytest.mark.parametrize("sport", NEW_SPORTS)
def test_the_anchor_is_the_safest_leg_not_the_longest(sport):
    board, _ = build_slate(sport, "2026-11-16")
    roles = assign_roles(board, sport)
    p = pd.to_numeric(roles[prob_col(sport)], errors="coerce")
    anchors = p[roles["role"] == "Anchor"]
    longs = p[roles["role"] == "Longshot"]
    assert not anchors.empty and not longs.empty
    assert anchors.min() > longs.max()


# --------------------------------------------------------------------------- #
# Odds fallback is labelled, never passed off as a market
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sport", NEW_SPORTS)
def test_model_implied_odds_say_so_and_carry_the_hold(sport):
    board, _ = build_slate(sport, "2026-11-16")
    priced = attach_model_odds(board, sport)
    assert (priced["odds_source"] == "model-implied").all()
    assert not priced["odds_is_live"].any()
    # Shaded by the hold, so a ticket built on them cannot read as free money.
    assert priced["edge_pct"].mean() < 0
    assert PROP_HOLD > 0


@pytest.mark.parametrize("sport", NEW_SPORTS)
def test_prices_land_inside_the_sport_realistic_band(sport):
    priced = attach_model_odds(build_slate(sport, "2026-11-16")[0], sport)
    spec = get_sport(sport)
    lo = min(b[0] for b in spec.tier_bands.values())
    hi = max(b[1] for b in spec.tier_bands.values())
    assert priced["book_odds"].between(lo, hi).all()


# --------------------------------------------------------------------------- #
# The shared builder works for every sport
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sport", ALL_SPORTS)
def test_a_ticket_builds_for_every_sport(sport):
    board, source = build_slate(sport, "2026-11-16")
    if sport == "mlb":
        from src.odds import attach_odds
        board = attach_odds(board, "2026-11-16", use_live=False)
    res = generate_parlay(board, sport, n_legs=3, strategy="ulx")
    legs = res["legs"]
    assert len(legs) == 3
    assert res["sport"] == sport
    # Diversification survives the translation: no two legs from one game.
    assert legs["game"].nunique() == 3
    assert legs["player"].nunique() == 3
    assert res["summary"]["combined_american_str"]


@pytest.mark.parametrize("sport", NEW_SPORTS)
def test_the_legs_come_back_in_the_sport_own_columns(sport):
    board = attach_model_odds(build_slate(sport, "2026-11-16")[0], sport)
    legs = generate_parlay(board, sport, n_legs=3)["legs"]
    assert prob_col(sport) in legs.columns
    assert score_col(sport) in legs.columns


def test_translation_round_trips():
    board, _ = build_slate("nfl", "2026-11-16")
    canon = to_canonical(board, "nfl")
    assert "hr_prob_game" in canon.columns and "season_hr" in canon.columns
    back = from_canonical(canon, "nfl")
    assert "td_prob_game" in back.columns
    assert back["td_prob_game"].equals(board["td_prob_game"])


def test_mlb_passes_through_the_translator_untouched():
    """Baseball must not pay for the other two sports being added."""
    board, _ = build_slate("mlb", "2026-11-16")
    assert to_canonical(board, "mlb").equals(board)
    legs = pd.DataFrame({"hr_prob_game": [0.2], "hr_score": [70.0]})
    assert from_canonical(legs, "mlb").equals(legs)


def test_build_slate_labels_synthetic_boards_as_synthetic():
    """The same guard baseball uses: modelled numbers never pass as real."""
    for sport in NEW_SPORTS:
        _, source = build_slate(sport, "2026-11-16")
        assert "DEMO" in source


# --------------------------------------------------------------------------- #
# The app actually renders each sport
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sport,expect", [
    ("nfl", "NFL Anytime TD"), ("nba", "NBA 3+ Threes")])
def test_the_app_renders_each_sport_without_error(sport, expect):
    """The sport picker lives in the SIDEBAR — a first pass at this test drove
    the main-area radio instead, never switched sports, and happily asserted
    against the baseball board it was still looking at."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(_APP, default_timeout=300).run()
    at.sidebar.radio(key="sport_picker").set_value(sport).run()
    # Force the synthetic path so the test does not depend on ESPN being
    # reachable — and so it can assert the thing that actually matters: a
    # modelled board must announce itself as modelled.
    at.sidebar.toggle(key=f"live_{sport}").set_value(False).run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    assert any(expect in t.value for t in at.title)
    assert any("modelled, not live" in w.value for w in at.warning)
    assert at.dataframe, "no board rendered"


def test_the_app_still_opens_on_baseball_by_default():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(_APP, default_timeout=300).run()
    assert not at.exception
    assert any("MLB Home Run" in t.value for t in at.title)
