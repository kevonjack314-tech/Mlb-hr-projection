"""Tests for the Trends Lab: tiers, tier-banded odds, roles, and the 12 trends."""

import numpy as np
import pandas as pd

from src.history import build_hr_history
from src.odds import (
    american_to_prob,
    model_market_odds,
    tier_banded_market_odds,
)
from src.parlay import assign_role
from src.trends import (
    MID_HR_MIN,
    STAR_HR_MIN,
    TIER_MID,
    TIER_ODDS_BAND,
    TIER_STAR,
    TIER_UNDER,
    compute_trends,
    rotation_hint,
    tier_of,
)


# --------------------------------------------------------------------------- #
# Tiers
# --------------------------------------------------------------------------- #
def test_tier_boundaries():
    assert tier_of(STAR_HR_MIN) == TIER_STAR          # 18 -> Star
    assert tier_of(30) == TIER_STAR
    assert tier_of(STAR_HR_MIN - 1) == TIER_MID       # 17 -> Mid
    assert tier_of(MID_HR_MIN) == TIER_MID            # 8 -> Mid
    assert tier_of(MID_HR_MIN - 1) == TIER_UNDER      # 7 -> Under
    assert tier_of(0) == TIER_UNDER


def test_tier_handles_missing():
    assert tier_of(None) == TIER_UNDER
    assert tier_of(float("nan")) == TIER_UNDER
    assert tier_of("not a number") == TIER_UNDER


# --------------------------------------------------------------------------- #
# Tier-banded model-implied odds
# --------------------------------------------------------------------------- #
def test_star_odds_stay_in_star_band():
    lo, hi = TIER_ODDS_BAND[TIER_STAR]
    # A true slugger day (high prob) and a soft day both stay in +200..+450.
    for prob in (0.05, 0.12, 0.20, 0.30):
        a = tier_banded_market_odds(prob, season_hr=25)
        assert lo <= a <= hi


def test_mid_and_under_bands():
    lo, hi = TIER_ODDS_BAND[TIER_MID]
    for prob in (0.04, 0.10, 0.18):
        assert lo <= tier_banded_market_odds(prob, season_hr=12) <= hi
    lo, hi = TIER_ODDS_BAND[TIER_UNDER]
    for prob in (0.02, 0.06, 0.12):
        assert lo <= tier_banded_market_odds(prob, season_hr=3) <= hi


def test_band_only_clamps_when_needed():
    # A price already inside the band should pass through unchanged.
    prob = 0.25                       # model_market_odds ~ +264 -> inside Star band
    raw = model_market_odds(prob)
    assert TIER_ODDS_BAND[TIER_STAR][0] <= raw <= TIER_ODDS_BAND[TIER_STAR][1]
    assert tier_banded_market_odds(prob, season_hr=20) == raw


def test_banded_odds_still_convert_to_probs():
    a = tier_banded_market_odds(0.08, season_hr=10)
    p = american_to_prob(a)
    assert 0.0 < p < 0.25


# --------------------------------------------------------------------------- #
# Tier-aware parlay roles
# --------------------------------------------------------------------------- #
def test_roles_follow_tiers():
    assert assign_role(0.16, season_hr=25) == "Anchor"     # star -> anchor
    assert assign_role(0.12, season_hr=12) == "Value"      # mid -> value
    assert assign_role(0.16, season_hr=3) == "Longshot"    # under stays longshot
    assert assign_role(0.02, season_hr=40) is None         # below floor: no ticket


def test_roles_fall_back_to_prob_bands():
    assert assign_role(0.16) == "Anchor"
    assert assign_role(0.12) == "Value"
    assert assign_role(0.05) == "Longshot"
    assert assign_role(0.01) is None
    assert assign_role(0.16, season_hr=float("nan")) == "Anchor"


# --------------------------------------------------------------------------- #
# The 12 trends on the simulated history
# --------------------------------------------------------------------------- #
def _events():
    events, _slate, source, _ = build_hr_history(
        "2026-06-01", "2026-06-14", prefer_live=False)
    assert source == "SIMULATED" and not events.empty
    return events


def test_compute_trends_returns_at_least_10():
    trends = compute_trends(_events())
    assert len(trends) >= 10
    keys = {t["key"] for t in trends}
    assert len(keys) == len(trends)          # unique
    for t in trends:
        assert t["title"] and isinstance(t["signal"], str) and t["signal"]


def test_key_trends_present():
    keys = {t["key"] for t in compute_trends(_events())}
    # The three the user specifically asked for:
    assert "dow_spot" in keys          # lineup spots x specific days
    assert "tier_rotation" in keys     # star day -> mid/under next day
    assert "back_to_back" in keys      # consecutive-day repeat HR hitters


def test_back_to_back_signal_has_rate():
    ev = _events()
    trends = {t["key"]: t for t in compute_trends(ev)}
    b2b = trends["back_to_back"]
    assert "%" in b2b["signal"]


def test_trends_handle_empty_and_junk():
    assert compute_trends(None) == []
    assert compute_trends(pd.DataFrame()) == []
    assert compute_trends(pd.DataFrame({"date": ["junk", None]})) == []


def test_rotation_hint_is_string_or_none():
    hint = rotation_hint(_events())
    assert hint is None or isinstance(hint, str)
    assert rotation_hint(pd.DataFrame()) is None


def test_prep_tiers_from_season_hr():
    ev = pd.DataFrame({
        "date": ["2026-06-01", "2026-06-02"],
        "player": ["A", "B"],
        "team": ["NYY", "LAD"],
        "lineup_spot": [3, 7],
        "hr_count": [1, 2],
        "season_hr": [22, np.nan],
    })
    trends = compute_trends(ev)
    assert len(trends) >= 5                   # thin window still yields trends
