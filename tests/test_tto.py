"""Times-through-the-order: the 3rd-look penalty, crossed with lineup spot."""

import datetime as dt

import pandas as pd

from src.demo import build_demo_slate
from src.model import matchup_multiplier, score_slate
from src.statcast import tto_penalties


def _pa(pid, thru, woba, n):
    return pd.DataFrame({"pitcher": [pid] * n, "n_thruorder_pitcher": [thru] * n,
                         "woba_value": [woba] * n, "woba_denom": [1] * n})


def test_tto_penalty_third_vs_early():
    # 1st/2nd time: .300 wOBA on 60 PA; 3rd time: .400 on 30 PA -> +.100.
    df = pd.concat([_pa(1, 1, 0.30, 30), _pa(1, 2, 0.30, 30), _pa(1, 3, 0.40, 30)])
    out = tto_penalties(df, min_pa=30).set_index("pitcher_id")
    assert out.loc[1, "sp_tto_penalty"] == 0.1


def test_tto_needs_samples_both_sides():
    df = pd.concat([_pa(1, 1, 0.30, 40), _pa(1, 3, 0.40, 5)])  # too few 3rd-time
    assert tto_penalties(df, min_pa=30).empty


def test_tto_empty_safe():
    assert tto_penalties(pd.DataFrame()).empty
    assert tto_penalties(pd.DataFrame({"pitcher": [1]})).empty


def test_tto_boosts_top_of_order_only():
    base = {"bats": "R", "pitcher_throws": "L", "pitcher_hr9": 1.1,
            "pitcher_barrel_pct_allowed": 8.0, "pitcher_lean": "NEU",
            "sp_tto_penalty": 0.045}
    leadoff = pd.Series({**base, "lineup_spot": 1})
    eighth = pd.Series({**base, "lineup_spot": 8})
    m_lead, _ = matchup_multiplier(leadoff)
    m_eight, _ = matchup_multiplier(eighth)
    # Leadoff reaches the 3rd look; the 8-hole barely does.
    assert m_lead > m_eight


def test_scored_slate_carries_tto():
    df = score_slate(build_demo_slate(dt.date(2026, 6, 18)))
    assert "sp_tto_penalty" in df.columns
    assert df["sp_tto_penalty"].notna().any()
