"""Self-calibration: learn from how the model *rated* bats vs. whether they homered.

The Trends backtest already records, for every hitter-day in the window, the
model's pre-game **HR Score** and whether the player actually homered. This module
turns that into a feedback signal:

  • `hit_rate_by_score` — empirical HR rate per HR-Score bucket (did high-rated
    bats actually go deep more often? where was the model right vs. wrong?).
  • `attach_calibrated_prob` — re-projects today's bats through that empirical
    curve, giving a **calibrated HR probability** and a **cal_edge** (calibrated −
    model). A positive cal_edge means history says this *kind* of rating homers
    more than the model currently credits — the parlay builder leans into it.

Because the curve is rebuilt from the trailing window every run (and the window
moves with the date), the system keeps adjusting as new games land — improving
parlay selection over time without any manual retuning.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def hit_rate_by_score(slate_hist: pd.DataFrame, bins: int = 8) -> pd.DataFrame:
    """Actual HR rate by pre-game HR-Score bucket. Columns: mid, n, hr_rate."""
    if (slate_hist is None or slate_hist.empty
            or "hr_score" not in slate_hist.columns or "hit_hr" not in slate_hist.columns):
        return pd.DataFrame()
    df = slate_hist.dropna(subset=["hr_score"]).copy()
    if df.empty:
        return pd.DataFrame()
    try:
        df["bucket"] = pd.cut(df["hr_score"], bins=bins)
    except Exception:
        return pd.DataFrame()
    g = df.groupby("bucket", observed=True)
    out = pd.DataFrame({
        "mid": [iv.mid for iv in g.size().index],
        "n": g.size().values,
        "hr_rate": g["hit_hr"].mean().values,
    })
    return out.sort_values("mid").reset_index(drop=True)


def calibrated_prob(score: float, table: pd.DataFrame) -> float:
    """Empirical HR probability for a given HR Score, interpolated from the curve."""
    if table is None or table.empty or score is None or (isinstance(score, float) and np.isnan(score)):
        return float("nan")
    return float(np.interp(score, table["mid"].values, table["hr_rate"].values))


def attach_calibrated_prob(slate: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """Add `calibrated_hr_prob` and `cal_edge_pct` (calibrated − model) to a slate."""
    slate = slate.copy()
    if table is None or table.empty or "hr_score" not in slate.columns:
        slate["calibrated_hr_prob"] = slate.get("hr_prob_game", np.nan)
        slate["cal_edge_pct"] = 0.0
        return slate
    slate["calibrated_hr_prob"] = slate["hr_score"].map(
        lambda s: calibrated_prob(s, table)).round(4)
    slate["cal_edge_pct"] = (
        (slate["calibrated_hr_prob"] - slate["hr_prob_game"]) * 100.0).round(1)
    return slate


def model_report_card(events: pd.DataFrame, slate_hist: pd.DataFrame | None = None) -> dict:
    """How well did the model flag the bats that actually homered?

    Returns capture rate (share of HR hitters the model rated 'live'), the HR
    hitters' average pre-game rating vs. the field, and a plain verdict.
    """
    if events is None or events.empty or "hr_score" not in events.columns:
        return {}
    hs = pd.to_numeric(events["hr_score"], errors="coerce").dropna()
    if hs.empty:
        return {}
    liked = float((hs >= 48).mean() * 100)          # model "liked" them pre-game
    loved = float((hs >= 60).mean() * 100)           # model "loved" them
    avg_hr_hitter = float(hs.mean())
    field = (float(pd.to_numeric(slate_hist["hr_score"], errors="coerce").mean())
             if slate_hist is not None and "hr_score" in slate_hist.columns else None)
    avg_prob = (float(pd.to_numeric(events.get("hr_prob_game"), errors="coerce").mean() * 100)
                if "hr_prob_game" in events.columns else None)
    if liked >= 60:
        verdict = "On point — most HRs came from bats the model already liked."
    elif liked >= 45:
        verdict = "Solid — the model flagged a healthy share of the HRs pre-game."
    else:
        verdict = "Lots of sleepers — many HRs came from bats the model underrated."
    return {
        "liked_pct": round(liked, 0),
        "loved_pct": round(loved, 0),
        "avg_hr_hitter_score": round(avg_hr_hitter, 1),
        "field_score": round(field, 1) if field is not None else None,
        "avg_prob": round(avg_prob, 0) if avg_prob is not None else None,
        "verdict": verdict,
    }
