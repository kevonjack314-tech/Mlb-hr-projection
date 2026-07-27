"""Feedback loop for the score predictor.

The HR model has had a graded record, a calibration curve and holdout
validation for a while. The score model shipped without any of it, which means
nothing could tell you whether a projected 5-4 was good work or a coin flip
dressed up in decimals. This closes that: every projection gets logged against
the actual final score, and the accumulated record answers the only questions
that matter —

  * how far off are the run projections (MAE, per side and on the total)?
  * how often does the projected winner actually win?
  * is the win probability CALIBRATED — do 60% sides win 60% of the time?
  * does the Over/Under lean beat a coin flip?
  * and the honest baseline: does any of it beat "always take the home team"?

A model that can't beat its own baselines should be told so, loudly, rather
than quietly shipping picks.
"""

from __future__ import annotations

import datetime as dt
import os

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
GAME_EVAL_LOG_PATH = os.path.join(_DATA_DIR, "game_eval_log.csv")

# What gets logged per graded game. The projection inputs come along so the
# record can later answer WHY it missed, not just that it did.
GAME_EVAL_COLS = [
    "date", "game", "home_team", "away_team",
    # what the model said, pre-game
    "home_runs_exp", "away_runs_exp", "total_exp", "most_likely_total",
    "winner", "win_prob", "p_home", "total_line", "p_over", "total_lean",
    "total_lean_prob", "confidence", "data_quality_pct", "gotd",
    # the inputs, for post-mortems
    "home_lineup_xwoba", "away_lineup_xwoba", "home_staff_mult",
    "away_staff_mult", "park_run_mult", "weather_run_mult",
    # what actually happened
    "home_runs_actual", "away_runs_actual", "total_actual", "innings",
    "winner_actual", "winner_correct", "total_over_actual", "total_lean_correct",
    "abs_err_home", "abs_err_away", "abs_err_total",
]


# --------------------------------------------------------------------------- #
# Log I/O
# --------------------------------------------------------------------------- #
def load_game_log() -> pd.DataFrame:
    if not os.path.exists(GAME_EVAL_LOG_PATH):
        return pd.DataFrame(columns=GAME_EVAL_COLS)
    try:
        return pd.read_csv(GAME_EVAL_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=GAME_EVAL_COLS)


def append_game_rows(rows: pd.DataFrame) -> int:
    """De-duped (date, game) append. Returns the number of new rows."""
    if rows is None or rows.empty:
        return 0
    rows = rows[[c for c in GAME_EVAL_COLS if c in rows.columns]].copy()
    existing = load_game_log()
    combined = pd.concat([existing, rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "game"], keep="last")
    os.makedirs(_DATA_DIR, exist_ok=True)
    combined.to_csv(GAME_EVAL_LOG_PATH, index=False)
    return len(combined) - len(existing)


# --------------------------------------------------------------------------- #
# Grading one date
# --------------------------------------------------------------------------- #
def grade_projections(games: pd.DataFrame, finals, date_iso: str) -> pd.DataFrame:
    """Join projections to real final scores. Pure — no network, so it tests.

    `finals` is whatever `sources.fetch_final_scores` returns (an iterable of
    dicts). Games without a final are dropped rather than half-graded.
    """
    if games is None or games.empty or not len(finals):
        return pd.DataFrame()
    fin = pd.DataFrame(list(finals))
    if fin.empty or "game" not in fin.columns:
        return pd.DataFrame()
    g = games.copy()
    g["date"] = date_iso
    merged = g.merge(fin[["game", "home_runs", "away_runs", "innings"]],
                     on="game", how="inner")
    if merged.empty:
        return pd.DataFrame()
    merged = merged.rename(columns={"home_runs": "home_runs_actual",
                                    "away_runs": "away_runs_actual"})
    ha = merged["home_runs_actual"].astype(float)
    aa = merged["away_runs_actual"].astype(float)
    merged["total_actual"] = ha + aa
    merged["winner_actual"] = np.where(ha > aa, merged["home_team"],
                                       merged["away_team"])
    merged["winner_correct"] = (merged["winner"] == merged["winner_actual"]).astype(int)
    line = pd.to_numeric(merged.get("total_line", 8.5), errors="coerce").fillna(8.5)
    merged["total_over_actual"] = np.where(
        merged["total_actual"] > line, 1,
        np.where(merged["total_actual"] < line, 0, -1))       # -1 = push
    lean_over = (merged["total_lean"] == "Over").astype(int)
    merged["total_lean_correct"] = np.where(
        merged["total_over_actual"] < 0, -1,
        (lean_over == merged["total_over_actual"]).astype(int))
    merged["abs_err_home"] = (ha - pd.to_numeric(merged["home_runs_exp"])).abs().round(3)
    merged["abs_err_away"] = (aa - pd.to_numeric(merged["away_runs_exp"])).abs().round(3)
    merged["abs_err_total"] = (merged["total_actual"]
                               - pd.to_numeric(merged["total_exp"])).abs().round(3)
    if "gotd" not in merged.columns:
        merged["gotd"] = 0
    return merged[[c for c in GAME_EVAL_COLS if c in merged.columns]]


def evaluate_game_day(date_iso: str, prefer_live: bool = True):
    """Project a completed date and grade it. Returns (rows | None, note).

    Mirrors `tuning.evaluate_day`: only real slates are ever graded, so a
    synthetic demo board can never leak into the record.
    """
    from .gamescore import game_of_the_day, predict_games
    from .model import score_slate
    from .sources import fetch_final_scores, get_slate

    game_date = dt.date.fromisoformat(date_iso)
    df, source, _ = get_slate(game_date, prefer_live=prefer_live)
    if df is None or df.empty:
        return None, "no slate"
    if not str(source).startswith("LIVE"):
        return None, f"slate source is {source} — skipping (real days only)"

    games = predict_games(score_slate(df))
    if games.empty:
        return None, "no projections"
    gotd = game_of_the_day(games)
    games["gotd"] = (games["game"] == gotd["game"]).astype(int) if gotd is not None else 0

    finals = fetch_final_scores(date_iso)
    if not finals:
        return None, "no final scores yet"
    rows = grade_projections(games, finals, date_iso)
    if rows.empty:
        return None, "no projections matched a completed game"
    return rows, f"{len(rows)} games graded"


# --------------------------------------------------------------------------- #
# The record
# --------------------------------------------------------------------------- #
def _brier(p, y) -> float | None:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    m = ~(np.isnan(p) | np.isnan(y))
    return round(float(((p[m] - y[m]) ** 2).mean()), 5) if m.any() else None


def game_record(log: pd.DataFrame | None = None) -> dict:
    """Headline accuracy of the score predictor, with honest baselines.

    The baselines are the point. "Winner picked correctly 54% of the time"
    means nothing until you know that always taking the home team would have
    gone 52%. Beating a coin flip is not the bar.
    """
    log = load_game_log() if log is None else log
    out = {"n": 0, "days": 0}
    if log is None or log.empty:
        return out
    d = log.dropna(subset=["home_runs_actual", "away_runs_actual"]).copy()
    if d.empty:
        return out

    ha = pd.to_numeric(d["home_runs_actual"], errors="coerce")
    aa = pd.to_numeric(d["away_runs_actual"], errors="coerce")
    p_home = pd.to_numeric(d["p_home"], errors="coerce")
    home_won = (ha > aa).astype(float)

    graded_totals = d[pd.to_numeric(d["total_lean_correct"], errors="coerce") >= 0]
    gotd = d[pd.to_numeric(d.get("gotd", 0), errors="coerce").fillna(0) == 1]

    out.update({
        "n": int(len(d)),
        "days": int(d["date"].nunique()),
        # run accuracy
        "mae_side": round(float(pd.concat([
            pd.to_numeric(d["abs_err_home"], errors="coerce"),
            pd.to_numeric(d["abs_err_away"], errors="coerce")]).mean()), 3),
        "mae_total": round(float(pd.to_numeric(d["abs_err_total"],
                                               errors="coerce").mean()), 3),
        "avg_total_proj": round(float(pd.to_numeric(d["total_exp"],
                                                    errors="coerce").mean()), 2),
        "avg_total_actual": round(float((ha + aa).mean()), 2),
        # side accuracy vs baselines
        "winner_pct": round(100.0 * float(pd.to_numeric(
            d["winner_correct"], errors="coerce").mean()), 1),
        "home_baseline_pct": round(100.0 * float(home_won.mean()), 1),
        "ml_brier": _brier(p_home, home_won),
        "ml_brier_baseline": _brier(np.full(len(d), float(home_won.mean())), home_won),
        # totals
        "total_pct": (round(100.0 * float(pd.to_numeric(
            graded_totals["total_lean_correct"], errors="coerce").mean()), 1)
            if not graded_totals.empty else None),
        "total_n": int(len(graded_totals)),
        # the featured pick
        "gotd_n": int(len(gotd)),
        "gotd_winner_pct": (round(100.0 * float(pd.to_numeric(
            gotd["winner_correct"], errors="coerce").mean()), 1)
            if not gotd.empty else None),
    })
    # Bias: is the model systematically high or low on runs?
    out["total_bias"] = round(out["avg_total_proj"] - out["avg_total_actual"], 2)
    out["beats_home_baseline"] = bool(out["winner_pct"] > out["home_baseline_pct"])
    out["beats_brier_baseline"] = bool(
        out["ml_brier"] is not None and out["ml_brier_baseline"] is not None
        and out["ml_brier"] < out["ml_brier_baseline"])
    return out


def win_prob_calibration(log: pd.DataFrame | None = None, bins: int = 5) -> pd.DataFrame:
    """Predicted vs actual home-win rate by probability bucket.

    The single most useful diagnostic: a model can pick winners at a fine clip
    and still be badly miscalibrated, which is what turns a real edge into a
    losing ticket once you're paying a price for it.
    """
    log = load_game_log() if log is None else log
    if log is None or log.empty:
        return pd.DataFrame()
    d = log.dropna(subset=["home_runs_actual", "p_home"]).copy()
    if d.empty:
        return pd.DataFrame()
    p = pd.to_numeric(d["p_home"], errors="coerce")
    y = (pd.to_numeric(d["home_runs_actual"], errors="coerce")
         > pd.to_numeric(d["away_runs_actual"], errors="coerce")).astype(float)
    edges = np.linspace(max(0.0, float(p.min()) - 1e-9), min(1.0, float(p.max()) + 1e-9),
                        bins + 1)
    if len(np.unique(edges)) < 2:
        return pd.DataFrame()
    grp = pd.cut(p, bins=np.unique(edges), include_lowest=True)
    tbl = pd.DataFrame({"p": p, "y": y}).groupby(grp, observed=True).agg(
        games=("y", "size"), predicted=("p", "mean"), actual=("y", "mean"))
    tbl = tbl[tbl["games"] >= 3].reset_index(drop=True)
    if tbl.empty:
        return tbl
    tbl["predicted"] = (tbl["predicted"] * 100).round(1)
    tbl["actual"] = (tbl["actual"] * 100).round(1)
    tbl["gap"] = (tbl["actual"] - tbl["predicted"]).round(1)
    return tbl


def total_bias_by_park(log: pd.DataFrame | None = None, min_games: int = 4) -> pd.DataFrame:
    """Where the run projections run hot or cold, by home park.

    A park whose run factor is stale shows up here as a persistent one-sided
    miss — which is exactly the signal needed to retune it.
    """
    log = load_game_log() if log is None else log
    if log is None or log.empty or "home_team" not in log.columns:
        return pd.DataFrame()
    d = log.dropna(subset=["total_actual", "total_exp"]).copy()
    if d.empty:
        return pd.DataFrame()
    d["err"] = (pd.to_numeric(d["total_exp"], errors="coerce")
                - pd.to_numeric(d["total_actual"], errors="coerce"))
    t = d.groupby("home_team").agg(games=("err", "size"),
                                   bias=("err", "mean"),
                                   mae=("err", lambda s: s.abs().mean()))
    t = t[t["games"] >= min_games].round(2).reset_index()
    return t.sort_values("bias", ascending=False)
