"""Daily self-improvement: persistent, real-outcome probability calibration.

The loop (runs daily via GitHub Actions, which has open network):
  1. Rebuild YESTERDAY's slate pre-game (live lineups + real metrics).
  2. Pull the actual HR outcomes from MLB box scores.
  3. Append prediction-vs-outcome rows to `data/eval_log.csv` (de-duped) —
     a growing, genuinely out-of-sample track record.
  4. Refit a monotonic probability-calibration curve on the FULL log and
     commit it to `data/model_tuning.json`.

The model applies that curve to every game HR probability at scoring time, so
as the record grows the probabilities — and everything built on them (roles,
parlays, EV, edges) — get more accurate every day.

Guardrails (so one weird day can't warp the model):
  • identity until ≥ MIN_ROWS_TO_APPLY evaluated hitter-days exist,
  • the correction is damped by sample size (full trust only after thousands),
  • the adjusted probability is clamped to a sane band around the raw one,
  • only REAL (live-source) days are ever logged — simulated slates are skipped.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
EVAL_LOG_PATH = os.path.join(_DATA_DIR, "eval_log.csv")
TUNING_PATH = os.path.join(_DATA_DIR, "model_tuning.json")

MIN_ROWS_TO_APPLY = 300     # ~2 live slates before any adjustment kicks in
FULL_TRUST_ROWS = 3000      # damping reaches full weight around here
N_BINS = 8

EVAL_COLS = ["date", "player", "team", "lineup_spot", "hr_prob_game",
             "hr_score", "ulx_checks", "parlay_role", "top_pick", "hit_hr"]

# Parlay-leg feedback needs at least this many logged legs per role.
MIN_LEGS_PER_ROLE = 25
ROLE_TRUST_LEGS = 500


# --------------------------------------------------------------------------- #
# Track record (eval log)
# --------------------------------------------------------------------------- #
def load_eval_log() -> pd.DataFrame:
    if not os.path.exists(EVAL_LOG_PATH):
        return pd.DataFrame(columns=EVAL_COLS)
    try:
        return pd.read_csv(EVAL_LOG_PATH)
    except Exception:
        return pd.DataFrame(columns=EVAL_COLS)


def append_eval_rows(rows: pd.DataFrame) -> int:
    """De-duped (date, player) append. Returns the number of new rows."""
    if rows is None or rows.empty:
        return 0
    rows = rows[[c for c in EVAL_COLS if c in rows.columns]].copy()
    existing = load_eval_log()
    combined = pd.concat([existing, rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["date", "player"], keep="last")
    os.makedirs(_DATA_DIR, exist_ok=True)
    combined.to_csv(EVAL_LOG_PATH, index=False)
    return len(combined) - len(existing)


def brier_score(log: pd.DataFrame) -> float | None:
    if log is None or log.empty:
        return None
    p = pd.to_numeric(log["hr_prob_game"], errors="coerce")
    y = pd.to_numeric(log["hit_hr"], errors="coerce")
    m = p.notna() & y.notna()
    if not m.any():
        return None
    return round(float(((p[m] - y[m]) ** 2).mean()), 5)


# --------------------------------------------------------------------------- #
# Calibration fit (daily) and apply (at scoring time)
# --------------------------------------------------------------------------- #
def fit_calibration(log: pd.DataFrame) -> dict:
    """Fit a monotonic predicted→actual curve from the accumulated record."""
    out = {"n": 0, "bins": [], "brier": None, "updated": None}
    if log is None or log.empty:
        return out
    df = log.dropna(subset=["hr_prob_game", "hit_hr"]).copy()
    out["n"] = int(len(df))
    out["brier"] = brier_score(df)
    if len(df) < MIN_ROWS_TO_APPLY:
        return out
    try:
        df["bucket"] = pd.qcut(df["hr_prob_game"].rank(method="first"), N_BINS)
        g = df.groupby("bucket", observed=True)
        pred = g["hr_prob_game"].mean().to_numpy(dtype=float)
        actual = g["hit_hr"].mean().to_numpy(dtype=float)
        order = np.argsort(pred)
        pred, actual = pred[order], actual[order]
        # Isotonic-style: enforce a monotone non-decreasing curve so calibration
        # never flips the model's ordering of hitters.
        actual = np.maximum.accumulate(actual)
        out["bins"] = [{"pred": round(float(p), 4), "actual": round(float(a), 4)}
                       for p, a in zip(pred, actual)]
    except Exception:
        out["bins"] = []
    return out


def fit_role_factors(log: pd.DataFrame) -> dict:
    """Per-role parlay-leg reliability from the logged legs.

    factor = (actual leg hit rate) / (mean predicted prob) for legs the builder
    actually picked in that role — damped by sample size, clamped to a sane
    band. Used to calibrate future ticket win% and lean role selection toward
    what has really been cashing.
    """
    out = {"role_factors": {}, "role_n": {}}
    if log is None or log.empty or "parlay_role" not in log.columns:
        return out
    legs = log[log["parlay_role"].astype(str).isin(["Anchor", "Value", "Longshot"])]
    for role, grp in legs.groupby("parlay_role"):
        n = len(grp)
        out["role_n"][role] = int(n)
        if n < MIN_LEGS_PER_ROLE:
            continue
        pred = pd.to_numeric(grp["hr_prob_game"], errors="coerce").mean()
        actual = pd.to_numeric(grp["hit_hr"], errors="coerce").mean()
        if not pred or np.isnan(pred) or np.isnan(actual):
            continue
        raw = float(actual / pred) if pred > 0 else 1.0
        w = min(1.0, n / ROLE_TRUST_LEGS)
        factor = w * raw + (1.0 - w) * 1.0
        out["role_factors"][role] = round(float(np.clip(factor, 0.6, 1.4)), 3)
    return out


def role_prob_factor(role: str) -> float:
    """Reliability multiplier for a parlay leg's probability, learned from the
    real track record of legs the builder picked in that role. 1.0 = neutral."""
    t = _load_tuning()
    return float((t.get("role_factors") or {}).get(role, 1.0))


def save_tuning(tuning: dict, when: str | None = None) -> None:
    tuning = dict(tuning)
    if when:
        tuning["updated"] = when
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(TUNING_PATH, "w") as f:
        json.dump(tuning, f, indent=1)
    reload_tuning()


@lru_cache(maxsize=1)
def _load_tuning() -> dict:
    if not os.path.exists(TUNING_PATH):
        return {}
    try:
        with open(TUNING_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def reload_tuning() -> None:
    _load_tuning.cache_clear()


def calibrate_game_prob(p: float) -> float:
    """Map a raw game HR probability through the learned real-outcome curve.

    Identity until enough real days are logged; damped by sample size; clamped
    so the correction stays sane.
    """
    t = _load_tuning()
    bins = t.get("bins") or []
    n = int(t.get("n") or 0)
    if not bins or n < MIN_ROWS_TO_APPLY or p is None or not np.isfinite(p):
        return p
    preds = np.array([b["pred"] for b in bins], dtype=float)
    actuals = np.array([b["actual"] for b in bins], dtype=float)
    p_cal = float(np.interp(p, preds, actuals))
    w = min(1.0, n / FULL_TRUST_ROWS)          # trust grows with the record
    p_out = w * p_cal + (1.0 - w) * float(p)
    # Clamp: never more than ±40% relative shift, always a sane absolute band.
    p_out = float(np.clip(p_out, 0.6 * p, 1.4 * p))
    return float(np.clip(p_out, 0.002, 0.35))


# --------------------------------------------------------------------------- #
# Daily evaluation (real outcomes only)
# --------------------------------------------------------------------------- #
def evaluate_day(date_iso: str, prefer_live: bool = True):
    """Grade the model on a completed date. Returns (rows_df | None, note).

    Only returns rows when BOTH sides are real: the slate came from the live
    path and at least one box score returned actual HRs — otherwise (None,
    reason), so simulated data can never pollute the track record.
    """
    from .model import score_slate            # lazy: avoid circular import
    from .odds import attach_odds
    from .parlay import generate_parlay
    from .sources import fetch_game_box_hrs, fetch_schedule, get_slate
    from .statcast import normalize_name
    import datetime as dt

    game_date = dt.date.fromisoformat(date_iso)
    df, source, _notes = get_slate(game_date, prefer_live=prefer_live)
    if df is None or df.empty:
        return None, "no slate"
    if not str(source).startswith("LIVE"):
        return None, f"slate source is {source} — skipping (real days only)"
    scored = score_slate(df)

    # What would the builder have PICKED pre-game? Log the ULX 3-leg parlay
    # (by role) and the top-5 HR-Score picks so their real hit rates feed back
    # into future parlay selection.
    scored["parlay_role"] = ""
    scored["top_pick"] = 0
    try:
        with_odds = attach_odds(scored, date_iso, use_live=False)
        legs = generate_parlay(with_odds, n_legs=3, strategy="ulx")["legs"]
        role_by_player = dict(zip(legs["player"], legs["role"]))
        scored["parlay_role"] = scored["player"].map(
            lambda p: role_by_player.get(p, ""))
    except Exception:
        pass
    top5 = set(scored.sort_values("hr_score", ascending=False)["player"].head(5))
    scored["top_pick"] = scored["player"].isin(top5).astype(int)

    hr_names: set = set()
    games_with_data = 0
    for g in fetch_schedule(date_iso):
        hrs = fetch_game_box_hrs(g.get("game_pk"))
        if hrs:
            games_with_data += 1
            for h in hrs:
                if h.get("player"):
                    hr_names.add(normalize_name(h["player"]))
    if games_with_data == 0:
        return None, "no box-score outcomes available yet"

    rows = scored.copy()
    rows["hit_hr"] = rows["player"].map(
        lambda nm: int(normalize_name(nm) in hr_names))
    rows["date"] = date_iso
    keep = rows[[c for c in EVAL_COLS if c in rows.columns]]
    note = (f"evaluated {len(keep)} hitters across {games_with_data} games; "
            f"{int(keep['hit_hr'].sum())} homered")
    return keep, note
