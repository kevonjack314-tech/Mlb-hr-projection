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

# Full feature vector logged with every graded hitter-day: the model's raw
# inputs + context. This is what lets the system eventually LEARN its weights
# from real outcomes (fit_feature_model) instead of the hand-set
# HR_SCORE_WEIGHTS — old rows without these columns simply carry NaN.
FEATURE_COLS = [
    # batted-ball power
    "barrel_pct", "brl_pa", "hard_hit_pct", "avg_ev", "max_ev", "launch_angle",
    # batted-ball mix & plate discipline
    "fb_pct", "gb_pct", "ld_pct", "pull_pct", "hr_fb",
    "whiff_pct", "chase_pct", "zone_contact_pct", "k_pct",
    # expected stats & season rates
    "xiso", "xslg", "xwoba", "iso", "sweet_spot_pct",
    "hr_per_pa", "season_hr", "pa",
    # recent form
    "hr_rate_7", "hr_rate_15", "hr_rate_30", "recent_form_score",
    # matchup & environment
    "platoon_adv", "pitch_matchup_score", "matchup_score", "env_score",
    "pitcher_hr9", "park_factor", "wind_mult", "temp_f", "expected_pa",
    # real platoon splits + bullpen exposure (may be sparse early on)
    "woba_vs_l", "woba_vs_r", "woba_vs_hand", "bullpen_hr9",
    # fence geometry x pull side
    "park_fit_mult", "park_porch_ft",
    # opposing starter's meatball (middle-middle) supply
    "sp_meatball_pct",
    # opposing starter's fastball velo trend (last start vs baseline)
    "sp_velo_delta", "sp_velo_last",
    # opposing starter's 3rd-time-through-the-order wOBA penalty
    "sp_tto_penalty",
    # opposing starter's fastball rate in hitter's counts (predictability)
    "sp_hitter_count_fb",
    # start-time park effect (day/night)
    "daynight_mult",
    # Statcast bat-tracking: swing speed & squared-up rate
    "bat_speed", "squared_up_pct", "fast_swing_pct",
]

EVAL_COLS = ["date", "player", "team", "lineup_spot", "hr_prob_game",
             "hr_score", "ulx_checks", "parlay_role", "top_pick", "hr_of_day",
             "hit_hr"] + FEATURE_COLS

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


# --------------------------------------------------------------------------- #
# Pick record: how the featured picks have ACTUALLY done
# --------------------------------------------------------------------------- #
def _streak(hits: list) -> str:
    """'W3' / 'L2' style current streak from a chronological 0/1 list."""
    if not hits:
        return "—"
    last, n = hits[-1], 0
    for h in reversed(hits):
        if h != last:
            break
        n += 1
    return f"{'W' if last else 'L'}{n}"


def pick_record(log: pd.DataFrame) -> dict:
    """Real win-loss record of the featured picks, from the graded eval log.

    Returns {"hotd": {...}, "top5": {...}, "roles": {...}} — each with wins,
    losses, hit rate, the model's expected rate (mean predicted prob), and a
    chronological results table for the UI.
    """
    out = {"hotd": None, "top5": None, "roles": {}}
    if log is None or log.empty or "hit_hr" not in log.columns:
        return out
    df = log.copy()
    df["hit_hr"] = pd.to_numeric(df["hit_hr"], errors="coerce").fillna(0).astype(int)
    df["hr_prob_game"] = pd.to_numeric(df["hr_prob_game"], errors="coerce")

    # --- 🔒 HR of the Day: one pick per date -> W/L record ---
    if "hr_of_day" in df.columns:
        h = df[pd.to_numeric(df["hr_of_day"], errors="coerce").fillna(0) == 1]
        h = h.sort_values("date").drop_duplicates("date", keep="last")
        if not h.empty:
            hits = h["hit_hr"].tolist()
            out["hotd"] = {
                "days": len(h),
                "wins": int(sum(hits)),
                "losses": int(len(hits) - sum(hits)),
                "hit_rate": round(100 * sum(hits) / len(hits), 1),
                "expected_rate": round(100 * float(h["hr_prob_game"].mean()), 1),
                "streak": _streak(hits),
                "rows": h[["date", "player", "team", "hr_prob_game", "hit_hr"]],
            }

    # --- ⭐ Top-5 picks: 5 picks per date ---
    if "top_pick" in df.columns:
        t = df[pd.to_numeric(df["top_pick"], errors="coerce").fillna(0) == 1]
        if not t.empty:
            by_day = (t.groupby("date")
                       .agg(picks=("hit_hr", "size"), hits=("hit_hr", "sum"))
                       .reset_index().sort_values("date"))
            out["top5"] = {
                "days": len(by_day),
                "picks": int(by_day["picks"].sum()),
                "wins": int(by_day["hits"].sum()),
                "hit_rate": round(100 * by_day["hits"].sum() / by_day["picks"].sum(), 1),
                "expected_rate": round(100 * float(t["hr_prob_game"].mean()), 1),
                "days_with_hit": int((by_day["hits"] > 0).sum()),
                "days_with_hit_pct": round(100 * (by_day["hits"] > 0).mean(), 1),
                "avg_hits_per_day": round(float(by_day["hits"].mean()), 2),
                "by_day": by_day,
            }

    # --- 🎰 Parlay legs by role ---
    if "parlay_role" in df.columns:
        legs = df[df["parlay_role"].astype(str).isin(["Anchor", "Value", "Longshot"])]
        for role, grp in legs.groupby("parlay_role"):
            w = int(grp["hit_hr"].sum())
            out["roles"][role] = {
                "legs": len(grp), "wins": w, "losses": len(grp) - w,
                "hit_rate": round(100 * w / len(grp), 1),
                "expected_rate": round(100 * float(grp["hr_prob_game"].mean()), 1),
            }
    return out


# --------------------------------------------------------------------------- #
# Learned feature weights (logistic model trained on the real record)
# --------------------------------------------------------------------------- #
FEATURE_MODEL_MIN_ROWS = 2000    # don't even fit below this
FEATURE_MODEL_TRUST_ROWS = 10000  # blend weight reaches its 0.5 cap here
_FM_MIN_COVERAGE = 0.7           # a feature must be present on ≥70% of rows


def _feature_matrix(df: pd.DataFrame, feats: list[str], medians: dict) -> np.ndarray:
    cols = []
    for f in feats:
        v = pd.to_numeric(df.get(f), errors="coerce")
        cols.append(v.fillna(medians.get(f, 0.0)).to_numpy(dtype=float))
    return np.column_stack(cols)


def fit_feature_model(log: pd.DataFrame) -> dict:
    """Learn HR-probability weights from the real graded record.

    A ridge-regularized logistic regression over FEATURE_COLS, validated on a
    time-ordered holdout (the most recent ~20% of days). It only goes ACTIVE
    when it beats the hand-weighted model's Brier score on those unseen days —
    until then it just trains and reports.
    """
    out = {"feature_model": {"n": 0, "active": False, "note": "warming up"}}
    if log is None or log.empty or "hit_hr" not in log.columns:
        return out
    df = log.dropna(subset=["hr_prob_game", "hit_hr"]).copy()
    feats = [f for f in FEATURE_COLS if f in df.columns
             and pd.to_numeric(df[f], errors="coerce").notna().mean() >= _FM_MIN_COVERAGE]
    rows_with_feats = df[feats].apply(pd.to_numeric, errors="coerce").notna().any(axis=1) \
        if feats else pd.Series(False, index=df.index)
    df = df[rows_with_feats]
    n = len(df)
    out["feature_model"]["n"] = int(n)
    if n < FEATURE_MODEL_MIN_ROWS or len(feats) < 8:
        out["feature_model"]["note"] = (
            f"warming up ({n} rows with features; needs {FEATURE_MODEL_MIN_ROWS})")
        return out

    df = df.sort_values("date")
    dates = sorted(df["date"].unique())
    split = dates[max(1, int(len(dates) * 0.8)) - 1]
    train, val = df[df["date"] <= split], df[df["date"] > split]
    if val.empty or train.empty:
        out["feature_model"]["note"] = "not enough distinct days for a holdout"
        return out

    medians = {f: float(pd.to_numeric(train[f], errors="coerce").median())
               for f in feats}
    Xt = _feature_matrix(train, feats, medians)
    mean, std = Xt.mean(axis=0), Xt.std(axis=0)
    std[std == 0] = 1.0
    Xt = (Xt - mean) / std
    yt = pd.to_numeric(train["hit_hr"], errors="coerce").to_numpy(dtype=float)

    # Plain-numpy ridge logistic regression (gradient descent).
    w = np.zeros(Xt.shape[1])
    b = float(np.log(max(yt.mean(), 1e-3) / max(1 - yt.mean(), 1e-3)))
    lam, lr = 1e-2, 0.5
    for _ in range(400):
        p = 1.0 / (1.0 + np.exp(-(Xt @ w + b)))
        g_w = Xt.T @ (p - yt) / len(yt) + lam * w
        g_b = float((p - yt).mean())
        w -= lr * g_w
        b -= lr * g_b

    Xv = (_feature_matrix(val, feats, medians) - mean) / std
    yv = pd.to_numeric(val["hit_hr"], errors="coerce").to_numpy(dtype=float)
    pv = 1.0 / (1.0 + np.exp(-(Xv @ w + b)))
    brier_model = float(((pv - yv) ** 2).mean())
    base = pd.to_numeric(val["hr_prob_game"], errors="coerce").to_numpy(dtype=float)
    brier_base = float(((base - yv) ** 2).mean())
    active = bool(brier_model < brier_base)

    out["feature_model"] = {
        "n": int(n), "features": feats,
        "medians": {k: round(v, 5) for k, v in medians.items()},
        "mean": [round(float(x), 5) for x in mean],
        "std": [round(float(x), 5) for x in std],
        "coef": [round(float(x), 5) for x in w],
        "intercept": round(b, 5),
        "val_days": int(val["date"].nunique()),
        "val_brier_model": round(brier_model, 5),
        "val_brier_baseline": round(brier_base, 5),
        "active": active,
        "note": ("ACTIVE — beats the hand-weighted model on held-out days"
                 if active else
                 "trained, not applied — hand-weighted model still wins on holdout"),
    }
    return out


def apply_feature_model(df: pd.DataFrame) -> pd.DataFrame:
    """Blend the learned-weights probability into hr_prob_game.

    No-op unless the learned model went ACTIVE (proved itself on held-out
    days). Even then the blend weight is damped by record size and capped at
    50%, and the result is clamped to the same sane band as calibration.
    """
    fm = _load_tuning().get("feature_model") or {}
    if not fm.get("active") or df is None or df.empty or "hr_prob_game" not in df.columns:
        return df
    try:
        feats = fm["features"]
        X = _feature_matrix(df, feats, fm.get("medians", {}))
        X = (X - np.array(fm["mean"], dtype=float)) / np.array(fm["std"], dtype=float)
        p_learn = 1.0 / (1.0 + np.exp(-(X @ np.array(fm["coef"], dtype=float)
                                        + float(fm["intercept"]))))
        w = 0.5 * min(1.0, float(fm.get("n", 0)) / FEATURE_MODEL_TRUST_ROWS)
        p_raw = pd.to_numeric(df["hr_prob_game"], errors="coerce").to_numpy(dtype=float)
        p_out = (1.0 - w) * p_raw + w * p_learn
        p_out = np.clip(p_out, 0.6 * p_raw, 1.4 * p_raw)
        df = df.copy()
        df["hr_prob_game"] = np.round(np.clip(p_out, 0.002, 0.35), 4)
    except Exception:
        return df
    return df


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
    # The featured 🔒 HR of the Day pick (same confidence formula as the app).
    scored["hr_of_day"] = 0
    try:
        from .model import hr_of_the_day
        hotd = hr_of_the_day(scored)
        if hotd is not None:
            scored.loc[scored["player"] == hotd["player"], "hr_of_day"] = 1
    except Exception:
        pass

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
    if "platoon_adv" in rows.columns:   # keep the CSV numeric (0/1, not True/False)
        rows["platoon_adv"] = rows["platoon_adv"].fillna(False).astype(int)
    keep = rows[[c for c in EVAL_COLS if c in rows.columns]]
    note = (f"evaluated {len(keep)} hitters across {games_with_data} games; "
            f"{int(keep['hit_hr'].sum())} homered")
    return keep, note
