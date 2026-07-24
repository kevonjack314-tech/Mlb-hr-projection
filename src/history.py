"""Trailing-month HR history, profile analysis, calibration, and top-5 lists.

What this does
--------------
1. Gather every home run hit in a date window (default: last ~30 days) together
   with the hitter's batted-ball profile and the game context (park, weather,
   matchup, platoon).
2. Summarize the *shared* characteristics of HR hitters — the empirical "HR
   profile" (avg barrel%, EV, max EV, launch angle, park factor, platoon share,
   handedness split, hottest parks) and how much HR hitters out-index the slate
   baseline on each metric.
3. Use that profile to (a) **calibrate / validate** the model — actual HR rate by
   predicted-probability decile — and (b) score every hitter on today's slate by
   **how closely they resemble recent HR hitters** (a profile-match %), which is
   folded into a calibrated ranking.
4. Produce **top-5 lists in each category** (Overall, Longshots, Consistent,
   Sneaky) for the projection slate.

Data path
---------
- LIVE: `pybaseball.statcast(start, end)` -> rows where `events == "home_run"`,
  joined to season batted-ball metrics. Requires baseballsavant.mlb.com on the
  network egress allowlist.
- OFFLINE: a deterministic simulation — for each date we build & score the demo
  slate and sample HR outcomes from each hitter's modeled game HR probability.
  HR hitters then naturally skew toward high barrel%/EV/good parks, so the
  profile analysis is a faithful demonstration of the live behavior.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from . import demo
from .model import score_slate
from .parks import get_park, park_hr_multiplier

# Metrics that define the "HR hitter profile" used for similarity matching.
# GB% and LD% are intentionally left out of the centroid (collinear with FB%)
# but still appear in the shared-profile lift table below.
PROFILE_METRICS = ["barrel_pct", "brl_pa", "hard_hit_pct", "avg_ev", "max_ev",
                   "launch_angle", "whiff_pct", "fb_pct", "pull_pct",
                   "hr_fb", "xiso", "park_factor"]

# Savant team codes -> our park abbreviations (handles the handful that differ).
_SAVANT_TEAM_FIX = {
    "CHW": "CWS", "WSN": "WSH", "SDP": "SD", "TBR": "TB", "KCR": "KC",
    "SFG": "SF", "AZ": "ARI", "ARZ": "ARI", "ATH": "ATH",
}


def _seed(*parts) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16)


# --------------------------------------------------------------------------- #
# 1. Gather HR history
# --------------------------------------------------------------------------- #
def build_hr_history(start_iso: str, end_iso: str, prefer_live: bool = True):
    """Return (events_df, slate_df, source, notes).

    events_df : one row per HR with the hitter's metrics + game context.
    slate_df  : every scored hitter-day in the window (for calibration).

    Deliberately NOT lru_cached: the app layer caches this with a TTL, and a
    process-lifetime cache here would pin a bad pull (blank metrics) until
    restart. The heavy sub-pulls are cached failure-safely in statcast.py.
    """
    notes: list[str] = []
    if prefer_live:
        live = _live_hr_history(start_iso, end_iso)
        if live is not None:
            events, slate = live
            notes.append("Live HR history from Baseball Savant (Statcast).")
            # Enrichment coverage: how many HR hitters carry real season
            # metrics. If it's low, say WHY (feed diagnostics) in the notes.
            cov = (float(events["barrel_pct"].notna().mean())
                   if "barrel_pct" in events.columns and len(events) else 0.0)
            notes.append(f"Season metrics attached for {cov*100:.0f}% of HR hitters.")
            if cov < 0.5:
                try:
                    from . import statcast as _sc
                    for src, msg in _sc.get_diagnostics().items():
                        notes.append(f"⚠️ feed issue — {src}: {msg}")
                except Exception:
                    pass
            return events, slate, "LIVE (Statcast)", notes
        notes.append("Statcast host unavailable — using simulated HR history.")

    events, slate = _simulated_hr_history(start_iso, end_iso)
    notes.append("Simulated HR history (deterministic) from modeled slates.")
    return events, slate, "SIMULATED", notes


def _date_range(start_iso: str, end_iso: str):
    start = dt.date.fromisoformat(start_iso)
    end = dt.date.fromisoformat(end_iso)
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def _simulated_hr_history(start_iso: str, end_iso: str):
    slates = []
    events = []
    for d in _date_range(start_iso, end_iso):
        day = score_slate(demo.build_demo_slate(d))
        day = day.copy()
        day["date"] = d.isoformat()
        # Deterministically sample HR outcomes from each hitter's game HR prob.
        rng = np.random.default_rng(_seed("hrsim", d.isoformat()))
        draws = rng.random(len(day))
        day["hit_hr"] = draws < day["hr_prob_game"].values
        # A hitter can occasionally go deep twice; sample a second independent HR.
        draws2 = rng.random(len(day))
        day["hr_count"] = day["hit_hr"].astype(int) + (
            (draws2 < day["hr_prob_game"].values * 0.18) & day["hit_hr"]
        ).astype(int)
        slates.append(day)
        events.append(day[day["hit_hr"]])
    slate_df = pd.concat(slates, ignore_index=True)
    events_df = pd.concat(events, ignore_index=True)
    return events_df, slate_df


# Cap the LIVE real-HR window (box-score calls) for responsiveness; the stat
# sheet is about *recent* HRs and this keeps first load fast.
_LIVE_HR_MAX_DAYS = 14

_PROFILE_METRIC_COLS = [
    "barrel_pct", "brl_pa", "hard_hit_pct", "avg_ev", "max_ev", "launch_angle",
    "whiff_pct", "chase_pct", "zone_contact_pct", "fb_pct", "gb_pct", "ld_pct",
    "pull_pct", "hr_fb", "xiso", "xslg", "sprint_speed", "xwoba", "hr_per_pa",
    "bat_speed", "squared_up_pct", "fast_swing_pct",
    "barrel_pct_14", "xwoba_14", "barrel_trend", "xwoba_trend",
]


def _eval_feature_index() -> dict:
    """{(date, normalized player): row dict} from the graded eval record.

    The eval log stores the model's full pre-game feature vector + rating for
    every hitter-day it graded — the exact 'what the model gave that player
    before the game' the Previous-HRs card should show."""
    try:
        from .statcast import normalize_name
        from .tuning import load_eval_log
        ev = load_eval_log()
        if ev.empty or "barrel_pct" not in ev.columns:
            return {}
        ev = ev[ev["barrel_pct"].notna()]
        keys = list(zip(ev["date"].astype(str),
                        ev["player"].map(normalize_name)))
        return dict(zip(keys, ev.to_dict("records")))
    except Exception:
        return {}


# Skip identity/outcome columns AND anything score_slate re-derives itself —
# re-adding those as inputs would create duplicate columns after scoring.
_EVAL_FEAT_SKIP = {"date", "player", "team", "lineup_spot", "hit_hr",
                   "parlay_role", "top_pick", "hr_score", "hr_prob_game",
                   "ulx_checks", "recent_form_score", "matchup_score",
                   "env_score", "pitch_matchup_score", "platoon_adv",
                   "expected_pa", "woba_vs_hand", "park_factor", "wind_mult",
                   "park_fit_mult", "park_porch_ft", "daynight_mult"}


def eval_features_for(eval_idx: dict, date, player):
    """(features, logged_scores) for one HR event from the eval record, or None."""
    try:
        from .statcast import normalize_name
        rec = eval_idx.get((str(date), normalize_name(player or "")))
        if not rec:
            return None
        feats = {k: v for k, v in rec.items()
                 if k not in _EVAL_FEAT_SKIP and v is not None and v == v}
        scored = {k: float(rec[k]) for k in ("hr_score", "hr_prob_game")
                  if rec.get(k) is not None and rec.get(k) == rec.get(k)}
        return (feats, scored) if feats else None
    except Exception:
        return None


def _live_hr_history(start_iso: str, end_iso: str):
    """Real HR hitters straight from MLB StatsAPI box scores (batter, HR count,
    lineup spot, opposing starter), enriched with season Statcast metrics and
    scored. Light and reliable — no multi-million-row Statcast pull."""
    try:
        from . import sources as src_mod
        from . import statcast as sc_mod

        end = dt.date.fromisoformat(end_iso)
        start = dt.date.fromisoformat(start_iso)
        if (end - start).days > _LIVE_HR_MAX_DAYS:
            start = end - dt.timedelta(days=_LIVE_HR_MAX_DAYS)

        rows = []
        d = end
        while d >= start:
            for g in src_mod.fetch_schedule(d.isoformat()):
                gpk = g.get("game_pk")
                # Prefer play-by-play (the actual pitcher per HR); fall back to the
                # box score (opposing starter) if the feed isn't available.
                hrs = src_mod.fetch_game_hr_details(gpk) or src_mod.fetch_game_box_hrs(gpk)
                for hr in hrs:
                    row = dict(hr)
                    row["date"] = d.isoformat()
                    rows.append(row)
            d -= dt.timedelta(days=1)
        if not rows:
            return None

        events_df = pd.DataFrame(rows)
        # Combine multiple HRs by the same batter off the same pitcher in a game.
        gcols = [c for c in ["date", "player", "mlbam_id", "team", "opponent",
                             "home_team", "is_home", "lineup_spot", "pitcher_name"]
                 if c in events_df.columns]
        if "hr_count" in events_df.columns and gcols:
            events_df = (events_df.groupby(gcols, dropna=False, as_index=False)["hr_count"]
                         .sum())
        # Enrich each HR hitter through the SAME per-player lookups the live
        # slate uses (matched by MLBAM id, then normalized name) — the code
        # path proven to attach real metrics on Streamlit Cloud. This is what
        # powers the pre-game metrics + "why they hit" insight.
        # PRIMARY source: the graded eval record. It stores every metric the
        # model actually gave the hitter BEFORE that game (logged pre-game by
        # the daily job / backfill), so it's point-in-time correct AND immune
        # to a live-feed outage blanking the card. Live lookups fill any gaps.
        eval_idx = _eval_feature_index()
        prof_rows, logged = [], []
        for _, r in events_df.iterrows():
            prof = {}
            got = eval_features_for(eval_idx, r.get("date"), r.get("player"))
            if got:
                feats, scored_vals = got
                prof.update(feats)
                logged.append(scored_vals)   # the model's REAL pre-game numbers
            else:
                logged.append(None)
            try:
                if not prof:
                    season = sc_mod.lookup_season(end.year, r.get("player"),
                                                  r.get("mlbam_id"))
                    if season:
                        prof.update({k: v for k, v in season.items()
                                     if k in _PROFILE_METRIC_COLS or k in
                                     ("pa", "season_hr", "power_tier", "k_pct")})
                if "hr_rate_7" not in prof:
                    recent = sc_mod.lookup_recent_form(end_iso, r.get("mlbam_id"))
                    if recent:
                        prof.update(recent)
                if "pitcher_hr9" not in prof:
                    peri = sc_mod.lookup_pitching(end.year, r.get("pitcher_name"))
                    if peri:
                        prof.update(peri)   # pitcher HR/9 etc. for the insight
            except Exception:
                pass
            prof_rows.append(prof)
        prof_df = pd.DataFrame(prof_rows, index=events_df.index)
        for c in prof_df.columns:
            events_df[c] = prof_df[c]

        if "bats" not in events_df.columns:
            events_df["bats"] = "R"
        if "pitcher_throws" not in events_df.columns:
            events_df["pitcher_throws"] = "R"
        # Score so the stat sheet shows metrics + "what we'd have rated them".
        try:
            events_df = score_slate(events_df)
        except Exception:
            pass
        # Where the eval record logged the ACTUAL pre-game rating, show that —
        # not a re-score with today's data.
        for col in ("hr_score", "hr_prob_game"):
            vals = [(lg or {}).get(col) for lg in logged]
            if any(v is not None for v in vals):
                cur = events_df.get(col, pd.Series(index=events_df.index, dtype=float))
                events_df[col] = [v if v is not None else c
                                  for v, c in zip(vals, cur)]
        # Calibration/report card still use the simulated full-slate-with-outcomes.
        _, slate_df = _simulated_hr_history(start_iso, end_iso)
        return events_df, slate_df
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 2. Profile summary  (the "similarities" across HR hitters)
# --------------------------------------------------------------------------- #
def summarize_hr_profile(events: pd.DataFrame, slate: pd.DataFrame) -> dict:
    """Compute the shared metric profile of HR hitters vs the slate baseline."""
    if events.empty:
        return {}
    n_days = events["date"].nunique()
    out = {
        "total_hr": int(events.get("hr_count", pd.Series([1] * len(events))).sum()),
        "hr_events": int(len(events)),
        "unique_hitters": int(events["player"].nunique()),
        "days": int(n_days),
        "hr_per_day": round(len(events) / max(1, n_days), 1),
    }

    metric_rows = []
    for m, label in [
        ("barrel_pct", "Barrel%"), ("brl_pa", "Barrel/PA%"),
        ("hard_hit_pct", "Hard-Hit%"),
        ("avg_ev", "Avg EV"), ("max_ev", "Max EV"),
        ("launch_angle", "Launch Angle"), ("whiff_pct", "Whiff%"),
        ("fb_pct", "Fly-Ball%"), ("gb_pct", "Ground-Ball%"),
        ("ld_pct", "Line-Drive%"), ("pull_pct", "Pull%"), ("hr_fb", "HR/FB"),
        ("xiso", "xISO"), ("hr_per_pa", "Season HR/PA"),
        ("park_factor", "Park Factor"),
    ]:
        if m not in events.columns:
            continue
        hr_mean = pd.to_numeric(events[m], errors="coerce").mean()
        base_mean = (pd.to_numeric(slate[m], errors="coerce").mean()
                     if (slate is not None and m in slate.columns) else np.nan)
        lift = ((hr_mean - base_mean) / base_mean * 100.0
                if base_mean and not np.isnan(base_mean) and base_mean != 0 else np.nan)
        metric_rows.append({
            "Metric": label,
            "HR hitters (avg)": round(float(hr_mean), 2) if not np.isnan(hr_mean) else None,
            "All hitters (avg)": round(float(base_mean), 2) if not np.isnan(base_mean) else None,
            "Lift vs baseline": (f"+{lift:.0f}%" if (lift is not None and not np.isnan(lift) and lift >= 0)
                                  else (f"{lift:.0f}%" if lift is not None and not np.isnan(lift) else None)),
        })
    out["metric_table"] = pd.DataFrame(metric_rows)

    # Platoon / handedness / park context.
    if "platoon_adv" in events.columns:
        out["platoon_share"] = round(100.0 * events["platoon_adv"].mean(), 1)
    if "bats" in events.columns:
        out["handedness"] = events["bats"].value_counts(normalize=True).mul(100).round(0).to_dict()
    if "park_factor" in events.columns:
        out["hr_friendly_share"] = round(100.0 * (pd.to_numeric(events["park_factor"],
                                          errors="coerce") >= 105).mean(), 1)
    if "home_team" in events.columns:
        out["top_parks"] = events["home_team"].value_counts().head(5).to_dict()
    if "hr_score" in events.columns and slate is not None and "hr_score" in slate.columns:
        out["mean_hr_score_hr_hitters"] = round(float(events["hr_score"].mean()), 1)
        out["mean_hr_score_all"] = round(float(slate["hr_score"].mean()), 1)
    return out


def _recency_weights(events: pd.DataFrame, end_date_iso: str | None,
                     half_life_days: float) -> np.ndarray | None:
    """Exponential-decay weights by how many days before `end_date` each HR fell.

    A half-life of `h` days means an HR from `h` days ago counts half as much as
    one today, so the profile tracks *what is working now*, not a month ago.
    """
    if end_date_iso is None or "date" not in events.columns or half_life_days <= 0:
        return None
    try:
        end = dt.date.fromisoformat(end_date_iso)
        dates = pd.to_datetime(events["date"], errors="coerce").dt.date
        days_ago = dates.map(lambda d: (end - d).days if pd.notna(d) else np.nan)
        w = np.power(0.5, days_ago.astype(float) / float(half_life_days))
        w = w.fillna(w[w.notna()].mean() if w.notna().any() else 1.0).to_numpy()
        return w
    except Exception:
        return None


def hr_profile_centroid(events: pd.DataFrame, end_date_iso: str | None = None,
                        half_life_days: float = 10.0) -> dict | None:
    """Recency-weighted mean + spread of the HR-hitter profile, for similarity.

    Recent HR hitters define "what's going deep now": weights decay with a
    configurable half-life (default 10 days). Pass half_life_days<=0 / no date to
    fall back to an unweighted centroid.
    """
    if events.empty:
        return None
    weights = _recency_weights(events, end_date_iso, half_life_days)
    centroid, scales = {}, {}
    for m in PROFILE_METRICS:
        if m not in events.columns:
            continue
        vals = pd.to_numeric(events[m], errors="coerce")
        mask = vals.notna()
        v = vals[mask].to_numpy()
        if not len(v):
            continue
        if weights is not None:
            w = weights[mask.to_numpy()]
            wmean = float(np.average(v, weights=w))
            wvar = float(np.average((v - wmean) ** 2, weights=w))
            centroid[m] = wmean
            scales[m] = float(np.sqrt(wvar) or 1.0)
        else:
            centroid[m] = float(v.mean())
            scales[m] = float(v.std() or 1.0)
    if not centroid:
        return None
    return {"centroid": centroid, "scales": scales,
            "half_life_days": half_life_days if weights is not None else None}


def recent_trend(events: pd.DataFrame, end_date_iso: str, recent_days: int = 7) -> pd.DataFrame:
    """What's trending among HR hitters: last `recent_days` vs the full window.

    Positive 'Trend' means HR hitters' average on that metric is higher in the
    most recent stretch than across the whole window — the profile is shifting.
    """
    if events.empty or "date" not in events.columns:
        return pd.DataFrame()
    end = dt.date.fromisoformat(end_date_iso)
    cutoff = end - dt.timedelta(days=recent_days)
    dates = pd.to_datetime(events["date"], errors="coerce").dt.date
    recent = events[dates > cutoff]
    if recent.empty:
        return pd.DataFrame()
    rows = []
    for m, label in [("barrel_pct", "Barrel%"), ("max_ev", "Max EV"),
                     ("fb_pct", "Fly-Ball%"), ("pull_pct", "Pull%"),
                     ("hr_fb", "HR/FB"), ("xiso", "xISO"), ("park_factor", "Park Factor")]:
        if m not in events.columns:
            continue
        full = pd.to_numeric(events[m], errors="coerce").mean()
        rec = pd.to_numeric(recent[m], errors="coerce").mean()
        if np.isnan(full) or np.isnan(rec) or full == 0:
            continue
        delta_pct = (rec - full) / abs(full) * 100.0
        rows.append({
            "Metric": label,
            f"Last {recent_days}d": round(float(rec), 3),
            "Full window": round(float(full), 3),
            "Trend": (f"+{delta_pct:.0f}%" if delta_pct >= 0 else f"{delta_pct:.0f}%"),
            "_sort": abs(delta_pct),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    return df


def add_profile_similarity(slate: pd.DataFrame, centroid: dict | None) -> pd.DataFrame:
    """Add `profile_match` (0-100): how closely each hitter resembles recent HR
    hitters, via a Gaussian kernel on the standardized metric distance."""
    slate = slate.copy()
    if not centroid:
        slate["profile_match"] = np.nan
        return slate
    c, s = centroid["centroid"], centroid["scales"]
    metrics = [m for m in PROFILE_METRICS if m in c and m in slate.columns]

    def match(row):
        d2 = 0.0
        for m in metrics:
            z = (float(row[m]) - c[m]) / (s.get(m, 1.0) or 1.0)
            d2 += z * z
        rms = (d2 / max(1, len(metrics))) ** 0.5
        return round(100.0 * float(np.exp(-0.5 * rms * rms)), 1)

    slate["profile_match"] = slate.apply(match, axis=1)
    # Calibrated score: mostly the model HR Score, nudged by recent-HR resemblance.
    slate["calibrated_score"] = (0.85 * slate["hr_score"]
                                 + 0.15 * slate["profile_match"]).round(1)
    return slate


# --------------------------------------------------------------------------- #
# 3. Calibration / validation
# --------------------------------------------------------------------------- #
def calibration_table(slate: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Actual HR rate vs predicted probability, by decile of predicted prob."""
    if slate is None or slate.empty or "hit_hr" not in slate.columns:
        return pd.DataFrame()
    df = slate.copy()
    df["bucket"] = pd.qcut(df["hr_prob_game"].rank(method="first"), bins,
                           labels=[f"D{i+1}" for i in range(bins)])
    g = df.groupby("bucket", observed=True)
    table = pd.DataFrame({
        "Predicted HR%": (g["hr_prob_game"].mean() * 100).round(1),
        "Actual HR%": (g["hit_hr"].mean() * 100).round(1),
        "Players": g.size(),
        "HRs": g["hit_hr"].sum().astype(int),
    }).reset_index(names="Decile")
    return table


# --------------------------------------------------------------------------- #
# 4. Top-5 lists per category for the projection slate
# --------------------------------------------------------------------------- #
CATEGORY_SORT = {
    "Overall (HR Score)": "hr_score",
    "Best Longshots": "longshot_score",
    "Consistent HR Hitters": "consistency_score",
    "Sneaky HR Chances": "sneaky_score",
}


def top5_by_category(slate: pd.DataFrame, n: int = 5) -> dict[str, pd.DataFrame]:
    """Top-N per category, including projected HR prob and profile match."""
    cols = ["player", "team", "opponent", "pitcher_name", "hr_prob_game",
            "hr_score", "longshot_score", "consistency_score", "sneaky_score",
            "barrel_pct", "max_ev", "park_factor"]
    if "profile_match" in slate.columns:
        cols.insert(5, "profile_match")
    if "calibrated_score" in slate.columns:
        cols.insert(6, "calibrated_score")
    out = {}
    for label, sort_col in CATEGORY_SORT.items():
        keep = [c for c in cols if c in slate.columns]
        out[label] = slate.sort_values(sort_col, ascending=False).head(n)[keep].reset_index(drop=True)
    return out
