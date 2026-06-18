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
from functools import lru_cache

import numpy as np
import pandas as pd

from . import demo
from .model import score_slate
from .parks import get_park, park_hr_multiplier

# Metrics that define the "HR hitter profile" used for similarity matching.
PROFILE_METRICS = ["barrel_pct", "hard_hit_pct", "avg_ev", "max_ev",
                   "launch_angle", "whiff_pct", "fb_pct", "park_factor"]

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
@lru_cache(maxsize=8)
def build_hr_history(start_iso: str, end_iso: str, prefer_live: bool = True):
    """Return (events_df, slate_df, source, notes).

    events_df : one row per HR with the hitter's metrics + game context.
    slate_df  : every scored hitter-day in the window (for calibration).
    """
    notes: list[str] = []
    if prefer_live:
        live = _live_hr_history(start_iso, end_iso)
        if live is not None:
            events, slate = live
            notes.append("Live HR history from Baseball Savant (Statcast).")
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


def _live_hr_history(start_iso: str, end_iso: str):
    try:
        from . import statcast as sc_mod
        if not sc_mod.is_available():
            return None
        import pybaseball as pyb

        sc = pyb.statcast(start_dt=start_iso, end_dt=end_iso, verbose=False)
        if sc is None or sc.empty or "events" not in sc.columns:
            return None
        hr = sc[sc["events"] == "home_run"].copy()
        if hr.empty:
            return None

        year = dt.date.fromisoformat(end_iso).year
        season = sc_mod.get_season_batter_table(year)
        season_by_id = season.set_index("mlbam_id") if not season.empty else None

        rows = []
        for _, r in hr.iterrows():
            home_abbr = _SAVANT_TEAM_FIX.get(r.get("home_team"), r.get("home_team"))
            stand = (r.get("stand") or "R")
            park = get_park(home_abbr)
            prof = {}
            if season_by_id is not None and r.get("batter") in season_by_id.index:
                srow = season_by_id.loc[r["batter"]]
                if isinstance(srow, pd.DataFrame):
                    srow = srow.iloc[0]
                prof = {m: srow.get(m) for m in
                        ["barrel_pct", "hard_hit_pct", "avg_ev", "max_ev",
                         "launch_angle", "whiff_pct", "chase_pct",
                         "zone_contact_pct", "fb_pct", "xwoba", "hr_per_pa"]}
            rows.append({
                "date": str(r.get("game_date")),
                "player": r.get("player_name"),
                "mlbam_id": r.get("batter"),
                "bats": stand,
                "home_team": home_abbr,
                "opponent": None,
                "pitcher_throws": r.get("p_throws", "R"),
                "hr_ev": r.get("launch_speed"),
                "hr_la": r.get("launch_angle"),
                "hr_distance": r.get("hit_distance_sc"),
                "park_factor": park_hr_multiplier(home_abbr, stand) * 100 if park else 100.0,
                **prof,
            })
        events_df = pd.DataFrame(rows)
        # No full scored slate available in the live HR-only pull; calibration
        # falls back to the simulated slate in that mode.
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
        ("barrel_pct", "Barrel%"), ("hard_hit_pct", "Hard-Hit%"),
        ("avg_ev", "Avg EV"), ("max_ev", "Max EV"),
        ("launch_angle", "Launch Angle"), ("whiff_pct", "Whiff%"),
        ("fb_pct", "Fly-Ball%"), ("hr_per_pa", "Season HR/PA"),
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


def hr_profile_centroid(events: pd.DataFrame) -> dict | None:
    """Mean vector + spread of the HR-hitter profile, for similarity scoring."""
    if events.empty:
        return None
    centroid, scales = {}, {}
    for m in PROFILE_METRICS:
        if m in events.columns:
            vals = pd.to_numeric(events[m], errors="coerce").dropna()
            if len(vals):
                centroid[m] = float(vals.mean())
                scales[m] = float(vals.std() or 1.0)
    return {"centroid": centroid, "scales": scales} if centroid else None


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
