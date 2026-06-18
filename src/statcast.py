"""Real Statcast / FanGraphs season + recent-form data.

This module pulls *live* batted-ball quality and HR rates and exposes them keyed
by MLBAM player id (the same id MLB StatsAPI uses for rosters/lineups) and by a
normalized player name, so `sources.py` can attach real numbers to the real slate.

Two cached tables are built per season/date:

  • Season batter table — Baseball Savant exit-velo & barrels leaderboard
    (barrel%, hard-hit%, avg/max EV, launch angle) merged with FanGraphs season
    stats (PA, HR, K%, xwOBA) → season HR/PA.
  • Recent-form table — one Statcast date-range pull aggregated by batter id into
    7 / 15 / 30-day HR rates.

Everything is best-effort: any failure returns an empty table and the caller
falls back to a deterministic modeled profile, so the app never breaks. Pulls are
heavy, so results are cached (in-process + on disk via pybaseball's own cache).

Required network egress hosts (add these to your environment's allowlist):
    baseballsavant.mlb.com, www.fangraphs.com, statsapi.mlb.com, api.open-meteo.com
"""

from __future__ import annotations

import datetime as dt
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

# pybaseball is optional; if it's missing or the network is blocked we degrade
# gracefully to the synthetic profiles in demo.py.
try:  # pragma: no cover - import guarded
    import pybaseball as pyb

    pyb.cache.enable()
    _HAS_PYB = True
except Exception:  # pragma: no cover
    _HAS_PYB = False


def normalize_name(name: str) -> str:
    """Accent/punctuation-insensitive 'first last' key for cross-source joins."""
    if not name:
        return ""
    # Savant exposes "Last, First"; normalize to "first last".
    if "," in name:
        last, first = [p.strip() for p in name.split(",", 1)]
        name = f"{first} {last}"
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = "".join(c for c in ascii_name if c.isalnum() or c.isspace())
    return " ".join(cleaned.lower().split())


def _tier_from_barrel(barrel_pct: float) -> int:
    if barrel_pct is None or np.isnan(barrel_pct):
        return 3
    if barrel_pct >= 15:
        return 5
    if barrel_pct >= 11:
        return 4
    if barrel_pct >= 7.5:
        return 3
    if barrel_pct >= 5:
        return 2
    return 1


def _coerce_pct(value) -> float:
    """FanGraphs returns rate stats as either fractions (0.25) or percents (25)."""
    try:
        v = float(value)
    except Exception:
        return np.nan
    return v * 100.0 if 0 < v <= 1.0 else v


@lru_cache(maxsize=4)
def get_season_batter_table(year: int, min_bbe: int = 25) -> pd.DataFrame:
    """Season batted-ball quality + counting stats, keyed by name and MLBAM id.

    Columns: barrel_pct, hard_hit_pct, avg_ev, max_ev, launch_angle, xwoba, pa,
    season_hr, hr_per_pa, k_pct, power_tier, mlbam_id, name_key.
    Returns an empty DataFrame on any failure.
    """
    if not _HAS_PYB:
        return pd.DataFrame()
    try:
        ev = pyb.statcast_batter_exitvelo_barrels(year, minBBE=min_bbe)
    except Exception:
        return pd.DataFrame()
    if ev is None or ev.empty:
        return pd.DataFrame()
    try:
        return _assemble_season_table(ev, year)
    except Exception:
        return pd.DataFrame()


def _assemble_season_table(ev: pd.DataFrame, year: int) -> pd.DataFrame:
    ev = ev.rename(
        columns={
            "avg_hit_speed": "avg_ev",
            "max_hit_speed": "max_ev",
            "ev95percent": "hard_hit_pct",
            "brl_percent": "barrel_pct",
            "avg_hit_angle": "launch_angle",
            "player_id": "mlbam_id",
        }
    )
    ev["name_full"] = (ev.get("first_name", "").astype(str).str.strip()
                       + " " + ev.get("last_name", "").astype(str).str.strip())
    ev["name_key"] = ev["name_full"].map(normalize_name)

    keep = ["name_key", "mlbam_id", "barrel_pct", "hard_hit_pct",
            "avg_ev", "max_ev", "launch_angle"]
    keep = [c for c in keep if c in ev.columns]
    table = ev[keep].copy()

    # Merge FanGraphs season counting stats (PA, HR, K%, xwOBA) by name.
    try:
        fg = pyb.batting_stats(year, qual=0)
        fg = fg.rename(columns={"Name": "name_full"})
        fg["name_key"] = fg["name_full"].map(normalize_name)
        cols = {}
        # 'Contact%' and 'SwStr%' are FanGraphs plate-discipline rates we use to
        # derive real swing-and-miss (whiff) rate; whiff% = 100 - Contact%.
        for src, dst in [("PA", "pa"), ("HR", "season_hr"), ("K%", "k_pct"),
                         ("xwOBA", "xwoba"), ("SO", "so"),
                         ("Contact%", "contact_pct"), ("SwStr%", "swstr_pct")]:
            if src in fg.columns:
                cols[src] = dst
        fg_small = fg[["name_key"] + list(cols)].rename(columns=cols)
        table = table.merge(fg_small, on="name_key", how="left")
    except Exception:
        pass

    # Derive / clean fields.
    if "pa" not in table:
        table["pa"] = np.nan
    if "season_hr" not in table:
        table["season_hr"] = np.nan
    if "k_pct" in table:
        table["k_pct"] = table["k_pct"].map(_coerce_pct)
    else:
        table["k_pct"] = np.nan
    if "xwoba" not in table:
        table["xwoba"] = np.nan

    # Real swing-and-miss (whiff) rate: prefer 100 - Contact%; if Contact% is
    # missing, approximate from SwStr% (whiffs/pitch) which runs ~0.45x of whiff%.
    if "contact_pct" in table:
        table["whiff_pct"] = 100.0 - table["contact_pct"].map(_coerce_pct)
    elif "swstr_pct" in table:
        table["whiff_pct"] = table["swstr_pct"].map(_coerce_pct) / 0.45
    else:
        table["whiff_pct"] = np.nan

    table["hr_per_pa"] = (table["season_hr"] / table["pa"]).replace([np.inf, -np.inf], np.nan)
    table["power_tier"] = table["barrel_pct"].map(_tier_from_barrel)

    # Drop dup names keeping the higher-PA / higher-attempt row.
    if "pa" in table:
        table = table.sort_values("pa", ascending=False)
    table = table.drop_duplicates("name_key", keep="first")
    return table.reset_index(drop=True)


@lru_cache(maxsize=8)
def get_recent_form_table(end_date_iso: str) -> pd.DataFrame:
    """7/15/30-day HR rates per MLBAM batter id from one Statcast date-range pull.

    Columns indexed by mlbam_id: hr_rate_7, hr_rate_15, hr_rate_30.
    Returns an empty DataFrame on any failure.
    """
    if not _HAS_PYB:
        return pd.DataFrame()
    end = dt.date.fromisoformat(end_date_iso)
    start = end - dt.timedelta(days=30)
    try:
        sc = pyb.statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
    except Exception:
        return pd.DataFrame()
    if sc is None or sc.empty or "batter" not in sc.columns:
        return pd.DataFrame()

    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce").dt.date
    # A plate appearance ends on a row with a non-null `events`.
    pa_rows = sc[sc["events"].notna()]

    def _window(days: int) -> pd.DataFrame:
        cutoff = end - dt.timedelta(days=days)
        w = pa_rows[pa_rows["game_date"] > cutoff]
        grp = w.groupby("batter")
        pa = grp.size().rename("pa")
        hr = w.assign(is_hr=(w["events"] == "home_run")).groupby("batter")["is_hr"].sum().rename("hr")
        out = pd.concat([pa, hr], axis=1).fillna(0)
        out[f"hr_rate_{days}"] = (out["hr"] / out["pa"]).replace([np.inf, -np.inf], 0).fillna(0)
        return out[[f"hr_rate_{days}"]]

    try:
        table = _window(7).join(_window(15), how="outer").join(_window(30), how="outer")
    except Exception:
        return pd.DataFrame()
    table = table.fillna(0.0)
    table.index.name = "mlbam_id"
    return table.reset_index()


@lru_cache(maxsize=1)
def _season_by_id_index(year: int):
    t = get_season_batter_table(year)
    if t.empty or "mlbam_id" not in t.columns:
        return None
    return t.set_index("mlbam_id")


@lru_cache(maxsize=1)
def _season_by_name_index(year: int):
    t = get_season_batter_table(year)
    if t.empty:
        return None
    return t.set_index("name_key")


@lru_cache(maxsize=1)
def _recent_by_id_index(end_date_iso: str):
    t = get_recent_form_table(end_date_iso)
    if t.empty:
        return None
    return t.set_index("mlbam_id")


def lookup_season(year: int, name: str | None, mlbam_id: int | None) -> dict | None:
    """Return a real season profile for a player, or None if not found."""
    by_id = _season_by_id_index(year)
    row = None
    if by_id is not None and mlbam_id is not None and mlbam_id in by_id.index:
        row = by_id.loc[mlbam_id]
    if row is None:
        by_name = _season_by_name_index(year)
        key = normalize_name(name) if name else ""
        if by_name is not None and key and key in by_name.index:
            row = by_name.loc[key]
    if row is None:
        return None
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    def g(col, default=np.nan):
        v = row.get(col, default)
        return None if (isinstance(v, float) and np.isnan(v)) else v

    profile = {
        "barrel_pct": g("barrel_pct"), "hard_hit_pct": g("hard_hit_pct"),
        "avg_ev": g("avg_ev"), "max_ev": g("max_ev"),
        "launch_angle": g("launch_angle"), "xwoba": g("xwoba"),
        "k_pct": g("k_pct"), "whiff_pct": g("whiff_pct"),
        "pa": g("pa"), "season_hr": g("season_hr"),
        "hr_per_pa": g("hr_per_pa"), "power_tier": int(row.get("power_tier", 3)),
    }
    # Drop keys that are None so the caller can fill gaps from the modeled profile.
    return {k: v for k, v in profile.items() if v is not None}


def lookup_recent_form(end_date_iso: str, mlbam_id: int | None) -> dict | None:
    """Return real {hr_rate_7,15,30} for a batter id, or None."""
    by_id = _recent_by_id_index(end_date_iso)
    if by_id is None or mlbam_id is None or mlbam_id not in by_id.index:
        return None
    row = by_id.loc[mlbam_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {
        "hr_rate_7": float(row.get("hr_rate_7", 0.0)),
        "hr_rate_15": float(row.get("hr_rate_15", 0.0)),
        "hr_rate_30": float(row.get("hr_rate_30", 0.0)),
    }


def is_available() -> bool:
    return _HAS_PYB
