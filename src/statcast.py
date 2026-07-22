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
import time as _time
import unicodedata
from functools import lru_cache  # noqa: F401  (still used by light lookups)

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


# --------------------------------------------------------------------------- #
# Caching + diagnostics
# --------------------------------------------------------------------------- #
_FAIL_TTL = 600.0   # seconds a FAILED pull is remembered before retrying

# Last error per feed, e.g. {"season_table": "HTTPError: 502 ..."} — surfaced
# in the app's data-provenance notes so a blank board explains itself.
_DIAG: dict = {}


def note_diag(source: str, msg) -> None:
    _DIAG[source] = str(msg)[:200]


def get_diagnostics() -> dict:
    return dict(_DIAG)


def _cache_ok(fn):
    """Memoize like lru_cache — but never remember a failure for long.

    lru_cache pinned EMPTY results for the process lifetime, so one transient
    Savant/FanGraphs hiccup at boot blanked every season metric until the app
    restarted. Here a successful (non-empty) result is cached forever, while
    an empty/None result is only cached _FAIL_TTL seconds and then retried.
    """
    store: dict = {}

    def wrapped(*args):
        hit = store.get(args)
        now = _time.time()
        if hit is not None:
            val, ts, ok = hit
            if ok or (now - ts) < _FAIL_TTL:
                return val
        val = fn(*args)
        ok = val is not None
        if ok and hasattr(val, "empty"):
            ok = not val.empty
        elif ok and isinstance(val, dict):
            ok = bool(val)
        store[args] = (val, now, ok)
        if ok:
            _DIAG.pop(fn.__name__, None)
        return val

    wrapped.cache_clear = store.clear
    wrapped.__name__ = fn.__name__
    return wrapped


_FG_API = "https://www.fangraphs.com/api/leaders/major-league/data"
_FG_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"),
    "Accept": "application/json",
}


def _fg_api_leaders(year: int, stats: str) -> pd.DataFrame:
    """FanGraphs leaders via their JSON API.

    Fallback when pybaseball's legacy-endpoint scrape gets 403'd (their CDN
    blocks some cloud IP ranges on the old .aspx page but serves the API).
    `stats` is 'bat' or 'pit'. Empty DataFrame on any failure.
    """
    try:
        import requests
        r = requests.get(_FG_API, params={
            "age": "", "pos": "all", "stats": stats, "lg": "all", "qual": "0",
            "season": str(year), "season1": str(year), "startdate": "",
            "enddate": "", "month": "0", "hand": "", "team": "0",
            "pageitems": "5000", "pagenum": "1", "ind": "0", "rost": "0",
            "players": "", "type": "8", "postseason": "",
            "sortdir": "default", "sortstat": "WAR",
        }, headers=_FG_HEADERS, timeout=30)
        r.raise_for_status()
        js = r.json()
        data = js.get("data") if isinstance(js, dict) else js
        df = pd.DataFrame(data or [])
        if df.empty:
            return df
        renames = {}
        if "Name" not in df.columns:
            if "PlayerName" in df.columns:
                renames["PlayerName"] = "Name"
            elif "PlayerNameRoute" in df.columns:
                renames["PlayerNameRoute"] = "Name"
        if "Team" not in df.columns:
            for cand in ("TeamNameAbb", "TeamName", "AbbName"):
                if cand in df.columns:
                    renames[cand] = "Team"
                    break
        return df.rename(columns=renames)
    except Exception as exc:
        note_diag(f"fangraphs_api ({stats})", exc)
        return pd.DataFrame()


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


@_cache_ok
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
    except Exception as exc:
        note_diag("season_table (Savant EV/barrels)", exc)
        return pd.DataFrame()
    if ev is None or ev.empty:
        note_diag("season_table (Savant EV/barrels)", "empty response")
        return pd.DataFrame()
    try:
        return _assemble_season_table(ev, year)
    except Exception as exc:
        note_diag("season_table (assemble)", exc)
        return pd.DataFrame()


def _assemble_season_table(ev: pd.DataFrame, year: int) -> pd.DataFrame:
    ev = ev.rename(
        columns={
            "avg_hit_speed": "avg_ev",
            "max_hit_speed": "max_ev",
            "ev95percent": "hard_hit_pct",
            "brl_percent": "barrel_pct",
            "brl_pa": "brl_pa",
            "avg_hit_angle": "launch_angle",
            "anglesweetspotpercent": "sweet_spot_pct",
            "player_id": "mlbam_id",
        }
    )
    # Savant has shipped several name layouts over time: first_name+last_name,
    # a single "last_name, first_name" column, or player_name. Handle them all
    # (a str default from .get() crashed here when the columns went missing).
    if "first_name" in ev.columns and "last_name" in ev.columns:
        ev["name_full"] = (ev["first_name"].astype(str).str.strip()
                           + " " + ev["last_name"].astype(str).str.strip())
    else:
        name_col = next(
            (c for c in ev.columns
             if c.strip().lower().replace(" ", "") in
             ("last_name,first_name", "player_name", "name", "player")),
            None)
        ev["name_full"] = ev[name_col].astype(str) if name_col else ""
    ev["name_key"] = ev["name_full"].map(normalize_name)

    keep = ["name_key", "mlbam_id", "barrel_pct", "brl_pa", "hard_hit_pct",
            "sweet_spot_pct",
            "avg_ev", "max_ev", "launch_angle"]
    keep = [c for c in keep if c in ev.columns]
    table = ev[keep].copy()

    # Merge FanGraphs season counting stats (PA, HR, K%, xwOBA) by name.
    try:
        try:
            fg = pyb.batting_stats(year, qual=0)
        except Exception as exc:
            note_diag("season_table (FanGraphs merge)", exc)
            fg = _fg_api_leaders(year, "bat")      # CDN-403 fallback
        if fg is None or fg.empty:
            raise ValueError("no FanGraphs batting data")
        fg = fg.rename(columns={"Name": "name_full"})
        fg["name_key"] = fg["name_full"].map(normalize_name)
        cols = {}
        # 'Contact%' and 'SwStr%' are FanGraphs plate-discipline rates we use to
        # derive real swing-and-miss (whiff) rate; whiff% = 100 - Contact%.
        for src, dst in [("PA", "pa"), ("HR", "season_hr"), ("K%", "k_pct"),
                         ("xwOBA", "xwoba"), ("SO", "so"),
                         ("Contact%", "contact_pct"), ("SwStr%", "swstr_pct"),
                         ("O-Swing%", "chase_pct"), ("Z-Contact%", "zone_contact_pct"),
                         ("FB%", "fb_pct"), ("GB%", "gb_pct"), ("LD%", "ld_pct"),
                         ("Pull%", "pull_pct"), ("HR/FB", "hr_fb"), ("ISO", "iso")]:
            if src in fg.columns:
                cols[src] = dst
        fg_small = fg[["name_key"] + list(cols)].rename(columns=cols)
        table = table.merge(fg_small, on="name_key", how="left")
    except Exception as exc:
        note_diag("season_table (FanGraphs merge)", exc)

    # Merge Statcast expected stats (quality-of-contact power) by MLBAM id.
    # xISO = expected SLG - expected BA: pure expected power, contact-quality based.
    try:
        xs = pyb.statcast_batter_expected_stats(year, minPA=10)
        xs = xs.rename(columns={"player_id": "mlbam_id"})
        if "est_slg" in xs.columns and "est_ba" in xs.columns:
            xs["xiso"] = xs["est_slg"] - xs["est_ba"]
            xs_small = xs[["mlbam_id", "est_slg", "xiso"]].rename(columns={"est_slg": "xslg"})
            table = table.merge(xs_small, on="mlbam_id", how="left")
    except Exception:
        pass

    # Merge Statcast sprint speed (ft/s) by MLBAM id — athletic context.
    try:
        sp = pyb.statcast_sprint_speed(year, 10)
        sp = sp.rename(columns={"player_id": "mlbam_id"})
        if "sprint_speed" in sp.columns:
            table = table.merge(sp[["mlbam_id", "sprint_speed"]], on="mlbam_id", how="left")
    except Exception:
        pass

    # Derive / clean fields.
    for col in ("xiso", "xslg", "brl_pa", "sprint_speed"):
        if col not in table:
            table[col] = np.nan
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
    # Contact% is the complement of whiff (contact made / swings); keep them
    # perfectly consistent and on a 0-100 percent scale.
    table["contact_pct"] = 100.0 - table["whiff_pct"]

    # Plate-discipline + batted-ball rates (real, FanGraphs), normalized to 0-100%.
    for col in ("chase_pct", "zone_contact_pct", "fb_pct", "gb_pct", "ld_pct",
                "pull_pct", "hr_fb", "sweet_spot_pct"):
        if col in table:
            table[col] = table[col].map(_coerce_pct)
        else:
            table[col] = np.nan
    # ISO is a rate (~.160), not a percent — keep as-is.
    if "iso" not in table:
        table["iso"] = np.nan

    table["hr_per_pa"] = (table["season_hr"] / table["pa"]).replace([np.inf, -np.inf], np.nan)
    table["power_tier"] = table["barrel_pct"].map(_tier_from_barrel)

    # Drop dup names keeping the higher-PA / higher-attempt row.
    if "pa" in table:
        table = table.sort_values("pa", ascending=False)
    table = table.drop_duplicates("name_key", keep="first")
    return table.reset_index(drop=True)


# --- Pitch-type families (Statcast pitch_type codes -> FB / breaking / offspeed).
_PITCH_FAMILY = {
    "FF": "fb", "FA": "fb", "SI": "fb", "FT": "fb", "FC": "fb",   # fastballs/cutter
    "SL": "br", "CU": "br", "KC": "br", "ST": "br", "SV": "br",   # breaking
    "CS": "br", "SC": "br", "KN": "br",
    "CH": "os", "FS": "os", "FO": "os",                            # offspeed
}


@_cache_ok
def _statcast_range(end_date_iso: str, lookback_days: int = 30):
    """One cached Statcast pitch-level pull for the window. Returns df or None.

    Shared by recent-form, pitch-mix, and batter-vs-pitch-type splits so the
    heavy pull happens once per (end_date, lookback).
    """
    if not _HAS_PYB:
        return None
    end = dt.date.fromisoformat(end_date_iso)
    start = end - dt.timedelta(days=lookback_days)
    try:
        sc = pyb.statcast(start_dt=start.isoformat(), end_dt=end.isoformat(), verbose=False)
    except Exception as exc:
        note_diag("statcast_range (recent form/splits)", exc)
        return None
    if sc is None or sc.empty or "batter" not in sc.columns:
        return None
    sc = sc.copy()
    sc["game_date"] = pd.to_datetime(sc["game_date"], errors="coerce").dt.date
    if "pitch_type" in sc.columns:
        sc["pitch_family"] = sc["pitch_type"].map(_PITCH_FAMILY)
    return sc


@_cache_ok
def get_recent_form_table(end_date_iso: str) -> pd.DataFrame:
    """7/15/30-day HR rates per MLBAM batter id from one Statcast date-range pull.

    Columns indexed by mlbam_id: hr_rate_7, hr_rate_15, hr_rate_30.
    Returns an empty DataFrame on any failure.
    """
    sc = _statcast_range(end_date_iso, 30)
    if sc is None:
        return pd.DataFrame()
    end = dt.date.fromisoformat(end_date_iso)
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


def meatball_rates(pitches: pd.DataFrame, min_pitches: int = 100) -> pd.DataFrame:
    """Middle-middle pitches per 100 ("meatballs") per pitcher.

    HRs are hit off mistakes, and mistake SUPPLY varies ~2x between starters.
    Statcast `zone` 5 is the dead-center cell of the strike zone — a direct
    measure of how many grooved pitches a pitcher serves, less noisy and less
    park-polluted than HR/9. Pure function so it's unit-testable offline.
    """
    if pitches is None or pitches.empty or not {"zone", "pitcher"} <= set(pitches.columns):
        return pd.DataFrame()
    df = pitches[pitches["zone"].notna()]
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby("pitcher")["zone"].agg(n="size", mb=lambda s: int((s == 5).sum()))
    grp = grp[grp["n"] >= min_pitches]
    if grp.empty:
        return pd.DataFrame()
    out = pd.DataFrame({"sp_meatball_pct": (grp["mb"] / grp["n"] * 100.0).round(2)})
    out.index.name = "pitcher_id"
    return out.reset_index()


@_cache_ok
def get_meatball_table(end_date_iso: str, lookback_days: int = 30) -> pd.DataFrame:
    return meatball_rates(_statcast_range(end_date_iso, lookback_days))


@_cache_ok
def _meatball_by_id(end_date_iso: str):
    t = get_meatball_table(end_date_iso)
    return t.set_index("pitcher_id") if not t.empty else None


def lookup_meatball(end_date_iso: str, pitcher_id) -> float | None:
    idx = _meatball_by_id(end_date_iso)
    if idx is None or pitcher_id is None or pitcher_id not in idx.index:
        return None
    v = idx.loc[pitcher_id, "sp_meatball_pct"]
    v = v.iloc[0] if hasattr(v, "iloc") else v
    return float(v) if v == v else None


def velo_deltas(pitches: pd.DataFrame, min_fb: int = 15) -> pd.DataFrame:
    """Fastball velocity change: last start vs the season-window baseline.

    A starter down 1+ mph on his fastball in his most recent outing is the best
    public early-warning of fatigue/injury — HR rates spike against diminished
    velo before ERA catches up, and books adjust slowly. Pure function.

    Returns per-pitcher: sp_velo_last, sp_velo_base, sp_velo_delta (last-base).
    """
    need = {"pitcher", "game_date", "release_speed", "pitch_family"}
    if pitches is None or pitches.empty or not need <= set(pitches.columns):
        return pd.DataFrame()
    fb = pitches[(pitches["pitch_family"] == "fb")
                 & pitches["release_speed"].notna()
                 & pitches["game_date"].notna()].copy()
    if fb.empty:
        return pd.DataFrame()
    rows = []
    for pid, g in fb.groupby("pitcher"):
        last_day = g["game_date"].max()
        last = g[g["game_date"] == last_day]["release_speed"]
        base = g[g["game_date"] < last_day]["release_speed"]
        if len(last) < min_fb or len(base) < min_fb:
            continue
        lv, bv = float(last.mean()), float(base.mean())
        rows.append({"pitcher_id": pid, "sp_velo_last": round(lv, 1),
                     "sp_velo_base": round(bv, 1), "sp_velo_delta": round(lv - bv, 1)})
    return pd.DataFrame(rows)


@_cache_ok
def get_velo_table(end_date_iso: str, lookback_days: int = 45) -> pd.DataFrame:
    # Wider window so a starter has multiple outings to baseline against.
    return velo_deltas(_statcast_range(end_date_iso, lookback_days))


@_cache_ok
def _velo_by_id(end_date_iso: str):
    t = get_velo_table(end_date_iso)
    return t.set_index("pitcher_id") if not t.empty else None


def lookup_velo(end_date_iso: str, pitcher_id) -> dict | None:
    idx = _velo_by_id(end_date_iso)
    if idx is None or pitcher_id is None or pitcher_id not in idx.index:
        return None
    row = idx.loc[pitcher_id]
    if hasattr(row, "iloc") and getattr(row, "ndim", 1) > 1:
        row = row.iloc[0]
    return {k: float(row[k]) for k in ("sp_velo_last", "sp_velo_base", "sp_velo_delta")
            if k in row and row[k] == row[k]} or None


@_cache_ok
def get_pitch_mix_table(end_date_iso: str, lookback_days: int = 30) -> pd.DataFrame:
    """Pitch mix (% fastball/breaking/offspeed) per pitcher MLBAM id.

    Columns indexed by pitcher id: pitcher_mix_fb, pitcher_mix_br, pitcher_mix_os.
    """
    sc = _statcast_range(end_date_iso, lookback_days)
    if sc is None or "pitch_family" not in sc.columns or "pitcher" not in sc.columns:
        return pd.DataFrame()
    df = sc[sc["pitch_family"].notna()]
    if df.empty:
        return pd.DataFrame()
    counts = df.groupby(["pitcher", "pitch_family"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1).replace(0, np.nan)
    out = pd.DataFrame(index=counts.index)
    for fam in ("fb", "br", "os"):
        out[f"pitcher_mix_{fam}"] = (counts.get(fam, 0) / totals * 100.0).round(1)
    out = out.dropna(how="all").fillna(0.0)
    out.index.name = "pitcher_id"
    return out.reset_index()


@_cache_ok
def get_batter_pitch_splits(end_date_iso: str, lookback_days: int = 45) -> pd.DataFrame:
    """Batter wOBA vs each pitch family (real, Statcast) per MLBAM batter id.

    Uses woba_value/woba_denom summed over batted/PA-ending events. Columns:
    vs_fb, vs_br, vs_os. A wider default window (45d) buffers the small samples.
    """
    sc = _statcast_range(end_date_iso, lookback_days)
    if sc is None or "pitch_family" not in sc.columns:
        return pd.DataFrame()
    if "woba_value" not in sc.columns or "woba_denom" not in sc.columns:
        return pd.DataFrame()
    df = sc[sc["woba_denom"].notna() & sc["pitch_family"].notna()]
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["batter", "pitch_family"]).agg(
        val=("woba_value", "sum"), den=("woba_denom", "sum")).reset_index()
    grp["woba"] = (grp["val"] / grp["den"]).replace([np.inf, -np.inf], np.nan)
    wide = grp.pivot(index="batter", columns="pitch_family", values="woba")
    out = pd.DataFrame(index=wide.index)
    for fam in ("fb", "br", "os"):
        out[f"vs_{fam}"] = wide.get(fam, np.nan).round(3)
    out.index.name = "mlbam_id"
    return out.reset_index()


@_cache_ok
def get_batter_platoon_table(end_date_iso: str, lookback_days: int = 45) -> pd.DataFrame:
    """REAL platoon splits per batter: wOBA vs LHP and vs RHP (Statcast).

    Shares the same cached 45-day pitch-level pull as the vs-pitch-type splits.
    Small samples (< 25 wOBA denominators vs a hand) are left NaN so a few
    lucky PAs can't fake a platoon edge. Columns: woba_vs_l, woba_vs_r.
    """
    sc = _statcast_range(end_date_iso, lookback_days)
    if sc is None or "p_throws" not in sc.columns:
        return pd.DataFrame()
    if "woba_value" not in sc.columns or "woba_denom" not in sc.columns:
        return pd.DataFrame()
    df = sc[sc["woba_denom"].notna() & sc["p_throws"].isin(["L", "R"])]
    if df.empty:
        return pd.DataFrame()
    grp = df.groupby(["batter", "p_throws"]).agg(
        val=("woba_value", "sum"), den=("woba_denom", "sum")).reset_index()
    grp["woba"] = np.where(grp["den"] >= 25, grp["val"] / grp["den"], np.nan)
    wide = grp.pivot(index="batter", columns="p_throws", values="woba")
    out = pd.DataFrame(index=wide.index)
    out["woba_vs_l"] = wide.get("L", np.nan).round(3)
    out["woba_vs_r"] = wide.get("R", np.nan).round(3)
    out = out.dropna(how="all")
    out.index.name = "mlbam_id"
    return out.reset_index()


@_cache_ok
def _fg_pitching_raw(year: int) -> pd.DataFrame:
    """One cached FanGraphs pitching pull per year, shared by the starter
    peripherals table and the team bullpen table."""
    if not _HAS_PYB:
        return _fg_api_leaders(year, "pit")
    try:
        fg = pyb.pitching_stats(year, qual=0)
    except Exception as exc:
        note_diag("pitching_table (FanGraphs)", exc)
        return _fg_api_leaders(year, "pit")        # CDN-403 fallback
    return fg if fg is not None else pd.DataFrame()


# FanGraphs team codes that differ from the MLB StatsAPI abbreviations the
# slate uses. Extra aliases are added so either style resolves.
_FG_TEAM_FIX = {"TBR": "TB", "KCR": "KC", "SDP": "SD", "SFG": "SF",
                "WSN": "WSH", "CHW": "CWS"}
_TEAM_ALIASES = {"AZ": "ARI", "ARI": "AZ", "ATH": "OAK", "OAK": "ATH"}


@_cache_ok
def get_bullpen_hr9_table(year: int) -> dict:
    """{team_abbr: bullpen HR/9} from FanGraphs — pure relievers (GS == 0).

    ~40% of a hitter's PAs come against the pen, so the opponent's bullpen
    homer-proneness is a real part of the matchup the starter-only view misses.
    """
    fg = _fg_pitching_raw(year)
    if fg is None or fg.empty or "Team" not in fg.columns:
        return {}
    df = fg.copy()
    df["GS"] = pd.to_numeric(df.get("GS"), errors="coerce").fillna(0)
    df["IP"] = pd.to_numeric(df.get("IP"), errors="coerce").fillna(0.0)
    df["HR"] = pd.to_numeric(df.get("HR"), errors="coerce").fillna(0.0)
    rp = df[(df["GS"] == 0) & (df["IP"] > 0)]
    if rp.empty:
        return {}
    agg = rp.groupby("Team").agg(hr=("HR", "sum"), ip=("IP", "sum"))
    agg = agg[agg["ip"] >= 30]                      # ignore tiny team samples
    out: dict = {}
    for team, r in agg.iterrows():
        abbr = _FG_TEAM_FIX.get(str(team), str(team))
        hr9 = round(float(9.0 * r["hr"] / r["ip"]), 2)
        out[abbr] = hr9
        alias = _TEAM_ALIASES.get(abbr)
        if alias:
            out.setdefault(alias, hr9)
    return out


def lookup_bullpen_hr9(year: int, team_abbr: str | None) -> float | None:
    try:
        return get_bullpen_hr9_table(year).get(str(team_abbr)) if team_abbr else None
    except Exception:
        return None


@_cache_ok
def get_pitching_table(year: int) -> pd.DataFrame:
    """Season pitcher peripherals from FanGraphs, keyed by normalized name.

    Columns: pitcher_hr9, pitcher_gb_pct, pitcher_fb_pct, pitcher_barrel_pct_allowed.
    Returns empty on failure.
    """
    fg = _fg_pitching_raw(year)
    if fg is None or fg.empty:
        return pd.DataFrame()
    fg = fg.rename(columns={"Name": "name_full"})
    fg["name_key"] = fg["name_full"].map(normalize_name)
    out = pd.DataFrame({"name_key": fg["name_key"]})
    out["pitcher_hr9"] = pd.to_numeric(fg.get("HR/9"), errors="coerce")
    out["pitcher_gb_pct"] = fg.get("GB%").map(_coerce_pct) if "GB%" in fg else np.nan
    out["pitcher_fb_pct"] = fg.get("FB%").map(_coerce_pct) if "FB%" in fg else np.nan
    if "Barrel%" in fg:
        out["pitcher_barrel_pct_allowed"] = fg["Barrel%"].map(_coerce_pct)
    else:
        out["pitcher_barrel_pct_allowed"] = np.nan
    if "IP" in fg:
        out["ip"] = pd.to_numeric(fg["IP"], errors="coerce")
        out = out.sort_values("ip", ascending=False)
    out = out.drop_duplicates("name_key", keep="first")
    return out.reset_index(drop=True)


@_cache_ok
def _pitch_mix_by_id(end_date_iso: str):
    t = get_pitch_mix_table(end_date_iso)
    return t.set_index("pitcher_id") if not t.empty else None


@_cache_ok
def _batter_splits_by_id(end_date_iso: str):
    t = get_batter_pitch_splits(end_date_iso)
    return t.set_index("mlbam_id") if not t.empty else None


@_cache_ok
def _pitching_by_name(year: int):
    t = get_pitching_table(year)
    return t.set_index("name_key") if not t.empty else None


@_cache_ok
def _platoon_by_id(end_date_iso: str):
    t = get_batter_platoon_table(end_date_iso)
    return t.set_index("mlbam_id") if not t.empty else None


def lookup_platoon(end_date_iso: str, batter_id) -> dict | None:
    """Real {woba_vs_l, woba_vs_r} for a batter, or None."""
    idx = _platoon_by_id(end_date_iso)
    if idx is None or batter_id is None or batter_id not in idx.index:
        return None
    row = idx.loc[batter_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {k: float(row[k]) for k in ("woba_vs_l", "woba_vs_r")
            if k in row and not pd.isna(row[k])} or None


def lookup_pitch_mix(end_date_iso: str, pitcher_id) -> dict | None:
    idx = _pitch_mix_by_id(end_date_iso)
    if idx is None or pitcher_id is None or pitcher_id not in idx.index:
        return None
    row = idx.loc[pitcher_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {k: float(row[k]) for k in ("pitcher_mix_fb", "pitcher_mix_br", "pitcher_mix_os")
            if k in row and not pd.isna(row[k])}


def lookup_batter_splits(end_date_iso: str, batter_id) -> dict | None:
    idx = _batter_splits_by_id(end_date_iso)
    if idx is None or batter_id is None or batter_id not in idx.index:
        return None
    row = idx.loc[batter_id]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    return {k: float(row[k]) for k in ("vs_fb", "vs_br", "vs_os")
            if k in row and not pd.isna(row[k])}


def lookup_pitching(year: int, name: str | None) -> dict | None:
    idx = _pitching_by_name(year)
    key = normalize_name(name) if name else ""
    if idx is None or not key or key not in idx.index:
        return None
    row = idx.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]
    out = {}
    for k in ("pitcher_hr9", "pitcher_gb_pct", "pitcher_fb_pct", "pitcher_barrel_pct_allowed"):
        v = row.get(k)
        if v is not None and not pd.isna(v):
            out[k] = float(v)
    return out or None


@_cache_ok
def _season_by_id_index(year: int):
    t = get_season_batter_table(year)
    if t.empty or "mlbam_id" not in t.columns:
        return None
    return t.set_index("mlbam_id")


@_cache_ok
def _season_by_name_index(year: int):
    t = get_season_batter_table(year)
    if t.empty:
        return None
    return t.set_index("name_key")


@_cache_ok
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
        "k_pct": g("k_pct"), "whiff_pct": g("whiff_pct"), "contact_pct": g("contact_pct"),
        "chase_pct": g("chase_pct"), "zone_contact_pct": g("zone_contact_pct"),
        "fb_pct": g("fb_pct"), "gb_pct": g("gb_pct"), "ld_pct": g("ld_pct"),
        "pull_pct": g("pull_pct"), "hr_fb": g("hr_fb"),
        "xiso": g("xiso"), "xslg": g("xslg"), "iso": g("iso"),
        "sweet_spot_pct": g("sweet_spot_pct"),
        "brl_pa": g("brl_pa"), "sprint_speed": g("sprint_speed"),
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
