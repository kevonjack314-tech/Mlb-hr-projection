"""HR Trends Lab — 12 pattern detectors over the rolling HR history.

Player tiers (user-defined, by season HR count):
    ⭐ Star  ≥ 18 HR   — books price these ~ +200 to +450 (never longshots)
    🔷 Mid   8–17 HR   — the classic "value" band, ~ +500 to +700
    🎯 Under ≤ 7 HR    — under-the-radar bats, +700 and up (longshots)

Every trend is computed purely from the HR events window (date, player, team,
lineup spot, HR count, season HR, home/away), returns a small table plus a
one-line SIGNAL takeaway, and degrades gracefully when the window is thin.
The same signals nudge parlay roles: after star-heavy days the rotation trend
says lean mid/under next; live back-to-back bats get flagged, etc.
"""

from __future__ import annotations

import pandas as pd

# --- User-defined tier thresholds (season HR count) ---
STAR_HR_MIN = 18
MID_HR_MIN = 8

TIER_STAR, TIER_MID, TIER_UNDER = "⭐ Star", "🔷 Mid", "🎯 Under"

# Model-implied odds bands per tier (American), per the user's market read.
TIER_ODDS_BAND = {
    TIER_STAR: (200, 450),
    TIER_MID: (500, 700),
    TIER_UNDER: (700, 2000),
}


def tier_of(season_hr) -> str:
    """Classify a player by season HR total (missing/NaN counts as Under)."""
    try:
        hr = float(season_hr)
    except (TypeError, ValueError):
        return TIER_UNDER
    if hr != hr:                      # NaN
        return TIER_UNDER
    if hr >= STAR_HR_MIN:
        return TIER_STAR
    if hr >= MID_HR_MIN:
        return TIER_MID
    return TIER_UNDER


# --------------------------------------------------------------------------- #
# Prep
# --------------------------------------------------------------------------- #
def _prep(events: pd.DataFrame) -> pd.DataFrame | None:
    if events is None or events.empty or "date" not in events.columns:
        return None
    ev = events.copy()
    ev["date"] = pd.to_datetime(ev["date"], errors="coerce")
    ev = ev.dropna(subset=["date"])
    if ev.empty:
        return None
    ev["dow"] = ev["date"].dt.day_name()
    ev["hr_n"] = pd.to_numeric(ev.get("hr_count", 1), errors="coerce").fillna(1)
    ev["tier"] = ev.get("season_hr", pd.Series(index=ev.index)).map(tier_of)
    return ev


def _t(key, title, signal, table=None):
    return {"key": key, "title": title, "signal": signal, "table": table}


# --------------------------------------------------------------------------- #
# The 12 trends
# --------------------------------------------------------------------------- #
def trend_dow_spot(ev):
    """1. Which lineup spots homer on which day of the week."""
    d = ev.dropna(subset=["lineup_spot"])
    if d.empty:
        return _t("dow_spot", "📅 Lineup spot × day of week", "Not enough data yet.")
    d = d.assign(spot=d["lineup_spot"].astype(int))
    pv = d.pivot_table(index="spot", columns="dow", values="hr_n",
                       aggfunc="sum", fill_value=0)
    order = [c for c in ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"] if c in pv.columns]
    pv = pv[order].astype(int)
    flat = pv.stack()
    (spot, dow), n = flat.idxmax(), int(flat.max())
    sig = (f"Hottest combo: the **{spot}-spot on {dow}s** — {n} HRs this window. "
           "Line bats up against the day they're playing.")
    return _t("dow_spot", "📅 Lineup spot × day of week", sig, pv.reset_index())


def trend_dow_tier(ev):
    """2. Which tier does the damage on which day of the week."""
    pv = ev.pivot_table(index="dow", columns="tier", values="hr_n",
                        aggfunc="sum", fill_value=0)
    if pv.empty:
        return _t("dow_tier", "📅 Tier × day of week", "Not enough data yet.")
    order = [c for c in ["Monday", "Tuesday", "Wednesday", "Thursday",
                         "Friday", "Saturday", "Sunday"] if c in pv.index]
    pv = pv.loc[order]
    shares = pv.div(pv.sum(axis=1), axis=0)
    if TIER_UNDER in shares.columns and shares[TIER_UNDER].notna().any():
        best_day = shares[TIER_UNDER].idxmax()
        sig = (f"**{best_day}s are underdog days** — under-radar bats take their "
               f"biggest share of HRs ({shares.loc[best_day, TIER_UNDER]*100:.0f}%). "
               "Longshot legs play best there.")
    else:
        sig = "Tier splits by weekday shown below."
    return _t("dow_tier", "📅 Tier × day of week", sig,
              (shares * 100).round(0).astype(int).reset_index())


def trend_tier_rotation(ev):
    """3. Star-heavy day → does the next day flip to mid/under bats?"""
    daily = ev.groupby([ev["date"].dt.date, "tier"])["hr_n"].sum().unstack(fill_value=0)
    if len(daily) < 4 or TIER_STAR not in daily.columns:
        return _t("tier_rotation", "🔄 Tier rotation (star day → underdog day?)",
                  "Needs a few more days of history.")
    shares = daily.div(daily.sum(axis=1), axis=0)
    thresh = shares[TIER_STAR].median()
    star_heavy = shares[TIER_STAR] > thresh
    nxt = shares.shift(-1).dropna()
    sh = star_heavy.iloc[:-1]
    non_star = [c for c in (TIER_MID, TIER_UNDER) if c in nxt.columns]
    after_heavy = nxt.loc[sh.values, non_star].sum(axis=1).mean() * 100
    after_light = nxt.loc[~sh.values, non_star].sum(axis=1).mean() * 100
    delta = after_heavy - after_light
    if delta > 2:
        sig = (f"**Confirmed rotation:** after star-heavy days, mid/under bats take "
               f"{after_heavy:.0f}% of the next day's HRs vs {after_light:.0f}% "
               f"otherwise (+{delta:.0f} pts). Star day yesterday → lean value/"
               "longshots today.")
    elif delta < -2:
        sig = (f"Stars stay hot: star-heavy days are followed by MORE star HRs "
               f"({100-after_heavy:.0f}% next-day star share). Ride the chalk.")
    else:
        sig = "No strong rotation either way in this window — tiers are independent day to day."
    tbl = pd.DataFrame({
        "Next-day mid+under HR share": [f"{after_heavy:.0f}%", f"{after_light:.0f}%"],
    }, index=["After a STAR-heavy day", "After a normal day"]).reset_index(names="Condition")
    return _t("tier_rotation", "🔄 Tier rotation (star day → underdog day?)", sig, tbl)


def trend_back_to_back(ev):
    """4. Back-to-back: who homers on consecutive days, and how often."""
    by_day = ev.groupby(ev["date"].dt.date)["player"].apply(set).sort_index()
    if len(by_day) < 2:
        return _t("back_to_back", "🔁 Back-to-back HR hitters", "Needs ≥2 days.")
    days = list(by_day.index)
    rows, repeats, prior_total = [], 0, 0
    for d1, d2 in zip(days[:-1], days[1:]):
        if (d2 - d1).days != 1:
            continue
        overlap = by_day[d1] & by_day[d2]
        repeats += len(overlap)
        prior_total += len(by_day[d1])
        rows.append({"Day": str(d2), "Repeat hitters": len(overlap),
                     "Names": ", ".join(sorted(overlap)[:4]) or "—"})
    rate = 100 * repeats / prior_total if prior_total else 0
    live = sorted(by_day[days[-1]] & by_day[days[-2]]) if (days[-1] - days[-2]).days == 1 else []
    sig = (f"**{rate:.0f}% of HR hitters homer again the very next day** "
           f"({repeats} repeats this window)")
    sig += (f" — 🔥 live right now: {', '.join(live[:5])}." if live
            else ". No one is riding a back-to-back streak into today.")
    return _t("back_to_back", "🔁 Back-to-back HR hitters", sig,
              pd.DataFrame(rows).tail(7) if rows else None)


def trend_streaks(ev):
    """5. Active multi-day HR streaks (3+ day heaters included)."""
    by_day = ev.groupby(ev["date"].dt.date)["player"].apply(set).sort_index()
    days = list(by_day.index)
    if not days:
        return _t("streaks", "🔥 Active HR streaks", "No data.")
    last = days[-1]
    streaks = {}
    for p in by_day[last]:
        n, d = 1, last
        while True:
            prev = d - pd.Timedelta(days=1)
            if prev in by_day.index and p in by_day[prev]:
                n += 1
                d = prev
            else:
                break
        if n >= 2:
            streaks[p] = n
    if not streaks:
        return _t("streaks", "🔥 Active HR streaks",
                  f"No active 2+ day streaks as of {last}. Fresh slate.")
    tbl = (pd.DataFrame({"Player": list(streaks), "Consecutive HR days": list(streaks.values())})
           .sort_values("Consecutive HR days", ascending=False))
    top = tbl.iloc[0]
    sig = (f"**{top['Player']} has homered {int(top['Consecutive HR days'])} days "
           f"in a row** — {len(tbl)} bat(s) carrying an active streak into today.")
    return _t("streaks", "🔥 Active HR streaks", sig, tbl)


def trend_multi_hr_follow(ev):
    """6. After a multi-HR game, does the player go deep again within 2 days?"""
    multi = ev[ev["hr_n"] >= 2]
    if multi.empty:
        return _t("multi_follow", "💥 Multi-HR game follow-up", "No multi-HR games yet.")
    by_day = ev.groupby(ev["date"].dt.date)["player"].apply(set)
    hits, n = 0, 0
    for _, r in multi.iterrows():
        d0 = r["date"].date()
        later = [d0 + pd.Timedelta(days=k) for k in (1, 2)]
        if all(d > max(by_day.index) for d in later):
            continue
        n += 1
        if any(d in by_day.index and r["player"] in by_day[d] for d in later):
            hits += 1
    rate = 100 * hits / n if n else 0
    sig = (f"After a **multi-HR game**, hitters homer again within 2 days "
           f"**{rate:.0f}%** of the time ({hits}/{n}). "
           + ("Multi-HR bats stay live — keep them on the board."
                 if rate >= 40 else "The bounce usually cools — don't overpay the day after."))
    return _t("multi_follow", "💥 Multi-HR game follow-up", sig)


def trend_tier_share_rolling(ev):
    """7. Last-7-days tier mix vs the full window — who's trending league-wide."""
    total = ev.groupby("tier")["hr_n"].sum()
    cutoff = ev["date"].max() - pd.Timedelta(days=7)
    recent = ev[ev["date"] > cutoff].groupby("tier")["hr_n"].sum()
    if total.sum() == 0:
        return _t("tier_roll", "📈 Tier momentum (last 7d vs window)", "No data.")
    tbl = pd.DataFrame({
        "Window share %": (total / total.sum() * 100).round(0),
        "Last 7d share %": (recent / max(recent.sum(), 1) * 100).round(0),
    }).fillna(0).astype(int).reset_index()
    tbl["Δ"] = tbl["Last 7d share %"] - tbl["Window share %"]
    mover = tbl.loc[tbl["Δ"].abs().idxmax()]
    sig = (f"**{mover['tier']} bats are {'surging' if mover['Δ'] > 0 else 'fading'}** — "
           f"{mover['Last 7d share %']}% of HRs the last 7 days vs "
           f"{mover['Window share %']}% across the window ({mover['Δ']:+d} pts).")
    return _t("tier_roll", "📈 Tier momentum (last 7d vs window)", sig, tbl)


def trend_weekend(ev):
    """8. Weekend vs weekday HR volume."""
    ev2 = ev.assign(weekend=ev["date"].dt.dayofweek >= 5)
    grp = ev2.groupby("weekend").agg(hrs=("hr_n", "sum"),
                                     days=("date", lambda s: s.dt.date.nunique()))
    if grp.empty or (grp["days"] == 0).any():
        return _t("weekend", "🗓️ Weekend vs weekday volume", "Not enough data.")
    wk = grp.loc[False, "hrs"] / grp.loc[False, "days"] if False in grp.index else 0
    we = grp.loc[True, "hrs"] / grp.loc[True, "days"] if True in grp.index else 0
    sig = (f"**{we:.1f} HRs/day on weekends vs {wk:.1f} on weekdays** — "
           + ("weekend slates run hotter; size up Saturday/Sunday."
              if we > wk * 1.05 else "no real weekend bump in this window."))
    return _t("weekend", "🗓️ Weekend vs weekday volume", sig)


def trend_team_stacks(ev):
    """9. Team stack days (2+/3+ HRs) and whether the team stays hot next day."""
    td = ev.groupby([ev["date"].dt.date, "team"])["hr_n"].sum()
    if td.empty:
        return _t("stacks", "🏟️ Team HR stacks", "No data.")
    n_days = ev["date"].dt.date.nunique()
    two = int((td >= 2).sum())
    three = int((td >= 3).sum())
    hot = td[td >= 3]
    again, n = 0, 0
    daymax = ev["date"].dt.date.max()
    for (d, tm) in hot.index:
        nx = d + pd.Timedelta(days=1)
        if nx > daymax:
            continue
        n += 1
        if (nx, tm) in td.index:
            again += 1
    follow = 100 * again / n if n else 0
    sig = (f"**{two/n_days:.1f} team stacks/day** (2+ HRs from one lineup); "
           f"{three} big 3+ HR eruptions. After an eruption the same team homers "
           f"again next day **{follow:.0f}%** of the time"
           + (" — hot lineups carry over." if follow >= 55 else " — eruptions don't reliably carry over."))
    return _t("stacks", "🏟️ Team HR stacks", sig)


def trend_spot_pairs(ev):
    """10. Which lineup-spot combos homer together on team stack days."""
    d = ev.dropna(subset=["lineup_spot"])
    if d.empty:
        return _t("spot_pairs", "👥 Lineup-spot pairs that hit together", "No data.")
    pairs = {}
    for (_, _), grp in d.assign(spot=d["lineup_spot"].astype(int)).groupby(
            [d["date"].dt.date, "team"]):
        spots = sorted(set(grp["spot"]))
        for i in range(len(spots)):
            for j in range(i + 1, len(spots)):
                pairs[(spots[i], spots[j])] = pairs.get((spots[i], spots[j]), 0) + 1
    if not pairs:
        return _t("spot_pairs", "👥 Lineup-spot pairs that hit together",
                  "No same-team multi-HR days yet.")
    tbl = (pd.DataFrame([{"Spots": f"#{a} + #{b}", "Times together": n}
                         for (a, b), n in pairs.items()])
           .sort_values("Times together", ascending=False).head(6))
    top = tbl.iloc[0]
    sig = (f"When a lineup stacks HRs, it's most often the **{top['Spots']}** spots "
           f"going deep together ({int(top['Times together'])}×) — a natural "
           "same-team pairing.")
    return _t("spot_pairs", "👥 Lineup-spot pairs that hit together", sig, tbl)


def trend_droughts(ev):
    """11. Drought-breakers: how often HRs come off a 7+ day quiet spell."""
    ev2 = ev.sort_values("date")
    gaps = []
    for _, grp in ev2.groupby("player"):
        ds = sorted(set(grp["date"].dt.date))
        gaps += [(b - a).days for a, b in zip(ds[:-1], ds[1:])]
    if not gaps:
        return _t("droughts", "⏳ Drought-breakers", "Not enough repeat hitters yet.")
    s = pd.Series(gaps)
    share7 = 100 * (s >= 7).mean()
    sig = (f"Median gap between a hitter's HRs: **{s.median():.0f} days**; "
           f"**{share7:.0f}%** of repeat HRs broke a 7+ day drought — "
           + ("cold streaks end loudly; 'due' bats are real here."
              if share7 >= 25 else "most repeat HRs come in tight clusters, not off long droughts."))
    return _t("droughts", "⏳ Drought-breakers", sig)


def trend_home_away(ev):
    """12. Home vs away HR tilt, by tier."""
    if "is_home" not in ev.columns or ev["is_home"].isna().all():
        return _t("home_away", "🏠 Home vs away", "Home/away not tracked in this window.")
    d = ev.dropna(subset=["is_home"])
    pv = d.pivot_table(index="tier", columns="is_home", values="hr_n",
                       aggfunc="sum", fill_value=0)
    pv.columns = ["Away" if not c else "Home" for c in pv.columns]
    home = pv.get("Home", pd.Series(0, index=pv.index)).sum()
    away = pv.get("Away", pd.Series(0, index=pv.index)).sum()
    tot = home + away
    sig = (f"**{100*home/tot:.0f}% of HRs come at home** vs {100*away/tot:.0f}% away"
           if tot else "No split available")
    if tot and home > away * 1.15:
        sig += " — a real home tilt; weight the home half of each matchup."
    return _t("home_away", "🏠 Home vs away", sig, pv.reset_index())


ALL_TRENDS = [trend_dow_spot, trend_dow_tier, trend_tier_rotation,
              trend_back_to_back, trend_streaks, trend_multi_hr_follow,
              trend_tier_share_rolling, trend_weekend, trend_team_stacks,
              trend_spot_pairs, trend_droughts, trend_home_away]


def compute_trends(events: pd.DataFrame) -> list[dict]:
    """Run all 12 detectors; each returns {key, title, signal, table}."""
    ev = _prep(events)
    if ev is None:
        return []
    out = []
    for fn in ALL_TRENDS:
        try:
            out.append(fn(ev))
        except Exception:
            continue
    return out


def rotation_hint(events: pd.DataFrame) -> str | None:
    """One-liner for TODAY based on yesterday's tier mix (used by the parlay tab)."""
    ev = _prep(events)
    if ev is None or ev["date"].dt.date.nunique() < 2:
        return None
    last = ev["date"].dt.date.max()
    y = ev[ev["date"].dt.date == last]
    shares = y.groupby("tier")["hr_n"].sum()
    if shares.sum() == 0:
        return None
    star_share = shares.get(TIER_STAR, 0) / shares.sum()
    if star_share >= 0.5:
        return (f"Yesterday was **star-heavy** ({star_share*100:.0f}% of HRs) — the "
                "rotation trend says lean 🔷 mid / 🎯 under bats today.")
    if star_share <= 0.25:
        return (f"Stars were quiet yesterday ({star_share*100:.0f}% of HRs) — "
                "rotation says the ⭐ big names show up today.")
    return None
