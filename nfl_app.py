"""NFL Matchup Lab — player/team analytics, scheme intel, matchup projections.

Run with:  streamlit run nfl_app.py
Deploy: point a second Streamlit Cloud app at this repo, main file nfl_app.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nfl.data import ARCHETYPE_LABEL, ROSTERS, TEAMS
from nfl.model import get_week_slate
from nfl.schemes import (DEF_SCHEMES, TEAM_DEF, TEAM_OFF, def_scheme_of,
                         scheme_desc, scheme_label)

st.set_page_config(page_title="NFL Matchup Lab", page_icon="🏈", layout="wide",
                   initial_sidebar_state="auto")

st.markdown("""
<style>
  #MainMenu, footer {visibility: hidden;}
  .block-container {padding-top: 1.6rem; max-width: 1400px;}
  [data-testid="stMetric"] {background:#161b26;border:1px solid #232a38;
    border-radius:12px;padding:12px 14px;}
  .stTabs [aria-selected="true"] {color:#31a354 !important;}
  @media (max-width: 640px) {
    .block-container {padding: 1rem 0.7rem 4rem !important;}
    h1 {font-size: 1.7rem !important;}
  }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Building the week's matchups…", ttl=1800)
def load_slate(week: int, season: int, prefer_live: bool):
    return get_week_slate(week, season, prefer_live)


def main():
    st.sidebar.title("🏈 NFL Matchup Lab")
    st.sidebar.caption("Stats + schemes → who does what this week")
    week = st.sidebar.slider("Week", 1, 18, 1)
    season = st.sidebar.number_input("Season", 2024, 2030, 2026)
    prefer_live = st.sidebar.toggle(
        "Try live data", value=True,
        help="Real prior-season stats & rosters via nflverse once the season "
             "starts; deterministic modeled baselines until then.")
    pos_filter = st.sidebar.multiselect("Position", ["QB", "RB", "WR", "TE"], [])

    st.title("🏈 NFL Matchup Lab")
    st.caption("Previous-season production × opponent defense × the **scheme "
               "they run** × player-vs-team history → who's a TD favorite and "
               "who's on 100-yd watch, with the reasoning shown.")

    df, source = load_slate(int(week), int(season), prefer_live)
    badge = "🟢" if source.startswith("LIVE") else "🟡"
    st.caption(f"{badge} {source} · Week {week}, {season} · {len(df)} players · "
               f"{df['game'].nunique()} games")
    if pos_filter:
        df = df[df["pos"].isin(pos_filter)]

    t1, t2, t3, t4, t5 = st.tabs(
        ["⭐ Favorites", "📊 Player Boards", "🥊 Matchup Explorer",
         "🧠 Team Schemes", "📖 Method"])

    # ---------------- Favorites ----------------
    with t1:
        st.markdown("#### 🎯 TD favorites")
        st.caption("Scheme + red-zone role + game environment say these players "
                   "find the end zone this week.")
        fav = df[df["td_favorite"]].sort_values("td_prob", ascending=False).head(12)
        if fav.empty:
            fav = df.sort_values("td_prob", ascending=False).head(8)
        for _, r in fav.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                c1.markdown(f"**{r['player']}** · {r['pos']}, {r['team']}")
                c1.caption(r["insight"])
                c2.metric("TD odds", f"{r['td_prob']*100:.0f}%")
        st.markdown("#### 💯 100-yard watch")
        st.caption("Projected volume + scheme leverage put these players in "
                   "century-mark range (275 for QBs).")
        w = df[df["watch_100"]].sort_values("p_100", ascending=False).head(10)
        for _, r in w.iterrows():
            with st.container(border=True):
                c1, c2 = st.columns([4, 1])
                proj = (r["proj_pass_yds"] if r["pos"] == "QB"
                        else (r["proj_scrimmage"] if r["pos"] == "RB"
                              else r["proj_rec_yds"]))
                c1.markdown(f"**{r['player']}** · {r['pos']}, {r['team']} — "
                            f"proj **{proj:.0f} yds**")
                c1.caption(r["insight"])
                c2.metric(r["watch_label"], f"{r['p_100']*100:.0f}%")

    # ---------------- Player boards ----------------
    with t2:
        board = st.selectbox("Board", ["TD likelihood", "Receiving yards",
                                       "Rushing yards", "Passing yards",
                                       "Best matchups (scheme edge)"])
        if board == "TD likelihood":
            b = df.sort_values("td_prob", ascending=False).head(35)
            out = pd.DataFrame({
                "Player": b["player"], "Pos": b["pos"], "Team": b["team"],
                "Opp": b["opponent"], "TD %": (b["td_prob"] * 100).round(0),
                "Exp TDs": b["exp_tds"],
                "Prev TDs": b["prev_tds"], "Insight": b["insight"]})
        elif board == "Best matchups (scheme edge)":
            b = df.sort_values("matchup_score", ascending=False).head(35)
            out = pd.DataFrame({
                "Player": b["player"], "Pos": b["pos"], "Team": b["team"],
                "Opp": b["opponent"], "Matchup": b["matchup_score"],
                "Opp scheme": b["def_scheme"], "Insight": b["insight"]})
        else:
            key = {"Receiving yards": ("proj_rec_yds", "rec_ypg"),
                   "Rushing yards": ("proj_rush_yds", "rush_ypg"),
                   "Passing yards": ("proj_pass_yds", "pass_ypg")}[board]
            b = df[df[key[0]] > 5].sort_values(key[0], ascending=False).head(35)
            out = pd.DataFrame({
                "Player": b["player"], "Pos": b["pos"], "Team": b["team"],
                "Opp": b["opponent"], "Projected": b[key[0]],
                "Last szn /g": b[key[1]],
                "P(100)%": (b["p_100"] * 100).round(0), "Insight": b["insight"]})
        st.dataframe(out, hide_index=True, use_container_width=True,
                     height=min(600, 60 + 35 * min(len(out), 15)),
                     column_config={"Matchup": st.column_config.ProgressColumn(
                         "Matchup", min_value=0, max_value=100, format="%.0f")})
        st.download_button("⬇️ Export CSV", out.to_csv(index=False).encode(),
                           file_name=f"nfl_{board.replace(' ', '_')}.csv")

    # ---------------- Matchup explorer ----------------
    with t3:
        game = st.selectbox("Game", sorted(df["game"].unique()))
        g = df[df["game"] == game]
        away, home = game.split(" @ ")
        c1, c2 = st.columns(2)
        for col, tm in ((c1, away), (c2, home)):
            opp = home if tm == away else away
            sch = def_scheme_of(opp)
            with col:
                st.markdown(f"### {tm} ({TEAMS[tm][0]})")
                st.caption(f"Offense: **{TEAM_OFF.get(tm, '—')}** · faces "
                           f"**{scheme_label(sch)}**")
                st.info(scheme_desc(sch))
                sub = g[g["team"] == tm].sort_values("matchup_score", ascending=False)
                for _, r in sub.iterrows():
                    flags = []
                    if r["td_favorite"]:
                        flags.append("🎯")
                    if r["watch_100"]:
                        flags.append("💯")
                    st.markdown(f"**{r['player']}** {' '.join(flags)} — "
                                f"matchup {r['matchup_score']:.0f}/100")
                    st.caption(r["insight"])

    # ---------------- Team schemes ----------------
    with t4:
        st.caption("What every defense runs, what it gives up, and which player "
                   "archetypes punish it. Bundled profiles — editable; swap in "
                   "charting data (man/zone rates) when live.")
        rows = []
        for tm, sch in TEAM_DEF.items():
            rows.append({"Team": f"{tm} ({TEAMS[tm][0]})",
                         "Defensive scheme": scheme_label(sch),
                         "Offense": TEAM_OFF.get(tm, ""),
                         "Def strength": TEAMS[tm][2]})
        st.dataframe(pd.DataFrame(rows).sort_values("Def strength", ascending=False),
                     hide_index=True, use_container_width=True, height=560)
        st.markdown("##### Scheme cheat sheet")
        for key, meta in DEF_SCHEMES.items():
            st.markdown(f"**{meta['label']}** — {meta['desc']}")

    # ---------------- Method ----------------
    with t5:
        st.markdown("""
### How projections are built
`projection = last-season per-game × opponent defense × scheme interaction ×
vs-team history × game environment`

- **Previous season is the baseline** — per-game yards/TDs/usage (modeled until
  the season starts; `nfl_data_py` overlays real numbers, same as the MLB app).
- **Scheme interaction** — every defense has a profile (press-man blitz,
  two-high zone, single-high, soft zone, attacking front) and every player an
  archetype (alpha X, deep threat, slot tech, YAC creator, power back, zone
  runner, receiving back, seam/red-zone TE, dual-threat/pocket QB). The matrix
  in `nfl/schemes.py` says who eats vs what — e.g. *alpha receivers vs
  press-man islands: +13% yards, +15% TD*. Interactions capped at ±15% —
  schemes tilt games, they don't decide them.
- **vs-team history** — a player's career line vs this opponent nudges the
  projection (damped, needs ≥2 games).
- **TD likelihood** — team expected TDs (implied points ÷ 7 × 85%) × the
  player's red-zone share × defense × scheme TD factor, blended with his
  actual TD rate last season. **🎯 TD favorite** = ≥35%.
- **💯 100-yd watch** — P(100+ scrimmage/receiving, 275+ passing) from the
  projection with position-calibrated volatility; flagged at ≥28%.
        """)


if __name__ == "__main__":
    main()
