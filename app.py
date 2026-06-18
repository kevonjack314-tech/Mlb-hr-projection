"""MLB Home Run Projection Tool — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import pandas as pd
import streamlit as st

from src.model import (
    HR_SCORE_WEIGHTS,
    POWER_QUALITY_WEIGHTS,
    RECENT_FORM_WEIGHTS,
    score_slate,
)
from src.sources import get_slate

st.set_page_config(
    page_title="MLB HR Projection Tool",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building & scoring the slate…", ttl=60 * 30)
def load_scored_slate(date_iso: str, prefer_live: bool):
    game_date = dt.date.fromisoformat(date_iso)
    df, source, notes = get_slate(game_date, prefer_live=prefer_live)
    scored = score_slate(df)
    return scored, source, notes


# --------------------------------------------------------------------------- #
# Tooltip / metric glossary
# --------------------------------------------------------------------------- #
GLOSSARY = {
    "HR Score": "Composite 0-100 rating blending batted-ball quality, season & recent HR rate, matchup, and park/weather.",
    "HR Prob (game)": "Modeled probability the hitter hits ≥1 HR in this game (per-PA rate compounded over ~4.1 PA).",
    "xHR": "Expected home runs in the game = per-PA HR rate × expected PA.",
    "Fair Odds": "Vig-free American odds implied by the game HR probability (handy for spotting +EV props).",
    "Longshot": "Boom-or-bust ceiling score: max exit velo + barrel% + park/weather, rewarding high-variance upside.",
    "Consistency": "High-floor score: hard-hit%, contact (low K), season HR rate, EV & xwOBA, weighted by sample size.",
    "Sneaky": "Under-the-radar value: strong matchup/park + recent surge vs season line + lower-profile bat.",
    "Barrel%": "Share of batted balls hit with the ideal EV/launch-angle combo for extra-base damage (best HR predictor).",
    "Hard-Hit%": "Share of batted balls ≥95 mph exit velocity.",
    "Avg EV": "Average exit velocity (mph).",
    "Max EV": "Top-end exit velocity (mph) — a raw-power ceiling indicator.",
    "xwOBA": "Expected weighted on-base average from quality of contact.",
    "Park Factor": "Handedness-aware HR park factor (100 = average; 110 = +10% HR).",
}


# --------------------------------------------------------------------------- #
# Sidebar — controls & methodology
# --------------------------------------------------------------------------- #
def sidebar_controls():
    st.sidebar.title("⚾ HR Projection Tool")
    st.sidebar.caption("Ranked HR upside for any MLB slate")

    game_date = st.sidebar.date_input(
        "Game date", value=dt.date.today(),
        help="Pick any MLB date. Defaults to today.",
    )
    prefer_live = st.sidebar.toggle(
        "Try live data (MLB StatsAPI + weather)", value=True,
        help="If off (or if the network is unavailable), a deterministic synthetic slate is used.",
    )
    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        load_scored_slate.clear()
        st.rerun()

    return game_date, prefer_live


def methodology_sidebar():
    with st.sidebar.expander("📖 Methodology & weights", expanded=False):
        st.markdown(
            "**Composite HR Score** is a weighted blend of five 0-100 sub-scores, "
            "each normalized against fixed league reference ranges so scores are "
            "comparable across dates."
        )
        wdf = pd.DataFrame(
            {"Component": list(HR_SCORE_WEIGHTS), "Weight": list(HR_SCORE_WEIGHTS.values())}
        )
        st.dataframe(wdf, hide_index=True, use_container_width=True)
        st.markdown("**Power-quality sub-weights** (Statcast):")
        st.dataframe(
            pd.DataFrame(
                {"Metric": list(POWER_QUALITY_WEIGHTS), "Weight": list(POWER_QUALITY_WEIGHTS.values())}
            ),
            hide_index=True, use_container_width=True,
        )
        st.markdown("**Recent-form window weights:**")
        st.dataframe(
            pd.DataFrame(
                {"Window": ["7d", "15d", "30d"], "Weight": list(RECENT_FORM_WEIGHTS.values())}
            ),
            hide_index=True, use_container_width=True,
        )
        st.markdown(
            "- **HR probability** compounds an adjusted per-PA HR rate "
            "(season + recent + quality-implied) over ~4.1 PA, scaled by "
            "matchup and park/weather multipliers.\n"
            "- **Park factors** are handedness-aware. **Wind** is resolved against "
            "each park's home-plate→CF orientation; **temp/humidity** adjust carry.\n"
            "- See the README for full formulas and assumptions."
        )

    with st.sidebar.expander("🧮 Metric glossary", expanded=False):
        for k, v in GLOSSARY.items():
            st.markdown(f"**{k}** — {v}")


def filter_controls(df: pd.DataFrame):
    st.sidebar.markdown("---")
    st.sidebar.subheader("Filters")
    teams = sorted(df["team"].unique())
    positions = sorted(df["position"].unique())

    sel_teams = st.sidebar.multiselect("Team", teams, default=[])
    sel_pos = st.sidebar.multiselect("Position", positions, default=[])
    sel_hand = st.sidebar.multiselect("Bats", ["L", "R", "S"], default=[])
    platoon_only = st.sidebar.checkbox("Platoon advantage only", value=False)
    min_pa = st.sidebar.slider(
        "Min plate appearances (season)", 0, int(df["pa"].max()), 0, step=10
    )

    f = df.copy()
    if sel_teams:
        f = f[f["team"].isin(sel_teams)]
    if sel_pos:
        f = f[f["position"].isin(sel_pos)]
    if sel_hand:
        f = f[f["bats"].isin(sel_hand)]
    if platoon_only:
        f = f[f["platoon_adv"]]
    f = f[f["pa"] >= min_pa]
    return f


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #
DISPLAY_COLUMNS = {
    "player": "Player",
    "team": "Team",
    "opponent": "Opp",
    "pitcher_name": "Probable Pitcher",
    "pitcher_throws": "P-Hand",
    "bats": "Bats",
    "position": "Pos",
    "hr_score": "HR Score",
    "hr_prob_game": "HR Prob (game)",
    "xhr": "xHR",
    "fair_odds": "Fair Odds",
    "longshot_score": "Longshot",
    "consistency_score": "Consistency",
    "sneaky_score": "Sneaky",
    "barrel_pct": "Barrel%",
    "hard_hit_pct": "Hard-Hit%",
    "avg_ev": "Avg EV",
    "max_ev": "Max EV",
    "xwoba": "xwOBA",
    "hr_per_pa": "HR/PA",
    "park_factor": "Park Factor",
    "wind_mult": "Wind x",
    "temp_f": "Temp °F",
    "recent_form_score": "Recent Form",
    "data_quality": "Data",
    "rationale": "Rationale",
    "sneaky_reasons": "Sneaky Reasons",
}

COLUMN_CONFIG = {
    "HR Score": st.column_config.ProgressColumn(
        "HR Score", help=GLOSSARY["HR Score"], min_value=0, max_value=100, format="%.1f"
    ),
    "HR Prob (game)": st.column_config.NumberColumn(
        "HR Prob (game)", help=GLOSSARY["HR Prob (game)"], format="%.1f%%"
    ),
    "xHR": st.column_config.NumberColumn("xHR", help=GLOSSARY["xHR"], format="%.2f"),
    "Fair Odds": st.column_config.NumberColumn("Fair Odds", help=GLOSSARY["Fair Odds"], format="%+d"),
    "Longshot": st.column_config.ProgressColumn(
        "Longshot", help=GLOSSARY["Longshot"], min_value=0, max_value=100, format="%.1f"
    ),
    "Consistency": st.column_config.ProgressColumn(
        "Consistency", help=GLOSSARY["Consistency"], min_value=0, max_value=100, format="%.1f"
    ),
    "Sneaky": st.column_config.ProgressColumn(
        "Sneaky", help=GLOSSARY["Sneaky"], min_value=0, max_value=100, format="%.1f"
    ),
    "Barrel%": st.column_config.NumberColumn("Barrel%", help=GLOSSARY["Barrel%"], format="%.1f"),
    "Hard-Hit%": st.column_config.NumberColumn("Hard-Hit%", help=GLOSSARY["Hard-Hit%"], format="%.1f"),
    "Avg EV": st.column_config.NumberColumn("Avg EV", help=GLOSSARY["Avg EV"], format="%.1f"),
    "Max EV": st.column_config.NumberColumn("Max EV", help=GLOSSARY["Max EV"], format="%.1f"),
    "xwOBA": st.column_config.NumberColumn("xwOBA", help=GLOSSARY["xwOBA"], format="%.3f"),
    "HR/PA": st.column_config.NumberColumn("HR/PA", format="%.3f"),
    "Park Factor": st.column_config.NumberColumn("Park Factor", help=GLOSSARY["Park Factor"], format="%.0f"),
    "Recent Form": st.column_config.ProgressColumn(
        "Recent Form", min_value=0, max_value=100, format="%.0f"
    ),
}


def prep_display(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    out = df[cols].rename(columns=DISPLAY_COLUMNS)
    if "HR Prob (game)" in out.columns:
        out["HR Prob (game)"] = out["HR Prob (game)"] * 100.0
    return out


def render_table(df: pd.DataFrame, columns: list[str], sort_col: str, key: str):
    disp = prep_display(df, columns)
    disp = disp.sort_values(sort_col, ascending=False) if sort_col in disp.columns else disp
    st.dataframe(
        disp,
        use_container_width=True,
        hide_index=True,
        height=min(620, 60 + 35 * len(disp)),
        column_config=COLUMN_CONFIG,
        key=key,
    )
    csv = disp.to_csv(index=False).encode()
    st.download_button(
        "⬇️ Export this view to CSV", csv, file_name=f"{key}.csv",
        mime="text/csv", key=f"dl_{key}",
    )


def leaderboard_cards(df: pd.DataFrame, score_col: str, label: str, n: int = 6):
    top = df.sort_values(score_col, ascending=False).head(n)
    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {row['player']}")
                st.caption(
                    f"{row['team']} vs {row['opponent']} · {row['position']} · "
                    f"bats {row['bats']} vs {row['pitcher_throws']}HP"
                )
                m1, m2, m3 = st.columns(3)
                m1.metric(label, f"{row[score_col]:.0f}")
                m2.metric("HR Prob", f"{row['hr_prob_game']*100:.0f}%")
                m3.metric("Barrel%", f"{row['barrel_pct']:.0f}")
                st.progress(min(1.0, row[score_col] / 100.0))
                if row.get("rationale"):
                    st.caption(f"💡 {row['rationale']}")


def metric_bar_chart(df: pd.DataFrame, score_col: str, title: str, n: int = 15):
    top = df.sort_values(score_col, ascending=False).head(n)
    chart = (
        alt.Chart(top)
        .mark_bar(color="#e63946")
        .encode(
            x=alt.X(f"{score_col}:Q", title=title),
            y=alt.Y("player:N", sort="-x", title=None),
            tooltip=["player", "team", "opponent", score_col, "hr_prob_game", "barrel_pct"],
        )
        .properties(height=28 * len(top))
    )
    st.altair_chart(chart, use_container_width=True)


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
def tab_longshots(df: pd.DataFrame):
    st.subheader("🚀 Best Longshots")
    st.caption(
        "High-upside, lower-probability boom-or-bust bats. Ranked by **explosiveness** "
        "(max EV + barrel% + favorable park/weather). Great for +EV HR props & DFS GPP."
    )
    leaderboard_cards(df, "longshot_score", "Longshot", n=6)
    st.markdown("##### Top 20 by Longshot Score")
    metric_bar_chart(df, "longshot_score", "Longshot Score", n=15)
    cols = ["player", "team", "opponent", "pitcher_name", "bats", "longshot_score",
            "hr_prob_game", "fair_odds", "max_ev", "barrel_pct", "park_factor",
            "wind_mult", "rationale"]
    render_table(df.sort_values("longshot_score", ascending=False).head(40),
                 cols, "Longshot", "longshots")


def tab_consistent(df: pd.DataFrame):
    st.subheader("🎯 Consistent HR Hitters")
    st.caption(
        "Reliable, high-floor power. Ranked by **consistency** (steady hard contact, "
        "low strikeouts, season HR pace, EV & xwOBA), weighted by sample size."
    )
    leaderboard_cards(df, "consistency_score", "Consistency", n=6)
    st.markdown("##### Top 20 by Consistency Score")
    metric_bar_chart(df, "consistency_score", "Consistency Score", n=15)
    cols = ["player", "team", "opponent", "pitcher_name", "bats", "consistency_score",
            "hr_score", "hr_prob_game", "hard_hit_pct", "barrel_pct", "avg_ev",
            "xwoba", "hr_per_pa", "rationale"]
    render_table(df.sort_values("consistency_score", ascending=False).head(40),
                 cols, "Consistency", "consistent")


def tab_sneaky(df: pd.DataFrame):
    st.subheader("🕵️ Sneaky Homerun Chances")
    st.caption(
        "Under-the-radar value plays: favorable matchup vs a hittable arm, hidden "
        "park/weather edge, or a hot streak not yet reflected in season stats."
    )
    sneaky = df[df["sneaky_reasons"].astype(str).str.len() > 0]
    sneaky = sneaky if not sneaky.empty else df
    leaderboard_cards(sneaky, "sneaky_score", "Sneaky", n=6)
    st.markdown("##### Why they're sneaky")
    cols = ["player", "team", "opponent", "pitcher_name", "sneaky_score",
            "hr_prob_game", "form_gap", "park_factor", "wind_mult",
            "barrel_pct", "sneaky_reasons"]
    render_table(sneaky.sort_values("sneaky_score", ascending=False).head(40),
                 cols, "Sneaky", "sneaky")


def tab_all(df: pd.DataFrame):
    st.subheader("📊 All Combined + Best Metrics")
    st.caption(
        "Master table for the full slate. Sort/filter by any column. The composite "
        "**HR Score** is the headline ranking."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hitters on slate", len(df))
    c2.metric("Games", df["game"].nunique())
    c3.metric("Avg HR Score", f"{df['hr_score'].mean():.1f}")
    top_row = df.sort_values("hr_score", ascending=False).iloc[0]
    c4.metric("Top bat", top_row["player"], f"{top_row['hr_score']:.0f}")

    st.markdown("##### 🏆 Top 20 overall by composite HR Score")
    metric_bar_chart(df, "hr_score", "HR Score", n=20)

    sort_options = {
        "HR Score": "hr_score", "HR Prob (game)": "hr_prob_game",
        "Longshot": "longshot_score", "Consistency": "consistency_score",
        "Sneaky": "sneaky_score", "Barrel%": "barrel_pct", "Max EV": "max_ev",
        "Park Factor": "park_factor",
    }
    sort_label = st.selectbox("Sort master table by", list(sort_options), index=0)
    sort_col = sort_options[sort_label]

    cols = list(DISPLAY_COLUMNS.keys())
    disp_sorted = df.sort_values(sort_col, ascending=False)
    render_table(disp_sorted, cols, DISPLAY_COLUMNS.get(sort_col, "HR Score"), "all_combined")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    game_date, prefer_live = sidebar_controls()
    methodology_sidebar()

    st.title("⚾ MLB Home Run Projection Tool")
    st.caption(
        "Ranked home-run upside built on Statcast-style batted-ball quality, recent "
        "form, pitcher matchup, park factors, and weather."
    )

    scored, source, notes = load_scored_slate(game_date.isoformat(), prefer_live)

    badge = "🟢" if source.startswith("LIVE") else "🟡"
    st.info(f"{badge} **Data source:** {source} — {game_date:%A, %B %-d, %Y}", icon=None)
    with st.expander("Data provenance & notes", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")
        st.markdown(
            "- **HR probabilities are model estimates, not guarantees.** Use for "
            "research/entertainment; not betting advice."
        )

    if scored is None or scored.empty:
        st.warning("No games/hitters available for this date.")
        return

    filtered = filter_controls(scored)
    if filtered.empty:
        st.warning("No hitters match the current filters.")
        return

    st.markdown(f"**{len(filtered)}** hitters across **{filtered['game'].nunique()}** games after filters.")

    t1, t2, t3, t4 = st.tabs(
        ["🚀 Best Longshots", "🎯 Consistent HR Hitters",
         "🕵️ Sneaky HR Chances", "📊 All Combined + Best Metrics"]
    )
    with t1:
        tab_longshots(filtered)
    with t2:
        tab_consistent(filtered)
    with t3:
        tab_sneaky(filtered)
    with t4:
        tab_all(filtered)


if __name__ == "__main__":
    main()
