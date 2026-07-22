"""MLB Home Run Projection Tool — Streamlit dashboard.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import datetime as dt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.history import (
    add_profile_similarity,
    build_hr_history,
    calibration_table,
    hr_profile_centroid,
    recent_trend,
    summarize_hr_profile,
    top5_by_category,
)
from src.model import (
    HR_SCORE_WEIGHTS,
    POWER_QUALITY_WEIGHTS,
    RECENT_FORM_WEIGHTS,
    hr_of_the_day,
    score_slate,
)
from src.learn import (
    attach_calibrated_prob,
    hit_rate_by_score,
    model_report_card,
)
from src.lineup import (
    attach_spot_signal,
    league_spot_table,
    player_spot_hr,
)
from src.odds import attach_odds, attach_prop_lines, format_american
from src.parlay import ROLE_EMOJI, generate_parlay, summarize_selection
from src.pitchers import attach_sp_spot_signal, sp_spot_counts_for
from src.props import BET_LABEL, BET_TYPES, attach_props, build_ladder_parlay
from src.sources import get_slate
from src.trends import (
    MID_HR_MIN, STAR_HR_MIN, TIER_UNDER,
    attach_trend_signals, compute_trends, rotation_hint, tier_of,
)

st.set_page_config(
    page_title="HR Hunter",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="auto",   # auto-collapses on phones for a full-screen feel
)


def inject_css():
    """Visual polish + a mobile-app feel (responsive layout, sticky scroll tabs)."""
    st.markdown(
        """
        <style>
          #MainMenu, footer, [data-testid="stToolbar"] {visibility: hidden;}
          .block-container {padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1400px;}
          h1 {font-weight: 800; letter-spacing: -0.5px;}
          /* Metric cards */
          [data-testid="stMetric"] {
            background: #161b26; border: 1px solid #232a38; border-radius: 12px;
            padding: 12px 14px;
          }
          [data-testid="stMetricLabel"] {opacity: .75; font-size: .8rem;}
          /* Bordered containers -> soft cards */
          div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 14px; border-color: #232a38 !important;
          }
          /* Tabs: pill-like, horizontally scrollable, sticky at the top */
          .stTabs [data-baseweb="tab-list"] {
            position: sticky; top: 0; z-index: 99; background: #0e1117;
            overflow-x: auto; scrollbar-width: none; gap: 2px;
            padding-bottom: 4px; scroll-snap-type: x proximity;
          }
          .stTabs [data-baseweb="tab-list"]::-webkit-scrollbar {display: none;}
          button[data-baseweb="tab"] {
            font-size: 0.95rem; font-weight: 600; white-space: nowrap;
            scroll-snap-align: start;
          }
          .stTabs [aria-selected="true"] {color: #ff5864 !important;}
          /* Buttons: big tap targets */
          .stDownloadButton button, .stButton button {border-radius: 10px; min-height: 42px;}
          /* Hero pick cards */
          .pickcard {background: linear-gradient(160deg,#1b2230,#141923);
            border: 1px solid #2a3346; border-radius: 16px; padding: 16px 18px; height: 100%;}
          .pickcard .lab {font-size:.72rem; text-transform:uppercase; letter-spacing:1px; opacity:.7;}
          .pickcard .name {font-size:1.18rem; font-weight:800; margin:.15rem 0;}
          .pickcard .sub {opacity:.8; font-size:.85rem;}
          .pickcard .big {font-size:1.5rem; font-weight:800; color:#ff5864;}
          .role-pill {display:inline-block; padding:2px 9px; border-radius:999px;
            font-size:.72rem; font-weight:700; margin-right:6px;}

          /* ---- Mobile (phones) ---- */
          @media (max-width: 640px) {
            .block-container {padding: 1rem 0.7rem 4.5rem !important;}
            h1 {font-size: 1.7rem !important; line-height: 1.15;}
            h2 {font-size: 1.25rem !important;}
            h3 {font-size: 1.08rem !important;}
            [data-testid="stMetricValue"] {font-size: 1.35rem !important;}
            button[data-baseweb="tab"] {font-size: 0.9rem; padding: 6px 10px !important;}
            .pickcard {padding: 13px 14px; border-radius: 14px;}
            .pickcard .name {font-size: 1.05rem;}
            .pickcard .big {font-size: 1.3rem;}
            /* tighten dataframes + make controls full width */
            .stButton button, .stDownloadButton button {width: 100%;}
            .stSlider, .stSelectbox, .stMultiSelect {margin-bottom: .2rem;}
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner="Building & scoring the slate…", ttl=60 * 30)
def load_scored_slate(date_iso: str, prefer_live: bool):
    game_date = dt.date.fromisoformat(date_iso)
    df, source, notes = get_slate(game_date, prefer_live=prefer_live)
    scored = score_slate(df)
    scored["player_tier"] = (scored["season_hr"].map(tier_of)
                             if "season_hr" in scored.columns else TIER_UNDER)
    return scored, source, notes


@st.cache_data(show_spinner="Loading pitcher HR-by-spot…", ttl=60 * 60)
def load_sp_spot_counts(date_iso: str, prefer_live: bool, pairs: tuple):
    """Cache {(game, pitcher): HR-by-spot} for the slate's probable starters."""
    return sp_spot_counts_for(pairs, date_iso, prefer_live)


@st.cache_data(show_spinner="Analyzing trailing-month HR history…", ttl=60 * 60)
def load_hr_history(start_iso: str, end_iso: str, prefer_live: bool, half_life_days: float):
    events, slate_hist, source, notes = build_hr_history(start_iso, end_iso, prefer_live)
    summary = summarize_hr_profile(events, slate_hist)
    # Recency-weighted centroid: recent HR hitters define "what's working now".
    centroid = hr_profile_centroid(events, end_date_iso=end_iso, half_life_days=half_life_days)
    calib = calibration_table(slate_hist)
    trend = recent_trend(events, end_iso, recent_days=7)
    # Lineup-spot HR data: aggregate the recurring log (REAL graded hitter-days,
    # grown daily by the GitHub Actions workflow from the eval record). The app
    # never writes the log — slate_hist is always simulated (see history.py),
    # so writing here would fill the log with fake rows.
    player_spot = player_spot_hr(slate_hist)
    league_spot = league_spot_table(slate_hist)
    # Self-calibration: empirical HR rate by model rating + a model report card.
    score_curve = hit_rate_by_score(slate_hist)
    report = model_report_card(events, slate_hist)
    return (events, summary, centroid, calib, trend, player_spot, league_spot,
            score_curve, report, source, notes)


# --------------------------------------------------------------------------- #
# Tooltip / metric glossary
# --------------------------------------------------------------------------- #
GLOSSARY = {
    "HR Score": "Composite 0-100 rating blending batted-ball quality, season & recent HR rate, matchup, and park/weather.",
    "HR Prob (game)": "Probability the hitter hits ≥1 HR in this game (per-PA rate compounded over ~4.1 PA). When live odds are on, it's sharpened by the market: nudged by the game's over/under total and blended 35% toward the de-vigged book price where one exists.",
    "Game Total": "The market's over/under run total for this game (live, median across books). Prices in park, weather, both pitchers and news — high totals lift every HR probability in the game a little.",
    "Opp Pen HR/9": "The OPPOSING bullpen's HR/9 (FanGraphs, relievers only). ~35% of a hitter's PAs come after the starter departs, so a leaky pen is a real part of the matchup.",
    "vs-Hand wOBA": "The hitter's REAL wOBA vs today's pitcher hand (Statcast, trailing 45 days, sample-gated) — replaces the flat platoon bonus with the bat's actual split.",
    "xHR (game)": "Expected home runs in THIS game = per-PA HR rate × expected PA.",
    "Fair Odds": "Vig-free American odds implied by the model's game HR probability.",
    "Calib HR%": "Self-calibrated HR probability — the ACTUAL HR rate that bats with this pre-game HR Score produced over the trailing window. The system relearns this each run.",
    "Calib Edge": "Calibrated HR% minus model HR%. Positive = history says this rating homers more than the model credits; the parlay builder leans into it.",
    "Book Odds": "American odds to hit ≥1 HR you'd actually bet — LIVE from a sportsbook when an ODDS_API_KEY is configured, else a model-implied market price (fair price + typical hold) clamped into the player's tier band: ⭐ Stars +200..+450, 🔷 Mid +500..+700, 🎯 Under +700 and up — so offline prices look like real board prices.",
    "Tier": f"Player tier by season HR total — ⭐ Star ≥{STAR_HR_MIN} HR (books ~+200 to +450, the Anchors), 🔷 Mid {MID_HR_MIN}-{STAR_HR_MIN-1} HR (~+500 to +700, the Value bats), 🎯 Under ≤{MID_HR_MIN-1} HR (+700 and up, the Longshots). Drives parlay roles and model-implied odds.",
    "Edge%": "Model HR% minus the book's implied HR%. Positive = +EV (model thinks the bat is underpriced). With model-implied odds it sits near −hold (the vig).",
    "Longshot": "Boom-or-bust ceiling score: max exit velo + barrel% + park/weather, rewarding high-variance upside.",
    "Consistency": "High-floor score: hard-hit%, contact (low K), season HR rate, EV & xwOBA, weighted by sample size.",
    "Sneaky": "Under-the-radar value: strong matchup/park + recent surge vs season line + lower-profile bat.",
    "ULX": "ULX power-checklist grade — 🟢 GREEN (≥7 of 9 minimums met, run it), 🟡 YELLOW (4-6, consider), 🔴 RED (<4, fade). Bet the profile, not the name.",
    "ULX ✓": "How many of the 9 ULX power minimums the bat meets: Barrel%≥8, Hard-Hit%≥40, xSLG≥.450, ISO≥.160, Sweet-Spot%≥30, AvgEV≥88, Launch 10-28°, Pull%≥35, HR/FB≥12.",
    "ISO": "Isolated power (SLG − AVG) — raw power output. ULX longshot minimum ≥ .160.",
    "Sweet-Spot%": "Share of batted balls in the 8-32° launch-angle sweet spot (real, Statcast). ULX minimum ≥ 30%.",
    "Barrel%": "Share of batted balls hit with the ideal EV/launch-angle combo for extra-base damage (best HR predictor). ULX minimum ≥ 8%.",
    "Barrel/PA%": "Barrels per plate appearance (real, Statcast) — barrel rate scaled by how often the bat puts a ball in play; an elite season-long HR signal.",
    "xHR (season)": "Season expected home runs from batted-ball quality (barrels/PA + fly-ball rate). The gap vs actual HR flags luck/regression.",
    "HR−xHR": "Actual HR minus expected HR. Negative = under-performing the quality of contact (a positive-regression / 'due' candidate, used in the Sneaky score).",
    "Sprint": "Sprint speed in ft/s (real, Statcast) — athletic context. Shown for color; it has no measurable effect on HR power, so xHR is NOT sprint-adjusted.",
    "Bat Speed": "Average swing speed in mph (real, Statcast bat-tracking — the newest public data). Exit velo says what HAPPENED on contact; bat speed says what a hitter is CAPABLE of. League average ~72 mph; a rising swing-speed trend often precedes a HR surge. Feeds power quality.",
    "Squared-Up%": "Share of swings squared up (real, Statcast bat-tracking) — how often the barrel meets the ball flush. High = premium contact quality.",
    "Fast-Swing%": "Share of swings at 75+ mph (real, Statcast bat-tracking) — how often a hitter really lets it rip; tracks raw power intent.",
    "Pitch Matchup": "Pitch-mix-weighted hitter performance vs the probable pitcher's arsenal (vs fastball/breaking/offspeed × the pitcher's usage). Higher = a better arsenal matchup.",
    "vs FB": "Hitter's wOBA-like performance vs fastballs (modeled; real version pulls Statcast run value by pitch type).",
    "vs BR": "Hitter's wOBA-like performance vs breaking balls (modeled).",
    "vs OS": "Hitter's wOBA-like performance vs offspeed pitches (modeled).",
    "Hard-Hit%": "Share of batted balls ≥95 mph exit velocity.",
    "Whiff%": "Swing-and-miss rate = swings that miss / total swings (real, from FanGraphs Contact%). High whiff = more boom-or-bust, lower contact floor.",
    "Contact%": "Contact rate = contact made / swings (real, from FanGraphs); the complement of Whiff%. High contact = better bat-to-ball skill / higher floor.",
    "Chase%": "O-Swing% — share of pitches OUTSIDE the zone the hitter swings at (real, FanGraphs). Higher chase = more volatile / boom-or-bust.",
    "Zone-Contact%": "Z-Contact% — contact rate on swings at pitches INSIDE the zone (real, FanGraphs). The cleanest repeatable-contact / floor signal.",
    "Fly-Ball%": "FB% — share of batted balls hit in the air (real, FanGraphs). Fly balls are the raw material of home runs, so above-average FB% earns a direct HR-rate boost.",
    "Ground-Ball%": "GB% — share of batted balls on the ground (real, FanGraphs). Grounders almost never leave the yard, so high GB% suppresses HR upside.",
    "Line-Drive%": "LD% — share of batted balls hit as line drives (real, FanGraphs). Great for hits, but line drives are usually too low to clear the wall.",
    "Pull%": "Share of batted balls hit to the pull side (real, FanGraphs). Pulled fly balls clear the wall most often, so pull power lifts the HR ceiling.",
    "HR/FB": "Home runs per fly ball (real, FanGraphs) — the fly-ball→HR conversion rate; a direct measure of game power. Drives a dedicated HR-rate multiplier.",
    "xISO": "Expected Isolated Power = xSLG − xBA (real, Statcast) — pure expected power based on quality of contact, independent of luck/defense. Feeds the power-quality score.",
    "xSLG": "Expected slugging (real, Statcast) — what a hitter's batted-ball quality should produce, regardless of outcomes.",
    "Avg EV": "Average exit velocity (mph).",
    "Max EV": "Top-end exit velocity (mph) — a raw-power ceiling indicator.",
    "xwOBA": "Expected weighted on-base average from quality of contact.",
    "Park Factor": "Handedness-aware HR park factor (100 = average; 110 = +10% HR).",
    "Porch Fit ×": "Real fence geometry × THIS hitter's pull side: his pull-field fence distance (wall height folded in — Fenway's 310-ft line plays deep behind the 37-ft Monster) vs the ~328-ft league norm, scaled by how pull-heavy he is. A dead-pull lefty gets the full boost of a short right-field porch; a spray hitter barely notices; >1.00 = the park's shape helps him.",
    "Porch (ft)": "The hitter's pull-side fence distance in this park, adjusted for wall height (~0.6 ft per extra foot of wall).",
    "Day/Night ×": "Start-time park effect (multiplier on the park factor). Some parks play very differently by first-pitch time: the marine layer settles at night in SF/SD/OAK (ball dies), Wrigley day games with the wind out are a launch pad. Roofed/domed parks are climate-controlled and stay 1.00. Derived from the game's real first-pitch time.",
    "Series Game": "Which game of the current series this is (1-4). Hitters see the SAME pitching staff on consecutive days and measurably improve within a series as they re-see arms and pitch shapes — so game 3 is a quietly better HR spot than game 1. Rarely modeled publicly.",
    "Spot": "Batting-order spot (1-9) for this game — live from the posted lineup when available, else estimated. Top-of-order bats get more PAs.",
    "xPA": "Expected plate appearances given the lineup spot (top of order bats more) — feeds the game HR probability.",
    "SP Meatball%": "The opposing starter's rate of MIDDLE-MIDDLE pitches (Statcast zone 5) per 100, over the trailing month. HRs are hit off mistakes, and mistake SUPPLY varies ~2x between starters — this is a less noisy, less park-polluted HR-risk signal than HR/9. League average ~5%; higher = more grooved pitches to punish.",
    "SP Velo Δ": "The opposing starter's average fastball velocity in his MOST RECENT start minus his season-window baseline (mph). Down 1+ mph is the best public early-warning of fatigue/injury — HR rates spike against diminished velo before ERA catches up, and books adjust slowly. Negative = losing giddy-up (a HR spot); positive = fresh/ramping.",
    "SP 3rd-Time Δ": "How much MORE the opposing starter gets hit the 3rd time through the order — his wOBA-allowed on the 3rd+ pass minus his 1st/2nd (Statcast n_thruorder_pitcher). Positive = he fades. Only top-of-order bats (spots 1-4) reliably reach that 3rd look before the bullpen, so this boosts THEM, not the bottom of the order.",
    "SP Auto-FB%": "The opposing starter's fastball rate in HITTER'S COUNTS (2-0, 3-1, 3-0, 2-1) — league average ~55%. A high number means he's predictable when behind, letting a hitter who mashes fastballs sit dead-red. The model rewards the interaction of this × the batter's damage vs fastballs.",
    "SP HRs@Spot": "HRs the opposing starting pitcher has allowed to THIS hitter's lineup spot over their last 10 games — a juicy-spot matchup signal that also boosts the bat in parlay selection.",
    "HRs@Spot": "HRs this hitter has hit from today's lineup spot in the recurring HR-by-spot log (data before the selected date).",
    "HR/G@Spot": "HR per game this hitter has produced from today's lineup spot (recurring log) — a parlay role-fit nudge.",
    "Profile Match": "How closely a hitter resembles the trailing-month HR-hitter profile (barrel%, EV, max EV, launch angle, park) — 100 = a dead-ringer for recent HR hitters.",
    "Calibrated": "HR Score nudged by recent-HR Profile Match (85% HR Score + 15% Profile Match).",
}


# --------------------------------------------------------------------------- #
# Sidebar — controls & methodology
# --------------------------------------------------------------------------- #
def sidebar_controls():
    st.sidebar.title("⚾ HR Hunter")
    st.sidebar.caption("Ranked home-run upside for any MLB slate")

    game_date = st.sidebar.date_input(
        "📅 Game date", value=dt.date.today(),
        help="Pick any MLB date. Defaults to today.",
    )
    simple = st.sidebar.toggle(
        "✨ Simple view", value=True,
        help="Show just the key columns (Player, odds, HR%, a couple of metrics). "
             "Turn off for the full metric set.",
    )
    st.session_state["simple_view"] = simple

    prefer_live = st.sidebar.toggle(
        "🛰️ Try live data", value=True,
        help="Pull the real schedule, lineups, Statcast metrics and weather. Falls "
             "back to a deterministic demo slate if the network is unavailable.",
    )

    with st.sidebar.expander("⚙️ Advanced settings", expanded=False):
        live_odds = st.toggle(
            "Use live HR odds (needs ODDS_API_KEY)", value=False,
            help="Real sportsbook HR odds from The Odds API when an ODDS_API_KEY "
                 "env var is set. Otherwise odds are model-implied (fair + hold).",
        )
        lookback = st.slider(
            "Backtest lookback (days)", min_value=7, max_value=45, value=31, step=1,
            help="Window of past HRs analyzed for the Trends tab (~1 month default).",
        )
        half_life = st.slider(
            "Trend recency half-life (days)", min_value=2, max_value=30, value=10, step=1,
            help="How fast older HRs fade in the Profile Match. Lower = more reactive "
                 "to the hottest recent profiles.",
        )

    if st.sidebar.button("🔄 Refresh data", use_container_width=True):
        load_scored_slate.clear()
        load_hr_history.clear()
        st.rerun()

    return game_date, prefer_live, lookback, half_life, live_odds


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
    "woba_vs_hand": "vs-Hand wOBA",
    "bullpen_hr9": "Opp Pen HR/9",
    "game_total": "Game Total",
    "bats": "Bats",
    "position": "Pos",
    "lineup_spot": "Spot",
    "player_tier": "Tier",
    "ulx_grade": "ULX",
    "ulx_checks": "ULX ✓",
    "hr_score": "HR Score",
    "calibrated_score": "Calibrated",
    "profile_match": "Profile Match",
    "hr_prob_game": "HR Prob (game)",
    "calibrated_hr_prob": "Calib HR%",
    "cal_edge_pct": "Calib Edge",
    "xhr": "xHR (game)",
    "fair_odds": "Fair Odds",
    "book_odds": "Book Odds",
    "edge_pct": "Edge%",
    "longshot_score": "Longshot",
    "consistency_score": "Consistency",
    "sneaky_score": "Sneaky",
    "barrel_pct": "Barrel%",
    "brl_pa": "Barrel/PA%",
    "hard_hit_pct": "Hard-Hit%",
    "whiff_pct": "Whiff%",
    "contact_pct": "Contact%",
    "chase_pct": "Chase%",
    "zone_contact_pct": "Zone-Contact%",
    "fb_pct": "Fly-Ball%",
    "gb_pct": "Ground-Ball%",
    "ld_pct": "Line-Drive%",
    "pull_pct": "Pull%",
    "hr_fb": "HR/FB",
    "avg_ev": "Avg EV",
    "max_ev": "Max EV",
    "xwoba": "xwOBA",
    "xiso": "xISO",
    "xslg": "xSLG",
    "iso": "ISO",
    "sweet_spot_pct": "Sweet-Spot%",
    "xhr_season": "xHR (season)",
    "hr_minus_xhr": "HR−xHR",
    "sprint_speed": "Sprint",
    "bat_speed": "Bat Speed",
    "squared_up_pct": "Squared-Up%",
    "fast_swing_pct": "Fast-Swing%",
    "pitch_matchup_score": "Pitch Matchup",
    "vs_fb": "vs FB",
    "vs_br": "vs BR",
    "vs_os": "vs OS",
    "hr_per_pa": "HR/PA",
    "expected_pa": "xPA",
    "sp_meatball_pct": "SP Meatball%",
    "sp_velo_delta": "SP Velo Δ",
    "sp_tto_penalty": "SP 3rd-Time Δ",
    "sp_hitter_count_fb": "SP Auto-FB%",
    "sp_hr_at_spot": "SP HRs@Spot",
    "spot_hr_at_current": "HRs@Spot",
    "spot_hr_rate": "HR/G@Spot",
    "park_factor": "Park Factor",
    "park_fit_mult": "Porch Fit ×",
    "park_porch_ft": "Porch (ft)",
    "daynight_mult": "Day/Night ×",
    "series_game": "Series Game",
    "wind_mult": "Wind x",
    "temp_f": "Temp °F",
    "recent_form_score": "Recent Form",
    "data_quality": "Data",
    "rationale": "Rationale",
    "sneaky_reasons": "Sneaky Reasons",
}

# Raw column keys kept in "Simple view" — the at-a-glance essentials. Detailed
# view shows everything. The active headline/sort column is always added back.
ESSENTIAL_KEYS = {
    "player", "team", "opponent", "pitcher_name", "lineup_spot", "player_tier",
    "ulx_grade", "ulx_checks",
    "hr_score", "hr_prob_game", "calibrated_hr_prob", "cal_edge_pct",
    "book_odds", "edge_pct", "fair_odds",
    "longshot_score", "consistency_score", "sneaky_score",
    "barrel_pct", "max_ev", "park_factor", "role", "sp_hr_at_spot",
    "rationale", "sneaky_reasons",
}

COLUMN_CONFIG = {
    "HR Score": st.column_config.ProgressColumn(
        "HR Score", help=GLOSSARY["HR Score"], min_value=0, max_value=100, format="%.1f"
    ),
    "Calibrated": st.column_config.ProgressColumn(
        "Calibrated", help=GLOSSARY["Calibrated"], min_value=0, max_value=100, format="%.1f"
    ),
    "Profile Match": st.column_config.NumberColumn(
        "Profile Match", help=GLOSSARY["Profile Match"], format="%.0f"
    ),
    "HR Prob (game)": st.column_config.NumberColumn(
        "HR Prob (game)", help=GLOSSARY["HR Prob (game)"], format="%.1f%%"
    ),
    "Calib HR%": st.column_config.NumberColumn("Calib HR%", help=GLOSSARY["Calib HR%"], format="%.0f%%"),
    "Calib Edge": st.column_config.NumberColumn("Calib Edge", help=GLOSSARY["Calib Edge"], format="%+.1f"),
    "Tier": st.column_config.TextColumn("Tier", help=GLOSSARY["Tier"]),
    "vs-Hand wOBA": st.column_config.NumberColumn("vs-Hand wOBA", help=GLOSSARY["vs-Hand wOBA"], format="%.3f"),
    "Opp Pen HR/9": st.column_config.NumberColumn("Opp Pen HR/9", help=GLOSSARY["Opp Pen HR/9"], format="%.2f"),
    "Game Total": st.column_config.NumberColumn("Game Total", help=GLOSSARY["Game Total"], format="%.1f"),
    "ULX": st.column_config.TextColumn("ULX", help=GLOSSARY["ULX"]),
    "ULX ✓": st.column_config.NumberColumn("ULX ✓", help=GLOSSARY["ULX ✓"], format="%d/9"),
    "ISO": st.column_config.NumberColumn("ISO", help=GLOSSARY["ISO"], format="%.3f"),
    "Sweet-Spot%": st.column_config.NumberColumn("Sweet-Spot%", help=GLOSSARY["Sweet-Spot%"], format="%.1f"),
    "xHR (game)": st.column_config.NumberColumn("xHR (game)", help=GLOSSARY["xHR (game)"], format="%.2f"),
    "xHR (season)": st.column_config.NumberColumn("xHR (season)", help=GLOSSARY["xHR (season)"], format="%.1f"),
    "HR−xHR": st.column_config.NumberColumn("HR−xHR", help=GLOSSARY["HR−xHR"], format="%+.1f"),
    "Sprint": st.column_config.NumberColumn("Sprint", help=GLOSSARY["Sprint"], format="%.1f"),
    "Bat Speed": st.column_config.NumberColumn("Bat Speed", help=GLOSSARY["Bat Speed"], format="%.1f"),
    "Squared-Up%": st.column_config.NumberColumn("Squared-Up%", help=GLOSSARY["Squared-Up%"], format="%.1f"),
    "Fast-Swing%": st.column_config.NumberColumn("Fast-Swing%", help=GLOSSARY["Fast-Swing%"], format="%.1f"),
    "Pitch Matchup": st.column_config.ProgressColumn("Pitch Matchup", help=GLOSSARY["Pitch Matchup"], min_value=0, max_value=100, format="%.0f"),
    "vs FB": st.column_config.NumberColumn("vs FB", help=GLOSSARY["vs FB"], format="%.3f"),
    "vs BR": st.column_config.NumberColumn("vs BR", help=GLOSSARY["vs BR"], format="%.3f"),
    "vs OS": st.column_config.NumberColumn("vs OS", help=GLOSSARY["vs OS"], format="%.3f"),
    "Fair Odds": st.column_config.NumberColumn("Fair Odds", help=GLOSSARY["Fair Odds"], format="%+d"),
    "Book Odds": st.column_config.NumberColumn("Book Odds", help=GLOSSARY["Book Odds"], format="%+d"),
    "Edge%": st.column_config.NumberColumn("Edge%", help=GLOSSARY["Edge%"], format="%+.1f"),
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
    "Barrel/PA%": st.column_config.NumberColumn("Barrel/PA%", help=GLOSSARY["Barrel/PA%"], format="%.1f"),
    "Hard-Hit%": st.column_config.NumberColumn("Hard-Hit%", help=GLOSSARY["Hard-Hit%"], format="%.1f"),
    "Whiff%": st.column_config.NumberColumn("Whiff%", help=GLOSSARY["Whiff%"], format="%.1f"),
    "Contact%": st.column_config.NumberColumn("Contact%", help=GLOSSARY["Contact%"], format="%.1f"),
    "Chase%": st.column_config.NumberColumn("Chase%", help=GLOSSARY["Chase%"], format="%.1f"),
    "Zone-Contact%": st.column_config.NumberColumn("Zone-Contact%", help=GLOSSARY["Zone-Contact%"], format="%.1f"),
    "Fly-Ball%": st.column_config.NumberColumn("Fly-Ball%", help=GLOSSARY["Fly-Ball%"], format="%.1f"),
    "Ground-Ball%": st.column_config.NumberColumn("Ground-Ball%", help=GLOSSARY["Ground-Ball%"], format="%.1f"),
    "Line-Drive%": st.column_config.NumberColumn("Line-Drive%", help=GLOSSARY["Line-Drive%"], format="%.1f"),
    "Pull%": st.column_config.NumberColumn("Pull%", help=GLOSSARY["Pull%"], format="%.1f"),
    "HR/FB": st.column_config.NumberColumn("HR/FB", help=GLOSSARY["HR/FB"], format="%.1f"),
    "Avg EV": st.column_config.NumberColumn("Avg EV", help=GLOSSARY["Avg EV"], format="%.1f"),
    "Max EV": st.column_config.NumberColumn("Max EV", help=GLOSSARY["Max EV"], format="%.1f"),
    "xwOBA": st.column_config.NumberColumn("xwOBA", help=GLOSSARY["xwOBA"], format="%.3f"),
    "xISO": st.column_config.NumberColumn("xISO", help=GLOSSARY["xISO"], format="%.3f"),
    "xSLG": st.column_config.NumberColumn("xSLG", help=GLOSSARY["xSLG"], format="%.3f"),
    "HR/PA": st.column_config.NumberColumn("HR/PA", format="%.3f"),
    "Park Factor": st.column_config.NumberColumn("Park Factor", help=GLOSSARY["Park Factor"], format="%.0f"),
    "Porch Fit ×": st.column_config.NumberColumn("Porch Fit ×", help=GLOSSARY["Porch Fit ×"], format="%.3f"),
    "Porch (ft)": st.column_config.NumberColumn("Porch (ft)", help=GLOSSARY["Porch (ft)"], format="%.0f"),
    "Day/Night ×": st.column_config.NumberColumn("Day/Night ×", help=GLOSSARY["Day/Night ×"], format="%.3f"),
    "Series Game": st.column_config.NumberColumn("Series Game", help=GLOSSARY["Series Game"], format="%d"),
    "Spot": st.column_config.NumberColumn("Spot", help=GLOSSARY["Spot"], format="%d"),
    "xPA": st.column_config.NumberColumn("xPA", help=GLOSSARY["xPA"], format="%.1f"),
    "SP Meatball%": st.column_config.NumberColumn("SP Meatball%", help=GLOSSARY["SP Meatball%"], format="%.1f"),
    "SP Velo Δ": st.column_config.NumberColumn("SP Velo Δ", help=GLOSSARY["SP Velo Δ"], format="%+.1f"),
    "SP 3rd-Time Δ": st.column_config.NumberColumn("SP 3rd-Time Δ", help=GLOSSARY["SP 3rd-Time Δ"], format="%+.3f"),
    "SP Auto-FB%": st.column_config.NumberColumn("SP Auto-FB%", help=GLOSSARY["SP Auto-FB%"], format="%.0f"),
    "SP HRs@Spot": st.column_config.NumberColumn("SP HRs@Spot", help=GLOSSARY["SP HRs@Spot"], format="%d"),
    "HRs@Spot": st.column_config.NumberColumn("HRs@Spot", help=GLOSSARY["HRs@Spot"], format="%d"),
    "HR/G@Spot": st.column_config.NumberColumn("HR/G@Spot", help=GLOSSARY["HR/G@Spot"], format="%.2f"),
    "Recent Form": st.column_config.ProgressColumn(
        "Recent Form", min_value=0, max_value=100, format="%.0f"
    ),
}


def prep_display(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cols = [c for c in columns if c in df.columns]
    out = df[cols].rename(columns=DISPLAY_COLUMNS)
    if "HR Prob (game)" in out.columns:
        out["HR Prob (game)"] = out["HR Prob (game)"] * 100.0
    if "Calib HR%" in out.columns:
        out["Calib HR%"] = out["Calib HR%"] * 100.0
    return out


def _simplify(columns: list[str], sort_col: str) -> list[str]:
    """In Simple view, keep only essential columns (plus the active sort col)."""
    if not st.session_state.get("simple_view", True):
        return columns
    # The sort_col is a display label; keep its raw key so sorting still works.
    keep_raw = {k for k, v in DISPLAY_COLUMNS.items() if v == sort_col}
    return [c for c in columns if c in ESSENTIAL_KEYS or c in keep_raw]


def render_table(df: pd.DataFrame, columns: list[str], sort_col: str, key: str):
    disp = prep_display(df, _simplify(columns, sort_col))
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
def _fmt(v, nd=1, suffix=""):
    """Format a metric value; '—' when missing/NaN."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return "—"
    try:
        return f"{float(v):.{nd}f}{suffix}"
    except (TypeError, ValueError):
        return str(v)


def _top3_card(rank_emoji: str, row: pd.Series, headline: str, headline_val: str):
    """One Top-3 card: matchup, odds, tier, key power metrics + the why."""
    with st.container(border=True):
        st.markdown(f"### {rank_emoji} {row['player']}")
        spot = row.get("lineup_spot")
        spot_txt = f" · batting {int(spot)}" if pd.notna(spot) else ""
        st.caption(
            f"{row.get('player_tier','')} · {row['team']} vs {row['opponent']}{spot_txt} · "
            f"vs {row.get('pitcher_throws','R')}HP {row.get('pitcher_name','—')}"
        )
        m1, m2, m3 = st.columns(3)
        m1.metric(headline, headline_val)
        m2.metric("HR Prob", f"{row['hr_prob_game']*100:.0f}%")
        m3.metric("Odds", format_american(row.get("book_odds")))
        st.caption(
            f"Barrel% **{_fmt(row.get('barrel_pct'))}** · "
            f"Hard-Hit% **{_fmt(row.get('hard_hit_pct'))}** · "
            f"FB% **{_fmt(row.get('fb_pct'))}** · "
            f"HR/FB **{_fmt(row.get('hr_fb'))}** · "
            f"Max EV **{_fmt(row.get('max_ev'))}** · "
            f"ULX {row.get('ulx_grade','—')}"
        )
        if row.get("rationale"):
            st.caption(f"💡 {row['rationale']}")


def tab_top3(df: pd.DataFrame):
    """The daily shortlist: top 3 HR picks, top 3 value plays, top 3 longshots."""
    from src.parlay import enrich
    st.subheader("🏆 Today's Top 3s")
    st.caption(
        "The daily shortlist — **Top 3 HR picks** (highest-confidence bats), "
        "**Top 3 Value plays** (🔷 mid-tier bats the books underprice, ~+500 to "
        "+700), and **Top 3 Longshots** (🎯 under-the-radar ceiling, +700 and up). "
        "No player appears twice."
    )
    e = enrich(df)
    used: set = set()

    def take3(pool: pd.DataFrame, sort_col: str):
        picks = pool[~pool["player"].isin(used)].sort_values(
            sort_col, ascending=False).head(3)
        used.update(picks["player"])
        return picks

    medals = ["🥇", "🥈", "🥉"]

    st.markdown("#### ⚾ Top 3 HR Picks")
    picks = take3(e, "hr_score")
    cols = st.columns(3)
    for i, (_, row) in enumerate(picks.iterrows()):
        with cols[i]:
            _top3_card(medals[i], row, "HR Score", f"{row['hr_score']:.0f}")

    st.markdown("#### 💰 Top 3 Value Plays")
    val_pool = e[e["role"] == "Value"]
    if len(val_pool) < 3:                       # thin slate: widen to mid-prob bats
        val_pool = e[e["hr_prob_game"] >= 0.08]
    val_pool = val_pool.assign(
        _v=val_pool["sneaky_score"] + 0.5 * val_pool["edge_pct"].fillna(0))
    picks = take3(val_pool, "_v")
    cols = st.columns(3)
    for i, (_, row) in enumerate(picks.iterrows()):
        with cols[i]:
            _top3_card(medals[i], row, "Sneaky", f"{row['sneaky_score']:.0f}")

    st.markdown("#### 🚀 Top 3 Longshots")
    ls_pool = e[e["role"] == "Longshot"]
    if len(ls_pool) < 3:
        ls_pool = e[e["book_odds"] >= 700]
    picks = take3(ls_pool, "longshot_score")
    cols = st.columns(3)
    for i, (_, row) in enumerate(picks.iterrows()):
        with cols[i]:
            _top3_card(medals[i], row, "Longshot", f"{row['longshot_score']:.0f}")

    st.caption("Roles follow the tier system — ⭐ stars anchor, 🔷 mid bats are the "
               "value band, 🎯 unknowns are the longshots. Full boards live in the "
               "other Picks views; build the ticket in 🎰 Parlays.")


def tab_longshots(df: pd.DataFrame):
    st.subheader("🚀 Best Longshots")
    st.caption(
        "High-upside, lower-probability boom-or-bust bats. Ranked by **explosiveness** "
        "(max EV + barrel% + favorable park/weather). Great for +EV HR props & DFS GPP."
    )
    leaderboard_cards(df, "longshot_score", "Longshot", n=6)
    st.markdown("##### Top 20 by Longshot Score")
    metric_bar_chart(df, "longshot_score", "Longshot Score", n=15)
    cols = ["player", "team", "opponent", "pitcher_name", "bats", "lineup_spot",
            "ulx_grade", "ulx_checks", "longshot_score", "hr_prob_game", "book_odds",
            "edge_pct", "sp_hr_at_spot", "max_ev", "barrel_pct", "iso",
            "sweet_spot_pct", "hr_fb", "pull_pct", "park_factor", "wind_mult", "rationale"]
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
            "hr_score", "hr_prob_game", "hard_hit_pct", "barrel_pct", "brl_pa", "avg_ev",
            "contact_pct", "zone_contact_pct", "xwoba", "xiso", "hr_per_pa", "rationale"]
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
    cols = ["player", "team", "opponent", "pitcher_name", "lineup_spot", "sneaky_score",
            "hr_prob_game", "sp_hr_at_spot", "form_gap", "hr_minus_xhr",
            "pitch_matchup_score", "park_factor", "wind_mult", "barrel_pct", "sneaky_reasons"]
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


def _top5_card_grid(tops: dict):
    """Render the four category top-5 lists as compact tables."""
    for label, t in tops.items():
        st.markdown(f"#### {label}")
        disp = t.copy()
        if "hr_prob_game" in disp:
            disp["hr_prob_game"] = (disp["hr_prob_game"] * 100).round(0)
        rename = {
            "player": "Player", "team": "Team", "opponent": "Opp",
            "pitcher_name": "Pitcher", "hr_prob_game": "HR Prob %",
            "profile_match": "Profile Match", "calibrated_score": "Calibrated",
            "hr_score": "HR Score", "longshot_score": "Longshot",
            "consistency_score": "Consistency", "sneaky_score": "Sneaky",
            "barrel_pct": "Barrel%", "max_ev": "Max EV", "park_factor": "Park",
        }
        st.dataframe(disp.rename(columns=rename), hide_index=True,
                     use_container_width=True)


def _model_verdict(score) -> str:
    """Did the model like this bat pre-game, given its HR Score?"""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "—"
    if s >= 60:
        return "✅ Loved"
    if s >= 48:
        return "👍 Liked"
    if s >= 35:
        return "😐 Lukewarm"
    return "⚠️ Missed"


def render_hr_stat_sheet(events, start_iso, end_iso):
    """The line-by-line previous-HR stat sheet (with lineup spot) + per-HR detail."""
    if events is None or events.empty:
        st.warning("No HR history available for this window.")
        return
    sheet = events.copy()
    spot_opts = (sorted(int(s) for s in sheet["lineup_spot"].dropna().unique())
                 if "lineup_spot" in sheet.columns else [])
    team_opts = sorted(sheet["team"].dropna().unique()) if "team" in sheet.columns else []
    fc1, fc2, fc3 = st.columns([1, 1, 1])
    sel_spots = fc1.multiselect("Lineup spot", spot_opts, default=[],
                                help="e.g. pick 5 and 6 to see HRs hit from the 5-6 holes.")
    sel_teams = fc2.multiselect("Team", team_opts, default=[])
    name_q = fc3.text_input("Player contains", "")
    if sel_spots and "lineup_spot" in sheet.columns:
        sheet = sheet[sheet["lineup_spot"].isin(sel_spots)]
    if sel_teams and "team" in sheet.columns:
        sheet = sheet[sheet["team"].isin(sel_teams)]
    if name_q:
        sheet = sheet[sheet["player"].str.contains(name_q, case=False, na=False)]

    sheet_sorted = sheet.sort_values("date", ascending=False).copy()
    # What the model would have rated them, pre-game.
    if "hr_prob_game" in sheet_sorted.columns:
        from src.parlay import assign_role
        sheet_sorted["role"] = sheet_sorted.apply(
            lambda r: assign_role(r["hr_prob_game"], r.get("season_hr")), axis=1)
    if "hr_score" in sheet_sorted.columns:
        sheet_sorted["verdict"] = sheet_sorted["hr_score"].map(_model_verdict)
    if "season_hr" in sheet_sorted.columns:
        sheet_sorted["player_tier"] = sheet_sorted["season_hr"].map(tier_of)
    sheet_cols = [c for c in ["date", "player", "team", "lineup_spot", "player_tier",
                              "pitcher_name",
                              "opponent", "hr_count", "hr_score", "hr_prob_game", "role",
                              "verdict",
                              # Pre-game metrics — the same inputs the model scores on.
                              "barrel_pct", "brl_pa", "hard_hit_pct", "avg_ev", "max_ev",
                              "launch_angle", "fb_pct", "gb_pct", "ld_pct", "pull_pct",
                              "hr_fb", "whiff_pct", "chase_pct", "zone_contact_pct",
                              "xiso", "xslg", "xwoba", "hr_per_pa", "season_hr",
                              "sprint_speed", "pitcher_hr9",
                              "park_factor", "recent_form_score", "hr_rate_7",
                              "rationale"]
                  if c in sheet_sorted.columns]
    show = sheet_sorted[sheet_cols].rename(columns={
        "date": "Date", "player": "Player", "team": "Team", "opponent": "Opp",
        "lineup_spot": "Spot", "player_tier": "Tier", "hr_count": "HR",
        "hr_score": "Model Score",
        "hr_prob_game": "Model HR%", "role": "Role", "verdict": "Model take",
        "pitcher_name": "Starting Pitcher", "park_factor": "Park",
        "barrel_pct": "Barrel%", "brl_pa": "Barrel/PA%", "hard_hit_pct": "Hard-Hit%",
        "avg_ev": "Avg EV", "max_ev": "Max EV", "launch_angle": "Launch°",
        "fb_pct": "Fly-Ball%", "gb_pct": "Ground-Ball%", "ld_pct": "Line-Drive%",
        "pull_pct": "Pull%", "hr_fb": "HR/FB", "whiff_pct": "Whiff%",
        "chase_pct": "Chase%", "zone_contact_pct": "Zone-Contact%",
        "xiso": "xISO", "xslg": "xSLG", "xwoba": "xwOBA", "hr_per_pa": "HR/PA",
        "season_hr": "Season HR", "sprint_speed": "Sprint",
        "pitcher_hr9": "SP HR/9", "hr_rate_7": "HR/PA (7d)",
        "recent_form_score": "Recent Form", "rationale": "Why they hit"})
    if "Spot" in show.columns:
        show["Spot"] = show["Spot"].astype("Int64")
    if "HR" in show.columns:
        show["HR"] = show["HR"].astype(int)
    if "Model HR%" in show.columns:
        show["Model HR%"] = (show["Model HR%"] * 100).round(0)
    if "Spot" in show.columns:
        show["Spot"] = show["Spot"].astype("Int64")
    st.markdown(f"**{len(show)}** home runs shown · _Model Score / HR% / Role / take = "
                "what the model rated them **before** the game_")
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        height=min(620, 60 + 35 * min(len(show), 16)),
        column_config={
            "Spot": st.column_config.NumberColumn("Spot", help=GLOSSARY["Spot"], format="%d"),
            "Model Score": st.column_config.ProgressColumn("Model Score", help="The model's pre-game HR Score (0-100) for this bat.", min_value=0, max_value=100, format="%.0f"),
            "Model HR%": st.column_config.NumberColumn("Model HR%", help="The model's pre-game ≥1 HR probability for this bat.", format="%.0f%%"),
            "Role": st.column_config.TextColumn("Role", help="ULX role the model put them in pre-game (Anchor/Value/Longshot)."),
            "Model take": st.column_config.TextColumn("Model take", help="Did the model like them pre-game? ✅ Loved / 👍 Liked / 😐 Lukewarm / ⚠️ Missed."),
            "Barrel%": st.column_config.NumberColumn("Barrel%", format="%.1f"),
            "Max EV": st.column_config.NumberColumn("Max EV", format="%.1f"),
            "HR/FB": st.column_config.NumberColumn("HR/FB", format="%.1f"),
            "xISO": st.column_config.NumberColumn("xISO", format="%.3f"),
            "Park": st.column_config.NumberColumn("Park", format="%.0f"),
            "Recent Form": st.column_config.ProgressColumn("Recent Form", min_value=0, max_value=100, format="%.0f"),
            "Why they hit": st.column_config.TextColumn("Why they hit", width="large"),
        },
    )
    st.download_button("⬇️ Export HR stat sheet to CSV", show.to_csv(index=False).encode(),
                       file_name="hr_stat_sheet.csv", mime="text/csv", key="dl_statsheet")

    with st.expander("🔍 HR detail — pre-game metrics & why they hit", expanded=False):
        opts = [f"{r['date']} · {r['player']} ({r.get('team','')}) — spot "
                f"{int(r['lineup_spot']) if pd.notna(r.get('lineup_spot')) else '?'}"
                for _, r in sheet_sorted.head(150).iterrows()]
        if opts:
            pick = st.selectbox("Pick a home run", opts, key="hr_detail_pick")
            _render_hr_detail(sheet_sorted.head(150).iloc[opts.index(pick)])


def tab_pick_record():
    """🏅 The real win-loss record of the featured picks, from the graded log."""
    st.subheader("🏅 Pick record — receipts, not vibes")
    rec = load_pick_record()
    if not rec.get("hotd") and not rec.get("top5"):
        st.info("No graded picks yet — the record grows each morning when the "
                "daily job grades yesterday's slate against real box scores.")
        return
    st.caption(
        "Every pick below was logged **pre-game** by the daily grader (and the "
        "backfill) and settled against real box scores. Hit = the player "
        "homered that day. *Expected* = the model's own average probability — "
        "beating it means the picks outperform what the model promised."
    )

    h = rec.get("hotd")
    if h:
        st.markdown("#### 🔒 HR of the Day")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Record", f"{h['wins']}-{h['losses']}")
        c2.metric("Hit rate", f"{h['hit_rate']}%",
                  delta=f"{h['hit_rate'] - h['expected_rate']:+.1f} vs expected")
        c3.metric("Expected", f"{h['expected_rate']}%")
        c4.metric("Streak", h["streak"])
        show = h["rows"].sort_values("date", ascending=False).copy()
        show["hr_prob_game"] = (show["hr_prob_game"] * 100).round(0)
        show["Result"] = show["hit_hr"].map({1: "✅ HR", 0: "❌ no HR"})
        st.dataframe(
            show.rename(columns={"date": "Date", "player": "Pick",
                                 "team": "Team", "hr_prob_game": "Model HR%"})
                [["Date", "Pick", "Team", "Model HR%", "Result"]],
            hide_index=True, use_container_width=True,
            height=min(420, 60 + 35 * len(show)),
        )

    t = rec.get("top5")
    if t:
        st.markdown("#### ⭐ Top-5 picks (daily board)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Picks graded", t["picks"])
        c2.metric("Hit rate", f"{t['hit_rate']}%",
                  delta=f"{t['hit_rate'] - t['expected_rate']:+.1f} vs expected")
        c3.metric("Days with ≥1 HR", f"{t['days_with_hit']}/{t['days']}",
                  help="Days when at least one of the five picks homered.")
        c4.metric("Avg HRs per day", t["avg_hits_per_day"])
        bd = t["by_day"].sort_values("date", ascending=False).copy()
        bd["Day result"] = bd["hits"].map(lambda x: "✅" * int(x) if x else "❌")
        st.dataframe(
            bd.rename(columns={"date": "Date", "picks": "Picks", "hits": "HRs hit"}),
            hide_index=True, use_container_width=True,
            height=min(380, 60 + 35 * len(bd)),
        )

    if rec.get("roles"):
        st.markdown("#### 🎰 Parlay legs by role")
        rows = [{"Role": f"{ROLE_EMOJI.get(r, '')} {r}", "Legs": v["legs"],
                 "Record": f"{v['wins']}-{v['losses']}",
                 "Hit rate %": v["hit_rate"], "Expected %": v["expected_rate"],
                 "Edge vs expected": round(v["hit_rate"] - v["expected_rate"], 1)}
                for r, v in rec["roles"].items()]
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption("These same numbers feed the parlay builder's per-role "
                   "reliability factors — roles that over-deliver get leaned into.")


def tab_previous_hrs(history):
    (events, _summary, _centroid, _calib, _trend, league_spot, _curve, report,
     source, _notes, start_iso, end_iso, _half_life) = history
    st.subheader("📋 Previous HRs & lineup spot")
    badge = "🟢" if source.startswith("LIVE") else "🟡"
    st.caption(
        f"{badge} **{source}** — every home run from **{start_iso} → {end_iso}**, the "
        "**batting-order spot** it was hit from, the pre-game metrics, **what the model "
        "rated them**, and *why* they went deep. (Live mode pulls the real lineup spot "
        "from each game's box score; the demo slate uses modeled spots.)"
    )
    # Blank-metrics tripwire: if the season feed came back empty, say so loudly
    # (with the feed diagnostics) instead of showing a wall of silent dashes.
    if len(events) and "barrel_pct" in events.columns:
        cov = float(events["barrel_pct"].notna().mean())
        if cov < 0.5:
            diag = [n for n in _notes if "feed issue" in n]
            st.warning(
                f"⚠️ Season metrics only resolved for **{cov*100:.0f}%** of these HR "
                "hitters — a Statcast/FanGraphs pull failed. The app retries "
                "automatically within ~10 minutes; refresh after that."
                + ("\n\n" + "\n".join(f"- {d}" for d in diag) if diag else "")
            )

    # --- Model report card: where it went right / wrong ---
    if report:
        st.markdown("##### 🎯 Model report card — right or wrong?")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("HRs the model liked", f"{report.get('liked_pct','—')}%",
                  help="Share of HR hitters the model rated 'live' pre-game (HR Score ≥ 48).")
        r2.metric("…loved", f"{report.get('loved_pct','—')}%",
                  help="Share rated elite pre-game (HR Score ≥ 60).")
        r3.metric("Avg rating — HR hitters", report.get("avg_hr_hitter_score", "—"),
                  delta=(round(report["avg_hr_hitter_score"] - report["field_score"], 1)
                         if report.get("field_score") is not None else None),
                  help="Average pre-game HR Score of players who homered vs. the field (delta).")
        r4.metric("Avg pre-game HR%", f"{report.get('avg_prob','—')}%")
        st.caption(f"📝 {report.get('verdict','')} — this feeds the **calibrated** "
                   "projection (and the parlay builder leans into ratings that "
                   "out-homer their model number).")

    if league_spot is not None and not league_spot.empty:
        ls = league_spot.sort_values("lineup_spot")
        cc1, cc2 = st.columns([2, 1])
        with cc1:
            st.altair_chart(
                alt.Chart(ls).mark_bar(color="#e63946").encode(
                    x=alt.X("lineup_spot:O", title="Lineup spot"),
                    y=alt.Y("hr:Q", title="HRs"),
                    tooltip=["lineup_spot", "hr", "games",
                             alt.Tooltip("hr_per_game:Q", format=".3f")]
                ).properties(height=210, title="HRs by lineup spot (window)"),
                use_container_width=True,
            )
        with cc2:
            top = ls.sort_values("hr", ascending=False).iloc[0]
            st.metric("Top HR spot", f"#{int(top['lineup_spot'])}", f"{int(top['hr'])} HRs")
            st.caption("Middle-order spots (3-5) usually lead — exactly why the parlay "
                       "builder anchors there.")

    # --- Batters by lineup spot: each HR with batter, starting pitcher & why ---
    if events is not None and not events.empty and "lineup_spot" in events.columns:
        ev = events.dropna(subset=["lineup_spot"]).copy()
        if not ev.empty:
            ev["lineup_spot"] = ev["lineup_spot"].astype(int)
            spots_present = sorted(ev["lineup_spot"].unique())
            st.markdown("##### 🔢 Batters by lineup spot")
            st.caption("Pick a spot to see every hitter who homered from it — with the "
                       "starting pitcher they faced, key metrics, and why they hit it.")
            spot_tabs = st.tabs([f"#{s}" for s in spots_present])
            for tab, s in zip(spot_tabs, spots_present):
                with tab:
                    grp = ev[ev["lineup_spot"] == s].sort_values("date", ascending=False)
                    n_hr = int(grp["hr_count"].sum()) if "hr_count" in grp.columns else len(grp)
                    st.caption(f"**{n_hr} HRs** from the {s}-spot · "
                               f"**{grp['player'].nunique()}** hitters")
                    cols = [c for c in ["date", "player", "team", "pitcher_name",
                                        "hr_score", "barrel_pct", "max_ev", "hr_fb",
                                        "rationale"] if c in grp.columns]
                    tbl = grp[cols].rename(columns={
                        "date": "Date", "player": "Player", "team": "Team",
                        "pitcher_name": "Starting Pitcher", "hr_score": "Model Score",
                        "barrel_pct": "Barrel%", "max_ev": "Max EV", "hr_fb": "HR/FB",
                        "rationale": "Why they hit"})
                    st.dataframe(
                        tbl, hide_index=True, use_container_width=True,
                        height=min(460, 60 + 35 * min(len(tbl), 11)),
                        column_config={
                            "Model Score": st.column_config.ProgressColumn("Model Score", min_value=0, max_value=100, format="%.0f"),
                            "Barrel%": st.column_config.NumberColumn("Barrel%", format="%.1f"),
                            "Max EV": st.column_config.NumberColumn("Max EV", format="%.1f"),
                            "HR/FB": st.column_config.NumberColumn("HR/FB", format="%.1f"),
                            "Why they hit": st.column_config.TextColumn("Why they hit", width="large"),
                        },
                    )

    render_hr_stat_sheet(events, start_iso, end_iso)


def tab_trends_lab(history):
    """12 pattern detectors over the HR-history window (src/trends.py)."""
    (events, _summary, _centroid, _calib, _trend, _league_spot, _curve, _report,
     source, _notes, start_iso, end_iso, _half_life) = history
    st.subheader("🔍 Trends Lab — 12 HR patterns")
    badge = "🟢" if source.startswith("LIVE") else "🟡"
    st.caption(
        f"{badge} **{source}** · window **{start_iso} → {end_iso}** · Tiers by season "
        f"HR total: ⭐ Star ≥{STAR_HR_MIN} · 🔷 Mid {MID_HR_MIN}-{STAR_HR_MIN-1} · "
        f"🎯 Under ≤{MID_HR_MIN-1}. Books price ⭐ ~+200..+450, 🔷 ~+500..+700, "
        "🎯 +700+ — the same bands the model uses when live odds are off."
    )
    hint = rotation_hint(events)
    if hint:
        st.info(f"🔄 **Today's rotation read:** {hint}")

    trends = compute_trends(events)
    if not trends:
        st.warning("No HR history available to mine for trends yet.")
        return
    st.markdown(f"**{len(trends)} trends** computed from this window — each with a "
                "one-line signal. Open any card for the numbers behind it.")
    for t in trends:
        with st.expander(t["title"], expanded=False):
            st.markdown(t["signal"])
            if t.get("table") is not None and len(t["table"]):
                st.dataframe(t["table"], hide_index=True, use_container_width=True)


def tab_trends(history, projection_slate):
    (events, summary, centroid, calib, trend, league_spot, _curve, _report,
     source, notes, start_iso, end_iso, half_life) = history
    st.subheader("📈 HR Trends & Backtest")
    badge = "🟢" if source.startswith("LIVE") else "🟡"
    st.caption(
        f"{badge} **{source}** — every home run from **{start_iso} → {end_iso}**, the "
        "shared profile of who went deep, model calibration, and how today's bats "
        f"resemble recent HR hitters (recency half-life **{half_life}d**)."
    )
    with st.expander("Data provenance", expanded=False):
        for n in notes:
            st.markdown(f"- {n}")

    if not summary:
        st.warning("No HR history available for this window.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("HRs in window", summary.get("total_hr"))
    c2.metric("HR games", summary.get("hr_events"))
    c3.metric("Unique HR hitters", summary.get("unique_hitters"))
    c4.metric("HR games / day", summary.get("hr_per_day"))
    st.caption("📋 See the **Previous HRs** tab for the full line-by-line HR log "
               "with each hitter's lineup spot and why they hit.")

    st.markdown("##### 🔬 Shared profile of HR hitters vs. all hitters")
    st.caption("How much HR hitters out-index the slate baseline on each metric.")
    st.dataframe(summary["metric_table"], hide_index=True, use_container_width=True)

    if trend is not None and not trend.empty:
        st.markdown("##### 🔥 Trend strength — what's shifting among HR hitters (last 7d vs. window)")
        st.caption(
            "Positive = HR hitters' recent average on that metric is running hotter "
            "than across the full window. The Profile Match uses a recency-weighted "
            f"centroid (half-life {half_life}d), so these shifts steer today's ranks."
        )
        st.dataframe(trend, hide_index=True, use_container_width=True)

    cc1, cc2 = st.columns(2)
    with cc1:
        st.markdown("**Context of home runs**")
        st.markdown(
            f"- Hit with a **platoon advantage:** {summary.get('platoon_share','—')}%\n"
            f"- In **HR-friendly parks** (PF ≥ 105): {summary.get('hr_friendly_share','—')}%\n"
            f"- Mean **HR Score** — HR hitters **{summary.get('mean_hr_score_hr_hitters','—')}** "
            f"vs all **{summary.get('mean_hr_score_all','—')}**"
        )
        hand = summary.get("handedness", {})
        if hand:
            st.markdown("- **Handedness:** " + ", ".join(f"{k} {v:.0f}%" for k, v in hand.items()))
    with cc2:
        st.markdown("**Hottest HR parks (by HR count)**")
        tp = summary.get("top_parks", {})
        if tp:
            tp_df = pd.DataFrame({"Park": list(tp), "HRs": list(tp.values())})
            st.altair_chart(
                alt.Chart(tp_df).mark_bar(color="#e63946").encode(
                    x=alt.X("HRs:Q"), y=alt.Y("Park:N", sort="-x", title=None),
                    tooltip=["Park", "HRs"]),
                use_container_width=True,
            )

    if league_spot is not None and not league_spot.empty:
        st.markdown("##### 🔢 HRs by lineup spot (recurring log)")
        st.caption(
            "Home runs by batting-order spot from the accumulating HR log "
            "(before the selected date). Middle-order spots produce the most HRs — "
            "the parlay builder reads this to fit Anchors (3-5), Value (6-7) and "
            "Longshots (7-9), and folds expected PA-by-spot into the HR probability."
        )
        ls = league_spot.sort_values("lineup_spot")
        st.altair_chart(
            alt.Chart(ls).mark_bar(color="#e63946").encode(
                x=alt.X("lineup_spot:O", title="Lineup spot"),
                y=alt.Y("hr:Q", title="HRs"),
                tooltip=["lineup_spot", "hr", "games", alt.Tooltip("hr_per_game:Q", format=".3f")]),
            use_container_width=True,
        )

    # --- Daily self-improvement track record (real outcomes, grows every day) ---
    from src.tuning import _load_tuning, brier_score, load_eval_log
    ev_log = load_eval_log()
    if not ev_log.empty:
        st.markdown("##### 🤖 Self-improvement track record")
        tun = _load_tuning()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Days graded", int(ev_log["date"].nunique()))
        c2.metric("Hitter-days", len(ev_log))
        b = brier_score(ev_log)
        c3.metric("Brier score", b if b is not None else "—",
                  help="Mean squared error of the HR probabilities vs reality — lower is better (~0.10 is solid for HR props).")
        cal_on = bool(tun.get("bins"))
        c4.metric("Auto-calibration", "🟢 active" if cal_on else "🟡 warming up",
                  help="Once ≥300 real hitter-days are logged, the model re-maps its "
                       "probabilities through the observed outcome curve — refit daily.")
        if "parlay_role" in ev_log.columns:
            legs = ev_log[ev_log["parlay_role"].astype(str).isin(["Anchor", "Value", "Longshot"])]
            if len(legs):
                rr = legs.groupby("parlay_role")["hit_hr"].agg(["mean", "count"])
                bits = [f"{ROLE_EMOJI.get(r, '')} {r}: {v['mean']*100:.0f}% hit ({int(v['count'])} legs)"
                        for r, v in rr.iterrows()]
                st.caption("**Parlay legs graded:** " + " · ".join(bits) +
                           " — these real hit rates recalibrate future ticket win% per role.")
        st.caption("Graded daily by GitHub Actions against real box scores; the "
                   "calibration + parlay role factors are refit on the full record "
                   "every morning, so the model gets a little sharper each day.")

    if calib is not None and not calib.empty:
        st.markdown("##### 🎯 Model calibration — predicted vs. actual HR rate")
        st.caption(
            "Each decile bins hitter-games by the model's predicted game HR probability; "
            "a well-calibrated model tracks the diagonal (actual ≈ predicted)."
        )
        melt = calib.melt(id_vars="Decile", value_vars=["Predicted HR%", "Actual HR%"],
                          var_name="Series", value_name="HR%")
        line = alt.Chart(melt).mark_line(point=True).encode(
            x=alt.X("Decile:N", sort=list(calib["Decile"])),
            y=alt.Y("HR%:Q"),
            color=alt.Color("Series:N", scale=alt.Scale(
                domain=["Predicted HR%", "Actual HR%"], range=["#888", "#e63946"])),
            tooltip=["Decile", "Series", "HR%"],
        ).properties(height=260)
        st.altair_chart(line, use_container_width=True)
        st.dataframe(calib, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.markdown(f"### 🏆 Top 5 per category — projections for {end_iso}")
    st.caption(
        "Built from the model and informed by the trailing-month HR profile. "
        "**Profile Match** = resemblance to recent HR hitters; **Calibrated** blends "
        "it with the HR Score."
    )
    tops = top5_by_category(projection_slate, n=5)
    _top5_card_grid(tops)

    flat = pd.concat([t.assign(Category=label) for label, t in tops.items()], ignore_index=True)
    st.download_button(
        "⬇️ Export all top-5 lists to CSV",
        flat.to_csv(index=False).encode(), file_name="top5_by_category.csv",
        mime="text/csv",
    )


def _render_parlay(result, stake, key: str = "parlay"):
    legs = result["legs"]
    s = result["summary"]
    if legs.empty or s.get("n_legs", 0) == 0:
        st.warning("Not enough qualifying bats to build this parlay. Loosen filters or change strategy.")
        return

    # Leg cards.
    cols = st.columns(min(len(legs), 3))
    for i, (_, leg) in enumerate(legs.iterrows()):
        with cols[i % len(cols)]:
            with st.container(border=True):
                role = leg.get("role", "Leg")
                st.markdown(f"### {ROLE_EMOJI.get(role, '•')} {role}")
                st.markdown(f"**{leg['player']}** · {leg['team']} vs {leg['opponent']}")
                live_tag = " 🟢LIVE" if leg.get("odds_is_live") else ""
                st.markdown(f"## {format_american(leg['book_odds'])}{live_tag}")
                m1, m2 = st.columns(2)
                m1.metric("HR Prob", f"{leg['hr_prob_game']*100:.0f}%")
                m2.metric("Edge", f"{leg['edge_pct']:+.1f}%")
                spot = leg.get("lineup_spot")
                spot_txt = f"bats {int(spot)}" if pd.notna(spot) else ""
                st.caption(f"_{leg['archetype']}_ · {spot_txt} · {leg.get('rationale','')}")
                sp = leg.get("sp_hr_at_spot")
                if pd.notna(sp) and sp and sp > 0:
                    st.caption(f"🎯 SP allowed **{int(sp)} HR** to the {int(spot)}-spot (last 10)")

    # Combined ticket.
    st.markdown("### 🎟️ The Ticket")
    light = s["light"]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Parlay odds", s["combined_american_str"])
    c2.metric("Model win %", f"{s['model_prob']}%")
    c3.metric("Implied %", f"{s['implied_prob']}%")
    c4.metric("EV / $1", f"{s['ev_pct']:+.0f}%")
    profit = stake * (s["combined_decimal"] - 1)
    c5.metric(f"${stake:.0f} pays", f"${profit:,.0f}")

    ev = s["ev_pct"]
    if s.get("any_live") and ev > 0:
        st.success(f"{light} · **+EV** at the current live price ({ev:+.0f}% per $1). Value spot.")
    elif s.get("any_live"):
        st.info(f"{light} · {ev:+.0f}% EV at live odds — no edge; shop for a better price.")
    else:
        st.info(f"{light} · EV is {ev:+.0f}% on model-implied odds (you pay the hold). "
                "Turn on live odds + shop books to find real +EV.")

    # Checklist (ULX 10-point).
    with st.expander(f"📋 ULX checklist — {s['checks_passed']}/{s['checks_total']} · {light}", expanded=True):
        cc = st.columns(2)
        for i, (label, ok) in enumerate(result["checklist"]):
            cc[i % 2].markdown(f"{'✅' if ok else '⬜'} {label}")

    disp = legs[[c for c in ["role", "player", "team", "opponent", "pitcher_name",
                             "book_odds", "hr_prob_game", "edge_pct", "archetype",
                             "rationale"] if c in legs.columns]].copy()
    st.download_button("⬇️ Export parlay to CSV", disp.to_csv(index=False).encode(),
                       file_name="hr_parlay.csv", mime="text/csv", key=f"dl_{key}")


def _hr_parlay_builder(df):
    st.caption(
        "Build HR parlays with **roles, not names** (the ULX formula): an **⚓ Anchor** "
        "(highest-confidence bat), **💰 Value** bats (underpriced profiles), and "
        "**🚀 Deep-Space Longshots** (overlooked ceiling). Diversified across games & "
        "archetypes, graded on the ULX checklist. *Research/entertainment only.*"
    )

    c1, c2, c3, c4 = st.columns(4)
    n_legs = c1.slider("Legs", 1, 5, 3)
    strategy = c2.selectbox(
        "Strategy",
        ["ulx", "safe", "value", "boom"],
        format_func={"ulx": "ULX role-based", "safe": "Safest (highest prob)",
                     "value": "Best value (edge)", "boom": "Boom (longshots)"}.get,
    )
    max_per_game = c3.selectbox("Max bats / game", [1, 2], index=0)
    stake = c4.number_input("Stake ($)", min_value=1.0, value=10.0, step=5.0)

    b1, b2 = st.columns([1, 3])
    if b1.button("🎲 Generate / shuffle legs", use_container_width=True):
        st.session_state["parlay_seed"] = st.session_state.get("parlay_seed", 0) + 1
    diversify = b2.checkbox("Diversify archetypes (avoid same-profile stacking)", value=True)

    seed = st.session_state.get("parlay_seed", 0)
    if seed:
        st.caption(f"🎲 Shuffle #{seed} — re-rolls among the top candidates per role. "
                   "Click again for a fresh mix; the leg slider/strategy still apply.")
    result = generate_parlay(df, n_legs=n_legs, strategy=strategy,
                             max_per_game=int(max_per_game), diversify_arch=diversify,
                             seed=(seed or None))
    _render_parlay(result, stake, key="builder")

    st.markdown("---")
    with st.expander("🛠️ Build your own parlay (pick the bats)"):
        choices = st.multiselect(
            "Players", sorted(df["player"].unique()),
            help="Pick 1-8 bats; roles are auto-assigned from each bat's HR odds band.",
        )
        if choices:
            _render_parlay(summarize_selection(df, choices), stake, key="custom")


def _mixed_ladder(df):
    st.caption(
        "The **ULX betting pyramid** — *don't get stuck on HRs*. One leg per bet "
        "type (💣 HR · 🟧 Double · 🟩 Total Bases · 🏃 Stolen Base · 🔷 Run), each from a "
        "different game: high-risk top, volume base. Non-HR cash rates are "
        "**modeled estimates** from the ULX hit-rate pyramid scaled by each bat's "
        "profile fit."
    )
    n = st.slider("Legs", 2, 5, 5, key="ladder_legs",
                  help="5 = the full pyramid (HR + 2B + TB + SB + Run).")
    if st.button("🎲 Rebuild ladder", key="ladder_btn"):
        st.session_state["ladder_seed"] = st.session_state.get("ladder_seed", 0) + 1
    res = build_ladder_parlay(df, n_legs=n)
    legs, s = res["legs"], res["summary"]
    if legs.empty:
        st.warning("Not enough qualifying bats for a ladder today.")
        return
    cols = st.columns(min(len(legs), 3))
    for i, (_, leg) in enumerate(legs.iterrows()):
        with cols[i % len(cols)]:
            with st.container(border=True):
                live_tag = " 🟢LIVE" if leg.get("bet_live") else ""
                st.markdown(f"**{BET_LABEL[leg['bet']]}**{live_tag}")
                st.markdown(f"**{leg['player']}** · {leg['team']} vs {leg['opponent']}")
                m1, m2 = st.columns(2)
                m1.metric("Est. cash", f"{leg['bet_prob']*100:.0f}%")
                m2.metric("Odds" if leg.get("bet_live") else "Est. odds",
                          format_american(leg["bet_odds"]))
                spot = leg.get("lineup_spot")
                spot_txt = f"bats {int(spot)} · " if pd.notna(spot) else ""
                st.caption(f"{spot_txt}fit {leg['bet_suit']:.0f}/100")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticket (est.)", format_american(s["combined_american"]))
    c2.metric("Est. win %", f"{s['combined_prob']}%")
    c3.metric("$10 pays (est.)", f"${s['payout_per_10']:,.0f}")
    st.caption("💡 ULX golden rules: never bet the same thing in every game · stack "
               "ways to cash · volume is king at the base of the ticket.")


def _prop_boards(df):
    st.caption(
        "**The prop ladder** — pick a bet type and see which bats fit it best "
        "(ULX cheat-sheet drivers + lineup-spot role). **ULX Best Bet** runs each "
        "player through the decision tree: elite HR profile → HR, else 2B → Run → "
        "SB → Hits/TB, else pass."
    )
    bet = st.selectbox("Bet type", BET_TYPES, format_func=BET_LABEL.get, key="prop_bet")
    b = df.copy()
    b = b.sort_values(f"suit_{bet}", ascending=False).head(30)
    src_col = b.get(f"odds_src_{bet}")
    has_live = src_col is not None and src_col.astype(str).str.startswith("LIVE").any()
    show = pd.DataFrame({
        "Player": b["player"], "Team": b["team"], "Opp": b["opponent"],
        "Spot": b["lineup_spot"].astype("Int64"),
        "Fit": b[f"suit_{bet}"].round(0),
        "Cash %": (b[f"prob_{bet}"] * 100).round(0),
        "Odds": b[f"odds_{bet}"],
        "Line": (src_col.map(lambda s: "🟢 " + s.split("· ")[-1]
                             if str(s).startswith("LIVE") else "🟡 est")
                 if src_col is not None else "🟡 est"),
        "Edge%": (b[f"edge_{bet}_pct"] if f"edge_{bet}_pct" in b.columns
                  else pd.Series(np.nan, index=b.index)),
        "ULX Best Bet": b["best_bet"].map(lambda x: BET_LABEL.get(x, "❌ Pass")),
        "Why": b["best_bet_reason"],
    })
    if has_live:
        st.caption("🟢 Real book lines loaded for this market — **Edge%** = model "
                   "cash prob − book implied (positive = +EV).")
    st.dataframe(
        show, hide_index=True, use_container_width=True,
        height=min(560, 60 + 35 * min(len(show), 14)),
        column_config={
            "Fit": st.column_config.ProgressColumn(
                "Fit", help="How well this bat fits the bet type (ULX drivers + lineup spot).",
                min_value=0, max_value=100, format="%.0f"),
            "Cash %": st.column_config.NumberColumn(
                "Cash %", help="Model cash probability (ULX pyramid base rate scaled "
                "by fit; HR uses the real model probability).", format="%.0f%%"),
            "Odds": st.column_config.NumberColumn(
                "Odds", help="Real book line when 🟢 LIVE (best price across books, "
                "TB = Over 1.5, Hits = Over 0.5), else the modeled estimate.",
                format="%+d"),
            "Edge%": st.column_config.NumberColumn(
                "Edge%", help="Model cash prob − book implied prob at the live line "
                "(only for 🟢 LIVE rows). Positive = +EV.", format="%+.1f"),
        },
    )
    st.download_button("⬇️ Export board to CSV", show.to_csv(index=False).encode(),
                       file_name=f"prop_board_{bet}.csv", mime="text/csv",
                       key=f"dl_prop_{bet}")


def tab_parlay(df, end_iso, live_odds, rot_hint=None):
    st.subheader("🎰 Parlays")
    if rot_hint:
        st.info(f"🔄 {rot_hint}  \n_(from the Trends Lab tier-rotation pattern — "
                "see 📚 History → 🔍 Trends Lab)_")
    fetch_lines = st.toggle(
        "📡 Fetch real TB/Hits lines", value=False, key="fetch_prop_lines",
        help="Pull live Total Bases (Over 1.5) and Hits (Over 0.5) prop lines from "
             "The Odds API for the Mixed Ladder and Prop Boards. Off by default — "
             "player-prop markets use extra API credits. Needs the live-odds toggle "
             "(sidebar → Advanced) and an ODDS_API_KEY.",
    )
    if fetch_lines:
        if live_odds:
            df = attach_prop_lines(df, end_iso, use_live=True)
            got_live = any(
                df.get(f"odds_src_{p}", pd.Series(dtype=str)).astype(str)
                  .str.startswith("LIVE").any() for p in ("TB", "H"))
            if not got_live:
                st.caption("⚠️ No live TB/Hits lines came back (check ODDS_API_KEY, "
                           "quota, or slate timing) — showing estimates.")
        else:
            st.caption("⚠️ Turn on **Use live HR odds** in the sidebar (⚙️ Advanced "
                       "settings) first — that enables the Odds API connection.")
    p1, p2, p3 = st.tabs(["⚾ HR Parlay", "🪜 Mixed Ladder", "📋 Prop Boards"])
    with p1:
        _hr_parlay_builder(df)
    with p2:
        _mixed_ladder(df)
    with p3:
        _prop_boards(df)


def _build_hr_insight(row) -> str:
    """A plain-language 'why they hit' narrative from the pre-game metrics."""
    spot = row.get("lineup_spot")
    spot_txt = f"the {int(spot)}-hole" if pd.notna(spot) else "the lineup"
    php = row.get("pitcher_throws", "R")
    pitcher = row.get("pitcher_name") or "the starter"
    parts = [f"**{row['player']}** ({row.get('team','')}) went deep from {spot_txt} "
             f"vs {php}HP {pitcher}."]
    pre = row.get("hr_prob_game")
    if pd.notna(pre):
        odds = format_american(row.get("fair_odds")) if pd.notna(row.get("fair_odds")) else ""
        parts.append(f"Pre-game model HR chance **{pre*100:.0f}%** ({odds} fair).")
    why = row.get("rationale", "")
    if why:
        parts.append(f"Drivers: {why}.")
    sneaky = row.get("sneaky_reasons", "")
    if sneaky:
        parts.append(f"Angle: {sneaky}.")
    r7 = row.get("hr_rate_7")
    if pd.notna(r7) and r7 > 0:
        parts.append(f"Was hot — ~{r7:.3f} HR/PA over the prior 7 days.")
    return " ".join(parts)


def _render_hr_detail(row):
    """A detail card: pre-game metrics grouped + the 'why they hit' insight."""
    st.markdown(_build_hr_insight(row))
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.markdown("**Power (pre-game)**")
        st.markdown(
            f"- Barrel%: {_fmt(row.get('barrel_pct'))}\n"
            f"- Barrel/PA%: {_fmt(row.get('brl_pa'))}\n"
            f"- Hard-Hit%: {_fmt(row.get('hard_hit_pct'))}\n"
            f"- Avg EV: {_fmt(row.get('avg_ev'))} · Max EV: {_fmt(row.get('max_ev'))}\n"
            f"- xISO: {_fmt(row.get('xiso'), 3)} · xSLG: {_fmt(row.get('xslg'), 3)}\n"
            f"- xwOBA: {_fmt(row.get('xwoba'), 3)}\n"
            f"- HR/FB: {_fmt(row.get('hr_fb'))} · Season HR: {_fmt(row.get('season_hr'), 0)}"
        )
    with g2:
        st.markdown("**Batted ball & discipline**")
        st.markdown(
            f"- Fly-Ball%: {_fmt(row.get('fb_pct'))}\n"
            f"- Ground-Ball%: {_fmt(row.get('gb_pct'))}\n"
            f"- Line-Drive%: {_fmt(row.get('ld_pct'))}\n"
            f"- Pull%: {_fmt(row.get('pull_pct'))}\n"
            f"- Launch angle: {_fmt(row.get('launch_angle'))}°\n"
            f"- Whiff%: {_fmt(row.get('whiff_pct'))} · Chase%: {_fmt(row.get('chase_pct'))}\n"
            f"- Zone-Contact%: {_fmt(row.get('zone_contact_pct'))}"
        )
    with g3:
        st.markdown("**Context**")
        st.markdown(
            f"- Lineup spot: {int(row['lineup_spot']) if pd.notna(row.get('lineup_spot')) else '—'}\n"
            f"- Tier: {tier_of(row.get('season_hr'))}\n"
            f"- Park factor: {_fmt(row.get('park_factor'), 0)}\n"
            f"- Wind ×: {_fmt(row.get('wind_mult'), 2)} · Temp: {_fmt(row.get('temp_f'), 0)}°F\n"
            f"- Platoon edge: {'yes' if row.get('platoon_adv') else 'no'}\n"
            f"- Pitcher HR/9: {_fmt(row.get('pitcher_hr9'), 2)}\n"
            f"- Sprint: {_fmt(row.get('sprint_speed'))}"
        )
    with g4:
        st.markdown("**Form & model**")
        st.markdown(
            f"- Recent form score: {_fmt(row.get('recent_form_score'), 0)}\n"
            f"- HR/PA 7d: {_fmt(row.get('hr_rate_7'), 3)}\n"
            f"- HR/PA 15d: {_fmt(row.get('hr_rate_15'), 3)}\n"
            f"- HR/PA 30d: {_fmt(row.get('hr_rate_30'), 3)}\n"
            f"- HR Score: {_fmt(row.get('hr_score'), 0)}\n"
            f"- Model game HR%: {row.get('hr_prob_game',0)*100:.0f}%"
        )


def tab_value_finder(df):
    st.subheader("💎 Value Finder")
    st.caption(
        "The biggest **model-vs-book edges** — where the model's HR% beats the "
        "book's implied HR%. Positive **Edge%** = +EV. *With model-implied odds "
        "everyone sits near −hold; flip on live odds (sidebar) to surface real "
        "value.* Research/entertainment only."
    )

    c1, c2, c3 = st.columns(3)
    min_prob = c1.slider("Min HR probability %", 0, 30, 6) / 100.0
    role_filter = c2.multiselect("Role", ["Anchor", "Value", "Longshot"], default=[])
    live_only = c3.checkbox("Live odds only", value=False)

    from src.parlay import assign_role
    v = df.copy()
    v["role"] = v.apply(
        lambda r: assign_role(r["hr_prob_game"], r.get("season_hr")), axis=1)
    v = v[v["hr_prob_game"] >= min_prob]
    if role_filter:
        v = v[v["role"].isin(role_filter)]
    if live_only:
        v = v[v.get("odds_is_live", False)]
    if v.empty:
        st.warning("No bats match these filters.")
        return

    v = v.sort_values("edge_pct", ascending=False)
    any_live = bool(v.get("odds_is_live", pd.Series([False])).any())
    pos = int((v["edge_pct"] > 0).sum())
    st.markdown(f"**{len(v)}** bats · **{pos}** with positive edge · "
                f"odds: {'🟢 LIVE' if any_live else '🟡 model-implied'}")

    cols = ["player", "team", "opponent", "pitcher_name", "role", "lineup_spot",
            "book_odds", "fair_odds", "hr_prob_game", "implied_prob", "edge_pct",
            "spot_hr_at_current", "rationale"]
    render_table(v.head(40), cols, "Edge%", "value_finder")

    st.markdown("##### 🎰 One-click value parlay")
    n = st.slider("Legs", 1, 5, 3, key="vf_legs")
    res = generate_parlay(v, n_legs=n, strategy="value")
    _render_parlay(res, 10.0, key="vf_parlay")


def _pick_card(label, name, sub, big, extra=""):
    st.markdown(
        f"<div class='pickcard'><div class='lab'>{label}</div>"
        f"<div class='name'>{name}</div><div class='sub'>{sub}</div>"
        f"<div class='big'>{big}</div><div class='sub'>{extra}</div></div>",
        unsafe_allow_html=True,
    )


def _col(d, name, default):
    return d[name].fillna(default) if name in d.columns else pd.Series(default, index=d.index)




@st.cache_data(ttl=60 * 30, show_spinner=False)
def load_pick_record():
    from src.tuning import load_eval_log, pick_record
    return pick_record(load_eval_log())


def render_hr_of_day(df):
    """Featured banner for the highest-confidence HR pick of the day."""
    row = hr_of_the_day(df)
    if row is None:
        return
    spot = row.get("lineup_spot")
    spot_txt = f"batting {int(spot)}" if pd.notna(spot) else ""
    cal = row.get("calibrated_hr_prob")
    cal_txt = f" · calibrated {cal*100:.0f}%" if pd.notna(cal) else ""
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"### 🔒 HR of the Day — **{row['player']}**")
            st.markdown(
                f"**{row['team']} vs {row['opponent']}** · {spot_txt} · "
                f"vs {row.get('pitcher_throws','R')}HP {row.get('pitcher_name','—')} · "
                f"**{format_american(row.get('book_odds'))}**"
            )
            reasons = []
            if row.get("rationale"):
                reasons.append(row["rationale"])
            if pd.notna(row.get("profile_match")) and row["profile_match"] >= 55:
                reasons.append(f"matches recent HR hitters ({row['profile_match']:.0f}% profile)")
            sp = row.get("sp_hr_at_spot")
            if pd.notna(sp) and sp and sp > 0 and pd.notna(spot):
                reasons.append(f"SP gave up {int(sp)} HR to the {int(spot)}-spot (last 10)")
            if pd.notna(row.get("hr_rate_7")) and row["hr_rate_7"] > 0.04:
                reasons.append("hot over the last 7 days")
            if row.get("consistency_score", 0) >= 65:
                reasons.append("high floor / steady hard contact")
            if pd.notna(row.get("ulx_checks")):
                reasons.insert(0, f"ULX profile {row.get('ulx_grade','')} "
                               f"({int(row['ulx_checks'])}/9 power checks)")
            st.markdown("**Why we're confident:** " + "; ".join(reasons[:4]) + ".")
        with c2:
            m1, m2 = st.columns(2)
            m1.metric("Confidence", f"{row['confidence']:.0f}/100")
            m2.metric("Model HR%", f"{row['hr_prob_game']*100:.0f}%")
            st.caption(f"HR Score {row.get('hr_score',0):.0f} · Consistency "
                       f"{row.get('consistency_score',0):.0f}{cal_txt}")
            st.progress(min(1.0, float(row["confidence"]) / 100.0))
    try:
        h = load_pick_record().get("hotd")
        if h and h["days"] >= 5:
            st.caption(
                f"📒 HR-of-the-Day record: **{h['wins']}-{h['losses']}** "
                f"({h['hit_rate']}% · expected {h['expected_rate']}%) · "
                f"streak {h['streak']} · full log in **📚 History → 🏅 Record**"
            )
    except Exception:
        pass


def render_top_picks(df):
    """At-a-glance hero cards: top pick, safest, best value, and a quick parlay."""
    if df.empty:
        return
    st.markdown("#### ⭐ Today's top picks")
    c1, c2, c3, c4 = st.columns(4)

    top = df.sort_values("hr_score", ascending=False).iloc[0]
    with c1:
        _pick_card("🥇 Top pick", top["player"],
                   f"{top['team']} vs {top['opponent']} · {format_american(top['book_odds'])}",
                   f"{top['hr_prob_game']*100:.0f}%", "to hit a HR (model)")

    safe = df.sort_values("hr_prob_game", ascending=False).iloc[0]
    with c2:
        _pick_card("🔒 Safest bat", safe["player"],
                   f"{safe['team']} vs {safe['opponent']} · HR Score {safe['hr_score']:.0f}",
                   f"{safe['hr_prob_game']*100:.0f}%", "highest HR probability")

    val = df.sort_values("edge_pct", ascending=False).iloc[0]
    with c3:
        _pick_card("💎 Best value", val["player"],
                   f"{val['team']} vs {val['opponent']} · {format_american(val['book_odds'])}",
                   f"{val['edge_pct']:+.1f}%", "model edge vs the book")

    with c4:
        res = generate_parlay(df, n_legs=3, strategy="ulx")
        s = res["summary"]
        names = " · ".join(res["legs"]["player"].tolist()) if not res["legs"].empty else "—"
        _pick_card("🎰 Suggested 3-leg", s.get("combined_american_str", "—"),
                   names, s.get("light", ""), f"model win {s.get('model_prob','—')}%")


def _pitcher_spot_card(pitcher_name, pitcher_id, end_iso, prefer_live):
    """Opposing starter's HRs allowed by lineup spot over their last 10 games.
    Returns the counts dict (used by the lineup table)."""
    from src.pitchers import hottest_spots, pitcher_recent_hr_by_spot
    counts, n, total, src = pitcher_recent_hr_by_spot(
        pitcher_id, pitcher_name, end_iso, 10, prefer_live)
    badge = "🟢 real" if src == "LIVE" else "🟡 modeled"
    st.markdown(f"**Opposing SP: {pitcher_name or '—'}**")
    st.caption(f"HRs allowed by lineup spot · last {n} games · {total} HR ({badge})")
    cdf = pd.DataFrame({"Spot": list(counts), "HRs": list(counts.values())})
    st.altair_chart(
        alt.Chart(cdf).mark_bar(color="#e63946").encode(
            x=alt.X("Spot:O", title="Lineup spot"),
            y=alt.Y("HRs:Q", title="HRs allowed"),
            tooltip=["Spot", "HRs"]).properties(height=160),
        use_container_width=True,
    )
    hot = hottest_spots(counts, 2)
    if hot:
        st.caption("🎯 Most vulnerable to spots " + ", ".join(f"**#{s}**" for s in hot))
    return counts


def _lineup_table(sub, opp_counts):
    t = sub[["lineup_spot", "player", "position", "hr_prob_game"]].copy()
    t["pit"] = t["lineup_spot"].map(
        lambda s: opp_counts.get(int(s), 0) if pd.notna(s) else 0)
    t["lineup_spot"] = t["lineup_spot"].astype("Int64")
    t["hr_prob_game"] = (t["hr_prob_game"] * 100).round(0)
    t = t.rename(columns={"lineup_spot": "Spot", "player": "Player",
                          "position": "Pos", "hr_prob_game": "Model HR%",
                          "pit": "SP HRs@Spot"})
    st.dataframe(
        t, hide_index=True, use_container_width=True, height=380,
        column_config={
            "Spot": st.column_config.NumberColumn("Spot", format="%d"),
            "Model HR%": st.column_config.NumberColumn("Model HR%", format="%.0f%%"),
            "SP HRs@Spot": st.column_config.NumberColumn(
                "SP HRs@Spot",
                help="HRs the opposing starter has allowed to THIS lineup spot over "
                     "their last 10 games — higher = a juicier spot to target.",
                format="%d"),
        },
    )


def tab_lineups(df, end_iso, prefer_live):
    st.subheader("🧾 Lineups & Pitcher HR Spots")
    st.caption(
        "Today's batting orders (1–9) for both teams in each game, next to the "
        "**opposing starter's HRs allowed by lineup spot over their last 10 games**. "
        "The **SP HRs@Spot** column flags which order positions have taken that "
        "pitcher deep. Updates with the selected date. *(Live mode uses posted "
        "lineups + real Statcast; demo uses modeled orders.)*"
    )
    games = sorted(df["game"].unique())
    game = st.selectbox("Game", games, key="lu_game")
    g = df[df["game"] == game]
    # ULX "HR hunting mode" read for the game's environment.
    env_count = int(g["hr_env_count"].max()) if "hr_env_count" in g.columns else 0
    if "hr_hunting" in g.columns and g["hr_hunting"].any():
        st.success(f"🔥 **HR Hunting Mode** — strong HR environment ({env_count}/5 "
                   "signals: wind out / warm / hitter park / homer-prone or fly-ball SP).")
    elif env_count >= 2:
        st.info(f"☀️ Decent HR environment ({env_count}/5 signals aligned).")
    is_home = g["is_home"].astype(bool)
    away = g[~is_home].dropna(subset=["lineup_spot"]).sort_values("lineup_spot")
    home = g[is_home].dropna(subset=["lineup_spot"]).sort_values("lineup_spot")

    c1, c2 = st.columns(2)
    for col, sub in ((c1, away), (c2, home)):
        with col:
            if sub.empty:
                st.info("Lineup not posted yet.")
                continue
            team = sub["team"].iloc[0]
            side = "home" if bool(sub["is_home"].iloc[0]) else "away"
            st.markdown(f"### {team} ({side})")
            pid = sub["pitcher_id"].iloc[0] if "pitcher_id" in sub.columns else None
            counts = _pitcher_spot_card(sub["pitcher_name"].iloc[0], pid, end_iso, prefer_live)
            _lineup_table(sub, counts)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    inject_css()
    game_date, prefer_live, lookback, half_life, live_odds = sidebar_controls()
    methodology_sidebar()

    st.title("⚾ MLB Home Run Hunter")
    st.caption(
        "Who's most likely to go deep today — ranked by a transparent model over "
        "Statcast power, recent form, matchup, park & weather."
    )
    with st.expander("👋 New here? How to use this", expanded=False):
        st.markdown(
            "1. **Pick a date** in the sidebar (defaults to today).\n"
            "2. **🏠 Today** — the HR of the Day and headline picks.\n"
            "3. **🎯 Picks** — Longshots, Consistent, Sneaky, Value, or the full board.\n"
            "4. **🎰 Parlays** — HR parlays, the **🪜 Mixed Ladder** (HR + doubles + "
            "total bases + steals + runs, per the ULX pyramid), and per-bet **Prop "
            "Boards** with each player's ULX Best Bet.\n"
            "5. **🧾 Lineups** & **📚 History** — today's orders and who's been homering.\n\n"
            "📲 **Install it like an app:** open this page in your phone browser → "
            "**Share / ⋮ menu → Add to Home Screen**.\n\n"
            "_Tip: tap the **»** (top-left) for the date picker & settings; turn off "
            "✨ Simple view there for every advanced metric._\n\n"
            "_Projections are model estimates for research/entertainment — not betting advice._"
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

    # Trailing-month HR history -> profile centroid -> enrich slate with the
    # "resemblance to recent HR hitters" signal used by the Trends tab.
    start_iso = (game_date - dt.timedelta(days=lookback)).isoformat()
    end_iso = game_date.isoformat()
    (events, summary, centroid, calib, trend, player_spot, league_spot,
     score_curve, report, h_source, h_notes) = load_hr_history(
        start_iso, end_iso, prefer_live, float(half_life))
    # Don't let a degraded pull sit in the cache for an hour: if live history
    # came back without season metrics (a feed hiccup), clear the entry so the
    # next refresh refetches instead of re-serving blanks.
    if (str(h_source).startswith("LIVE") and len(events)
            and "barrel_pct" in events.columns
            and events["barrel_pct"].notna().mean() < 0.5):
        load_hr_history.clear()
    if prefer_live and "data_quality" in scored.columns \
            and not (scored["data_quality"] == "real").any():
        load_scored_slate.clear()
    scored = add_profile_similarity(scored, centroid)
    scored = attach_spot_signal(scored, player_spot)
    scored = attach_calibrated_prob(scored, score_curve)   # learn from past ratings
    # Opposing-starter HRs allowed by lineup spot (last 10 games) -> per hitter.
    _has_pid = "pitcher_id" in scored.columns
    _pairs = tuple(
        (game, name, (grp["pitcher_id"].iloc[0] if _has_pid else None))
        for (game, name), grp in scored.groupby(["game", "pitcher_name"])
    )
    sp_counts = load_sp_spot_counts(end_iso, prefer_live, _pairs)
    scored = attach_sp_spot_signal(scored, sp_counts)
    # Live Trends Lab signals (streaks, tier rotation, spot×weekday heat) —
    # these now carry more pick weight than the ULX checklist.
    scored = attach_trend_signals(scored, events, game_date.strftime("%A"))
    scored = attach_odds(scored, end_iso, use_live=live_odds)
    scored = attach_props(scored)   # ULX prop ladder: per-bet-type fit & est. odds
    # Real TB/Hits lines are opt-in inside the Parlays tab (extra API credits).
    history = (events, summary, centroid, calib, trend, league_spot, score_curve,
               report, h_source, h_notes, start_iso, end_iso, half_life)

    filtered = filter_controls(scored)
    if filtered.empty:
        st.warning("No hitters match the current filters.")
        return

    odds_badge = "🟢 LIVE" if (filtered.get("odds_is_live", pd.Series([False])).any()) else "🟡 model-implied"
    st.caption(f"{len(filtered)} hitters · {filtered['game'].nunique()} games · "
               f"HR odds: {odds_badge}")

    # 5 simple top-level tabs; deeper views live inside each one.
    t_today, t_picks, t_parlay, t_lineups, t_history = st.tabs(
        ["🏠 Today", "🎯 Picks", "🎰 Parlays", "🧾 Lineups", "📚 History"]
    )
    with t_today:
        render_hr_of_day(filtered)
        st.markdown("")
        render_top_picks(filtered)
        st.caption("Head to **🎯 Picks → 🏆 Top 3** for the daily shortlist (top 3 HR "
                   "picks / value plays / longshots), **🎰 Parlays** to build "
                   "tickets, **🧾 Lineups** for today's orders, **📚 History** for "
                   "previous HRs & trends.")
    with t_picks:
        view = st.radio(
            "View", ["🏆 Top 3", "🚀 Longshots", "🎯 Consistent", "🕵️ Sneaky",
                     "💎 Value Finder", "📊 Full Board"],
            horizontal=True, label_visibility="collapsed", key="picks_view",
        )
        if view == "🏆 Top 3":
            tab_top3(filtered)
        elif view == "🚀 Longshots":
            tab_longshots(filtered)
        elif view == "🎯 Consistent":
            tab_consistent(filtered)
        elif view == "🕵️ Sneaky":
            tab_sneaky(filtered)
        elif view == "💎 Value Finder":
            tab_value_finder(filtered)
        else:
            tab_all(filtered)
    with t_parlay:
        tab_parlay(filtered, end_iso, live_odds, rot_hint=rotation_hint(events))
    with t_lineups:
        tab_lineups(filtered, end_iso, prefer_live)
    with t_history:
        hview = st.radio(
            "View", ["📋 Previous HRs", "🏅 Record", "🔍 Trends Lab",
                     "📈 Trends & Backtest"],
            horizontal=True, label_visibility="collapsed", key="hist_view",
        )
        if hview == "📋 Previous HRs":
            tab_previous_hrs(history)
        elif hview == "🏅 Record":
            tab_pick_record()
        elif hview == "🔍 Trends Lab":
            tab_trends_lab(history)
        else:
            tab_trends(history, filtered)


if __name__ == "__main__":
    main()
