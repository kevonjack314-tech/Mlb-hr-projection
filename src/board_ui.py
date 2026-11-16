"""Streamlit board for the non-baseball sports.

Baseball has five tabs built over a season of graded outcomes — trends, a
record, a score predictor, a lineup view. Football and basketball have none of
that yet, and pretending otherwise with empty tabs would be worse than an
honest single board. This renders what actually exists: the ranked board, the
pick of the day, the parlay builder, and a plain statement of what the model
does and does not know.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from .multisport import (
    attach_model_odds, build_slate, generate_parlay, prob_col, score_col,
)
from .odds import format_american
from .sports import get_sport

# Columns worth showing per sport, in the order a person reads them.
BOARD_COLUMNS = {
    "nfl": [("player", "Player"), ("team", "Tm"), ("position", "Pos"),
            ("game", "Game"), ("td_prob_game", "TD%"), ("book_odds", "Odds"),
            ("td_score", "Score"), ("snap_share", "Snap%"),
            ("inside5_share", "Inside-5%"), ("rz_tgt_share", "RZ Tgt%"),
            ("expected_team_tds", "Team TDs"), ("rationale", "Why")],
    "nba": [("player", "Player"), ("team", "Tm"), ("game", "Game"),
            ("threes_prob_game", "3+ 3PM%"), ("book_odds", "Odds"),
            ("threes_score", "Score"), ("proj_minutes", "Min"),
            ("fg3a_per36", "3PA/36"), ("proj_3pa", "Proj 3PA"),
            ("proj_3p_pct", "3P%"), ("proj_3pm", "Proj 3PM"),
            ("game_pace", "Pace"), ("rationale", "Why")],
}

PCT_COLS = {"td_prob_game", "threes_prob_game", "snap_share", "inside5_share",
            "rz_tgt_share", "proj_3p_pct"}


def _fmt_board(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    cols = [(c, label) for c, label in BOARD_COLUMNS[sport] if c in df.columns]
    out = df[[c for c, _ in cols]].copy()
    for c in out.columns:
        if c in PCT_COLS:
            out[c] = (pd.to_numeric(out[c], errors="coerce") * 100).round(1)
    if "book_odds" in out.columns:
        out["book_odds"] = out["book_odds"].map(format_american)
    return out.rename(columns=dict(cols))


def _pick_card(row: pd.Series, sport: str) -> None:
    spec = get_sport(sport)
    p = float(row[prob_col(sport)])
    with st.container(border=True):
        st.markdown(f"### {spec.emoji} {spec.event_short} of the Day")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"## {row['player']}")
            st.markdown(
                f"**{row.get('team','')}** · {row.get('game','')}"
                + (f" · {row.get('position','')}" if row.get("position") else ""))
            if row.get("rationale"):
                st.caption(f"💡 {row['rationale']}")
        with c2:
            st.metric(spec.prop_label, f"{p*100:.0f}%",
                      f"{format_american(row.get('book_odds'))} model-implied",
                      delta_color="off")


def render_sport(sport: str, date_iso: str, n_legs: int = 3) -> None:
    spec = get_sport(sport)
    st.title(f"{spec.emoji} {spec.label}")
    st.caption(spec.notes)

    board, source = build_slate(sport, date_iso)
    if board is None or board.empty:
        st.warning("No slate available for this date.")
        return
    board = attach_model_odds(board, sport)
    pcol, scol = prob_col(sport), score_col(sport)
    board = board.sort_values([pcol, scol], ascending=False).reset_index(drop=True)

    live = source.startswith("LIVE")
    st.info(f"{'🟢' if live else '🟡'} **Data source:** {source}")
    if not live:
        st.warning(
            "**This board is modelled, not live.** The projections come from a "
            "deterministic synthetic slate, and the prices are the model's own "
            "numbers shaded by a typical prop hold — not a market. The "
            "football and basketball live feeds are not wired up yet, so treat "
            "everything here as a demonstration of the model, not as picks.")

    _pick_card(board.iloc[0], sport)

    st.markdown(f"#### 📋 Full board — {len(board)} {spec.player_noun}s, "
                f"{board['game'].nunique()} games")
    st.dataframe(_fmt_board(board, sport), hide_index=True,
                 use_container_width=True, height=440)
    st.download_button(
        "⬇️ Export board to CSV", board.to_csv(index=False).encode(),
        file_name=f"{sport}_{spec.event_short.lower()}_{date_iso}.csv",
        mime="text/csv", key=f"dl_{sport}")

    st.markdown("#### 🎰 Parlay")
    legs_n = st.slider("Legs", 1, 5, n_legs, key=f"legs_{sport}")
    strategy = st.selectbox(
        "Strategy", ["ulx", "model", "safe", "value", "boom"],
        format_func={"ulx": "ULX role-based", "model": "Model-driven",
                     "safe": "Safest (highest prob)", "value": "Best value (edge)",
                     "boom": "Boom (longshots)"}.get, key=f"strat_{sport}")
    try:
        res = generate_parlay(board, sport, n_legs=legs_n, strategy=strategy)
    except Exception as exc:                      # a thin board can leave a role empty
        st.warning(f"Couldn't build a {legs_n}-leg ticket: {exc}")
        return
    legs, summary = res["legs"], res["summary"]
    if legs.empty:
        st.warning("No qualifying legs for that strategy.")
        return
    for _, leg in legs.iterrows():
        st.markdown(
            f"**{leg.get('role','Leg')} · {leg['player']}** "
            f"({leg.get('team','')}) — {float(leg[pcol])*100:.0f}% · "
            f"{format_american(leg.get('book_odds'))}  \n"
            f"<span style='opacity:0.72;font-size:0.85em'>{leg.get('game','')}"
            f"{' · ' + leg['rationale'] if leg.get('rationale') else ''}</span>",
            unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Ticket", summary.get("combined_american_str", "—"))
    c2.metric("Model win", f"{summary.get('model_prob','—')}%")
    c3.metric("EV", f"{summary.get('ev_pct', 0):+.0f}%")
    st.caption(
        f"{summary.get('light','')} — {summary.get('checks_passed','?')}/"
        f"{summary.get('checks_total','?')} checks. EV is against the "
        "model-implied prices above, so it mostly reflects the assumed hold; "
        "it is not an edge against a real book.")

    with st.expander(f"How the {spec.event_short} model works"):
        st.markdown(_METHOD[sport])


_METHOD = {
    "nfl": """
**Touchdowns are a fixed, shared resource.** A team scores about **2.3**
offensive touchdowns a game. The model estimates how many this team will score
(from the implied team total, which already contains what the market knows),
then splits them by **role near the goal line** — not by yardage.

- **Inside-the-5 carries** dominate rushing touchdowns; **red-zone targets**
  dominate receiving ones. A 20%-snap goal-line back beats an every-down
  receiver who never sees the red zone, and the board says so.
- **Weather moves touchdowns, it does not delete them.** Wind and cold suppress
  passing, so the model shifts the touchdown pool toward the run rather than
  penalising everybody. Domes and retractable roofs are neutral.
- **Conservation:** every player's share is estimated independently, so they
  do not naturally add up. They are normalised against the team's own supply —
  without that pass the board was handing out up to 1.6× the touchdowns a team
  actually scores.
- The per-game probability is **Poisson**: 0.5 expected touchdowns is 39%, not
  50%. The gap is multi-touchdown games.
""",
    "nba": """
**Minutes first, then attempts, then accuracy.** Most public error in
three-point props is a minutes error, not a shooting error — a 40% shooter
playing 18 minutes is a worse bet than a 34% shooter playing 36.

- **Minutes** come from the last five games rather than the season, adjusted
  for a back-to-back and for a starter being out ahead of him.
- **Attempts** scale with pace (the matchup average, not one team's number) and
  with how freely the opponent lets threes go up.
- **Accuracy** is shrunk toward league average on a thin sample: 48% on 40
  attempts is a 38% shooter who has had a good month.
- The probability is **exact, not simulated**. Poisson attempts, each converted
  independently, gives Poisson makes — so P(3+) has a closed form that handles
  a fractional attempt projection honestly.
- **Conservation:** the rotation's projected attempts are normalised against
  the team's actual three-point volume, for the same reason football normalises
  touchdowns.
""",
}
