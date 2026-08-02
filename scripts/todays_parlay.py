#!/usr/bin/env python3
"""Print today's 3-leg HR parlay using the FULL live model.

Runs the same pipeline the app does (live slate -> score -> profile match ->
lineup-spot + SP-spot signals -> calibration -> trend signals -> odds), then
generates parlays and prints them for the workflow log.

Usage:
    python scripts/todays_parlay.py [YYYY-MM-DD] [n_legs]

Needs open network (GitHub Actions runners have it; sandboxes may not).
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.history import (  # noqa: E402
    add_profile_similarity, build_hr_history, hr_profile_centroid,
)
from src.learn import attach_calibrated_prob, hit_rate_by_score  # noqa: E402
from src.lineup import attach_spot_signal, player_spot_hr  # noqa: E402
from src.model import hr_of_the_day, score_slate  # noqa: E402
from src.odds import attach_odds, format_american  # noqa: E402
from src.parlay import generate_parlay  # noqa: E402
from src.pitchers import attach_sp_spot_signal, sp_spot_counts_for  # noqa: E402
from src.sources import get_slate  # noqa: E402
from src.trends import attach_trend_signals  # noqa: E402


def _leg_line(i, leg):
    spot = leg.get("lineup_spot")
    spot_txt = f"bats {int(spot)}" if spot == spot and spot is not None else "spot ?"
    bvp = leg.get("bvp_hr")
    bvp_txt = (f" | {int(bvp)} career HR vs this SP"
               if bvp is not None and bvp == bvp and bvp >= 1 else "")
    print(f"  {i}. [{leg.get('role','Leg')}] {leg['player']} ({leg['team']}) "
          f"{leg['game']}")
    print(f"     {leg['hr_prob_game']*100:.1f}% HR | {format_american(leg.get('book_odds'))}"
          f" ({leg.get('odds_source','model')}) | {spot_txt} | "
          f"vs {leg.get('pitcher_throws','R')}HP {leg.get('pitcher_name','—')}{bvp_txt}")
    print(f"     HR Score {leg.get('hr_score',0):.0f} | Barrel% "
          f"{leg.get('barrel_pct','—')} | HR/FB {leg.get('hr_fb','—')} | "
          f"MaxEV {leg.get('max_ev','—')} | ULX {leg.get('ulx_grade','—')}")
    if leg.get("rationale"):
        print(f"     WHY: {leg['rationale']}")


def _show(title, res):
    legs, s = res["legs"], res["summary"]
    print(f"\n=== {title} ===")
    if legs.empty:
        print("  (no qualifying legs)")
        return
    for i, (_, leg) in enumerate(legs.iterrows(), 1):
        _leg_line(i, leg)
    print(f"  TICKET: {s['combined_american_str']} | model win {s['model_prob']}% | "
          f"implied {s['implied_prob']}% | EV {s['ev_pct']:+.0f}% | "
          f"{s['light']} ({s['checks_passed']}/{s['checks_total']} checks) | "
          f"$10 pays ${s['payout_per_10']:,.2f}")


def main() -> None:
    date_iso = sys.argv[1] if len(sys.argv) > 1 else dt.date.today().isoformat()
    n_legs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    game_date = dt.date.fromisoformat(date_iso)

    df, source, notes = get_slate(game_date, prefer_live=True)
    print(f"DATE: {date_iso}\nSLATE SOURCE: {source}")
    for n in notes[:4]:
        print(" -", n)
    if not str(source).startswith("LIVE"):
        print("\n!! NOT LIVE DATA — refusing to print picks from a synthetic slate.")
        sys.exit(1)

    scored = score_slate(df)
    start_iso = (game_date - dt.timedelta(days=30)).isoformat()
    events, slate_hist, h_src, _ = build_hr_history(start_iso, date_iso, prefer_live=True)
    print(f"HISTORY SOURCE: {h_src} ({len(events)} HR events)")

    scored = add_profile_similarity(
        scored, hr_profile_centroid(events, end_date_iso=date_iso, half_life_days=10))
    scored = attach_spot_signal(scored, player_spot_hr(slate_hist))
    scored = attach_calibrated_prob(scored, hit_rate_by_score(slate_hist))
    has_pid = "pitcher_id" in scored.columns
    pairs = tuple((g, n, (grp["pitcher_id"].iloc[0] if has_pid else None))
                  for (g, n), grp in scored.groupby(["game", "pitcher_name"]))
    scored = attach_sp_spot_signal(scored, sp_spot_counts_for(pairs, date_iso, True))
    scored = attach_trend_signals(scored, events, game_date.strftime("%A"))
    scored = attach_odds(scored, date_iso, use_live=True)

    live_odds = bool(scored.get("odds_is_live", False).any()) \
        if "odds_is_live" in scored.columns else False
    print(f"GAMES: {scored['game'].nunique()} | HITTERS: {len(scored)} | "
          f"ODDS: {'LIVE' if live_odds else 'model-implied'}")

    hotd = hr_of_the_day(scored)
    if hotd is not None:
        print(f"\n*** HR OF THE DAY: {hotd['player']} ({hotd['team']}) "
              f"{hotd['hr_prob_game']*100:.1f}% | "
              f"{format_american(hotd.get('book_odds'))} | "
              f"confidence {hotd['confidence']:.0f}/100 ***")
        print(f"    {hotd.get('rationale','')}")

    _show(f"{n_legs}-LEG ULX (Anchor/Value/Longshot)",
          generate_parlay(scored, n_legs=n_legs, strategy="ulx"))
    _show(f"{n_legs}-LEG BEST-VALUE (record says Value legs over-deliver)",
          generate_parlay(scored, n_legs=n_legs, strategy="value"))
    _show(f"{n_legs}-LEG SAFEST (highest HR probability)",
          generate_parlay(scored, n_legs=n_legs, strategy="safe"))


if __name__ == "__main__":
    main()
