#!/usr/bin/env python3
"""Grade the score predictor against real final scores.

Projects each completed date exactly as the app would have pre-game, then
scores it against the actual line score from MLB StatsAPI. Appends to
data/game_eval_log.csv (de-duped by date+game) and prints the record.

Usage:
    python scripts/grade_games.py [END_DATE] [DAYS] [--regrade] [--minutes=N]

Defaults: END_DATE = yesterday (UTC), DAYS = 1. Dates already in the log are
skipped unless --regrade is passed. Needs live network (GitHub Actions runners
have it; sandboxes may not).
"""
import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gamegrade import (  # noqa: E402
    append_game_rows, evaluate_game_day, game_record, load_game_log,
    total_bias_by_park, win_prob_calibration,
)


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    regrade = "--regrade" in sys.argv
    end = (dt.date.fromisoformat(argv[0]) if argv
           else dt.date.today() - dt.timedelta(days=1))
    days = int(argv[1]) if len(argv) > 1 else 1

    budget_min = next((float(a.split("=", 1)[1]) for a in sys.argv
                       if a.startswith("--minutes=")), 25.0)
    deadline = time.time() + budget_min * 60.0
    have = set() if regrade else set(load_game_log().get("date", []))
    graded = 0
    for k in range(days - 1, -1, -1):                 # oldest -> newest
        d = (end - dt.timedelta(days=k)).isoformat()
        if d in have:
            print(f"[{d}] already graded")
            continue
        if time.time() > deadline:
            print(f"[{d}] STOPPING — time budget spent; re-run to continue")
            break
        try:
            rows, note = evaluate_game_day(d, prefer_live=True)
        except Exception as e:                        # one bad date can't kill the run
            print(f"[{d}] error: {e}")
            continue
        if rows is None:
            print(f"[{d}] skipped: {note}")
            continue
        added = append_game_rows(rows)
        graded += 1
        hit = int(rows["winner_correct"].sum())
        print(f"[{d}] {note} -> {added} new | winners {hit}/{len(rows)} | "
              f"total MAE {rows['abs_err_total'].mean():.2f}")

    rec = game_record()
    print(f"\n=== SCORE PREDICTOR RECORD ({rec.get('n', 0)} games over "
          f"{rec.get('days', 0)} days) ===")
    if not rec.get("n"):
        print("nothing graded yet")
        return
    print(f"  runs MAE (per side)  {rec['mae_side']}")
    print(f"  total MAE            {rec['mae_total']}")
    print(f"  projected total avg  {rec['avg_total_proj']}  vs actual "
          f"{rec['avg_total_actual']}  (bias {rec['total_bias']:+})")
    print(f"  winner picked        {rec['winner_pct']}%   "
          f"[always-home baseline {rec['home_baseline_pct']}%] "
          f"{'BEATS' if rec['beats_home_baseline'] else 'does NOT beat'} baseline")
    print(f"  moneyline Brier      {rec['ml_brier']}  "
          f"[base-rate baseline {rec['ml_brier_baseline']}] "
          f"{'BEATS' if rec['beats_brier_baseline'] else 'does NOT beat'} baseline")
    if rec.get("total_pct") is not None:
        print(f"  O/U lean             {rec['total_pct']}% on {rec['total_n']} graded")
    if rec.get("gotd_winner_pct") is not None:
        print(f"  Game of the Day      {rec['gotd_winner_pct']}% on {rec['gotd_n']} picks")

    cal = win_prob_calibration()
    if not cal.empty:
        print("\n  win-prob calibration (home side)")
        print("    predicted%  actual%   gap   games")
        for _, r in cal.iterrows():
            print(f"    {r['predicted']:>9}  {r['actual']:>7}  {r['gap']:>+5}  "
                  f"{int(r['games']):>5}")

    park = total_bias_by_park()
    if not park.empty:
        print("\n  total bias by park (+ = model projects too many runs)")
        for _, r in park.head(6).iterrows():
            print(f"    {r['home_team']:>4}  bias {r['bias']:>+5}  "
                  f"mae {r['mae']:>4}  ({int(r['games'])} g)")


if __name__ == "__main__":
    main()
