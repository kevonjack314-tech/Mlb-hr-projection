#!/usr/bin/env python3
"""Recurring updater for the HR-by-lineup-spot log.

Copies REAL graded hitter-days from the eval record (data/eval_log.csv — built
daily by scripts/daily_improve.py from live lineups + box-score outcomes) into
`data/lineup_hr_log.csv`, de-duped by (date, player), so the log accumulates
real data over time. Run it daily AFTER daily_improve.py.

The old version rebuilt a history slate on the fly; that slate is always
simulated (see src/history.py), which silently filled the log with fake rows.
Sourcing from the eval log makes it impossible to log a simulated hitter-day.

Usage:
    python scripts/update_lineup_log.py [YYYY-MM-DD] [lookback_days]

Defaults to today and a 3-day lookback (grabs any recently graded days that
aren't in the log yet). Use a larger lookback to backfill.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.lineup import update_log_from_history     # noqa: E402
from src.tuning import load_eval_log                # noqa: E402


def main() -> None:
    end = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
    lookback = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    start = end - dt.timedelta(days=lookback)

    ev = load_eval_log()
    if ev.empty:
        print("eval log is empty — nothing real to log yet "
              "(run scripts/daily_improve.py first).")
        return
    window = ev[(ev["date"] >= start.isoformat()) & (ev["date"] <= end.isoformat())]
    if window.empty:
        print(f"no graded hitter-days in {start}..{end} — log left untouched.")
        return
    added = update_log_from_history(window)   # uses hit_hr as the HR outcome
    print(f"[EVAL-LOG (real)] {start}..{end}: logged {added} new hitter-days "
          f"to the lineup HR log.")


if __name__ == "__main__":
    main()
