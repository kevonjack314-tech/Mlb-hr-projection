#!/usr/bin/env python3
"""Recurring updater for the HR-by-lineup-spot log.

Appends the trailing window's HRs (with each hitter's lineup spot) to
`data/lineup_hr_log.csv`, de-duped by (date, player), so the log accumulates
over time. Run it daily (cron, a CI schedule, or the `/loop` skill).

Usage:
    python scripts/update_lineup_log.py [YYYY-MM-DD] [lookback_days]

Defaults to today and a 1-day lookback (just the prior day's results). Use a
larger lookback for a backfill. With live Statcast hosts allowlisted it logs real
games; otherwise it logs the deterministic simulated slate.
"""
import datetime as dt
import sys

sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.dirname(__file__)))

from src.history import build_hr_history          # noqa: E402
from src.lineup import update_log_from_history     # noqa: E402


def main() -> None:
    end = dt.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else dt.date.today()
    lookback = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    start = end - dt.timedelta(days=lookback)
    _events, slate_hist, source, _notes = build_hr_history(
        start.isoformat(), end.isoformat(), prefer_live=True)
    if not str(source).startswith("LIVE"):
        print(f"[{source}] {start}..{end}: live data unavailable — log left "
              "untouched (never write simulated rows to the real log).")
        return
    added = update_log_from_history(slate_hist)
    print(f"[{source}] {start}..{end}: logged {added} new hitter-days to the "
          f"lineup HR log.")


if __name__ == "__main__":
    main()
