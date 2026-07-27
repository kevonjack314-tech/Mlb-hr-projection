#!/usr/bin/env python3
"""Backfill the eval record over past dates, then retune everything.

The self-calibration, parlay role factors, and the learned feature model all
improve with sample size — and every completed date's lineups and box scores
are still on MLB StatsAPI. This grades each past date exactly like the daily
job does and appends the rows (de-duped), turning weeks of waiting into one
run. Needs live network (run it on GitHub Actions, not a sandbox).

Known trade-off: season-stat lookups are "as of today", so backfilled rows
carry mild lookahead bias. For calibration purposes that's an accepted trade.

Usage:
    python scripts/backfill_eval.py [END_DATE] [DAYS] [--regrade]

Defaults: END_DATE = yesterday (UTC), DAYS = 30. Dates already in the eval
log are skipped, so re-running is cheap and idempotent. Pass --regrade to
re-grade dates already present and overwrite their rows — needed after a
feature-window fix, since old rows carry the old (leaky) feature values.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tuning import (  # noqa: E402
    append_eval_rows, brier_score, evaluate_day, fit_calibration,
    fit_feature_model, fit_role_factors, leakage_report, load_eval_log,
    save_tuning,
)


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    regrade = "--regrade" in sys.argv
    end = (dt.date.fromisoformat(argv[0]) if argv
           else dt.date.today() - dt.timedelta(days=1))
    days = int(argv[1]) if len(argv) > 1 else 30

    # append_eval_rows keeps the LAST row for a (date, player), so re-grading
    # simply overwrites the stale rows in place.
    have = set() if regrade else set(load_eval_log().get("date", []))
    if regrade:
        print("REGRADE: re-grading dates already in the log (overwrites rows)")
    graded = skipped = 0
    for k in range(days, -1, -1):          # oldest → newest
        d = (end - dt.timedelta(days=k)).isoformat()
        if d in have:
            skipped += 1
            continue
        try:
            rows, note = evaluate_day(d, prefer_live=True)
        except Exception as e:              # one bad date shouldn't kill the run
            print(f"[{d}] error: {e}")
            continue
        if rows is None:
            print(f"[{d}] skipped: {note}")
            continue
        added = append_eval_rows(rows)
        graded += 1
        print(f"[{d}] {note} -> {added} new rows")

    log = load_eval_log()
    tuning = fit_calibration(log)
    tuning.update(fit_role_factors(log))
    tuning.update(fit_feature_model(log))
    save_tuning(tuning, when=dt.date.today().isoformat())

    fm = tuning.get("feature_model") or {}
    print(f"\nbackfill done: {graded} dates graded, {skipped} already logged")
    print(f"track record: {tuning.get('n', 0)} hitter-days over "
          f"{log['date'].nunique() if not log.empty else 0} days | Brier {brier_score(log)}")
    print(f"calibration: {'ACTIVE' if tuning.get('bins') else 'warming up'}"
          f" | role factors: {tuning.get('role_factors') or 'warming up'}")
    print(f"learned feature model: {fm.get('note', 'n/a')}")

    leak = leakage_report(log)
    print("leakage check: " + ("CLEAN" if not leak["leaked"] else "LEAKED — " + ", ".join(
        k for k, v in leak["features"].items() if v["leaked"])))
    for k, v in leak["features"].items():
        print(f"  {k:20s} zero-rate  HR {v['zero_rate_hr']:.3f} | no-HR "
              f"{v['zero_rate_no_hr']:.3f}{'   <-- LEAK' if v['leaked'] else ''}")


if __name__ == "__main__":
    main()
