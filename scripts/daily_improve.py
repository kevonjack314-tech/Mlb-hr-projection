#!/usr/bin/env python3
"""Daily self-improvement: grade yesterday, grow the record, retune the model.

Run daily (GitHub Actions has open network, so this grades REAL games):
    python scripts/daily_improve.py [YYYY-MM-DD]     # default: yesterday (UTC)

Steps:
  1. Rebuild the date's slate pre-game (live lineups + metrics) and log every
     hitter's predicted HR probability, the parlay legs the builder would have
     picked, and the top-5 picks — joined to the ACTUAL HR outcomes from box
     scores — into data/eval_log.csv (de-duped; simulated days are skipped).
  2. Refit the probability-calibration curve and the per-role parlay-leg
     reliability factors on the FULL accumulated record.
  3. Save data/model_tuning.json, which the model + parlay builder load at
     scoring time. Commit both files (the workflow does this) and the model is
     a little more accurate tomorrow than it was today.
"""
import datetime as dt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tuning import (  # noqa: E402
    append_eval_rows, brier_score, evaluate_day, fit_calibration,
    fit_feature_model, fit_role_factors, load_eval_log, save_tuning,
)


def main() -> None:
    date_iso = (sys.argv[1] if len(sys.argv) > 1
                else (dt.date.today() - dt.timedelta(days=1)).isoformat())

    rows, note = evaluate_day(date_iso, prefer_live=True)
    if rows is None:
        print(f"[{date_iso}] eval skipped: {note}")
    else:
        added = append_eval_rows(rows)
        print(f"[{date_iso}] {note} -> {added} new rows in the eval log")

    log = load_eval_log()
    tuning = fit_calibration(log)
    tuning.update(fit_role_factors(log))
    tuning.update(fit_feature_model(log))
    save_tuning(tuning, when=dt.date.today().isoformat())

    n = tuning.get("n", 0)
    print(f"track record: {n} hitter-days over {log['date'].nunique() if not log.empty else 0} days"
          f" | Brier {brier_score(log)}")
    if tuning.get("bins"):
        print(f"calibration: ACTIVE ({len(tuning['bins'])} bins)")
    else:
        print(f"calibration: warming up ({n} rows; needs 300)")
    if tuning.get("role_factors"):
        print(f"parlay role factors: {tuning['role_factors']} (n={tuning.get('role_n')})")
    else:
        print(f"parlay role factors: warming up (legs so far: {tuning.get('role_n', {})})")
    fm = tuning.get("feature_model") or {}
    print(f"learned feature model: {fm.get('note', 'n/a')}"
          + (f" | holdout Brier {fm.get('val_brier_model')} vs "
             f"hand-weighted {fm.get('val_brier_baseline')}"
             if fm.get("val_brier_model") is not None else ""))


if __name__ == "__main__":
    main()
