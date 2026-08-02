"""ULX methodology, encoded — the thresholds and rules from the ULX playbook.

Instilled directly from the ULX infographics so the model "bets the profile, not
the name":

POWER CHECKLIST (minimums a HR longshot should meet):
    Barrel% ≥ 8 · Hard-Hit% ≥ 40 · xSLG ≥ .450 · ISO ≥ .160 · Sweet-Spot% ≥ 30 ·
    Avg EV ≥ 88 · Launch Angle 10–28° · Pull% ≥ 35 · HR/FB ≥ 12.
  → the more green checks, the more confidence in a longshot.

PLATOON IS THE FIRST FILTER: LHB vs RHP / RHB vs LHP (and switch hitters) get on
the list; same-handed matchups are a red light UNLESS the bat is a "same-handed
smasher" (elite power that homers regardless of handedness).

LINEUP SPOT: longshots ideally hit 5–9 (overlooked); anchors 3–5.

HR ENVIRONMENT ("HR hunting mode"): wind blowing out, warm temps, hitter-friendly
park, and a homer-prone / fly-ball starting pitcher. When several align, it's a
HR day.

GRADES: GREEN (run it) ≥7 checks · YELLOW (consider) 4–6 · RED (fade) <4.
"""

from __future__ import annotations

import numpy as np

# (label, key, predicate) — the ULX power-checklist minimums.
POWER_CHECKS = [
    ("Barrel% ≥ 8", "barrel_pct", lambda v: v >= 8.0),
    ("Hard-Hit% ≥ 40", "hard_hit_pct", lambda v: v >= 40.0),
    ("xSLG ≥ .450", "xslg", lambda v: v >= 0.450),
    ("ISO ≥ .160", "iso", lambda v: v >= 0.160),
    ("Sweet-Spot% ≥ 30", "sweet_spot_pct", lambda v: v >= 30.0),
    ("Avg EV ≥ 88", "avg_ev", lambda v: v >= 88.0),
    ("Launch 10–28°", "launch_angle", lambda v: 10.0 <= v <= 28.0),
    ("Pull% ≥ 35", "pull_pct", lambda v: v >= 35.0),
    ("HR/FB ≥ 12", "hr_fb", lambda v: v >= 12.0),
]
N_POWER_CHECKS = len(POWER_CHECKS)


def _val(row, key):
    v = row.get(key)
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return None if (isinstance(v, float) and np.isnan(v)) else v


def power_checks(row) -> dict:
    """Evaluate the ULX power checklist. Returns met count, total, passed items,
    a 0-100 score, a grade, and whether the bat is a 'same-handed smasher'."""
    passed, items = 0, []
    for label, key, pred in POWER_CHECKS:
        v = _val(row, key)
        ok = bool(v is not None and pred(v))
        items.append((label, ok))
        passed += int(ok)
    grade = "🟢 GREEN" if passed >= 7 else ("🟡 YELLOW" if passed >= 4 else "🔴 RED")
    # Same-handed smasher: elite raw power that bombs regardless of platoon.
    barrel = _val(row, "barrel_pct") or 0
    iso = _val(row, "iso") or _val(row, "xiso") or 0
    smasher = barrel >= 13.0 or iso >= 0.230 or passed >= 8
    return {
        "ulx_checks": passed,
        "ulx_total": N_POWER_CHECKS,
        "ulx_score": round(100.0 * passed / N_POWER_CHECKS, 1),
        "ulx_grade": grade,
        "ulx_items": items,
        "same_handed_smasher": smasher,
    }


# Per-game HR-environment signals (the "HR hunting mode" board read).
def hr_environment(row) -> dict:
    """Score the home-run environment of a hitter's game from park, weather, and
    the opposing starter. Returns a 0-100 score, the count of green flags, and a
    'hunting mode' flag when several align."""
    flags = []
    wind = _val(row, "wind_mult")
    flags.append(("Wind blowing out", wind is not None and wind >= 1.05))
    temp = _val(row, "temp_f")
    flags.append(("Warm temps (≥80°)", temp is not None and temp >= 80.0))
    park = _val(row, "park_factor")
    flags.append(("Hitter-friendly park", park is not None and park >= 105.0))
    hr9 = _val(row, "pitcher_hr9")
    flags.append(("Homer-prone starter (HR/9 ≥ 1.3)", hr9 is not None and hr9 >= 1.3))
    flags.append(("Fly-ball starter", str(row.get("pitcher_lean", "")).upper() == "FB"))
    n = sum(1 for _, ok in flags if ok)
    return {
        "hr_env_flags": flags,
        "hr_env_count": n,
        "hr_env_score": round(100.0 * n / len(flags), 1),
        "hr_hunting": n >= 3,
    }
