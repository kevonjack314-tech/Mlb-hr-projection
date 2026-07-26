"""Probability shaping: soft ceiling + end-extrapolated calibration.

Regression tests for the "everything prints 35.0%" bug. A hard clip flattened
every elite bat onto one number, and np.interp's flat tail collapsed anything
above the top calibration bin — both destroyed the ordering exactly where the
picks get made.
"""

import numpy as np
import pytest

from src.tuning import (
    CAL_MAX_MULT,
    CAL_MIN_MULT,
    P_GAME_CEIL,
    P_GAME_KNEE,
    _interp_calibration,
    soft_cap,
)


# --------------------------------------------------------------------------- #
# soft_cap
# --------------------------------------------------------------------------- #
def test_soft_cap_is_strictly_increasing_no_ties():
    xs = np.linspace(0.0, 1.5, 400)
    ys = soft_cap(xs)
    assert np.all(np.diff(ys) > 0), "soft cap must never flat-top"
    assert len(np.unique(np.round(ys, 6))) == len(ys)


def test_soft_cap_identity_below_knee():
    # Identity between the floor and the knee — no compression down there.
    for v in (0.01, 0.05, 0.15, P_GAME_KNEE):
        assert soft_cap(v) == pytest.approx(v, abs=1e-9)


def test_soft_cap_floors_at_a_nonzero_probability():
    # A zero HR probability is never a useful output.
    from src.tuning import P_GAME_FLOOR
    assert soft_cap(0.0) == pytest.approx(P_GAME_FLOOR)
    assert soft_cap(-1.0) == pytest.approx(P_GAME_FLOOR)


def test_soft_cap_bounded_and_never_reaches_ceiling():
    assert soft_cap(10.0) < P_GAME_CEIL
    assert soft_cap(0.60) < P_GAME_CEIL
    # ...but it lifts the old hard-cap region well above 0.35.
    assert soft_cap(0.50) > 0.35


def test_soft_cap_continuous_at_the_knee():
    lo, hi = soft_cap(P_GAME_KNEE - 1e-6), soft_cap(P_GAME_KNEE + 1e-6)
    assert abs(hi - lo) < 1e-5


def test_soft_cap_vectorizes():
    out = soft_cap(np.array([0.1, 0.3, 0.5]))
    assert isinstance(out, np.ndarray) and len(out) == 3
    assert isinstance(soft_cap(0.1), float)


# --------------------------------------------------------------------------- #
# calibration end-extrapolation
# --------------------------------------------------------------------------- #
_PREDS = np.array([0.05, 0.12, 0.20, 0.28])
_ACTS = np.array([0.02, 0.11, 0.24, 0.36])      # model under-predicts the top


def test_interp_extrapolates_above_top_bin_instead_of_flattening():
    a = _interp_calibration(0.30, _PREDS, _ACTS)
    b = _interp_calibration(0.40, _PREDS, _ACTS)
    c = _interp_calibration(0.50, _PREDS, _ACTS)
    assert a < b < c, "must keep rising above the top bin, not flat-line"
    # np.interp would have pinned all three to _ACTS[-1].
    assert b != pytest.approx(_ACTS[-1])


def test_interp_extrapolates_below_bottom_bin():
    a = _interp_calibration(0.04, _PREDS, _ACTS)
    b = _interp_calibration(0.01, _PREDS, _ACTS)
    assert b < a
    assert b < _ACTS[0]


def test_interp_continuous_at_both_boundaries():
    top = _interp_calibration(_PREDS[-1], _PREDS, _ACTS)
    assert top == pytest.approx(_ACTS[-1], abs=1e-9)
    bot = _interp_calibration(_PREDS[0], _PREDS, _ACTS)
    assert bot == pytest.approx(_ACTS[0], abs=1e-9)


def test_interp_matches_np_interp_inside_the_range():
    for v in (0.08, 0.15, 0.24):
        assert _interp_calibration(v, _PREDS, _ACTS) == pytest.approx(
            float(np.interp(v, _PREDS, _ACTS)))


def test_interp_preserves_ordering_across_the_whole_domain():
    xs = np.linspace(0.001, 0.6, 300)
    ys = [_interp_calibration(float(x), _PREDS, _ACTS) for x in xs]
    assert all(b >= a for a, b in zip(ys, ys[1:]))


def test_interp_empty_bins_is_identity():
    assert _interp_calibration(0.2, np.array([]), np.array([])) == 0.2


# --------------------------------------------------------------------------- #
# The clamp band still guards against a runaway calibration fit
# --------------------------------------------------------------------------- #
def test_calibration_clamp_band_is_sane():
    assert 0 < CAL_MIN_MULT < 1 < CAL_MAX_MULT
    # Wide enough to let the top correct (the record needed ~1.5-1.7x)...
    assert CAL_MAX_MULT >= 1.5
    # ...but not so wide the curve can invent probabilities.
    assert CAL_MAX_MULT <= 2.5
