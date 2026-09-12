"""Tests for variance ratios.

This module exists to corroborate the AR(1) verdicts without sharing their
assumptions, so the tests check it against cases with known analytic answers
rather than against the AR(1) estimator itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.reversion import half_life_to_kappa, simulate_ar1, simulate_ou
from src.analysis.variance_ratio import profile_variance_ratios, variance_ratio

N = 2185  # the real sample size


# --------------------------------------------------------------------------
# Known analytic answers
# --------------------------------------------------------------------------


@pytest.mark.parametrize("q", [2, 5, 10, 20])
def test_white_noise_gives_one_over_q(q):
    """For an i.i.d. series the variance ratio is exactly 1/q.

    An i.i.d. level series reverts completely every period, which is the
    fastest reversion possible, so this pins the bottom of the scale.
    """
    rng = np.random.default_rng(0)
    vr = variance_ratio(rng.standard_normal(20_000), q)
    assert vr.ratio == pytest.approx(1.0 / q, rel=0.10)


@pytest.mark.parametrize("q", [2, 5, 10, 20])
def test_random_walk_gives_one(q):
    """No memory, so variance accumulates linearly and the ratio is 1."""
    vr = variance_ratio(simulate_ar1(1.0, 20_000, seed=1), q)
    assert vr.ratio == pytest.approx(1.0, abs=0.08)
    assert not vr.rejects_random_walk
    assert vr.verdict == "random walk"


@pytest.mark.parametrize("b", [0.0, 0.3, 0.6, 0.9])
@pytest.mark.parametrize("q", [2, 5, 10])
def test_matches_the_ar1_closed_form(b, q):
    """For a stationary AR(1) in LEVELS the ratio has an analytic value.

    From the autocovariance, Var(x[t+q] - x[t]) = 2*g0*(1 - b^q), so

        VR(q) = (1 - b^q) / (q * (1 - b))

    which reduces to 1/q at b=0 (i.i.d. levels) and to 1 as b approaches 1
    (a random walk). Checked here to validate the implementation; the
    relationship is NOT used in reporting, since the point of the module is
    to measure the ratio empirically and see whether it agrees with the
    separately fitted AR(1).
    """
    expected = (1 - b**q) / (q * (1 - b))
    observed = variance_ratio(simulate_ar1(b, 200_000, seed=2), q).ratio
    assert observed == pytest.approx(expected, rel=0.03)


# --------------------------------------------------------------------------
# Behaviour the reporting relies on
# --------------------------------------------------------------------------


def test_faster_reversion_gives_lower_ratio():
    fast = profile_variance_ratios(simulate_ou(half_life_to_kappa(0.5), N, seed=1), ticker="F")
    slow = profile_variance_ratios(simulate_ou(half_life_to_kappa(10.0), N, seed=1), ticker="S")
    for a, b in zip(fast.results, slow.results):
        assert a.ratio < b.ratio


def test_ratio_falls_with_horizon_for_a_reverting_series():
    """Reversion has more time to act over a longer window."""
    p = profile_variance_ratios(simulate_ou(half_life_to_kappa(3.0), N, seed=1), ticker="OU")
    assert p.monotone_decreasing
    assert p.all_below_one
    assert p.confirms_mean_reversion


def test_random_walk_is_not_called_reverting():
    p = profile_variance_ratios(simulate_ar1(1.0, N, seed=1), ticker="RW")
    assert not p.confirms_mean_reversion


def test_trending_series_gives_ratio_above_one():
    """The measure is two-sided; positive drift in the differences shows up
    as VR > 1 rather than being mistaken for reversion."""
    rng = np.random.default_rng(3)
    trend = np.cumsum(rng.normal(0, 1, 20_000) + 0.3 * np.sign(rng.normal(0, 1, 20_000)))
    momentum = np.cumsum(np.convolve(rng.normal(0, 1, 20_000), np.ones(3) / 3, mode="same"))
    assert variance_ratio(momentum, 10).ratio > 1.0


# --------------------------------------------------------------------------
# Agreement with the AR(1) verdict -- the reason the module exists
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_hl", [0.05, 0.2])
def test_unresolved_regime_shows_strong_reversion_model_free(true_hl):
    """The corroboration that matters.

    A fund the AR(1) leaves unresolved
    should also show a very low variance ratio. If instead VR sat near 1, the
    censoring would more likely mean the AR(1) simply failed to fit.
    """
    p = profile_variance_ratios(
        simulate_ou(half_life_to_kappa(true_hl), N, seed=1), ticker="FAST"
    )
    assert p.results[0].ratio < 0.7
    assert p.confirms_mean_reversion
    assert p.results[0].rejects_random_walk


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_horizon_below_two_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        variance_ratio(np.random.default_rng(0).standard_normal(500), 1)


def test_short_series_rejected():
    with pytest.raises(ValueError, match="need at least"):
        variance_ratio(np.random.default_rng(0).standard_normal(50), 20)


def test_constant_series_rejected():
    with pytest.raises(ValueError, match="zero variance"):
        variance_ratio(np.full(500, 3.0), 5)


def test_profile_exposes_rows():
    p = profile_variance_ratios(simulate_ou(half_life_to_kappa(3.0), N, seed=1), ticker="OU")
    rows = p.as_rows()
    assert len(rows) == 4
    assert set(rows[0]) == {"ticker", "horizon", "vr", "z", "p", "verdict"}
