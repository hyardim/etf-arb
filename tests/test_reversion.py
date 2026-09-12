"""Tests for AR(1) estimation and half-life recovery.

This is the highest-value test file in the project. Half-life estimation has
several standard sign and log errors that produce plausible-looking but wrong
numbers, and no amount of inspecting real data reveals them -- the output looks
reasonable either way. A synthetic process with a known parameter catches all
of them at once, and answers "how do you know your estimator is right" with
something specific.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.reversion import (
    estimate_half_life,
    fit_ar1,
    half_life_from_b,
    half_life_to_kappa,
    simulate_ar1,
    simulate_ou,
)


# --------------------------------------------------------------------------
# Parameter recovery -- the core validation
# --------------------------------------------------------------------------


def test_recovers_known_half_life():
    """The benchmark case from the build guide."""
    true_hl = 3.0
    path = simulate_ou(kappa=half_life_to_kappa(true_hl), n=20_000, seed=0)
    assert abs(estimate_half_life(path) - true_hl) < 0.15


@pytest.mark.parametrize("true_hl", [0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0])
def test_recovers_across_the_plausible_range(true_hl):
    """Spans the range real ETFs are expected to occupy, from large-cap equity
    at the fast end to EM debt at the slow end."""
    path = simulate_ou(kappa=half_life_to_kappa(true_hl), n=50_000, seed=1)
    assert estimate_half_life(path) == pytest.approx(true_hl, rel=0.10)


def test_recovery_is_not_seed_specific():
    """Passing on one lucky seed proves nothing."""
    true_hl = 2.0
    estimates = [
        estimate_half_life(simulate_ou(half_life_to_kappa(true_hl), 20_000, seed=s))
        for s in range(10)
    ]
    assert np.mean(estimates) == pytest.approx(true_hl, rel=0.05)
    assert all(abs(e - true_hl) < 0.4 for e in estimates)


def test_precision_improves_with_sample_size():
    true_hl = 3.0
    kappa = half_life_to_kappa(true_hl)
    small = [estimate_half_life(simulate_ou(kappa, 500, seed=s)) for s in range(40)]
    large = [estimate_half_life(simulate_ou(kappa, 20_000, seed=s)) for s in range(40)]
    assert np.std(large) < np.std(small)


# --------------------------------------------------------------------------
# The sign and log errors this file exists to catch
# --------------------------------------------------------------------------


def test_ordering_is_preserved():
    """A sign error typically inverts the ranking, which would silently
    reverse the entire cross-sectional conclusion."""
    fast = estimate_half_life(simulate_ou(half_life_to_kappa(0.5), 50_000, seed=3))
    slow = estimate_half_life(simulate_ou(half_life_to_kappa(8.0), 50_000, seed=3))
    assert fast < slow


def test_half_life_of_b_equals_half():
    """b = 0.5 must give exactly a one-period half-life. Off-by-a-log errors
    fail here immediately."""
    assert half_life_from_b(0.5) == pytest.approx(1.0)


def test_half_life_is_monotone_in_b():
    values = [half_life_from_b(b) for b in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99)]
    assert values == sorted(values)


def test_half_life_matches_closed_form():
    for b in (0.2, 0.5, 0.75, 0.9):
        assert half_life_from_b(b) == pytest.approx(np.log(2) / -np.log(b))


def test_decay_actually_halves():
    """The operational meaning: after one half-life, a deviation should be
    about half its original size."""
    b = 0.7
    hl = half_life_from_b(b)
    assert b**hl == pytest.approx(0.5, rel=1e-9)


# --------------------------------------------------------------------------
# Guard clauses -- no finite half-life exists
# --------------------------------------------------------------------------


def test_random_walk_coefficient_has_no_half_life():
    """b >= 1: shocks never decay, so there is nothing to halve."""
    assert np.isnan(half_life_from_b(1.0))
    assert np.isnan(half_life_from_b(1.2))


@pytest.mark.parametrize("n,expected_hl", [(500, 135), (2_185, 293), (5_000, 536)])
def test_algebraic_guard_does_not_catch_a_sampled_random_walk(n, expected_hl):
    """The b >= 1 guard is necessary but nowhere near sufficient, and this is
    the reason the ADF test exists.

    OLS on a true random walk estimates b slightly BELOW 1 in every finite
    sample (the Kendall downward bias), so the algebraic guard never fires.
    What comes back instead is a large, finite, entirely spurious half-life
    that scales with sample size rather than with anything about the process:
    135 days at n=500, 293 at n=2185, 536 at n=5000, 16700 at n=50000.

    "Unstable across subsamples" is not a figure of speech here -- the number
    is a function of how much data happened to be available. Nothing about the
    fit itself flags this, which is why stationarity must be tested separately
    before any half-life is interpreted.
    """
    estimate = estimate_half_life(simulate_ar1(1.0, n, seed=1))
    assert np.isfinite(estimate)
    assert estimate == pytest.approx(expected_hl, rel=0.05)


def test_spurious_half_life_scales_with_sample_size():
    """The diagnostic signature: a genuine half-life is a property of the
    process and does not grow as more data arrives."""
    walk = [estimate_half_life(simulate_ar1(1.0, n, seed=1)) for n in (500, 5_000)]
    real = [
        estimate_half_life(simulate_ou(half_life_to_kappa(3.0), n, seed=1))
        for n in (500, 5_000)
    ]

    # The spurious estimate grows several-fold with sample size.
    assert walk[1] / walk[0] > 3.0

    # The genuine one converges rather than growing. It does move a little --
    # OLS on AR(1) is biased downward in b at order 1/n (Kendall), which at
    # n=500 pulls a true half-life of 3.0 down to roughly 2.2 -- but it
    # approaches the truth as n rises instead of running away from it.
    assert real[1] / real[0] < 2.0
    assert real[1] == pytest.approx(3.0, rel=0.15)


def test_negative_b_has_no_half_life():
    """b <= 0: oscillatory rather than decaying. Bid-ask bounce in a closing
    price series induces exactly this shape."""
    assert np.isnan(half_life_from_b(0.0))
    assert np.isnan(half_life_from_b(-0.5))
    assert np.isnan(estimate_half_life(simulate_ar1(-0.5, 20_000, seed=1)))


def test_guards_return_nan_rather_than_raising():
    """A cross-sectional run must be able to report 'no finite half-life' for
    one fund and carry on with the rest."""
    for b in (-1.0, 0.0, 1.0, 2.0, np.nan, np.inf):
        assert np.isnan(half_life_from_b(b))


# --------------------------------------------------------------------------
# Regression mechanics
# --------------------------------------------------------------------------


def test_fit_recovers_coefficient():
    fit = fit_ar1(simulate_ar1(0.6, 50_000, seed=1))
    assert fit.b == pytest.approx(0.6, abs=0.02)


def test_fit_recovers_intercept_and_long_run():
    """long_run = a/(1-b) is the equilibrium premium level."""
    fit = fit_ar1(simulate_ar1(0.5, 100_000, seed=2, a=10.0))
    assert fit.a == pytest.approx(10.0, rel=0.1)
    assert fit.long_run == pytest.approx(20.0, rel=0.1)


def test_white_noise_has_b_near_zero():
    """Pure noise has no persistence; b should be statistically zero."""
    rng = np.random.default_rng(0)
    fit = fit_ar1(rng.standard_normal(20_000))
    assert abs(fit.b) < 0.05
    assert fit.b_pvalue > 0.01


def test_standard_error_shrinks_with_sample():
    small = fit_ar1(simulate_ou(half_life_to_kappa(3.0), 1_000, seed=1))
    large = fit_ar1(simulate_ou(half_life_to_kappa(3.0), 50_000, seed=1))
    assert large.b_se < small.b_se


def test_n_reflects_regression_pairs():
    """One observation is consumed forming the lag."""
    assert fit_ar1(simulate_ou(0.5, 1_000, seed=1)).n == 999


def test_nan_values_are_dropped():
    path = simulate_ou(half_life_to_kappa(3.0), 20_000, seed=1)
    holed = path.copy()
    holed[::500] = np.nan
    assert estimate_half_life(holed) == pytest.approx(3.0, rel=0.15)


def test_too_few_observations_raises():
    with pytest.raises(ValueError, match="at least 3"):
        fit_ar1(np.array([1.0, 2.0]))


def test_accepts_pandas_series():
    import pandas as pd

    path = simulate_ou(half_life_to_kappa(3.0), 20_000, seed=0)
    series = pd.Series(path, index=pd.bdate_range("2018-01-01", periods=len(path)))
    assert estimate_half_life(series) == pytest.approx(estimate_half_life(path))
