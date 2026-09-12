"""Tests for the OU simulator.

The simulator is the yardstick the half-life estimator is measured against,
so it is verified against the analytic properties of the process rather than
assumed correct. A bent yardstick would let a recovery test pass while both
simulator and estimator are wrong in the same direction.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.reversion import (
    half_life_to_kappa,
    kappa_to_half_life,
    simulate_ar1,
    simulate_ou,
)


# --------------------------------------------------------------------------
# Conversions
# --------------------------------------------------------------------------


def test_half_life_kappa_roundtrip():
    for hl in (0.5, 1.0, 3.0, 10.0):
        assert kappa_to_half_life(half_life_to_kappa(hl)) == pytest.approx(hl)


def test_kappa_for_unit_half_life():
    """A one-period half-life means b = 0.5 exactly."""
    assert np.exp(-half_life_to_kappa(1.0)) == pytest.approx(0.5)


@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_non_positive_conversions_raise(bad):
    with pytest.raises(ValueError):
        half_life_to_kappa(bad)
    with pytest.raises(ValueError):
        kappa_to_half_life(bad)


# --------------------------------------------------------------------------
# Analytic properties
# --------------------------------------------------------------------------


def test_lag1_autocorrelation_matches_exp_minus_kappa():
    """The defining property: for an OU sampled at dt, corr(x_t, x_t+1) is
    exp(-kappa*dt), which is precisely the AR(1) coefficient b."""
    kappa = half_life_to_kappa(3.0)
    path = simulate_ou(kappa, 200_000, seed=42)
    observed = np.corrcoef(path[:-1], path[1:])[0, 1]
    assert observed == pytest.approx(np.exp(-kappa), abs=0.01)


def test_stationary_mean():
    path = simulate_ou(half_life_to_kappa(2.0), 200_000, seed=1, mu=5.0)
    assert path.mean() == pytest.approx(5.0, abs=0.05)


def test_stationary_variance():
    """Var = sigma^2 / (2*kappa) for the stationary OU distribution."""
    kappa, sigma = half_life_to_kappa(4.0), 2.0
    path = simulate_ou(kappa, 200_000, seed=2, sigma=sigma)
    assert path.var() == pytest.approx(sigma**2 / (2 * kappa), rel=0.05)


def test_no_burn_in_needed():
    """Starting from the stationary distribution means the first tenth of the
    path has the same variance as the last tenth."""
    path = simulate_ou(half_life_to_kappa(5.0), 100_000, seed=3)
    head, tail = path[:10_000], path[-10_000:]
    assert head.var() == pytest.approx(tail.var(), rel=0.15)


@pytest.mark.parametrize("hl", [0.5, 1.0, 3.0, 10.0])
def test_recovers_across_half_lives(hl):
    path = simulate_ou(half_life_to_kappa(hl), 200_000, seed=7)
    b = np.corrcoef(path[:-1], path[1:])[0, 1]
    assert np.log(2) / -np.log(b) == pytest.approx(hl, rel=0.05)


def test_faster_reversion_is_less_autocorrelated():
    fast = simulate_ou(half_life_to_kappa(0.5), 50_000, seed=5)
    slow = simulate_ou(half_life_to_kappa(10.0), 50_000, seed=5)
    assert np.corrcoef(fast[:-1], fast[1:])[0, 1] < np.corrcoef(slow[:-1], slow[1:])[0, 1]


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_seed_is_reproducible():
    a = simulate_ou(0.5, 1000, seed=99)
    b = simulate_ou(0.5, 1000, seed=99)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, simulate_ou(0.5, 1000, seed=100))


def test_explicit_start_respected():
    assert simulate_ou(0.5, 100, seed=1, x0=42.0)[0] == 42.0


def test_zero_sigma_is_deterministic_decay():
    """With no noise the path decays geometrically toward mu, halving each
    half-life."""
    path = simulate_ou(half_life_to_kappa(1.0), 5, seed=1, sigma=0.0, mu=0.0, x0=1.0)
    np.testing.assert_allclose(path, [1.0, 0.5, 0.25, 0.125, 0.0625], atol=1e-12)


def test_length_and_shape():
    path = simulate_ou(0.3, 500, seed=1)
    assert path.shape == (500,)
    assert np.isfinite(path).all()


@pytest.mark.parametrize(
    "kwargs", [{"kappa": 0.0}, {"kappa": -1.0}, {"n": 1}, {"sigma": -1.0}, {"dt": 0.0}]
)
def test_invalid_parameters_raise(kwargs):
    params = {"kappa": 0.5, "n": 100} | kwargs
    with pytest.raises(ValueError):
        simulate_ou(**params)


# --------------------------------------------------------------------------
# AR(1) simulator for cases OU cannot represent
# --------------------------------------------------------------------------


def test_ar1_random_walk_is_non_stationary():
    """b = 1 has no finite half-life; needed to exercise the unit-root guard.

    Non-stationarity shows up in the variance of the LEVEL across independent
    realisations, growing linearly in t. It does not show up in the sample
    variance of a fixed-length window, which for a random walk depends on the
    window length and not on where the window sits.
    """
    t_mid, t_end = 500, 2_000
    levels = np.array(
        [simulate_ar1(1.0, t_end + 1, seed=s)[[t_mid, t_end]] for s in range(300)]
    )
    var_mid, var_end = levels[:, 0].var(), levels[:, 1].var()

    assert var_end > 2.0 * var_mid
    assert var_end / var_mid == pytest.approx(t_end / t_mid, rel=0.35)


def test_ar1_negative_b_oscillates():
    """Negative serial correlation, the shape bid-ask bounce induces in a
    closing-price series. No valid half-life exists."""
    path = simulate_ar1(-0.5, 20_000, seed=1)
    assert np.corrcoef(path[:-1], path[1:])[0, 1] < -0.3


def test_ar1_matches_requested_coefficient():
    path = simulate_ar1(0.7, 100_000, seed=1)
    assert np.corrcoef(path[:-1], path[1:])[0, 1] == pytest.approx(0.7, abs=0.02)
