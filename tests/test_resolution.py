"""Tests for the resolution guards.

These prove the pipeline refuses to report a number when the data cannot
support one. Each test corresponds to a way a half-life can be meaningless
while still looking perfectly reasonable in a results table.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.reversion import (
    Resolution,
    estimate_reversion,
    half_life_to_kappa,
    simulate_ar1,
    simulate_ou,
)

SAMPLE_N = 2185  # the real HYG sample size


# --------------------------------------------------------------------------
# Too slow to resolve: unit root
# --------------------------------------------------------------------------


def test_random_walk_is_labelled_unit_root():
    """The case the algebraic b >= 1 guard cannot catch. Unguarded, this same
    series reports a confident 293-day half-life."""
    r = estimate_reversion(simulate_ar1(1.0, SAMPLE_N, seed=1), ticker="WALK")
    assert r.resolution is Resolution.UNIT_ROOT
    assert np.isnan(r.half_life)
    assert not r.is_identified
    assert "unit root" in r.reason


def test_unit_root_suppresses_the_spurious_number():
    """Suppression is the point: a value IS computable, and it is nonsense."""
    from src.analysis.reversion import fit_ar1, half_life_from_b

    path = simulate_ar1(1.0, SAMPLE_N, seed=1)
    naive = half_life_from_b(fit_ar1(path).b)

    assert np.isfinite(naive)
    assert naive > 200  # a confident, entirely spurious figure
    assert np.isnan(estimate_reversion(path, ticker="WALK").half_life)


@pytest.mark.parametrize("b", [0.995, 0.999, 1.0])
def test_near_unit_root_is_caught(b):
    r = estimate_reversion(simulate_ar1(b, SAMPLE_N, seed=2), ticker="SLOW")
    assert r.resolution is Resolution.UNIT_ROOT


# --------------------------------------------------------------------------
# Too fast to resolve: censoring
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_hl", [0.02, 0.05, 0.1])
def test_sub_resolution_half_lives_are_unresolved(true_hl):
    """Daily sampling aliases anything below ~0.1 days to b = 0. The estimator
    must refuse a point estimate rather than invent one from noise."""
    path = simulate_ou(half_life_to_kappa(true_hl), SAMPLE_N, seed=1)
    r = estimate_reversion(path, ticker="FAST")

    assert r.resolution is Resolution.UNRESOLVED
    assert np.isnan(r.half_life)
    assert np.isfinite(r.half_life_upper_bound)
    assert "unresolved" in r.display


def test_white_noise_is_censored_not_measured():
    """Pure noise has no reversion speed at all. A naive estimator returns a
    plausible ~0.2 days here, which is the exact trap this guard closes."""
    rng = np.random.default_rng(0)
    r = estimate_reversion(rng.standard_normal(SAMPLE_N), ticker="NOISE")
    assert r.resolution is Resolution.UNRESOLVED
    assert np.isnan(r.half_life)


def test_censor_bound_is_an_upper_bound():
    """The bound must exceed the truth -- it is a statement about what can be
    ruled out, not an estimate."""
    true_hl = 0.05
    r = estimate_reversion(
        simulate_ou(half_life_to_kappa(true_hl), SAMPLE_N, seed=3), ticker="FAST"
    )
    assert r.half_life_upper_bound > true_hl
    assert r.half_life_upper_bound < 1.0  # still an informative bound


def test_unresolved_display_leads_with_the_bound():
    """The bound is the claim; the fitted value is shown after it so a loose
    bound cannot be mistaken for a measurement of speed."""
    r = estimate_reversion(
        simulate_ou(half_life_to_kappa(0.03), SAMPLE_N, seed=4), ticker="FAST"
    )
    assert r.display.startswith("<")
    assert "unresolved" in r.display


# --------------------------------------------------------------------------
# Oscillatory
# --------------------------------------------------------------------------


def test_negative_autocorrelation_is_labelled_oscillatory():
    """The shape bid-ask bounce induces in a closing-price series."""
    r = estimate_reversion(simulate_ar1(-0.4, SAMPLE_N, seed=1), ticker="BOUNCE")
    assert r.resolution is Resolution.OSCILLATORY
    assert np.isnan(r.half_life)
    assert "bounce" in r.reason.lower()


# --------------------------------------------------------------------------
# The guard must not swallow genuine results
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_hl", [1.0, 2.0, 3.0, 5.0, 10.0])
def test_resolvable_half_lives_are_identified(true_hl):
    """A guard that rejects everything is useless. The funds that carry the
    cross-sectional result -- credit and EM, at multi-day half-lives -- must
    come back identified with accurate values."""
    path = simulate_ou(half_life_to_kappa(true_hl), SAMPLE_N, seed=1)
    r = estimate_reversion(path, ticker="OK")

    assert r.resolution is Resolution.IDENTIFIED
    assert r.is_identified
    assert r.half_life == pytest.approx(true_hl, rel=0.25)
    assert np.isnan(r.half_life_upper_bound)


def test_identified_display_is_a_number():
    r = estimate_reversion(
        simulate_ou(half_life_to_kappa(3.0), SAMPLE_N, seed=1), ticker="OK"
    )
    assert r.display.endswith(" d")
    assert not r.display.startswith("<")


def test_boundary_between_unresolved_and_identified():
    """Around 0.25 days the regimes meet. Either verdict is legitimate there,
    but the two must never be confused: a censored result carries a bound and
    no estimate, an identified one carries an estimate and no bound."""
    r = estimate_reversion(
        simulate_ou(half_life_to_kappa(0.25), SAMPLE_N, seed=1), ticker="EDGE"
    )
    assert r.resolution in (
        Resolution.UNRESOLVED,
        Resolution.IDENTIFIED_SUBDAILY,
        Resolution.IDENTIFIED,
    )
    if r.is_identified:
        assert np.isfinite(r.half_life)
        assert np.isnan(r.half_life_upper_bound)
    else:
        assert np.isnan(r.half_life)
        assert np.isfinite(r.half_life_upper_bound)


# --------------------------------------------------------------------------
# Reporting contract
# --------------------------------------------------------------------------


def test_every_result_is_labelled():
    """No unlabelled numbers in the results table."""
    cases = [
        simulate_ar1(1.0, SAMPLE_N, seed=1),
        simulate_ou(half_life_to_kappa(0.05), SAMPLE_N, seed=1),
        simulate_ou(half_life_to_kappa(3.0), SAMPLE_N, seed=1),
        simulate_ar1(-0.4, SAMPLE_N, seed=1),
    ]
    for path in cases:
        row = estimate_reversion(path, ticker="X").as_row()
        assert row["resolution"] in {r.value for r in Resolution}
        if row["resolution"] != "identified":
            assert np.isnan(row["half_life"])


def test_alpha_is_configurable():
    path = simulate_ar1(0.99, SAMPLE_N, seed=5)
    assert estimate_reversion(path, alpha=0.01).alpha == 0.01


def test_short_series_raises():
    with pytest.raises(ValueError, match="at least 20"):
        estimate_reversion(np.arange(10.0), ticker="TINY")


# --------------------------------------------------------------------------
# Sub-daily identification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_hl", [0.3, 0.5, 0.8])
def test_half_life_below_sampling_interval_is_flagged_subdaily(true_hl):
    """Real, but describing decay that mostly happens between observations.

    These pass both statistical tests -- persistence is distinguishable from
    zero and from a unit root -- so censoring them would discard a genuine
    result. But a half-life shorter than the gap between observations is not
    a measurement in the same sense as a multi-day one, and it must not sit
    unmarked in a column beside LQD's 1.49 days.
    """
    r = estimate_reversion(simulate_ou(half_life_to_kappa(true_hl), SAMPLE_N, seed=1), ticker="F")
    assert r.resolution is Resolution.IDENTIFIED_SUBDAILY
    assert r.half_life == pytest.approx(true_hl, rel=0.3)
    assert "sub-daily" in r.display


@pytest.mark.parametrize("true_hl", [1.5, 3.0, 8.0])
def test_multi_day_half_lives_stay_plain_identified(true_hl):
    """The guard must not swallow the funds carrying the cross-sectional
    result. HYG, EMB and LQD all live here."""
    r = estimate_reversion(simulate_ou(half_life_to_kappa(true_hl), SAMPLE_N, seed=1), ticker="F")
    assert r.resolution is Resolution.IDENTIFIED
    assert "sub-daily" not in r.display


def test_subdaily_still_counts_as_identified():
    """is_identified means 'a usable point estimate exists', which sub-daily
    results do have -- unlike censored or unit-root ones."""
    r = estimate_reversion(simulate_ou(half_life_to_kappa(0.4), SAMPLE_N, seed=1), ticker="F")
    assert r.is_identified
    assert r.is_subdaily
    assert np.isfinite(r.half_life)


def test_too_fast_is_unresolved_rather_than_subdaily():
    """The three tiers must stay distinct. Below roughly 0.1 days the series
    is indistinguishable from white noise, and then no point estimate is
    warranted at all."""
    r = estimate_reversion(simulate_ou(half_life_to_kappa(0.05), SAMPLE_N, seed=1), ticker="F")
    assert r.resolution is Resolution.UNRESOLVED
    assert np.isnan(r.half_life)


def test_sampling_interval_is_configurable():
    """Nothing about the boundary is specific to daily data. With hourly
    observations the same series would be fully resolved."""
    path = simulate_ou(half_life_to_kappa(0.5), SAMPLE_N, seed=1)
    daily = estimate_reversion(path, ticker="F", sampling_interval=1.0)
    finer = estimate_reversion(path, ticker="F", sampling_interval=0.1)
    assert daily.resolution is Resolution.IDENTIFIED_SUBDAILY
    assert finer.resolution is Resolution.IDENTIFIED


def test_subdaily_results_still_get_intervals():
    r = estimate_reversion(
        simulate_ou(half_life_to_kappa(0.4), SAMPLE_N, seed=1),
        ticker="F", intervals=True, n_boot=200,
    )
    assert r.bootstrap_ci is not None and r.bootstrap_ci.is_defined
    assert r.delta_ci is not None


def test_reason_explains_the_limitation():
    r = estimate_reversion(simulate_ou(half_life_to_kappa(0.4), SAMPLE_N, seed=1), ticker="F")
    assert "between observations" in r.reason
