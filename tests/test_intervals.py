"""Tests for half-life confidence intervals.

The decisive test is coverage: an interval claiming 95% confidence must
actually contain the true value about 95% of the time. Everything else about
a confidence interval is bookkeeping; if coverage is wrong, the interval is
worse than none, because it invites a comparison it cannot support.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.reversion import (
    Interval,
    bootstrap_interval,
    delta_method_interval,
    fit_ar1,
    half_life_from_b,
    half_life_to_kappa,
    simulate_ar1,
    simulate_ou,
    stationary_bootstrap_indices,
)

SAMPLE_N = 2185  # the real sample size


# --------------------------------------------------------------------------
# Delta method
# --------------------------------------------------------------------------


def test_delta_interval_brackets_the_estimate():
    b, se = 0.4767, 0.0505  # HYG
    ci = delta_method_interval(b, se)
    assert ci.lower < half_life_from_b(b) < ci.upper


def test_delta_matches_closed_form():
    """SE(h) = |ln2 / (b (ln b)^2)| * SE(b), then +/- 1.96 SE."""
    b, se = 0.4767, 0.0505
    expected_se = abs(np.log(2) / (b * np.log(b) ** 2)) * se
    ci = delta_method_interval(b, se)
    assert ci.width / 2 == pytest.approx(1.96 * expected_se, rel=0.01)


def test_delta_amplification_grows_with_b():
    """The derivative is an amplification factor that rises steeply as b
    approaches 1. It is why LQD's interval is far wider than HYG's despite a
    b standard error only twice as large."""
    same_se = 0.05
    widths = [delta_method_interval(b, same_se).width for b in (0.1, 0.3, 0.5, 0.7, 0.9)]
    assert widths == sorted(widths)


def test_delta_wider_with_noisier_b():
    assert delta_method_interval(0.5, 0.10).width > delta_method_interval(0.5, 0.02).width


def test_delta_lower_bound_never_negative():
    """A negative half-life is meaningless."""
    assert delta_method_interval(0.1, 0.5).lower == 0.0


def test_delta_undefined_where_no_half_life_exists():
    for b in (0.0, 1.0, 1.5, -0.3):
        assert not delta_method_interval(b, 0.05).is_defined


# --------------------------------------------------------------------------
# Block resampling mechanics
# --------------------------------------------------------------------------


def test_bootstrap_indices_right_length_and_in_range():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(1000, 12.0, rng)
    assert idx.shape == (1000,)
    assert idx.min() >= 0 and idx.max() < 1000


def test_blocks_are_contiguous_runs():
    """The essential property. Resampling individual days would destroy the
    serial dependence that is the quantity being measured."""
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(5000, 20.0, rng)
    steps = np.diff(idx)
    # Most transitions continue a block, i.e. advance by exactly one.
    assert (steps == 1).mean() > 0.8


def test_shorter_expected_block_breaks_more_often():
    rng = np.random.default_rng(0)
    short = np.diff(stationary_bootstrap_indices(5000, 3.0, rng))
    long = np.diff(stationary_bootstrap_indices(5000, 50.0, rng))
    assert (short != 1).mean() > (long != 1).mean()


def test_shuffling_destroys_the_signal_blocks_preserve_it():
    """Why blocks are used at all: an i.i.d. bootstrap of a mean-reverting
    series collapses b toward zero and reports no reversion."""
    path = simulate_ou(half_life_to_kappa(3.0), 5000, seed=1)
    rng = np.random.default_rng(0)

    block = path[stationary_bootstrap_indices(5000, 17.0, rng)]
    shuffled = rng.permutation(path)

    assert abs(fit_ar1(shuffled).b) < 0.05                     # memory destroyed
    assert fit_ar1(block).b > 0.5 * fit_ar1(path).b            # largely retained


def test_resampling_raw_values_attenuates_b():
    """Documents the bug the coverage test caught.

    Gluing resampled blocks of RAW VALUES together splices each block onto an
    unrelated predecessor, and that seam pair carries no autocorrelation.
    Those spurious pairs pull b down, which pulled every interval below the
    truth: on simulated paths of known half-life 3.0 the raw-series bootstrap
    produced intervals centred near 2.25 with 0% coverage.

    Resampling blocks of (lagged, current) PAIRS instead keeps every pair
    genuine, which is what the implementation now does.
    """
    path = simulate_ou(half_life_to_kappa(3.0), 5000, seed=1)
    rng = np.random.default_rng(0)
    b_true = fit_ar1(path).b

    # Short blocks mean many seams, so the attenuation is stark.
    spliced = path[stationary_bootstrap_indices(5000, 4.0, rng)]
    assert fit_ar1(spliced).b < 0.8 * b_true


# --------------------------------------------------------------------------
# Coverage -- the test that matters
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_hl", [1.0, 3.0])
def test_bootstrap_coverage_is_near_nominal(true_hl):
    """A 95% interval must contain the truth about 95% of the time.

    Run over independent simulated samples of the real sample size. The band
    is generous because 60 trials is itself a noisy estimate of coverage --
    the test is designed to catch an interval that is badly wrong, not to
    certify it to two decimals.
    """
    kappa = half_life_to_kappa(true_hl)
    hits = 0
    trials = 60
    for s in range(trials):
        path = simulate_ou(kappa, SAMPLE_N, seed=1000 + s)
        ci = bootstrap_interval(path, n_boot=200, seed=s)
        if ci.is_defined and ci.contains(true_hl):
            hits += 1
    coverage = hits / trials
    assert 0.80 <= coverage <= 1.0, f"coverage {coverage:.0%} for half-life {true_hl}"


def test_bootstrap_brackets_the_point_estimate():
    path = simulate_ou(half_life_to_kappa(3.0), SAMPLE_N, seed=1)
    ci = bootstrap_interval(path, n_boot=400, seed=0)
    assert ci.contains(half_life_from_b(fit_ar1(path).b))


def test_bootstrap_narrows_with_more_data():
    kappa = half_life_to_kappa(3.0)
    small = bootstrap_interval(simulate_ou(kappa, 500, seed=1), n_boot=300, seed=0)
    large = bootstrap_interval(simulate_ou(kappa, 8000, seed=1), n_boot=300, seed=0)
    assert large.width < small.width


# --------------------------------------------------------------------------
# Transform order -- documenting a subtlety
# --------------------------------------------------------------------------


def test_percentiles_commute_with_the_transform():
    """Quantiles are equivariant under a monotone transform, so bootstrapping
    b and transforming the endpoints gives the SAME percentile interval as
    transforming each replicate first.

    Transforming first is still the right choice, but for a different reason:
    a replicate landing on b <= 0 has no half-life, and transforming first
    lets those be counted and excluded explicitly.
    """
    def b_stat(pairs):
        lag, cur = pairs[:, 0], pairs[:, 1]
        design = np.column_stack([np.ones(lag.size), lag])
        return float(np.linalg.lstsq(design, cur, rcond=None)[0][1])

    path = simulate_ou(half_life_to_kappa(3.0), 1500, seed=1)
    on_half_life = bootstrap_interval(path, n_boot=300, seed=7)
    on_b = bootstrap_interval(path, n_boot=300, seed=7, statistic=b_stat)

    # Agreement is to within quantile interpolation, not to machine precision:
    # np.quantile interpolates linearly BETWEEN order statistics, and linear
    # interpolation does not commute with a nonlinear transform. The
    # equivariance is exact for the order statistics themselves.
    assert half_life_from_b(on_b.lower) == pytest.approx(on_half_life.lower, rel=1e-3)
    assert half_life_from_b(on_b.upper) == pytest.approx(on_half_life.upper, rel=1e-3)


def test_invalid_replicates_are_excluded_not_coerced():
    """A random walk yields few valid half-lives; the interval must come back
    undefined rather than be manufactured from whichever replicates survived."""
    ci = bootstrap_interval(simulate_ar1(1.0, 1000, seed=1), n_boot=200, seed=0)
    assert not ci.is_defined or ci.n_valid >= 100


def test_white_noise_gives_few_valid_replicates():
    rng = np.random.default_rng(0)
    ci = bootstrap_interval(rng.standard_normal(1500), n_boot=200, seed=0)
    assert ci.n_valid < 200  # many replicates land on b <= 0


# --------------------------------------------------------------------------
# Interval object
# --------------------------------------------------------------------------


def test_overlap_detection():
    """The operation behind every ranking claim in the cross-section."""
    hyg = Interval(0.67, 1.20, "delta")
    emb = Interval(0.72, 1.91, "delta")
    ivv = Interval(0.21, 0.38, "delta")

    assert hyg.overlaps(emb)       # indistinguishable
    assert not ivv.overlaps(hyg)   # genuinely separated
    assert hyg.overlaps(hyg)


def test_undefined_intervals_never_claimed_distinguishable():
    undefined = Interval(float("nan"), float("nan"), "bootstrap")
    assert undefined.overlaps(Interval(1.0, 2.0, "delta"))


def test_short_series_raises():
    with pytest.raises(ValueError, match="at least 20"):
        bootstrap_interval(np.arange(10.0))
