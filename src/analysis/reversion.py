"""Mean-reversion estimation: OU simulation, AR(1) fitting, half-lives.

The premium is modelled as a discretely sampled Ornstein-Uhlenbeck process,
which in discrete time is an AR(1):

    x[t+1] = a + b * x[t] + e[t+1]

    kappa     = -ln(b)            mean reversion speed, per period
    half_life = ln(2) / kappa     periods to close half the gap
    long_run  = a / (1 - b)       equilibrium premium level

Half-life is the headline number of this project, and half-life estimators are
easy to get subtly wrong -- sign errors and log errors produce plausible-looking
but incorrect values that no amount of staring at real data will reveal. The
simulator below exists so the estimator can be validated against a process
whose parameter is known by construction, which is the difference between a
measurement and a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


def half_life_to_kappa(half_life: float) -> float:
    """Convert a half-life in periods to an OU mean-reversion speed."""
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return float(np.log(2.0) / half_life)


def kappa_to_half_life(kappa: float) -> float:
    """Convert an OU mean-reversion speed to a half-life in periods."""
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    return float(np.log(2.0) / kappa)


def simulate_ou(
    kappa: float,
    n: int,
    *,
    seed: int | None = None,
    sigma: float = 1.0,
    mu: float = 0.0,
    dt: float = 1.0,
    x0: float | None = None,
) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck path with a known mean-reversion speed.

    Uses the EXACT discrete transition rather than an Euler-Maruyama step:

        x[t+1] = mu + (x[t] - mu) * exp(-kappa*dt) + sigma * sqrt(v) * z

    with ``v = (1 - exp(-2*kappa*dt)) / (2*kappa)``.

    This matters for the purpose at hand. Euler discretisation carries an
    O(dt) bias in the autoregressive coefficient, which maps straight into a
    biased half-life. Since these paths are the yardstick the estimator is
    measured against, the yardstick must not itself be bent -- otherwise a
    recovery test would be comparing one approximation to another and could
    pass while both are wrong.

    Args:
        kappa: Mean-reversion speed. Half-life is ln(2)/kappa.
        n: Number of observations.
        seed: Seed for reproducibility.
        sigma: Instantaneous volatility.
        mu: Long-run mean.
        dt: Sampling interval, in the same units as kappa.
        x0: Starting value. Defaults to a draw from the stationary
            distribution, so the path needs no burn-in.

    Returns:
        Array of length ``n``.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    rng = np.random.default_rng(seed)

    decay = np.exp(-kappa * dt)
    step_var = sigma**2 * (1.0 - decay**2) / (2.0 * kappa)
    step_sd = np.sqrt(step_var)

    # Stationary variance of the OU process; starting here avoids burn-in.
    stationary_sd = sigma / np.sqrt(2.0 * kappa)

    path = np.empty(n, dtype=float)
    path[0] = mu + stationary_sd * rng.standard_normal() if x0 is None else x0

    shocks = rng.standard_normal(n - 1) * step_sd
    for t in range(1, n):
        path[t] = mu + (path[t - 1] - mu) * decay + shocks[t - 1]

    return path


def simulate_ar1(
    b: float,
    n: int,
    *,
    seed: int | None = None,
    a: float = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Simulate a plain AR(1), including cases an OU cannot represent.

    ``simulate_ou`` requires kappa > 0, so b must lie in (0, 1). This function
    accepts any b, which is needed to exercise the estimator's guard clauses:
    b >= 1 (a random walk, no finite half-life) and b <= 0 (oscillatory, the
    shape bid-ask bounce induces in a closing-price series).
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n) * sigma

    path = np.empty(n, dtype=float)
    path[0] = shocks[0] / np.sqrt(1 - b**2) if abs(b) < 1 else shocks[0]
    for t in range(1, n):
        path[t] = a + b * path[t - 1] + shocks[t]

    return path


# ==========================================================================
# AR(1) estimation
# ==========================================================================

import statsmodels.api as sm  # noqa: E402


@dataclass(frozen=True)
class AR1Fit:
    """Raw AR(1) regression output, before any half-life interpretation."""

    n: int
    a: float
    b: float
    b_se: float
    b_tstat: float
    b_pvalue: float
    """Two-sided p-value for H0: b = 0, i.e. no persistence at all."""
    resid_sd: float

    @property
    def long_run(self) -> float:
        """Equilibrium level a/(1-b). Undefined at b = 1."""
        return float(self.a / (1.0 - self.b)) if self.b != 1.0 else float("nan")


def fit_ar1(x: np.ndarray | "pd.Series", *, cov_type: str = "HC1") -> AR1Fit:
    """Fit x[t+1] = a + b*x[t] + e by OLS with robust standard errors.

    HC1 is used because premium residuals are strongly heteroskedastic --
    variance is an order of magnitude larger in stress periods than in calm
    ones, as the 2020 column of the HYG sample shows. Classical standard
    errors would understate uncertainty exactly where it is largest.
    """
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError(f"need at least 3 finite observations, got {values.size}")

    lagged, current = values[:-1], values[1:]
    design = sm.add_constant(lagged, has_constant="add")
    res = sm.OLS(current, design).fit(cov_type=cov_type)

    return AR1Fit(
        n=int(current.size),
        a=float(res.params[0]),
        b=float(res.params[1]),
        b_se=float(res.bse[1]),
        b_tstat=float(res.tvalues[1]),
        b_pvalue=float(res.pvalues[1]),
        resid_sd=float(np.std(res.resid, ddof=2)),
    )


def half_life_from_b(b: float) -> float:
    """Convert an AR(1) coefficient to a half-life in periods.

    Returns NaN where no finite half-life exists:

    - ``b >= 1``  non-stationary. Shocks never decay, so there is nothing to
      halve. A "half-life" computed anyway is arbitrarily large and unstable
      across subsamples.
    - ``b <= 0``  oscillatory rather than decaying. The series alternates sign
      instead of relaxing toward a mean, so the half-life concept does not
      apply. Bid-ask bounce in a closing-price series induces exactly this.

    Both cases are returned as NaN rather than raising, so a cross-sectional
    run can report "no finite half-life" for one fund and carry on.
    """
    if not np.isfinite(b) or b <= 0.0 or b >= 1.0:
        return float("nan")
    return float(np.log(2.0) / -np.log(b))


def estimate_half_life(x: np.ndarray | "pd.Series", *, cov_type: str = "HC1") -> float:
    """Estimate the half-life of mean reversion, in sampling periods.

    The direct interface used by the validation tests: feed a simulated path
    with a known half-life and confirm the value comes back.
    """
    return half_life_from_b(fit_ar1(x, cov_type=cov_type).b)


# ==========================================================================
# Resolution: deciding whether a half-life is identified at all
# ==========================================================================
#
# Daily close-to-close sampling can resolve a half-life only inside a window,
# and the two ends fail for opposite reasons.
#
# TOO SLOW. A near-unit-root series has no finite half-life, but OLS never
# returns b >= 1 in a finite sample, so the algebraic guard never fires. It
# returns a large spurious value that scales with sample size instead: a true
# random walk gives 135 days at n=500, 293 at n=2185, 16700 at n=50000.
# Detected with an augmented Dickey-Fuller test.
#
# TOO FAST. If the premium deviates, reverts, and deviates again WITHIN a
# session, close-to-close data cannot see it. This is aliasing -- a property of
# the sampling interval, not something a better estimator fixes. Sampled
# daily, b = exp(-kappa*dt), so:
#
#     true half-life      implied daily b
#     0.02 d (~10 min)    0.000000
#     0.05 d (~25 min)    0.000001
#     0.10 d (~50 min)    0.000977
#     0.25 d              0.0625
#     1.00 d              0.5000
#
# Everything below ~0.1 days collapses to b = 0. Twenty minutes and two hours
# are the same observation.
#
# That is dangerous rather than merely limiting, because the estimator does
# not go quiet -- it returns confident nonsense. At n = 2185, SE(b) is about
# 1/sqrt(n) = 0.021, so sampling noise alone moves b across +/-0.04, which
# maps to half-lives of 0.15 to 0.30 days. Those look like measurements, and a
# bootstrap will faithfully resample the noise and dress it in a confidence
# interval. Large-cap equity funds are expected to sit exactly here.
#
# So a half-life is reported only when the series is distinguishable from BOTH
# a random walk and white noise. Otherwise it is labelled, and where the
# process is too fast a censored upper bound is given instead of a point
# estimate.

from enum import StrEnum  # noqa: E402

from scipy import stats as scipy_stats  # noqa: E402
from statsmodels.tsa.stattools import adfuller  # noqa: E402

DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOTSTRAP = 2000
DEFAULT_CI_LEVEL = 0.95


class Resolution(StrEnum):
    """Whether a half-life is identified, and if not, why not."""

    IDENTIFIED = "identified"
    """Distinguishable from both a random walk and white noise, with a
    half-life longer than the sampling interval."""

    IDENTIFIED_SUBDAILY = "identified_subdaily"
    """Statistically identified, but the point estimate falls BELOW one
    sampling interval.

    The estimate is real -- persistence is distinguishable from zero -- yet it
    describes a process that mostly completes between observations. Daily
    closes can establish that reversion is fast without pinning down how fast,
    and the fitted value depends on the tail of the distribution rather than
    on directly observed decay. Reported as a number, labelled as partial."""

    CENSORED_FAST = "censored_fast"
    """Reverts faster than daily sampling can resolve. Only an upper bound."""

    UNIT_ROOT = "unit_root"
    """Did not mean-revert over the sample. No finite half-life."""

    OSCILLATORY = "oscillatory"
    """Significantly negative persistence, e.g. bid-ask bounce."""


@dataclass(frozen=True)
class ReversionResult:
    """A half-life estimate together with whether it means anything."""

    ticker: str
    n: int
    fit: AR1Fit
    adf_stat: float
    adf_pvalue: float
    resolution: Resolution
    half_life: float
    """NaN unless resolution is IDENTIFIED."""
    half_life_upper_bound: float
    """For CENSORED_FAST, the bound implied by the upper confidence limit on
    b. NaN otherwise."""
    reason: str
    alpha: float
    bootstrap_ci: Interval | None = None
    delta_ci: Interval | None = None

    @property
    def is_identified(self) -> bool:
        """True when a usable point estimate exists, sub-daily included."""
        return self.resolution in (
            Resolution.IDENTIFIED,
            Resolution.IDENTIFIED_SUBDAILY,
        )

    @property
    def is_subdaily(self) -> bool:
        return self.resolution is Resolution.IDENTIFIED_SUBDAILY

    @property
    def display(self) -> str:
        """Human-readable half-life, never a bare number."""
        if self.resolution is Resolution.IDENTIFIED:
            return f"{self.half_life:.2f} d"
        if self.resolution is Resolution.IDENTIFIED_SUBDAILY:
            return f"{self.half_life:.2f} d (sub-daily)"
        if self.resolution is Resolution.CENSORED_FAST:
            return f"< {self.half_life_upper_bound:.2f} d (censored)"
        return f"n/a ({self.resolution.value})"

    def summary(self) -> str:
        return (
            f"{self.ticker}: b={self.fit.b:+.4f} (se {self.fit.b_se:.4f}) "
            f"adf_p={self.adf_pvalue:.4f} p(b=0)={self.fit.b_pvalue:.3g} "
            f"-> {self.resolution.value}: {self.display}"
        )

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "n": self.n,
            "b": round(self.fit.b, 6),
            "b_se": round(self.fit.b_se, 6),
            "adf_pvalue": round(self.adf_pvalue, 6),
            "b_zero_pvalue": float(f"{self.fit.b_pvalue:.3g}"),
            "resolution": self.resolution.value,
            "half_life": self.half_life,
            "half_life_upper_bound": self.half_life_upper_bound,
        }


def estimate_reversion(
    x: np.ndarray | "pd.Series",
    *,
    ticker: str = "",
    alpha: float = DEFAULT_ALPHA,
    cov_type: str = "HC1",
    intervals: bool = False,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    seed: int | None = 0,
    sampling_interval: float = 1.0,
) -> ReversionResult:
    """Estimate a half-life and classify whether it is identified.

    Applies two tests that bracket the resolvable range from opposite ends:

    - Augmented Dickey-Fuller. Failing to reject a unit root means the
      deviation did not mean-revert over the sample. That is a publishable
      result and is reported as such, rather than as an arbitrary number.
    - H0: b = 0. Failing to reject means reversion completes inside one
      sampling interval, so the speed is unresolvable. A censored upper bound
      is reported instead of a point estimate.

    Args:
        x: The premium series, typically in bps.
        ticker: Label for reporting.
        alpha: Significance level for both tests.
        cov_type: Covariance estimator for the AR(1) fit.
        sampling_interval: One observation spacing, in the units the half-life
            is reported in. Daily closes give 1.0. An identified half-life
            below this is labelled IDENTIFIED_SUBDAILY.
    """
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size < 20:
        raise ValueError(f"{ticker}: need at least 20 observations, got {values.size}")

    fit = fit_ar1(values, cov_type=cov_type)
    # result_object=False pins the current tuple return; statsmodels 0.16
    # switches the default and warns until then.
    adf = adfuller(values, autolag="AIC", result_object=False)
    adf_stat, adf_pvalue = adf[0], adf[1]

    half_life = float("nan")
    upper_bound = float("nan")

    if adf_pvalue > alpha:
        resolution = Resolution.UNIT_ROOT
        reason = (
            f"ADF p={adf_pvalue:.3f} > {alpha}: cannot reject a unit root, so the "
            f"deviation did not mean-revert over this sample. Any half-life computed "
            f"here would scale with sample size rather than describe the process."
        )
    elif fit.b_pvalue > alpha:
        # Cannot distinguish from white noise: reversion completes within one
        # interval. The upper confidence limit on b bounds how slow it could
        # plausibly be, which is the strongest honest statement available.
        b_upper = fit.b + 1.96 * fit.b_se
        upper_bound = half_life_from_b(b_upper)
        resolution = Resolution.CENSORED_FAST
        reason = (
            f"p(b=0)={fit.b_pvalue:.3f} > {alpha}: persistence is indistinguishable "
            f"from zero, so reversion completes inside one sampling interval and its "
            f"speed is not resolvable at daily frequency. Reported as an upper bound "
            f"from the upper confidence limit on b."
        )
    elif fit.b <= 0:
        resolution = Resolution.OSCILLATORY
        reason = (
            f"b={fit.b:.4f} is significantly negative: the series alternates rather "
            f"than decaying, so no half-life exists. Bid-ask bounce in the closing "
            f"print produces this."
        )
    else:
        half_life = half_life_from_b(fit.b)
        if half_life < sampling_interval:
            resolution = Resolution.IDENTIFIED_SUBDAILY
            reason = (
                f"Distinguishable from both a random walk (ADF p={adf_pvalue:.3g}) and "
                f"white noise (p(b=0)={fit.b_pvalue:.3g}), but the estimated half-life "
                f"of {half_life:.2f} is SHORTER than the {sampling_interval:g}-period "
                f"sampling interval. Most of the decay happens between observations, so "
                f"the value establishes that reversion is fast without pinning down how "
                f"fast. Treat as an order of magnitude, not a measurement."
            )
        else:
            resolution = Resolution.IDENTIFIED
            reason = (
                f"Distinguishable from both a random walk (ADF p={adf_pvalue:.3g}) and "
                f"white noise (p(b=0)={fit.b_pvalue:.3g})."
            )

    boot_ci = delta_ci = None
    if intervals and resolution in (Resolution.IDENTIFIED, Resolution.IDENTIFIED_SUBDAILY):
        # Only meaningful where a point estimate exists. A censored or
        # unit-root series has no half-life to put an interval around.
        delta_ci = delta_method_interval(fit.b, fit.b_se, level=1.0 - alpha)
        boot_ci = bootstrap_interval(values, n_boot=n_boot, level=1.0 - alpha, seed=seed)

    result = ReversionResult(
        ticker=ticker,
        n=fit.n,
        fit=fit,
        adf_stat=float(adf_stat),
        adf_pvalue=float(adf_pvalue),
        resolution=resolution,
        half_life=half_life,
        half_life_upper_bound=upper_bound,
        reason=reason,
        alpha=alpha,
        bootstrap_ci=boot_ci,
        delta_ci=delta_ci,
    )
    logger.info("%s", result.summary())
    return result


# ==========================================================================
# Confidence intervals
# ==========================================================================
#
# A half-life without an interval cannot support the claim this project is
# for. The headline is a COMPARISON -- credit reverts more slowly than equity
# -- and a comparison only holds if the intervals do not overlap. "0.94 days"
# beside "1.32 days" means nothing until you know how far either could have
# landed by luck.
#
# Two methods, reported together, because their disagreement is a diagnostic.
#
# DELTA METHOD propagates SE(b) through h(b) = ln2 / -ln(b) using a
# first-order Taylor expansion: over a small range the curve is approximately
# a straight line, so SE(h) = |dh/db| * SE(b). The derivative is an
# amplification factor, and it varies a lot across these funds -- 1.3x for
# IVV, 2.6x for HYG, 5.1x for LQD. That is why LQD's interval is so much
# wider: its b is both less precisely measured AND sits where the curve is
# steepest. The approximation degrades as b approaches 0 or 1, where the
# derivative diverges, which is exactly where several of these funds live.
#
# BLOCK BOOTSTRAP resamples the series in contiguous blocks -- preserving the
# serial dependence that IS the signal here, and which an ordinary bootstrap
# would destroy -- then re-estimates the half-life on each replicate. It
# assumes far less, and is the primary number.
#
# Expect the bootstrap WIDER than delta for the credit funds: residuals are
# strongly heteroskedastic and cluster in 2020. Narrower would be a bug signal.

from typing import Callable  # noqa: E402


@dataclass(frozen=True)
class Interval:
    """A confidence interval and how it was produced."""

    lower: float
    upper: float
    method: str
    level: float = DEFAULT_CI_LEVEL
    n_valid: int = 0
    """Bootstrap only: replicates that produced a finite half-life."""

    @property
    def width(self) -> float:
        return float(self.upper - self.lower)

    @property
    def is_defined(self) -> bool:
        return bool(np.isfinite(self.lower) and np.isfinite(self.upper))

    def contains(self, value: float) -> bool:
        return bool(self.lower <= value <= self.upper)

    def overlaps(self, other: "Interval") -> bool:
        """Whether two intervals overlap -- the test behind every ranking
        claim in the cross-section."""
        if not (self.is_defined and other.is_defined):
            return True  # undefined cannot be claimed distinguishable
        return self.lower <= other.upper and other.lower <= self.upper

    def __str__(self) -> str:
        return f"[{self.lower:.2f}, {self.upper:.2f}]" if self.is_defined else "[undefined]"


def delta_method_interval(b: float, b_se: float, *, level: float = DEFAULT_CI_LEVEL) -> Interval:
    """Half-life interval by first-order delta method.

        h(b)  = ln2 / -ln(b)
        dh/db = ln2 / (b * (ln b)^2)
        SE(h) = |dh/db| * SE(b)

    The derivative diverges as b approaches 0 or 1. That is the method
    honestly reporting that the transformation is badly behaved there, not a
    numerical artefact to clip away.
    """
    half_life = half_life_from_b(b)
    if not np.isfinite(half_life) or b_se <= 0:
        return Interval(float("nan"), float("nan"), "delta", level)

    z = float(scipy_stats.norm.ppf(0.5 + level / 2.0))
    derivative = np.log(2.0) / (b * np.log(b) ** 2)
    se = abs(derivative) * b_se

    # A negative half-life is meaningless; floor the lower bound at zero.
    return Interval(
        lower=float(max(0.0, half_life - z * se)),
        upper=float(half_life + z * se),
        method="delta",
        level=level,
    )


def stationary_bootstrap_indices(
    n: int, expected_block: float, rng: np.random.Generator
) -> np.ndarray:
    """Indices for one Politis-Romano stationary bootstrap replicate.

    Blocks are contiguous runs of the original series, wrapping at the end,
    with GEOMETRICALLY distributed lengths averaging ``expected_block``.

    Contiguity is the essential part: this series' day-to-day memory is the
    quantity being measured, and resampling individual days would destroy it,
    driving every replicate's b toward zero. Random rather than fixed block
    lengths are what make the resampled series stationary, avoiding an
    artefact at the block seams.
    """
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=np.int64)
    current = int(rng.integers(0, n))
    for i in range(n):
        idx[i] = current
        if rng.random() < p:
            current = int(rng.integers(0, n))  # start a new block
        else:
            current = (current + 1) % n  # continue the current one
    return idx


def bootstrap_interval(
    x: np.ndarray | "pd.Series",
    *,
    n_boot: int = DEFAULT_N_BOOTSTRAP,
    level: float = DEFAULT_CI_LEVEL,
    expected_block: float | None = None,
    seed: int | None = 0,
    statistic: Callable[[np.ndarray], float] | None = None,
) -> Interval:
    """Half-life interval by stationary block bootstrap.

    Each replicate rebuilds a same-length series from random contiguous
    blocks, refits the AR(1), and converts to a half-life. The interval is
    read off the percentiles of those replicate half-lives.

    Each replicate is transformed to a half-life BEFORE percentiles are taken.
    For percentile intervals this is not about skew -- quantiles are
    equivariant under a monotone transform, so the two orders agree exactly.
    It matters because a replicate landing on b <= 0 or b >= 1 has no
    half-life at all, and transforming first lets those be counted and
    excluded explicitly rather than silently folded in.

    Args:
        x: Series, typically premium in bps.
        n_boot: Replicates.
        level: Coverage, e.g. 0.95.
        expected_block: Mean block length; defaults to n**(1/3), the usual
            rule of thumb -- long enough to span local dependence, short
            enough to stay small relative to the sample.
        seed: For reproducibility.
        statistic: Override the per-replicate statistic. Receives an
            (n, 2) array of (lagged, current) pairs. Used by tests to
            bootstrap b instead of the half-life.

    Returns:
        Interval, with ``n_valid`` recording how many replicates yielded a
        finite half-life. If fewer than half did, the series sits near a
        boundary and the interval is returned undefined rather than
        manufactured from the survivors.
    """
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = values.size
    if n < 20:
        raise ValueError(f"need at least 20 observations to bootstrap, got {n}")

    if expected_block is None:
        expected_block = max(2.0, float(n ** (1.0 / 3.0)))

    # Resample blocks of consecutive PAIRS, not of raw values.
    #
    # Resampling the raw series and refitting attenuates b badly. Every seam
    # between two blocks splices a value onto an unrelated predecessor, and
    # that pair carries no autocorrelation. With blocks averaging n**(1/3)
    # about one transition in thirteen is such a seam, and those spurious
    # uncorrelated pairs pull b down. Measured on simulated paths of known
    # half-life 3.0, the raw-series version returned intervals centred near
    # 2.25 with 0% coverage -- a systematic bias, not noise.
    #
    # Forming the (lagged, current) pairs first and resampling blocks of those
    # keeps every pair genuine: seams now fall BETWEEN pairs rather than
    # through them. Blocking still preserves the serial dependence between
    # nearby pairs, which is what makes the standard errors valid.
    pairs = np.column_stack([values[:-1], values[1:]])
    n_pairs = pairs.shape[0]

    def _fit_pairs(sample: np.ndarray) -> float:
        lag, cur = sample[:, 0], sample[:, 1]
        design = np.column_stack([np.ones(lag.size), lag])
        coef, *_ = np.linalg.lstsq(design, cur, rcond=None)
        return half_life_from_b(float(coef[1]))

    stat = statistic or _fit_pairs
    rng = np.random.default_rng(seed)

    estimates = np.empty(n_boot, dtype=float)
    for r in range(n_boot):
        idx = stationary_bootstrap_indices(n_pairs, expected_block, rng)
        try:
            estimates[r] = stat(pairs[idx])
        except (ValueError, np.linalg.LinAlgError):
            estimates[r] = np.nan

    valid = estimates[np.isfinite(estimates)]
    if valid.size < 0.5 * n_boot:
        return Interval(float("nan"), float("nan"), "bootstrap", level, int(valid.size))

    alpha = (1.0 - level) / 2.0
    lower, upper = np.quantile(valid, [alpha, 1.0 - alpha])
    return Interval(float(lower), float(upper), "bootstrap", level, int(valid.size))
