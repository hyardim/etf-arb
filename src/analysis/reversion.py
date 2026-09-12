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

from statsmodels.tsa.stattools import adfuller  # noqa: E402

DEFAULT_ALPHA = 0.05


class Resolution(StrEnum):
    """Whether a half-life is identified, and if not, why not."""

    IDENTIFIED = "identified"
    """Distinguishable from both a random walk and white noise."""

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

    @property
    def is_identified(self) -> bool:
        return self.resolution is Resolution.IDENTIFIED

    @property
    def display(self) -> str:
        """Human-readable half-life, never a bare number."""
        if self.resolution is Resolution.IDENTIFIED:
            return f"{self.half_life:.2f} d"
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
        resolution = Resolution.IDENTIFIED
        reason = (
            f"Distinguishable from both a random walk (ADF p={adf_pvalue:.3g}) and "
            f"white noise (p(b=0)={fit.b_pvalue:.3g})."
        )

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
    )
    logger.info("%s", result.summary())
    return result
