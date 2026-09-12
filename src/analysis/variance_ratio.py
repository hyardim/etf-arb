"""Variance ratios -- a model-free check on the reversion verdicts.

Every half-life in this project comes from one model: an AR(1) fitted by OLS.
That creates a blind spot. When the estimator reports "reverts faster than
daily data resolves," two very different situations produce that answer:

    (a) the deviation genuinely does revert inside a day, or
    (b) an AR(1) is the wrong shape for this series and the fit is empty.

The censored funds -- SPY, ITOT, EWJ -- rest entirely on that verdict, so it
is worth corroborating with something that shares no assumptions with it.

THE IDEA, which needs no model. For a series with no memory, variance
accumulates in proportion to time: the variance of a 5-day change is five
times the variance of a 1-day change. Mean reversion pulls excursions back, so
variance grows more slowly than that. Define

    VR(q) = Var(x[t+q] - x[t]) / (q * Var(x[t+1] - x[t]))

and read it directly:

    VR = 1    no memory; a random walk
    VR < 1    mean reverting; the faster the reversion, the lower the ratio
    VR > 1    trending

There is no regression here and nothing is fitted. If the AR(1) says a fund
reverts quickly, VR(q) should be well below 1 and should keep falling as q
grows, because a longer horizon gives the reversion more time to act. Two
independent methods agreeing is worth considerably more than either alone.

For a pure AR(1) the two are linked analytically, which gives a way to check
the implementation:

    VR(q) = 1 + 2b/(1-b) * (1 - (1-b^q)/(q*(1-b)))

That relationship is used in the tests, not in the reporting -- the point of
this module is to measure the ratio empirically and see whether it agrees.

The standard error uses the Lo-MacKinlay heteroskedasticity-robust form, since
premium volatility clusters violently in stress periods and the homoskedastic
version would understate uncertainty exactly where it matters.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

DEFAULT_HORIZONS = (2, 5, 10, 20)


@dataclass(frozen=True)
class VarianceRatioResult:
    """Variance ratio at one horizon."""

    ticker: str
    horizon: int
    ratio: float
    z_stat: float
    """Lo-MacKinlay robust test statistic against the null VR = 1."""
    p_value: float
    n: int

    @property
    def rejects_random_walk(self) -> bool:
        return self.p_value < 0.05

    @property
    def verdict(self) -> str:
        if not self.rejects_random_walk:
            return "random walk"
        return "mean reverting" if self.ratio < 1 else "trending"


@dataclass(frozen=True)
class VarianceRatioProfile:
    """Variance ratios across horizons for one fund."""

    ticker: str
    results: tuple[VarianceRatioResult, ...] = field(default=())

    @property
    def all_below_one(self) -> bool:
        return all(r.ratio < 1.0 for r in self.results)

    @property
    def monotone_decreasing(self) -> bool:
        """Reversion should bite harder at longer horizons."""
        ratios = [r.ratio for r in self.results]
        return all(a >= b for a, b in zip(ratios, ratios[1:]))

    @property
    def confirms_mean_reversion(self) -> bool:
        """Model-free agreement with an AR(1) verdict of 'reverts'."""
        return self.all_below_one and any(r.rejects_random_walk for r in self.results)

    def summary(self) -> str:
        parts = " ".join(f"q={r.horizon}:{r.ratio:.2f}" for r in self.results)
        return f"{self.ticker}: {parts} -> {'reverts' if self.confirms_mean_reversion else 'inconclusive'}"

    def as_rows(self) -> list[dict]:
        return [
            {
                "ticker": r.ticker,
                "horizon": r.horizon,
                "vr": round(r.ratio, 4),
                "z": round(r.z_stat, 2),
                "p": round(r.p_value, 5),
                "verdict": r.verdict,
            }
            for r in self.results
        ]


def variance_ratio(x: np.ndarray | pd.Series, q: int, *, ticker: str = "") -> VarianceRatioResult:
    """Lo-MacKinlay variance ratio at horizon ``q``, robust to heteroskedasticity.

    Args:
        x: The series in levels -- the premium itself, not its changes. The
           differencing happens here.
        q: Horizon in periods. Must be at least 2.
    """
    values = np.asarray(x, dtype=float).ravel()
    values = values[np.isfinite(values)]
    n = values.size
    if q < 2:
        raise ValueError(f"horizon must be at least 2, got {q}")
    if n < q * 10:
        raise ValueError(f"need at least {q * 10} observations for q={q}, got {n}")

    diffs = np.diff(values)
    mu = diffs.mean()
    m = diffs.size

    var_1 = np.sum((diffs - mu) ** 2) / (m - 1)

    # Overlapping q-period differences, with the standard unbiasing constant.
    q_diffs = values[q:] - values[:-q]
    denom = (m - q + 1) * (1.0 - q / m)
    var_q = np.sum((q_diffs - q * mu) ** 2) / (q * denom)

    if var_1 <= 0:
        raise ValueError(f"{ticker}: zero variance in first differences")

    ratio = float(var_q / var_1)

    # Lo-MacKinlay heteroskedasticity-robust variance of the ratio. Premium
    # volatility clusters hard in stress, so the homoskedastic form would
    # understate uncertainty precisely where the data is most informative.
    centred = (diffs - mu) ** 2
    theta = 0.0
    for j in range(1, q):
        num = np.sum(centred[j:] * centred[:-j])
        delta = num / (np.sum(centred) ** 2 / m)
        theta += (2.0 * (q - j) / q) ** 2 * delta / m

    se = np.sqrt(theta) if theta > 0 else np.nan
    z = float((ratio - 1.0) / se) if np.isfinite(se) and se > 0 else float("nan")
    p = float(2.0 * (1.0 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else float("nan")

    return VarianceRatioResult(
        ticker=ticker, horizon=q, ratio=ratio, z_stat=z, p_value=p, n=n
    )


def profile_variance_ratios(
    x: np.ndarray | pd.Series,
    *,
    ticker: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> VarianceRatioProfile:
    """Variance ratios across several horizons."""
    results = tuple(variance_ratio(x, q, ticker=ticker) for q in horizons)
    profile = VarianceRatioProfile(ticker=ticker, results=results)
    logger.info("%s", profile.summary())
    return profile
