"""Premium / discount construction.

The premium is the RELATIVE deviation of market price from NAV:

    premium = price / nav - 1

A ratio rather than a difference, because the cross-sectional comparison
depends on it. A ten-cent gap means something entirely different on a $40 fund
than on a $400 one, and half-lives from series measured in different units are
not comparable. The ratio makes every fund's deviation dimensionless and so
rankable.

Reported in basis points. For large-cap equity funds these deviations live in
single digits of bps, where percentages read as rounding noise.

Sign convention: positive is a PREMIUM (price above NAV), negative is a
DISCOUNT (price below NAV).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

BPS_PER_UNIT = 1e4


def add_premium(
    frame: pd.DataFrame,
    *,
    price_col: str = "close",
    nav_col: str = "nav",
) -> pd.DataFrame:
    """Attach ``premium`` and ``premium_bps`` columns to an aligned frame.

    Args:
        frame: Aligned output of ``align_price_nav``.
        price_col: Must be the UNADJUSTED close, to match the NAV convention.
        nav_col: Issuer-published NAV.

    Returns:
        A copy of ``frame`` with ``premium`` (decimal) and ``premium_bps``.

    Raises:
        ValueError: if a required column is missing or NAV is non-positive,
            which would make the ratio meaningless rather than merely wrong.
    """
    for col in (price_col, nav_col):
        if col not in frame.columns:
            raise ValueError(f"frame has no {col!r} column; got {list(frame.columns)}")

    nav = frame[nav_col]
    if not (nav > 0).all():
        bad = nav[~(nav > 0)]
        raise ValueError(
            f"{len(bad)} non-positive NAV values make the premium ratio undefined, "
            f"e.g. {bad.head(3).to_dict()}"
        )

    out = frame.copy()
    out["premium"] = out[price_col] / nav - 1.0
    out["premium_bps"] = out["premium"] * BPS_PER_UNIT
    return out


@dataclass(frozen=True)
class PremiumStats:
    """Descriptive statistics for one fund's premium series."""

    ticker: str
    n: int
    mean_bps: float
    median_bps: float
    sd_bps: float
    min_bps: float
    max_bps: float
    frac_premium: float
    """Fraction of days trading above NAV."""
    t_stat: float
    p_value: float
    """Two-sided test that the mean premium is zero."""

    @property
    def mean_differs_from_zero(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        flag = " [MEAN != 0]" if self.mean_differs_from_zero else ""
        return (
            f"{self.ticker}: n={self.n} mean={self.mean_bps:+.2f}bps "
            f"sd={self.sd_bps:.2f} range=[{self.min_bps:+.1f}, {self.max_bps:+.1f}] "
            f"premium_days={self.frac_premium:.1%} p={self.p_value:.3g}{flag}"
        )


def describe_premium(frame: pd.DataFrame, *, ticker: str) -> PremiumStats:
    """Summarise a premium series, including a test that its mean is zero.

    The zero-mean test is a DATA-QUALITY alarm, not a finding. A persistently
    non-zero average premium is far more likely to be a convention mismatch --
    an adjusted price series, a NAV struck at a different time, a currency
    difference -- than a persistent arbitrage the entire market has left on
    the table. Treat a significant result as a reason to re-examine ingestion
    before interpreting anything.

    The exception is structural: SPY is a unit investment trust and cannot
    reinvest dividends between distributions, so a small persistent premium
    drag there is expected rather than suspicious.
    """
    if "premium_bps" not in frame.columns:
        raise ValueError("frame has no 'premium_bps'; call add_premium first")

    series = frame["premium_bps"].dropna()
    if series.empty:
        raise ValueError(f"{ticker}: premium series is empty")

    result = stats.ttest_1samp(series, popmean=0.0)

    stats_obj = PremiumStats(
        ticker=ticker,
        n=len(series),
        mean_bps=float(series.mean()),
        median_bps=float(series.median()),
        sd_bps=float(series.std(ddof=1)),
        min_bps=float(series.min()),
        max_bps=float(series.max()),
        frac_premium=float((series > 0).mean()),
        t_stat=float(result.statistic),
        p_value=float(result.pvalue),
    )
    logger.info("%s", stats_obj.summary())
    return stats_obj
