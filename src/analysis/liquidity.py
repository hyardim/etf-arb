"""Liquidity proxies for the underlying basket.

The project's hypothesis is that reversion speed tracks how tradable the
underlying is. Until this module, tradability was an assumption encoded in the
sleeve labels -- "credit is less liquid than equity" -- rather than something
measured. These proxies turn that into a quantity, so the cross-sectional
claim can be tested rather than asserted.

Two measures, both computed from data already downloaded:

DOLLAR VOLUME is the easy one: close * volume, averaged. It says how much
capital can move without difficulty.

EFFECTIVE SPREAD is the one that matters more, and daily bars do not contain
it -- there are no quotes, only trades. The Corwin-Schultz (2012) estimator
recovers it from daily highs and lows, exploiting the fact that the two
components scale differently with time:

  - The variance of the true price grows LINEARLY with the interval. A
    two-day range therefore reflects twice the variance of a one-day range.
  - The bid-ask spread inflates the observed range by a roughly CONSTANT
    amount regardless of interval, since the high is struck at an ask and the
    low at a bid either way.

Comparing a single-day range against the two-day range spanning the same
period separates the two effects. The spread is what does not scale.

A caveat worth stating plainly: this measures the spread of the ETF, which is
not the same thing as the liquidity of its underlying basket. They are related
-- an ETF on illiquid collateral is expensive to make markets in, so the
spread widens -- but a heavily traded ETF can quote tightly while its
constituents barely trade at all. HYG is precisely that case. The proxy is
therefore a lower bound on how hard the arbitrage is, and is treated as
suggestive rather than decisive.

OVERNIGHT GAPS BREAK IT, AND THE FIX IS NOT ENOUGH. This is a negative result
and it is recorded rather than buried, because it determines which proxy the
cross-section actually uses.

Corwin-Schultz assumes a continuously evolving price. Real markets close. On
the SPY sample, 38.5% of days open entirely above or below the prior close,
with a median overnight move of 32bps -- an order of magnitude larger than the
~1bp spread being estimated. A gap inflates the two-day range while inflating
neither one-day range, so the "does not scale with the interval" component is
overstated and attributed to a negative spread.

Aligning the second day of each window onto the first day's close removes most
of it, and matters enormously: SPY moves from -33.0bps to -0.8bps. But it does
not rescue the estimator. Tested against simulated bars carrying a known 10bp
spread with increasing overnight volatility:

    overnight vol    true   unadjusted   adjusted
        0.0%         10.0        6.8        7.8     ok
        0.5%         10.0      -24.3       -1.9     broken
        1.0%         10.0      -89.5      -11.6     broken
        2.0%         10.0     -256.1      -22.7     broken

SPY's overnight moves sit around 0.32%, squarely inside the broken region. On
the real universe every fund returns a negative estimate, and the ordering is
INVERTED -- EWJ and EEM, which gap hardest because their underlying markets
trade while the US is closed, come out most negative rather than widest.

So the spread estimate is reported but NOT used as the liquidity proxy, and
``spread_is_usable`` says so per fund. Median dollar volume is the proxy the
cross-section leans on: directly measured, no estimator, no assumptions.

WHICH SUMMARY STATISTIC -- this matters more than it looks. The daily
estimates are heavily right-skewed and roughly 40% of them come out NEGATIVE,
because on any given pair of days the two-day range can happen to be smaller
than consecutive one-day ranges imply. The estimator is unbiased in the MEAN,
and only in the mean. Validated against simulated bars carrying a known
spread:

    true spread   raw mean    raw median   clipped mean
      0.0 bps      -0.85         20.31        36.41
     10.0 bps       8.48         29.23        41.91

The median and the clipped mean overstate by three- to fourfold. The negative
values are not errors to be discarded -- they are the left half of a noisy
distribution, and removing them destroys the property that makes the estimator
work. So the headline number here is the mean of the UNCLIPPED estimates, and
a negative result is reported as "not distinguishable from zero" rather than
quietly floored.

Reference: Corwin, S. A. and Schultz, P. (2012), "A Simple Way to Estimate
Bid-Ask Spreads from Daily High and Low Prices", Journal of Finance 67(2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Constants from the closed-form solution in the paper.
_K1 = 3.0 - 2.0 * np.sqrt(2.0)
_K2 = np.sqrt(8.0 / np.pi)


@dataclass(frozen=True)
class LiquidityProfile:
    """Liquidity measures for one fund."""

    ticker: str
    n_obs: int
    median_dollar_volume: float
    mean_dollar_volume: float
    spread_bps: float
    """Corwin-Schultz effective spread in bps: the MEAN of unclipped daily
    estimates, with overnight adjustment. Reported for completeness; see
    ``spread_is_usable`` and the module docstring before relying on it."""
    spread_bps_clipped: float
    """Mean after setting negative daily estimates to zero. Reported only for
    comparison; it overstates the spread by roughly 3-4x in testing."""
    pct_negative: float
    """Share of days with a negative raw estimate. Around 40% is normal and
    is not a defect -- see the docstring."""
    median_range_bps: float
    """Daily high-low range as a fraction of close -- a crude volatility and
    liquidity signal that needs no estimator at all."""

    def summary(self) -> str:
        return (
            f"{self.ticker}: spread {self.spread_bps:.2f}bps "
            f"ADV ${self.median_dollar_volume / 1e6:,.0f}m "
            f"range {self.median_range_bps:.0f}bps (n={self.n_obs})"
        )

    @property
    def spread_is_usable(self) -> bool:
        """Whether the spread estimate can be taken at face value.

        A negative effective spread is not a small number -- it is an
        impossible one, and it signals the estimator has failed on this
        series rather than that the fund is unusually cheap to trade.
        """
        return self.spread_bps > 0.0

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "spread_bps": round(self.spread_bps, 3),
            "spread_usable": self.spread_is_usable,
            "adv_usd_m": round(self.median_dollar_volume / 1e6, 1),
            "range_bps": round(self.median_range_bps, 1),
        }


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series | None = None,
    *,
    clip_negative: bool = False,
) -> pd.Series:
    """Estimate the effective bid-ask spread from daily high and low prices.

    Args:
        high: Daily high, indexed by date.
        low: Daily low, same index.
        close: Daily close. When given, the SECOND day of each two-day window
            is shifted so its range is contiguous with the first day's close.
            This is required on real data -- see the overnight-gap note in the
            module docstring. Omit only for synthetic gap-free series.
        clip_negative: Set negative daily estimates to zero. Defaults to
            False; see the note on summary statistics in the module docstring.

    Returns:
        Series of spread estimates as a proportion of price (multiply by 1e4
        for bps), indexed by the second day of each pair. The first entry is
        NaN since a pair needs two days.
    """
    if not high.index.equals(low.index):
        raise ValueError("high and low must share an index")
    if (high < low).any():
        raise ValueError("high below low on some dates")
    if not ((high > 0) & (low > 0)).all():
        raise ValueError("non-positive prices present")

    # Day 1 of each window is the previous day, taken as-is.
    h1, l1 = high.shift(1), low.shift(1)
    h2, l2 = high.copy(), low.copy()

    if close is not None:
        if not close.index.equals(high.index):
            raise ValueError("close must share the high/low index")
        # Translate ONLY day 2, onto day 1's close. The adjustment must be
        # relative to the other day IN THIS WINDOW; shifting every day
        # relative to its own predecessor leaves each pair misaligned and
        # only partly removes the gap.
        prev_close = close.shift(1)
        gap_up = (l2 - prev_close).where(l2 > prev_close, 0.0).fillna(0.0)
        gap_down = (prev_close - h2).where(h2 < prev_close, 0.0).fillna(0.0)
        shift = gap_down - gap_up
        h2, l2 = h2 + shift, l2 + shift

    # beta: the two single-day squared log ranges. Variance grows linearly
    # with the interval, so this is the "scales with time" piece.
    beta = np.log(h1 / l1) ** 2 + np.log(h2 / l2) ** 2

    # gamma: the squared log range across the whole two-day window.
    high_2 = pd.concat([h1, h2], axis=1).max(axis=1)
    low_2 = pd.concat([l1, l2], axis=1).min(axis=1)
    gamma = np.log(high_2 / low_2) ** 2

    # alpha isolates the component that does NOT scale with the interval,
    # which is the spread.
    alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / _K1 - np.sqrt(gamma / _K1)

    spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread.name = "spread"

    if clip_negative:
        spread = spread.clip(lower=0.0)
    return spread


def profile_liquidity(frame: pd.DataFrame, *, ticker: str) -> LiquidityProfile:
    """Compute all liquidity proxies for one fund.

    Args:
        frame: Aligned frame carrying ``high``, ``low``, ``close``, ``volume``.
    """
    required = ("high", "low", "close", "volume")
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"{ticker}: frame missing {missing}")

    dollar_volume = frame["close"] * frame["volume"]
    raw = corwin_schultz_spread(
        frame["high"], frame["low"], frame["close"], clip_negative=False
    ).dropna()
    daily_range = (frame["high"] - frame["low"]) / frame["close"] * 1e4

    profile = LiquidityProfile(
        ticker=ticker,
        n_obs=len(frame),
        median_dollar_volume=float(dollar_volume.median()),
        mean_dollar_volume=float(dollar_volume.mean()),
        spread_bps=float(raw.mean() * 1e4),
        spread_bps_clipped=float(raw.clip(lower=0.0).mean() * 1e4),
        pct_negative=float((raw < 0).mean() * 100.0),
        median_range_bps=float(daily_range.median()),
    )
    logger.info("%s", profile.summary())
    return profile
