"""Staleness and outlier checks.

These checks FLAG observations. They do not remove them, and nothing in this
module drops a row.

That is a deliberate methodological choice, not caution. An outlier filter
applied silently is indistinguishable from cherry-picking, and in this
particular study the extremes are the most economically meaningful data in
the sample. The largest bond ETF dislocations on record -- March 2020, when
credit ETFs traded far from NAV in both directions -- were genuine price
discovery, not feed errors. Deleting them without comment would remove the
part of the sample that actually tests the hypothesis.

So: every check produces a count and a list of dates. Those counts are
reported in the README, which turns a judgement call into a documented
methodology choice. Whether to exclude anything is then a decision taken in
the open, by hand, with reasons.

Two things are checked:

1. STALE NAV -- NAV identical to the previous day while the price moved
   materially. Usually a non-trading day for the underlying (a foreign
   holiday, or a bond market close that does not match the equity calendar),
   or a feed that failed to update. A stale NAV creates an artificial
   deviation that reverts when the real value arrives, which is exactly the
   pattern the reversion estimator is looking for.

2. EXTREME PREMIUM -- absolute premium beyond a per-sleeve plausibility band.
   Wider bands for credit and EM than for equity, because the underlying is
   genuinely harder to value there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# A NAV can legitimately repeat when the underlying genuinely did not move.
# It is only suspicious when the ETF price moved meaningfully on the same day.
DEFAULT_PRICE_MOVE_BPS = 25.0

# How little NAV must move to count as "not updated". Exact equality is too
# literal to be useful: iShares publishes NAV to six decimal places, so an
# exact repeat essentially never occurs, and on the real HYG sample an
# equality test flags nothing at all.
#
# It also fails on the case that matters most. When Tokyo is shut, EWJ's NAV
# is struck on the same stale closes as the day before, but FX still moves it
# fractionally -- so a genuinely stale valuation is never exactly equal.
#
# A threshold in bps catches both, and setting it to 0.0 recovers strict
# equality for anyone who wants it.
DEFAULT_NAV_MOVE_BPS = 1.0


@dataclass(frozen=True)
class QualityReport:
    """Counts and dates for every check, for disclosure in the README."""

    ticker: str
    n_obs: int
    n_stale_nav: int
    n_extreme_premium: int
    longest_nav_run: int
    stale_dates: tuple[pd.Timestamp, ...]
    extreme_dates: tuple[pd.Timestamp, ...]
    band_bps: float
    price_move_bps: float
    nav_move_bps: float

    @property
    def pct_stale(self) -> float:
        return 100.0 * self.n_stale_nav / self.n_obs if self.n_obs else 0.0

    @property
    def pct_extreme(self) -> float:
        return 100.0 * self.n_extreme_premium / self.n_obs if self.n_obs else 0.0

    def summary(self) -> str:
        return (
            f"{self.ticker}: n={self.n_obs} "
            f"stale_nav={self.n_stale_nav} ({self.pct_stale:.2f}%) "
            f"extreme>{self.band_bps:.0f}bps={self.n_extreme_premium} "
            f"({self.pct_extreme:.2f}%) longest_nav_run={self.longest_nav_run}"
        )

    def as_row(self) -> dict:
        """Flat record for the per-fund README table."""
        return {
            "ticker": self.ticker,
            "n_obs": self.n_obs,
            "n_stale_nav": self.n_stale_nav,
            "pct_stale": round(self.pct_stale, 3),
            "n_extreme_premium": self.n_extreme_premium,
            "pct_extreme": round(self.pct_extreme, 3),
            "longest_nav_run": self.longest_nav_run,
            "band_bps": self.band_bps,
        }


def _longest_run(mask: pd.Series) -> int:
    """Length of the longest consecutive True run."""
    if mask.empty or not mask.any():
        return 0
    groups = (~mask).cumsum()[mask]
    return int(groups.value_counts().max())


def add_quality_flags(
    frame: pd.DataFrame,
    *,
    band_bps: float,
    price_move_bps: float = DEFAULT_PRICE_MOVE_BPS,
    nav_move_bps: float = DEFAULT_NAV_MOVE_BPS,
) -> pd.DataFrame:
    """Attach boolean quality flags. Adds columns, never removes rows.

    Args:
        frame: Aligned frame carrying ``close``, ``nav`` and ``premium_bps``.
        band_bps: Absolute premium beyond which to flag, per sleeve.
        price_move_bps: Price move that makes a static NAV suspicious.
        nav_move_bps: NAV move below which NAV counts as not updated.
            0.0 requires exact equality.

    Returns:
        A copy with ``nav_unchanged``, ``stale_nav`` and ``extreme_premium``.
    """
    for col in ("close", "nav", "premium_bps"):
        if col not in frame.columns:
            raise ValueError(f"frame has no {col!r}; got {list(frame.columns)}")
    if band_bps <= 0:
        raise ValueError(f"band_bps must be positive, got {band_bps}")

    out = frame.copy()

    nav_move_abs_bps = (out["nav"].pct_change().abs() * 1e4).fillna(np.inf)
    price_move_abs_bps = (out["close"].pct_change().abs() * 1e4).fillna(0.0)

    nav_unchanged = nav_move_abs_bps <= nav_move_bps
    out["nav_unchanged"] = nav_unchanged
    # Unchanged NAV alone is unremarkable; unchanged NAV while the ETF moved
    # is what indicates a stale or non-trading underlying.
    out["stale_nav"] = nav_unchanged & (price_move_abs_bps > price_move_bps)
    out["extreme_premium"] = out["premium_bps"].abs() > band_bps

    return out


def check_quality(
    frame: pd.DataFrame,
    *,
    ticker: str,
    band_bps: float,
    price_move_bps: float = DEFAULT_PRICE_MOVE_BPS,
    nav_move_bps: float = DEFAULT_NAV_MOVE_BPS,
) -> tuple[pd.DataFrame, QualityReport]:
    """Run all checks and report what they found.

    Returns:
        ``(flagged_frame, report)``. The frame has the SAME number of rows as
        the input -- flags are added, nothing is filtered.
    """
    flagged = add_quality_flags(
        frame,
        band_bps=band_bps,
        price_move_bps=price_move_bps,
        nav_move_bps=nav_move_bps,
    )

    stale = flagged.index[flagged["stale_nav"]]
    extreme = flagged.index[flagged["extreme_premium"]]

    report = QualityReport(
        ticker=ticker,
        n_obs=len(flagged),
        n_stale_nav=len(stale),
        n_extreme_premium=len(extreme),
        longest_nav_run=_longest_run(flagged["nav_unchanged"]),
        stale_dates=tuple(stale),
        extreme_dates=tuple(extreme),
        band_bps=float(band_bps),
        price_move_bps=float(price_move_bps),
        nav_move_bps=float(nav_move_bps),
    )

    logger.info("%s", report.summary())
    if report.n_extreme_premium:
        worst = flagged.loc[extreme, "premium_bps"].abs().nlargest(5)
        logger.info(
            "%s: largest deviations (inspect by hand, do not delete): %s",
            ticker,
            ", ".join(f"{d:%Y-%m-%d}={v:+.0f}bps" for d, v in worst.items()),
        )

    assert len(flagged) == len(frame), "quality checks must not drop rows"
    return flagged, report
