"""Intraday premium range -- seeing what close-to-close data cannot.

Every other number in this project is built from one observation per day, the
4pm close. That makes a whole class of behaviour invisible: a deviation that
opens and closes inside a single session leaves no trace in a close-to-close
series. The estimator cannot see it, and no amount of statistical care
recovers it, because the information was never sampled.

Daily highs and lows are sampled, though, and they bound what happened in
between. Comparing the day's price range against NAV gives a floor on how far
the premium travelled during the session:

    premium_high = high / nav - 1
    premium_low  = low  / nav - 1

A wide intraday range around a near-zero closing premium is direct evidence
that a round trip occurred.

THE CAVEAT, which has to be stated. NAV is struck once, at the close. It is
not observed intraday. So for a domestic fund the intraday range conflates
two different things:

    (a) the ETF deviating from its basket -- what we want, and
    (b) the basket itself moving during the day -- what we do not.

On a day when the S&P 500 falls 2%, IVV's price range will be wide and its
"intraday premium range" will look large, but almost all of that is the
basket moving with NAV unobserved in between. For domestic funds this is a
DIAGNOSTIC, not an estimate, and it is an upper bound on deviation rather
than a measurement of it.

THE EXCEPTION is what makes this worth building. When the underlying market is
closed during the US session, the basket is genuinely frozen: its constituents
cannot reprice, so NAV would not move even if it were observed. Term (b)
vanishes. For EWJ, whose Japanese holdings are shut throughout US trading
hours, the intraday price range IS the intraday premium range, unconfounded.

That is the one place in this project where intraday deviation can be measured
honestly from free daily data, and it is the evidence behind the claim that
EWJ's premium is price discovery rather than mispricing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def add_intraday_premium(frame: pd.DataFrame, *, nav_col: str = "nav") -> pd.DataFrame:
    """Attach intraday premium bounds from the daily high and low.

    Returns a copy with ``premium_high_bps``, ``premium_low_bps`` and
    ``premium_range_bps`` (the high-to-low span).
    """
    required = ("high", "low", nav_col)
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"frame missing {missing}")

    out = frame.copy()
    nav = out[nav_col]
    out["premium_high_bps"] = (out["high"] / nav - 1.0) * 1e4
    out["premium_low_bps"] = (out["low"] / nav - 1.0) * 1e4
    out["premium_range_bps"] = out["premium_high_bps"] - out["premium_low_bps"]
    return out


@dataclass(frozen=True)
class IntradayProfile:
    """How much of a fund's deviation is invisible at the close."""

    ticker: str
    n_obs: int
    foreign_session: bool
    """True when the underlying market is shut during US trading, which is
    what makes the range interpretable as a premium range rather than an
    upper bound."""

    median_range_bps: float
    median_abs_close_bps: float
    """Median |closing premium| -- what the daily series actually records."""
    range_to_close_ratio: float
    """Median range divided by median |closing premium|. How much wider the
    within-day excursion is than the number the close-to-close analysis
    sees."""
    pct_straddling_zero: float
    """Share of days where the premium was positive at some point and
    negative at another. On those days the deviation demonstrably changed
    sign inside the session -- a round trip the close cannot show."""
    median_unwound_bps: float
    """Median distance from the day's furthest excursion back to the close.
    What reverted before the bell."""

    def summary(self) -> str:
        kind = "frozen basket" if self.foreign_session else "upper bound only"
        return (
            f"{self.ticker}: range {self.median_range_bps:.0f}bps vs close "
            f"{self.median_abs_close_bps:.0f}bps ({self.range_to_close_ratio:.1f}x), "
            f"sign flips on {self.pct_straddling_zero:.0f}% of days [{kind}]"
        )

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "foreign_session": self.foreign_session,
            "range_bps": round(self.median_range_bps, 1),
            "abs_close_bps": round(self.median_abs_close_bps, 1),
            "range_over_close": round(self.range_to_close_ratio, 2),
            "pct_sign_flip": round(self.pct_straddling_zero, 1),
            "unwound_bps": round(self.median_unwound_bps, 1),
        }


def profile_intraday(
    frame: pd.DataFrame,
    *,
    ticker: str,
    foreign_session: bool = False,
) -> IntradayProfile:
    """Summarise intraday premium behaviour for one fund.

    Args:
        frame: Aligned frame with ``high``, ``low``, ``nav`` and
            ``premium_bps``.
        ticker: Label.
        foreign_session: Whether the underlying market is closed during US
            trading. Recorded on the result because it determines whether the
            range is a premium measurement or merely an upper bound.
    """
    if "premium_bps" not in frame.columns:
        raise ValueError("frame has no 'premium_bps'; call add_premium first")

    out = add_intraday_premium(frame)
    hi, lo, close = out["premium_high_bps"], out["premium_low_bps"], out["premium_bps"]

    straddles = (hi > 0) & (lo < 0)

    # Distance from whichever extreme is further from zero back to the close.
    furthest = np.where(hi.abs() >= lo.abs(), hi, lo)
    unwound = (furthest - close).abs()

    median_range = float(out["premium_range_bps"].median())
    median_abs_close = float(close.abs().median())

    profile = IntradayProfile(
        ticker=ticker,
        n_obs=len(out),
        foreign_session=foreign_session,
        median_range_bps=median_range,
        median_abs_close_bps=median_abs_close,
        range_to_close_ratio=float(median_range / median_abs_close)
        if median_abs_close > 0
        else float("inf"),
        pct_straddling_zero=float(straddles.mean() * 100.0),
        median_unwound_bps=float(pd.Series(unwound).median()),
    )
    logger.info("%s", profile.summary())
    return profile
