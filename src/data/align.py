"""Calendar alignment of price and NAV series.

One rule governs this module: price and NAV must refer to the same date, and
a date where either is missing is DROPPED, never filled.

Forward-filling NAV across a missing day manufactures the exact pattern the
rest of the project sets out to measure. A stale NAV carried forward while the
price moves creates an artificial premium on the gap day, and when the real
NAV arrives the next day it creates an artificial reversal. That is a textbook
mean-reverting deviation, and an AR(1) estimator cannot tell it apart from a
real one. Filling would let the pipeline discover its own imputation and
report it as arbitrage.

Dropping costs a handful of observations out of ~2,180 and costs nothing else.

The need is not hypothetical. In the real HYG sample, NAV exists on
2024-03-29 (Good Friday) and 2024-03-31 (Easter Sunday) -- days the US market
was closed and no price exists. Those NAVs are real issuer publications, but
there is no contemporaneous price to compare them against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlignmentReport:
    """What alignment kept, dropped, and why.

    Recorded per fund so the counts can be reported in the README. An
    undisclosed drop is indistinguishable from cherry-picking, so every
    discarded observation is accounted for here.
    """

    ticker: str
    n_price: int
    n_nav: int
    n_aligned: int
    price_only: tuple[pd.Timestamp, ...] = field(default=())
    nav_only: tuple[pd.Timestamp, ...] = field(default=())
    n_dropped_na: int = 0

    @property
    def n_dropped(self) -> int:
        return len(self.price_only) + len(self.nav_only) + self.n_dropped_na

    def summary(self) -> str:
        return (
            f"{self.ticker}: {self.n_aligned} aligned rows "
            f"({self.n_price} price, {self.n_nav} nav); dropped "
            f"{len(self.price_only)} price-only, {len(self.nav_only)} nav-only, "
            f"{self.n_dropped_na} incomplete"
        )


def align_price_nav(
    prices: pd.DataFrame,
    nav: pd.Series,
    *,
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> tuple[pd.DataFrame, AlignmentReport]:
    """Join price and NAV strictly on date.

    Args:
        prices: Daily OHLCV indexed by date, from ``load_prices``.
        nav: Daily NAV indexed by date, from ``load_nav_file``.
        ticker: Used for logging and the report.
        start: Optional inclusive window start (ISO date).
        end: Optional inclusive window end (ISO date).

    Returns:
        ``(frame, report)`` where frame is indexed by the dates present in
        BOTH inputs, carrying the price columns plus ``nav``.

    Raises:
        ValueError: if either input is empty, or the two series share no
            dates at all -- which means a convention mismatch rather than a
            thin sample, and silently returning an empty frame would hide it.
    """
    if prices.empty:
        raise ValueError(f"{ticker}: price frame is empty")
    if nav.empty:
        raise ValueError(f"{ticker}: nav series is empty")

    px = prices.sort_index()
    nv = nav.sort_index()

    if start is not None:
        px, nv = px.loc[px.index >= start], nv.loc[nv.index >= start]
    if end is not None:
        px, nv = px.loc[px.index <= end], nv.loc[nv.index <= end]

    price_dates, nav_dates = set(px.index), set(nv.index)
    common = price_dates & nav_dates
    if not common:
        raise ValueError(
            f"{ticker}: price and NAV share no dates in {start}..{end}. "
            f"Price spans {px.index.min():%Y-%m-%d}..{px.index.max():%Y-%m-%d}, "
            f"NAV spans {nv.index.min():%Y-%m-%d}..{nv.index.max():%Y-%m-%d}. "
            f"This is a convention mismatch, not a thin sample."
        )

    # how="inner" is the whole point: no reindexing, no fill, no asof merge.
    frame = px.join(nv.to_frame("nav"), how="inner")

    n_before = len(frame)
    frame = frame.dropna(subset=["close", "nav"])
    n_dropped_na = n_before - len(frame)

    report = AlignmentReport(
        ticker=ticker,
        n_price=len(px),
        n_nav=len(nv),
        n_aligned=len(frame),
        price_only=tuple(sorted(price_dates - nav_dates)),
        nav_only=tuple(sorted(nav_dates - price_dates)),
        n_dropped_na=n_dropped_na,
    )

    logger.info("%s", report.summary())
    for label, dates in (("price-only", report.price_only), ("nav-only", report.nav_only)):
        if dates:
            shown = ", ".join(d.strftime("%Y-%m-%d") for d in dates[:10])
            more = f" (+{len(dates) - 10} more)" if len(dates) > 10 else ""
            logger.info("%s: dropped %s dates: %s%s", ticker, label, shown, more)

    if frame.empty:
        raise ValueError(f"{ticker}: alignment produced no usable rows")

    return frame, report
