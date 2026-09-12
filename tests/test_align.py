"""Tests for calendar alignment.

The central test is test_gap_is_dropped_not_filled. Forward-filling NAV
manufactures the precise signal the project exists to measure, so that
behaviour is pinned down explicitly rather than left to convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.align import align_price_nav


def _prices(dates, close=100.0):
    idx = pd.DatetimeIndex(dates, name="date")
    n = len(idx)
    return pd.DataFrame(
        {
            "open": np.full(n, close),
            "high": np.full(n, close * 1.001),
            "low": np.full(n, close * 0.999),
            "close": np.full(n, close, dtype=float),
            "adj_close": np.full(n, close * 0.9),
            "volume": np.full(n, 1e7),
        },
        index=idx,
    )


def _nav(dates, value=100.0):
    return pd.Series(
        np.full(len(dates), value, dtype=float),
        index=pd.DatetimeIndex(dates, name="date"),
        name="nav",
    )


# --------------------------------------------------------------------------
# The no-fill rule
# --------------------------------------------------------------------------


def test_gap_is_dropped_not_filled():
    """A NAV gap must remove the row, never carry the previous value.

    Carrying it would create a fake premium on the gap day and a fake reversal
    the next -- a clean mean-reverting deviation that the AR(1) estimator
    would happily measure as real.
    """
    days = pd.bdate_range("2024-01-01", periods=5)
    prices = _prices(days)
    nav = _nav(days.delete(2))  # NAV missing on day 3

    frame, report = align_price_nav(prices, nav, ticker="TEST")

    assert len(frame) == 4
    assert days[2] not in frame.index
    assert report.price_only == (days[2],)
    assert not frame["nav"].isna().any()


def test_no_synthetic_dates_introduced():
    """Output dates must be a subset of both inputs -- no reindexing to a
    calendar that invents rows."""
    days = pd.bdate_range("2024-01-01", periods=10)
    prices = _prices(days)
    nav = _nav(days.delete([3, 7]))

    frame, _ = align_price_nav(prices, nav, ticker="TEST")
    assert set(frame.index) <= set(prices.index)
    assert set(frame.index) <= set(nav.index)


def test_nav_values_are_not_altered():
    days = pd.bdate_range("2024-01-01", periods=6)
    nav = pd.Series(
        [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
        index=pd.DatetimeIndex(days, name="date"),
        name="nav",
    )
    frame, _ = align_price_nav(_prices(days), nav, ticker="TEST")
    assert frame["nav"].tolist() == [10.0, 11.0, 12.0, 13.0, 14.0, 15.0]


# --------------------------------------------------------------------------
# Real-world shape: NAV published on non-trading days
# --------------------------------------------------------------------------


def test_nav_on_non_trading_day_is_dropped():
    """Reproduces the real HYG case: iShares published NAV on Good Friday
    2024-03-29 and Easter Sunday 2024-03-31, when no US price exists."""
    trading = pd.DatetimeIndex(["2024-03-27", "2024-03-28", "2024-04-01"])
    nav_dates = pd.DatetimeIndex(
        ["2024-03-27", "2024-03-28", "2024-03-29", "2024-03-31", "2024-04-01"]
    )

    frame, report = align_price_nav(_prices(trading), _nav(nav_dates), ticker="HYG")

    assert len(frame) == 3
    assert report.nav_only == (pd.Timestamp("2024-03-29"), pd.Timestamp("2024-03-31"))
    assert report.n_dropped == 2


# --------------------------------------------------------------------------
# Windowing and reporting
# --------------------------------------------------------------------------


def test_window_is_applied():
    days = pd.bdate_range("2017-01-02", periods=600)
    frame, _ = align_price_nav(
        _prices(days), _nav(days), ticker="TEST", start="2018-01-01", end="2018-12-31"
    )
    assert frame.index.min() >= pd.Timestamp("2018-01-01")
    assert frame.index.max() <= pd.Timestamp("2018-12-31")


def test_report_accounts_for_every_dropped_row():
    """Counts feed the README; an undisclosed drop reads as cherry-picking."""
    days = pd.bdate_range("2024-01-01", periods=10)
    prices = _prices(days)
    nav = _nav(days.delete([2, 5]))

    frame, report = align_price_nav(prices, nav, ticker="TEST")
    assert report.n_price == 10
    assert report.n_nav == 8
    assert report.n_aligned == 8
    assert report.n_dropped == 2
    assert "TEST" in report.summary()


def test_nan_rows_are_dropped_and_counted():
    days = pd.bdate_range("2024-01-01", periods=6)
    prices = _prices(days)
    prices.loc[days[2], "close"] = np.nan

    frame, report = align_price_nav(prices, _nav(days), ticker="TEST")
    assert len(frame) == 5
    assert report.n_dropped_na == 1


def test_output_sorted_ascending():
    days = pd.bdate_range("2024-01-01", periods=8)
    frame, _ = align_price_nav(_prices(days[::-1]), _nav(days[::-1]), ticker="TEST")
    assert frame.index.is_monotonic_increasing


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_disjoint_dates_raise():
    """No overlap means a convention mismatch; an empty frame would hide it."""
    px = _prices(pd.bdate_range("2024-01-01", periods=5))
    nav = _nav(pd.bdate_range("2020-01-01", periods=5))
    with pytest.raises(ValueError, match="share no dates"):
        align_price_nav(px, nav, ticker="TEST")


def test_empty_inputs_raise():
    days = pd.bdate_range("2024-01-01", periods=3)
    with pytest.raises(ValueError, match="price frame is empty"):
        align_price_nav(_prices([]), _nav(days), ticker="TEST")
    with pytest.raises(ValueError, match="nav series is empty"):
        align_price_nav(_prices(days), _nav([]), ticker="TEST")
