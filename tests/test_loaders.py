"""Tests for price ingestion.

These run offline against synthetic frames shaped exactly like yfinance
output. The point is to prove the guards fire, particularly the adjustment
guard, which defends the most consequential correctness rule in the project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.loaders import PriceDataError, _normalise


def _yf_frame(
    n: int = 300,
    ticker: str = "HYG",
    *,
    with_adj_close: bool = True,
    adj_equals_close: bool = False,
) -> pd.DataFrame:
    """Build a frame with the MultiIndex (Price, Ticker) columns that yfinance
    returns even for a single-ticker download."""
    idx = pd.bdate_range("2023-01-02", periods=n, name="Date")
    close = pd.Series(np.linspace(70, 80, n), index=idx)
    adj = close if adj_equals_close else close * 0.85

    fields = {
        "Open": close,
        "High": close * 1.002,
        "Low": close * 0.998,
        "Close": close,
        "Volume": pd.Series(np.full(n, 1e7), index=idx),
    }
    if with_adj_close:
        fields["Adj Close"] = adj

    order = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    order = [c for c in order if c in fields]
    frame = pd.concat([fields[c] for c in order], axis=1)
    frame.columns = pd.MultiIndex.from_product([order, [ticker]], names=["Price", "Ticker"])
    return frame


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------


def test_normalise_shape_and_columns():
    out = _normalise(_yf_frame(), "HYG")
    assert list(out.columns) == ["open", "high", "low", "close", "adj_close", "volume"]
    assert out.index.name == "date"
    assert out.index.tz is None
    assert out.index.is_monotonic_increasing


def test_close_is_unadjusted():
    """close must carry the raw print, not the adjusted series."""
    out = _normalise(_yf_frame(), "HYG")
    assert not np.allclose(out["close"], out["adj_close"])
    assert (out["close"] > out["adj_close"]).all()


def test_flattens_multiindex_for_single_ticker():
    """yfinance nests columns even for one ticker; ``raw['Close']`` would be a
    DataFrame rather than a Series without flattening."""
    raw = _yf_frame()
    assert raw.columns.nlevels == 2
    assert _normalise(raw, "HYG").columns.nlevels == 1


# --------------------------------------------------------------------------
# The adjustment guard -- the highest-consequence check in the project
# --------------------------------------------------------------------------


def test_missing_adj_close_is_rejected():
    """auto_adjust=True drops 'Adj Close' and puts adjusted values in 'Close'.
    Both shapes expose a 'Close' column and both return plausible numbers, so
    absence of 'Adj Close' is the only positive proof adjustment leaked in."""
    raw = _yf_frame(with_adj_close=False)
    with pytest.raises(PriceDataError, match="auto_adjust"):
        _normalise(raw, "HYG")


def test_identical_close_and_adj_close_rejected():
    """A distributing ETF cannot have identical series over 250+ sessions."""
    raw = _yf_frame(n=300, adj_equals_close=True)
    with pytest.raises(PriceDataError, match="identical"):
        _normalise(raw, "HYG")


def test_short_identical_sample_allowed():
    """The identical-series check needs enough history to be meaningful; a
    short window between distributions can legitimately match."""
    out = _normalise(_yf_frame(n=50, adj_equals_close=True), "HYG")
    assert len(out) == 50


# --------------------------------------------------------------------------
# Structural guards
# --------------------------------------------------------------------------


def test_wrong_ticker_rejected():
    with pytest.raises(PriceDataError, match="not present"):
        _normalise(_yf_frame(ticker="LQD"), "HYG")


def test_empty_frame_rejected():
    with pytest.raises(PriceDataError, match="no rows"):
        _normalise(_yf_frame(n=0), "HYG")


def test_non_positive_close_rejected():
    raw = _yf_frame()
    raw.loc[raw.index[10], ("Close", "HYG")] = 0.0
    with pytest.raises(PriceDataError, match="non-positive"):
        _normalise(raw, "HYG")


def test_duplicate_dates_rejected():
    raw = _yf_frame(n=100)
    raw = pd.concat([raw, raw.iloc[[50]]])
    with pytest.raises(PriceDataError, match="duplicate dates"):
        _normalise(raw, "HYG")


def test_unsorted_input_is_sorted_not_rejected():
    """Ordering is a normalisation concern, not a data-integrity failure."""
    raw = _yf_frame(n=100).iloc[::-1]
    out = _normalise(raw, "HYG")
    assert out.index.is_monotonic_increasing


def test_timezone_is_stripped():
    """NAV files carry plain dates; a tz-aware price index would break the
    strict date join in alignment."""
    raw = _yf_frame(n=100)
    raw.index = raw.index.tz_localize("America/New_York")
    assert _normalise(raw, "HYG").index.tz is None
