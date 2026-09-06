"""Market price ingestion.

The single most consequential rule in this project lives here: prices compared
against NAV must be UNADJUSTED. NAV as published by the issuer is not
dividend-adjusted, so pairing it with an adjusted price series introduces a
drift that grows with every distribution and masquerades as a persistent
premium or discount.

The magnitude is not subtle. For HYG on 2024-01-02, Close was 77.13 while
Adj Close was 65.86 -- a 15% gap of pure accumulated distributions. Used
against NAV that would read as a ~1500bp discount, dwarfing every real signal
in the study.

What makes it dangerous is that it fails silently. yfinance called with
auto_adjust=True returns a frame that still has a "Close" column; it simply
holds adjusted values and drops "Adj Close" entirely. Code that reads
frame["Close"] works fine in both cases and returns plausible numbers either
way. The guard below therefore checks for the PRESENCE of "Adj Close", whose
absence is positive proof that adjustment leaked in.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

PRICE_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "prices"

# Column names after normalisation. "close" is deliberately the UNADJUSTED
# close; "adj_close" is carried only so the guard can prove the two differ.
PRICE_COLUMNS = ("open", "high", "low", "close", "adj_close", "volume")

_YF_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


class PriceDataError(RuntimeError):
    """Raised when downloaded price data fails a correctness guard."""


def _flatten_columns(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """yfinance returns MultiIndex columns (Price, Ticker) even for a single
    ticker, so ``raw["Close"]`` is a DataFrame rather than a Series. Flatten
    to the price level and keep this quirk contained to one function."""
    if raw.columns.nlevels == 1:
        return raw

    tickers = set(raw.columns.get_level_values(1))
    if ticker not in tickers:
        raise PriceDataError(f"{ticker}: not present in download; got {sorted(tickers)}")
    return raw.xs(ticker, axis=1, level=1)


def _validate(frame: pd.DataFrame, ticker: str) -> None:
    """Guards against the failure modes that would silently corrupt premiums."""
    if frame.empty:
        raise PriceDataError(f"{ticker}: download returned no rows")

    # THE adjustment guard. auto_adjust=True drops "Adj Close" and puts
    # adjusted values in "Close" -- indistinguishable downstream.
    if "adj_close" not in frame.columns:
        raise PriceDataError(
            f"{ticker}: no 'Adj Close' column, which means auto_adjust was True and "
            f"'Close' holds ADJUSTED prices. NAV is unadjusted, so this pairing would "
            f"manufacture a dividend-driven phantom premium."
        )

    missing = [c for c in PRICE_COLUMNS if c not in frame.columns]
    if missing:
        raise PriceDataError(f"{ticker}: missing columns {missing}")

    if frame["close"].isna().all():
        raise PriceDataError(f"{ticker}: close is entirely NaN")

    if not (frame["close"].dropna() > 0).all():
        raise PriceDataError(f"{ticker}: non-positive close prices present")

    if not frame.index.is_monotonic_increasing:
        raise PriceDataError(f"{ticker}: index is not sorted ascending")

    if frame.index.has_duplicates:
        dupes = frame.index[frame.index.duplicated()].tolist()[:5]
        raise PriceDataError(f"{ticker}: duplicate dates, e.g. {dupes}")

    # Sanity check on the guard above: for a distribution-paying fund the two
    # series must actually diverge. Identical series would mean the download
    # is adjusted-but-mislabelled, which the presence check alone would miss.
    both = frame[["close", "adj_close"]].dropna()
    if len(both) > 250 and both["close"].equals(both["adj_close"]):
        raise PriceDataError(
            f"{ticker}: 'Close' and 'Adj Close' are identical across "
            f"{len(both)} rows, which is implausible for a distributing ETF "
            f"and suggests the feed is already adjusted."
        )


def _normalise(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = _flatten_columns(raw, ticker)
    frame = frame.rename(columns=_YF_RENAME)
    keep = [c for c in PRICE_COLUMNS if c in frame.columns]
    frame = frame[keep].copy()

    frame.index = pd.to_datetime(frame.index).tz_localize(None).normalize()
    frame.index.name = "date"
    frame = frame.sort_index()

    _validate(frame, ticker)
    return frame


def load_prices(
    ticker: str,
    start: str,
    end: str,
    *,
    cache_dir: Path | str = PRICE_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load unadjusted daily OHLCV for one ticker, cached to parquet.

    High and low are carried alongside the close because the intraday range is
    used later as a diagnostic for deviations that open and close within a
    single session -- a round trip that close-to-close data cannot otherwise
    see.

    Returns:
        DataFrame indexed by tz-naive normalised date with columns
        ``open, high, low, close, adj_close, volume``, where ``close`` is
        UNADJUSTED and matches the NAV convention.

    Raises:
        PriceDataError: if the download is empty, malformed, or adjusted.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ticker}.parquet"

    if cache_path.exists() and not force_refresh:
        cached = pd.read_parquet(cache_path)
        window = cached.loc[(cached.index >= start) & (cached.index <= end)]
        if not window.empty:
            logger.info("%s: %d rows from cache", ticker, len(window))
            return window

    logger.info("%s: downloading %s to %s", ticker, start, end)
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,  # never change this; see module docstring
        progress=False,
        actions=False,
    )
    if raw is None or raw.empty:
        raise PriceDataError(f"{ticker}: yfinance returned nothing for {start}..{end}")

    frame = _normalise(raw, ticker)

    cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path)
    logger.info("%s: cached %d rows to %s", ticker, len(frame), cache_path)

    return frame
