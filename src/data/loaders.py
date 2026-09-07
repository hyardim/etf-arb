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


# ==========================================================================
# NAV ingestion
# ==========================================================================
#
# NAV files are downloaded by hand from issuer websites, so this loader has a
# different trust model from the price loader above: the question is not
# "is this the right convention" but "are these bytes the data I think they
# are".
#
# That distinction is not theoretical. Scripted download of iShares NAV was
# attempted and rejected during design, and the way it failed is the reason
# every guard below exists. Requesting the exact NAV CSV URL scraped off the
# fund's own product page -- with a browser user-agent, a session cookie and a
# correct referer -- returns:
#
#     HTTP 200
#     Content-Type: text/csv; charset=UTF-8
#     <!DOCTYPE html> ... the fund product page ...
#
# Status code, content type and file extension all report success while the
# body is a web page. SSGA answered 301 -> 404 HTML, Vanguard served a
# client-rendered shell, Invesco returned 406. A loader that trusted any
# ordinary signal would have written a parsed web page into the NAV column and
# raised nothing.
#
# So: verify the bytes, never the metadata.

NAV_DIR = PROJECT_ROOT / "data" / "raw" / "nav"

# Issuers vary the header wording and none of them are stable across time, so
# match on a normalised alias set rather than a fixed column name. Matching is
# explicit and closed -- an unrecognised header fails loudly rather than being
# guessed at positionally.
_DATE_ALIASES = frozenset(
    {"as of", "as of date", "asof", "date", "trade date", "record date", "nav date"}
)
_NAV_ALIASES = frozenset(
    {
        "nav per share",
        "nav per share ($)",
        "nav per share (usd)",
        "nav",
        "nav ($)",
        "net asset value",
        "net asset value per share",
        "closing nav",
    }
)

# A 2018->present daily series has ~2180 US trading days. Anything far short
# of that means a truncated download or a date filter left on the issuer page.
MIN_NAV_ROWS = 1000

# Issuer NAV should never be far from a plausible ETF share price. This is a
# last-resort catch for a column matched to the wrong field (shares
# outstanding, total net assets), not a data-quality filter.
_NAV_PLAUSIBLE_RANGE = (1.0, 10_000.0)


class NavDataError(RuntimeError):
    """Raised when a NAV file is not what it claims to be."""


def _reject_markup(text: str, path: Path) -> None:
    """The iShares failure mode, caught on the only signal that did not lie.

    Checks the first non-whitespace byte rather than the content type or the
    file extension, both of which reported ``text/csv`` while serving HTML.
    """
    stripped = text.lstrip()
    if stripped.startswith(("<", "﻿<")):
        first = stripped.splitlines()[0][:80] if stripped.splitlines() else ""
        raise NavDataError(
            f"{path.name} is markup, not CSV (starts with {first!r}). Issuer sites "
            f"serve their product page with Content-Type: text/csv when a download "
            f"link is fetched without a browser -- re-download it by hand from the "
            f"fund page and save the file the browser produces."
        )


def _normalise_header(value: object) -> str:
    return " ".join(str(value).strip().strip('"').lower().split())


def _find_header_row(lines: list[str], path: Path) -> int:
    """Locate the real header row.

    Issuer exports carry a preamble of fund name, inception date and
    disclaimers before the table starts, and its depth varies by issuer and
    changes over time. Rather than hardcode a skiprows count that silently
    rots, find the row that actually contains both a date column and a NAV
    column.
    """
    for i, line in enumerate(lines[:80]):
        cells = {_normalise_header(c) for c in line.split(",")}
        if cells & _DATE_ALIASES and cells & _NAV_ALIASES:
            return i
    raise NavDataError(
        f"{path.name}: no header row containing both a date column "
        f"({sorted(_DATE_ALIASES)[:3]}...) and a NAV column "
        f"({sorted(_NAV_ALIASES)[:3]}...) in the first 80 lines. "
        f"Check the file is a NAV history export and not holdings or performance."
    )


def _pick_column(frame: pd.DataFrame, aliases: frozenset[str], label: str, path: Path) -> str:
    matches = [c for c in frame.columns if _normalise_header(c) in aliases]
    if not matches:
        raise NavDataError(
            f"{path.name}: no {label} column found in {list(frame.columns)}"
        )
    if len(matches) > 1:
        raise NavDataError(
            f"{path.name}: ambiguous {label} column, matched {matches}. "
            f"Remove the duplicate column from the export."
        )
    return matches[0]


def _to_numeric(series: pd.Series) -> pd.Series:
    """Strip currency formatting that issuer exports apply inconsistently."""
    cleaned = (
        series.astype("string")
        .str.replace(r"[$,\s]", "", regex=True)
        .str.replace(r"^\((.*)\)$", r"-\1", regex=True)  # (1.23) => -1.23
        .replace({"": None, "-": None, "--": None, "N/A": None, "n/a": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def load_nav_csv(
    path: Path | str,
    *,
    min_rows: int = MIN_NAV_ROWS,
) -> pd.Series:
    """Load a hand-downloaded issuer NAV history file.

    Every check refuses rather than repairs. A NAV series that is quietly
    wrong produces a premium series that is quietly wrong, and nothing
    downstream -- not the ADF test, not the bootstrap -- would flag it.

    Returns:
        Series named ``nav`` indexed by tz-naive normalised date, ascending.

    Raises:
        NavDataError: if the file is markup, has no recognisable header, has
            duplicate or unparseable dates, contains non-positive NAV, or is
            shorter than ``min_rows``.
    """
    path = Path(path)
    if not path.exists():
        raise NavDataError(
            f"NAV file not found: {path}. It must be downloaded by hand from the "
            f"issuer -- see docs/NAV_SOURCES.md."
        )

    text = path.read_text(encoding="utf-8-sig", errors="replace")
    _reject_markup(text, path)

    lines = text.splitlines()
    if not lines:
        raise NavDataError(f"{path.name} is empty")

    header_row = _find_header_row(lines, path)
    frame = pd.read_csv(path, skiprows=header_row, encoding="utf-8-sig")
    if frame.empty:
        raise NavDataError(f"{path.name}: no data rows below the header")

    date_col = _pick_column(frame, _DATE_ALIASES, "date", path)
    nav_col = _pick_column(frame, _NAV_ALIASES, "NAV", path)

    dates = pd.to_datetime(frame[date_col], errors="coerce", format="mixed")
    nav = _to_numeric(frame[nav_col])

    series = (
        pd.Series(nav.to_numpy(), index=pd.DatetimeIndex(dates), name="nav")
        .dropna()
        .sort_index()
    )
    # Issuer exports often end with disclaimer rows that parse to NaT/NaN;
    # those are dropped above. A large loss means the wrong column was matched.
    parsed_ratio = len(series) / len(frame)
    if parsed_ratio < 0.5:
        raise NavDataError(
            f"{path.name}: only {len(series)}/{len(frame)} rows parsed to a "
            f"date and a number. The matched columns ({date_col!r}, {nav_col!r}) "
            f"are probably not the intended ones."
        )

    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series.index = series.index.normalize()
    series.index.name = "date"

    if series.index.has_duplicates:
        dupes = series.index[series.index.duplicated()].strftime("%Y-%m-%d").tolist()
        raise NavDataError(
            f"{path.name}: duplicate dates ({len(dupes)}), e.g. {dupes[:5]}. "
            f"Overlapping exports must be de-duplicated before use, not silently "
            f"collapsed -- which value wins would otherwise be arbitrary."
        )

    if len(series) < min_rows:
        raise NavDataError(
            f"{path.name}: only {len(series)} rows, expected at least {min_rows}. "
            f"A truncated file usually means a date filter was left applied on the "
            f"issuer page. Range found: {series.index.min():%Y-%m-%d} to "
            f"{series.index.max():%Y-%m-%d}."
        )

    if not (series > 0).all():
        bad = series[series <= 0]
        raise NavDataError(
            f"{path.name}: {len(bad)} non-positive NAV values, e.g. "
            f"{bad.head(3).to_dict()}"
        )

    lo, hi = _NAV_PLAUSIBLE_RANGE
    if not series.between(lo, hi).all():
        outside = series[~series.between(lo, hi)]
        raise NavDataError(
            f"{path.name}: {len(outside)} NAV values outside [{lo}, {hi}], e.g. "
            f"{outside.head(3).to_dict()}. This usually means the matched column "
            f"({nav_col!r}) is shares outstanding or total net assets, not NAV per share."
        )

    logger.info(
        "%s: %d NAV rows, %s to %s",
        path.name,
        len(series),
        series.index.min().date(),
        series.index.max().date(),
    )
    return series
