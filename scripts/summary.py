#!/usr/bin/env python
"""Run the whole pipeline for one fund and print what it found.

    python scripts/summary.py HYG

A fast text view of the same evidence the notebook plots. Useful for checking
a fund after a new NAV download without opening Jupyter.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.premium import add_premium, describe_premium  # noqa: E402
from src.analysis.reversion import estimate_reversion  # noqa: E402
from src.config import load_universe  # noqa: E402
from src.data.align import align_price_nav  # noqa: E402
from src.data.loaders import load_nav_file, load_prices  # noqa: E402
from src.data.quality import check_quality  # noqa: E402

NAV_DIR = ROOT / "data" / "raw" / "nav"


def find_nav_file(ticker: str) -> Path:
    """NAV files keep whatever extension the issuer produced."""
    for suffix in (".xls", ".csv", ".xlsx", ".xml"):
        candidate = NAV_DIR / f"{ticker}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No NAV file for {ticker} in {NAV_DIR}.\n"
        f"Download it by hand from the issuer -- see docs/NAV_SOURCES.md."
    )


def rule(char="=", n=68):
    print(char * n)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", help="ticker in configs/universe.yaml, e.g. HYG")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)

    universe = load_universe()
    ticker = args.ticker.upper()
    fund = universe[ticker]
    start = args.start or universe.start
    end = args.end or universe.end_or_today

    prices = load_prices(ticker, start, end)
    nav = load_nav_file(find_nav_file(ticker))
    frame, alignment = align_price_nav(prices, nav, ticker=ticker, start=start, end=end)
    frame = add_premium(frame)
    frame, quality = check_quality(frame, ticker=ticker, band_bps=fund.outlier_band_bps)
    stats = describe_premium(frame, ticker=ticker)
    result = estimate_reversion(frame["premium_bps"], ticker=ticker)

    print()
    rule()
    print(f"  {ticker} — {fund.name}")
    print(f"  {frame.index.min():%Y-%m-%d} to {frame.index.max():%Y-%m-%d}"
          f"   sleeve={fund.sleeve}  issuer={fund.issuer}")
    rule()

    print("\n  DATA")
    print(f"    price rows              {alignment.n_price:>10,}")
    print(f"    nav rows (in window)    {alignment.n_nav:>10,}")
    print(f"    aligned                 {alignment.n_aligned:>10,}")
    print(f"    dropped (no price)      {len(alignment.nav_only):>10,}")
    print(f"    dropped (no nav)        {len(alignment.price_only):>10,}")

    print("\n  QUALITY  (flagged, never removed)")
    print(f"    stale nav               {quality.n_stale_nav:>10,}  ({quality.pct_stale:.2f}%)")
    print(f"    beyond {quality.band_bps:>4.0f}bps band     {quality.n_extreme_premium:>10,}  ({quality.pct_extreme:.2f}%)")
    print(f"    longest nav run         {quality.longest_nav_run:>10,}")

    print("\n  PREMIUM (bps)")
    print(f"    mean                    {stats.mean_bps:>+10.2f}")
    print(f"    median                  {stats.median_bps:>+10.2f}")
    print(f"    std dev                 {stats.sd_bps:>10.2f}")
    print(f"    min / max               {stats.min_bps:>+10.1f} / {stats.max_bps:+.1f}")
    print(f"    days at premium         {stats.frac_premium:>10.1%}")
    print(f"    p(mean = 0)             {stats.p_value:>10.2g}"
          f"{'   <- investigate, likely convention' if stats.mean_differs_from_zero else ''}")

    print("\n  REVERSION")
    print(f"    AR(1) b                 {result.fit.b:>+10.4f}  (robust se {result.fit.b_se:.4f})")
    print(f"    ADF p-value             {result.adf_pvalue:>10.2e}")
    print(f"    p(b = 0)                {result.fit.b_pvalue:>10.2e}")
    print(f"    long-run level          {result.fit.long_run:>+10.2f} bps")

    print()
    rule("-")
    print(f"  RESOLUTION   {result.resolution.value}")
    print(f"  HALF-LIFE    {result.display}")
    rule("-")
    print(f"\n  {result.reason}\n")


if __name__ == "__main__":
    main()
