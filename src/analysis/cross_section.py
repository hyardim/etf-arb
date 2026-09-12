"""The cross-sectional result.

This is the part that makes the project research rather than description: a
set of per-fund half-lives, related to how tradable each fund's underlying is.

WHAT THE SAMPLE SUPPORTS, AND WHAT IT DOES NOT.

With eight funds there is no statistical power to estimate a relationship,
only to describe one. So no line is fitted and no R-squared is quoted.
Overclaiming on a tiny cross-section is a worse error than having a small
sample, and stating the limit plainly is the point.

The claim the data does support is a TWO-GROUP separation. Using bootstrap
confidence intervals, every equity-vs-credit pair separates cleanly and no
credit-vs-credit pair does:

    equity-vs-credit pairs overlapping:  0 of 6
    credit-vs-credit pairs overlapping:  3 of 3

So: equity reverts faster than credit and EM debt, firmly. The ordering WITHIN
credit is not resolvable at this sample size, and LQD vs HYG in particular --
which looks like an interesting inversion in the point estimates -- is not
distinguishable from noise.

WHY THE LIQUIDITY PROXY IS CATEGORICAL. The planned quantitative proxies both
failed, and the failure is informative rather than an inconvenience. The
Corwin-Schultz spread estimator breaks on assets that gap overnight, returning
impossible negative spreads for all eight funds. Dollar volume works perfectly
well but measures the wrong thing: HYG trades $2,353m a day against IVV's
$1,855m, so the high-yield fund is MORE liquid at the ETF level while its
underlying bonds barely trade. That gap between ETF liquidity and basket
liquidity is not a measurement problem -- it is the mechanism that makes the
premium exist. Sleeve is therefore the classifier, and it is honest about
being a category rather than a measurement.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.intraday import profile_intraday
from src.analysis.liquidity import profile_liquidity
from src.analysis.premium import add_premium, describe_premium
from src.analysis.reversion import Interval, Resolution, estimate_reversion
from src.analysis.variance_ratio import profile_variance_ratios
from src.config import PROJECT_ROOT, Universe, load_universe
from src.data.align import align_price_nav
from src.data.loaders import load_nav_file, load_prices
from src.data.quality import check_quality

logger = logging.getLogger(__name__)

NAV_DIR = PROJECT_ROOT / "data" / "raw" / "nav"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Sleeves whose underlying trades continuously on exchange, versus those whose
# underlying trades OTC or not at all during the US session.
EQUITY_SLEEVES = frozenset(
    {"equity_us_large", "equity_us_total", "equity_intl_developed", "equity_em"}
)


def nav_path(ticker: str, nav_dir: Path = NAV_DIR) -> Path:
    """NAV files keep whatever extension the issuer produced."""
    for suffix in (".xls", ".xlsx", ".csv", ".xml"):
        candidate = nav_dir / f"{ticker}{suffix}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No NAV file for {ticker} in {nav_dir}. Download it from the issuer "
        f"-- see docs/NAV_SOURCES.md."
    )


@dataclass(frozen=True)
class FundResult:
    """Everything measured for one fund."""

    ticker: str
    sleeve: str
    role: str
    foreign_session: bool
    n: int
    mean_premium_bps: float
    sd_premium_bps: float
    min_premium_bps: float
    max_premium_bps: float
    pct_days_premium: float
    resolution: Resolution
    half_life: float
    half_life_display: str
    bootstrap_ci: Interval | None
    delta_ci: Interval | None
    adf_pvalue: float
    b: float
    adv_usd_m: float
    intraday_range_bps: float
    vr_2: float
    vr_20: float
    n_stale: int
    n_extreme: int

    @property
    def asset_group(self) -> str:
        """The two groups the result separates into."""
        return "equity" if self.sleeve in EQUITY_SLEEVES else "credit"

    @property
    def has_estimate(self) -> bool:
        return np.isfinite(self.half_life)


def analyse_fund(fund, universe: Universe, *, n_boot: int = 2000) -> FundResult:
    """Run the whole pipeline for one fund."""
    prices = load_prices(fund.ticker, universe.start, universe.end_or_today)
    nav = load_nav_file(nav_path(fund.ticker))
    frame, _ = align_price_nav(
        prices, nav, ticker=fund.ticker, start=universe.start, end=universe.end_or_today
    )
    frame = add_premium(frame)
    frame, quality = check_quality(frame, ticker=fund.ticker, band_bps=fund.outlier_band_bps)

    stats = describe_premium(frame, ticker=fund.ticker)
    rev = estimate_reversion(
        frame["premium_bps"], ticker=fund.ticker, intervals=True, n_boot=n_boot
    )
    liq = profile_liquidity(frame, ticker=fund.ticker)
    intra = profile_intraday(frame, ticker=fund.ticker, foreign_session=fund.foreign_session)
    vr = profile_variance_ratios(frame["premium_bps"], ticker=fund.ticker)

    return FundResult(
        ticker=fund.ticker,
        sleeve=fund.sleeve,
        role=fund.role,
        foreign_session=fund.foreign_session,
        n=stats.n,
        mean_premium_bps=stats.mean_bps,
        sd_premium_bps=stats.sd_bps,
        min_premium_bps=stats.min_bps,
        max_premium_bps=stats.max_bps,
        pct_days_premium=stats.frac_premium * 100.0,
        resolution=rev.resolution,
        half_life=rev.half_life,
        half_life_display=rev.display,
        bootstrap_ci=rev.bootstrap_ci,
        delta_ci=rev.delta_ci,
        adf_pvalue=rev.adf_pvalue,
        b=rev.fit.b,
        adv_usd_m=liq.median_dollar_volume / 1e6,
        intraday_range_bps=intra.median_range_bps,
        vr_2=vr.results[0].ratio,
        vr_20=vr.results[-1].ratio,
        n_stale=quality.n_stale_nav,
        n_extreme=quality.n_extreme_premium,
    )


def analyse_universe(
    universe: Universe | None = None, *, n_boot: int = 2000
) -> list[FundResult]:
    """Run every fund in the universe."""
    universe = universe or load_universe()
    return [analyse_fund(f, universe, n_boot=n_boot) for f in universe.funds]


def results_table(results: list[FundResult]) -> pd.DataFrame:
    """Tidy per-fund table. Every half-life carries its resolution label."""
    rows = []
    for r in results:
        rows.append(
            {
                "ticker": r.ticker,
                "sleeve": r.sleeve,
                "group": r.asset_group,
                "role": r.role,
                "n": r.n,
                "mean_bps": round(r.mean_premium_bps, 2),
                "sd_bps": round(r.sd_premium_bps, 2),
                "pct_prem": round(r.pct_days_premium, 1),
                "resolution": r.resolution.value,
                "half_life": round(r.half_life, 3) if r.has_estimate else np.nan,
                "ci_low": round(r.bootstrap_ci.lower, 3) if r.bootstrap_ci else np.nan,
                "ci_high": round(r.bootstrap_ci.upper, 3) if r.bootstrap_ci else np.nan,
                "half_life_display": r.half_life_display,
                "adv_usd_m": round(r.adv_usd_m, 0),
                "vr_2": round(r.vr_2, 3),
                "vr_20": round(r.vr_20, 3),
                "n_stale": r.n_stale,
                "n_extreme": r.n_extreme,
            }
        )
    df = pd.DataFrame(rows)
    return df.sort_values("half_life", na_position="last").reset_index(drop=True)


@dataclass(frozen=True)
class SeparationTest:
    """Which funds can actually be told apart."""

    cross_group_pairs: int
    cross_group_overlapping: int
    within_credit_pairs: int
    within_credit_overlapping: int
    overlap_matrix: pd.DataFrame

    @property
    def groups_separate(self) -> bool:
        return self.cross_group_overlapping == 0

    @property
    def within_credit_resolved(self) -> bool:
        return self.within_credit_overlapping == 0

    def summary(self) -> str:
        return (
            f"equity-vs-credit overlapping: {self.cross_group_overlapping}"
            f"/{self.cross_group_pairs}; "
            f"credit-vs-credit overlapping: {self.within_credit_overlapping}"
            f"/{self.within_credit_pairs}"
        )


def measure_separation(results: list[FundResult]) -> SeparationTest:
    """Test which pairs of funds have non-overlapping confidence intervals.

    Interval overlap is the operation behind every ranking claim here. Two
    funds whose intervals overlap cannot be ordered, no matter how different
    their point estimates look.
    """
    usable = [r for r in results if r.has_estimate and r.bootstrap_ci is not None]
    tickers = [r.ticker for r in usable]

    matrix = pd.DataFrame(True, index=tickers, columns=tickers)
    for a, b in combinations(usable, 2):
        overlap = a.bootstrap_ci.overlaps(b.bootstrap_ci)
        matrix.loc[a.ticker, b.ticker] = overlap
        matrix.loc[b.ticker, a.ticker] = overlap

    cross = [(a, b) for a, b in combinations(usable, 2) if a.asset_group != b.asset_group]
    within = [
        (a, b)
        for a, b in combinations(usable, 2)
        if a.asset_group == b.asset_group == "credit"
    ]

    test = SeparationTest(
        cross_group_pairs=len(cross),
        cross_group_overlapping=sum(a.bootstrap_ci.overlaps(b.bootstrap_ci) for a, b in cross),
        within_credit_pairs=len(within),
        within_credit_overlapping=sum(
            a.bootstrap_ci.overlaps(b.bootstrap_ci) for a, b in within
        ),
        overlap_matrix=matrix,
    )
    logger.info("separation: %s", test.summary())
    return test
