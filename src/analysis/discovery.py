"""Price discovery: does the premium carry information, or is it mispricing?

The premium is the same statistic for every fund, but it does not mean the
same thing in every fund. This module distinguishes the cases by asking what
the premium PREDICTS.

If the ETF price leads a stale NAV, today's premium should be followed by NAV
moving in the same direction -- the ETF saw something first and NAV catches
up. That is price discovery, and it is not an arbitrage opportunity: the ETF
was right and the NAV was late.

If instead the premium were simple mispricing, the ETF price would fall back
toward an already-correct NAV, and the premium would predict the ETF's own
return rather than NAV's.

THE TRAP, and why the headline test is run against an external index.

``premium_t = price_t / nav_t - 1`` and ``nav_return_{t+1} = nav_{t+1}/nav_t - 1``
share ``nav_t``. If NAV carries measurement error, a spurious positive
relationship appears between them from the shared denominator alone, with no
information involved.

A placebo at t+2 does NOT rule this out, because the mechanical effect lives
only at t+1 as well -- both explanations predict the same decay pattern. It
was checked and discarded for that reason.

The test that does discriminate uses a benchmark computed OUTSIDE the fund's
NAV process. For a fund whose market is closed during US hours, the home index
is exactly that: the Nikkei is calculated in Tokyo and shares no input with
iShares' valuation. NAV noise cannot move it. Measured on EWJ, today's premium
predicts the next Nikkei session with slope +0.92 (t = 18.2, R^2 = 0.140),
while the same-session and previous-session placebos sit at R^2 = 0.006 and
0.010. A 100bp closing premium is followed by the Nikkei opening about 92bp
higher.

WHAT THIS DOES NOT MEAN. Establishing that the premium carries information is
not the same as establishing that the information is new. CME Nikkei futures
trade nearly 24 hours, so the price of Japan risk keeps updating while Tokyo
cash is shut. Regressed against futures rather than the cash index, the EWJ
premium moves one-for-one with the SAME day's futures move (slope +1.031,
t = 23.0) and predicts essentially nothing the next day (R^2 = 0.003).

So the premium mirrors what futures have already priced; the Nikkei CASH index
is the stale object that makes the forward regression look predictive. The
result is a correct statement about price discovery and a false lead about
tradeability: the premium measures how far the stale Tokyo cash close has
drifted from a live market that is already quoted continuously elsewhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DiscoveryResult:
    """How much of a premium is realised later in the thing it prices."""

    ticker: str
    target: str
    """What the premium was tested against: 'nav' or an external index."""
    n: int
    slope: float
    """bps of subsequent target move per bp of premium. A slope near 1 means
    the premium is realised almost fully."""
    t_stat: float
    r_squared: float
    placebo_r_squared: float
    """Same regression against the PREVIOUS period's move. Should be near
    zero; a large value means the relationship is not forward-looking."""
    uses_external_target: bool
    """True when the target is computed outside the fund's NAV process, which
    is what rules out the shared-denominator artefact."""

    @property
    def is_price_discovery(self) -> bool:
        return (
            self.slope > 0.2
            and abs(self.t_stat) > 3.0
            and self.r_squared > 0.02
            and self.r_squared > 4.0 * self.placebo_r_squared
        )

    def summary(self) -> str:
        verdict = "leads the target" if self.is_price_discovery else "no lead"
        external = "external" if self.uses_external_target else "shared-nav"
        return (
            f"{self.ticker}->{self.target}: slope {self.slope:+.2f} "
            f"t={self.t_stat:+.1f} R2={self.r_squared:.3f} "
            f"(placebo {self.placebo_r_squared:.3f}) [{external}] {verdict}"
        )

    def as_row(self) -> dict:
        return {
            "ticker": self.ticker,
            "target": self.target,
            "slope": round(self.slope, 3),
            "t_stat": round(self.t_stat, 1),
            "r_squared": round(self.r_squared, 4),
            "placebo_r2": round(self.placebo_r_squared, 4),
            "external": self.uses_external_target,
            "price_discovery": self.is_price_discovery,
        }


def _regress(x: pd.Series, y: pd.Series) -> tuple[float, float, float, int]:
    d = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    d = d[np.isfinite(d).all(axis=1)]
    if len(d) < 30:
        raise ValueError(f"only {len(d)} usable observations")
    r = stats.linregress(d["x"], d["y"])
    return float(r.slope), float(r.slope / r.stderr), float(r.rvalue**2), len(d)


def measure_price_discovery(
    premium_bps: pd.Series,
    target_level: pd.Series,
    *,
    ticker: str,
    target_name: str,
    external: bool,
    horizon: int = 1,
) -> DiscoveryResult:
    """Regress the next period's target return on today's premium.

    Args:
        premium_bps: Closing premium series.
        target_level: Price or NAV level whose future return is predicted.
        ticker: Label.
        target_name: What the target is, for reporting.
        external: Whether the target is computed independently of the fund's
            NAV. Only an external target rules out the shared-denominator
            artefact described in the module docstring.
        horizon: Periods ahead. 1 means the next session.

    Returns:
        DiscoveryResult carrying the forward regression and a backward-looking
        placebo.
    """
    target_return = target_level.pct_change() * 1e4

    slope, t_stat, r2, n = _regress(premium_bps, target_return.shift(-horizon))
    # Placebo: the premium cannot cause a move that already happened.
    _, _, placebo_r2, _ = _regress(premium_bps, target_return.shift(horizon))

    result = DiscoveryResult(
        ticker=ticker,
        target=target_name,
        n=n,
        slope=slope,
        t_stat=t_stat,
        r_squared=r2,
        placebo_r_squared=placebo_r2,
        uses_external_target=external,
    )
    logger.info("%s", result.summary())
    return result
