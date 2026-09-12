"""Mean-reversion estimation: OU simulation, AR(1) fitting, half-lives.

The premium is modelled as a discretely sampled Ornstein-Uhlenbeck process,
which in discrete time is an AR(1):

    x[t+1] = a + b * x[t] + e[t+1]

    kappa     = -ln(b)            mean reversion speed, per period
    half_life = ln(2) / kappa     periods to close half the gap
    long_run  = a / (1 - b)       equilibrium premium level

Half-life is the headline number of this project, and half-life estimators are
easy to get subtly wrong -- sign errors and log errors produce plausible-looking
but incorrect values that no amount of staring at real data will reveal. The
simulator below exists so the estimator can be validated against a process
whose parameter is known by construction, which is the difference between a
measurement and a guess.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def half_life_to_kappa(half_life: float) -> float:
    """Convert a half-life in periods to an OU mean-reversion speed."""
    if half_life <= 0:
        raise ValueError(f"half_life must be positive, got {half_life}")
    return float(np.log(2.0) / half_life)


def kappa_to_half_life(kappa: float) -> float:
    """Convert an OU mean-reversion speed to a half-life in periods."""
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    return float(np.log(2.0) / kappa)


def simulate_ou(
    kappa: float,
    n: int,
    *,
    seed: int | None = None,
    sigma: float = 1.0,
    mu: float = 0.0,
    dt: float = 1.0,
    x0: float | None = None,
) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck path with a known mean-reversion speed.

    Uses the EXACT discrete transition rather than an Euler-Maruyama step:

        x[t+1] = mu + (x[t] - mu) * exp(-kappa*dt) + sigma * sqrt(v) * z

    with ``v = (1 - exp(-2*kappa*dt)) / (2*kappa)``.

    This matters for the purpose at hand. Euler discretisation carries an
    O(dt) bias in the autoregressive coefficient, which maps straight into a
    biased half-life. Since these paths are the yardstick the estimator is
    measured against, the yardstick must not itself be bent -- otherwise a
    recovery test would be comparing one approximation to another and could
    pass while both are wrong.

    Args:
        kappa: Mean-reversion speed. Half-life is ln(2)/kappa.
        n: Number of observations.
        seed: Seed for reproducibility.
        sigma: Instantaneous volatility.
        mu: Long-run mean.
        dt: Sampling interval, in the same units as kappa.
        x0: Starting value. Defaults to a draw from the stationary
            distribution, so the path needs no burn-in.

    Returns:
        Array of length ``n``.
    """
    if kappa <= 0:
        raise ValueError(f"kappa must be positive, got {kappa}")
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative, got {sigma}")
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")

    rng = np.random.default_rng(seed)

    decay = np.exp(-kappa * dt)
    step_var = sigma**2 * (1.0 - decay**2) / (2.0 * kappa)
    step_sd = np.sqrt(step_var)

    # Stationary variance of the OU process; starting here avoids burn-in.
    stationary_sd = sigma / np.sqrt(2.0 * kappa)

    path = np.empty(n, dtype=float)
    path[0] = mu + stationary_sd * rng.standard_normal() if x0 is None else x0

    shocks = rng.standard_normal(n - 1) * step_sd
    for t in range(1, n):
        path[t] = mu + (path[t - 1] - mu) * decay + shocks[t - 1]

    return path


def simulate_ar1(
    b: float,
    n: int,
    *,
    seed: int | None = None,
    a: float = 0.0,
    sigma: float = 1.0,
) -> np.ndarray:
    """Simulate a plain AR(1), including cases an OU cannot represent.

    ``simulate_ou`` requires kappa > 0, so b must lie in (0, 1). This function
    accepts any b, which is needed to exercise the estimator's guard clauses:
    b >= 1 (a random walk, no finite half-life) and b <= 0 (oscillatory, the
    shape bid-ask bounce induces in a closing-price series).
    """
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")

    rng = np.random.default_rng(seed)
    shocks = rng.standard_normal(n) * sigma

    path = np.empty(n, dtype=float)
    path[0] = shocks[0] / np.sqrt(1 - b**2) if abs(b) < 1 else shocks[0]
    for t in range(1, n):
        path[t] = a + b * path[t - 1] + shocks[t]

    return path
