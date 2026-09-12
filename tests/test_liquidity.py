"""Tests for liquidity proxies.

The Corwin-Schultz estimator claims to recover a quantity that is not present
in the input -- there are no quotes in daily bars, only trades. So it is
validated the same way the half-life estimator was: simulate a price series
with a KNOWN spread built in, and check the number comes back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.liquidity import corwin_schultz_spread, profile_liquidity


def simulate_bars(
    n_days: int = 2000,
    *,
    spread: float = 0.001,
    daily_vol: float = 0.01,
    steps_per_day: int = 78,  # 5-minute bars in a US session
    seed: int = 0,
    p0: float = 100.0,
) -> pd.DataFrame:
    """Daily OHLC from an intraday random walk observed through a spread.

    The efficient price wanders during the session; the observed high is
    struck at an ask and the observed low at a bid, so the quoted spread
    widens the recorded range by a fixed proportion. That is exactly the
    structure Corwin-Schultz exploits.
    """
    rng = np.random.default_rng(seed)
    step_vol = daily_vol / np.sqrt(steps_per_day)

    highs, lows, closes = [], [], []
    log_p = np.log(p0)
    for _ in range(n_days):
        path = log_p + np.cumsum(rng.normal(0.0, step_vol, steps_per_day))
        efficient = np.exp(path)
        highs.append(efficient.max() * (1.0 + spread / 2.0))
        lows.append(efficient.min() * (1.0 - spread / 2.0))
        closes.append(efficient[-1])
        log_p = path[-1]

    idx = pd.bdate_range("2018-01-02", periods=n_days, name="date")
    return pd.DataFrame(
        {
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": np.full(n_days, 1e7),
        },
        index=idx,
    )


# --------------------------------------------------------------------------
# Recovery of a known spread
# --------------------------------------------------------------------------


@pytest.mark.parametrize("true_spread", [0.0005, 0.001, 0.003])
def test_recovers_known_spread(true_spread):
    """The estimator's whole claim, checked directly.

    The MEAN is the statistic that recovers the spread. The estimator is
    unbiased in expectation and only in expectation -- see
    test_median_and_clipping_both_overstate for what the alternatives do.
    """
    bars = simulate_bars(n_days=3000, spread=true_spread, seed=1)
    estimated = corwin_schultz_spread(bars["high"], bars["low"]).dropna().mean()
    assert estimated == pytest.approx(true_spread, abs=0.0003)


def test_median_and_clipping_both_overstate():
    """Documents a real error made while building this.

    Roughly 40% of daily estimates come out negative, because on any given
    pair of days the two-day range can be smaller than consecutive one-day
    ranges imply. Those negatives are the left half of a noisy distribution,
    not errors: discarding them, or taking a median of a right-skewed
    distribution, destroys the property that makes the estimator work.

    Measured on a known 10bp spread: mean 8.5, median 29.2, clipped mean 41.9.
    """
    bars = simulate_bars(n_days=3000, spread=0.001, seed=1)
    raw = corwin_schultz_spread(bars["high"], bars["low"]).dropna()

    assert raw.mean() == pytest.approx(0.001, abs=0.0003)
    assert raw.median() > 2 * 0.001
    assert raw.clip(lower=0).mean() > 3 * 0.001
    assert 0.30 < (raw < 0).mean() < 0.50


def test_ordering_is_preserved():
    """The cross-sectional use only needs the ranking to survive, and the
    ranking is more robust than the level."""
    tight_bars = simulate_bars(spread=0.0002, seed=2)
    wide_bars = simulate_bars(spread=0.004, seed=2)
    tight = corwin_schultz_spread(tight_bars["high"], tight_bars["low"]).mean()
    wide = corwin_schultz_spread(wide_bars["high"], wide_bars["low"]).mean()
    assert tight < wide


def test_zero_spread_estimates_near_zero():
    """A slightly negative answer is correct here, not a failure: it means the
    spread is not distinguishable from zero."""
    bars = simulate_bars(n_days=3000, spread=0.0, seed=3)
    assert abs(corwin_schultz_spread(bars["high"], bars["low"]).dropna().mean()) < 0.0002


def test_spread_separated_from_volatility():
    """The point of the estimator: a volatile asset with a tight spread must
    not be mistaken for a calm one with a wide spread. A naive high-low range
    would conflate them entirely."""
    volatile_tight = simulate_bars(spread=0.0002, daily_vol=0.04, seed=4)
    calm_wide = simulate_bars(spread=0.004, daily_vol=0.005, seed=4)

    s_vol = corwin_schultz_spread(volatile_tight["high"], volatile_tight["low"]).mean()
    s_calm = corwin_schultz_spread(calm_wide["high"], calm_wide["low"]).mean()

    # Raw range says the volatile asset is "wider" -- the estimator disagrees.
    raw_vol = ((volatile_tight["high"] - volatile_tight["low"]) / volatile_tight["close"]).median()
    raw_calm = ((calm_wide["high"] - calm_wide["low"]) / calm_wide["close"]).median()
    assert raw_vol > raw_calm
    assert s_vol < s_calm


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_first_value_is_nan():
    """A two-day window needs two days."""
    bars = simulate_bars(n_days=50, seed=5)
    assert np.isnan(corwin_schultz_spread(bars["high"], bars["low"]).iloc[0])


def test_negatives_retained_by_default():
    """Clipping is off by default because it destroys the estimator."""
    bars = simulate_bars(n_days=500, spread=0.0, seed=6)
    assert (corwin_schultz_spread(bars["high"], bars["low"]).dropna() < 0).any()


def test_clipping_available_but_opt_in():
    bars = simulate_bars(n_days=500, spread=0.0, seed=6)
    clipped = corwin_schultz_spread(bars["high"], bars["low"], clip_negative=True).dropna()
    assert (clipped >= 0).all()


def test_mismatched_index_rejected():
    bars = simulate_bars(n_days=50, seed=7)
    with pytest.raises(ValueError, match="share an index"):
        corwin_schultz_spread(bars["high"], bars["low"].iloc[:-1])


def test_high_below_low_rejected():
    bars = simulate_bars(n_days=50, seed=8)
    bars.loc[bars.index[10], "high"] = bars.loc[bars.index[10], "low"] - 1
    with pytest.raises(ValueError, match="high below low"):
        corwin_schultz_spread(bars["high"], bars["low"])


# --------------------------------------------------------------------------
# Profile
# --------------------------------------------------------------------------


def test_estimator_is_noisy_at_small_samples():
    """Worth knowing before reading the real numbers: this estimator needs a
    lot of days. The real sample is ~2,185, so single-fund spread estimates
    carry meaningful uncertainty and only the ranking should be leaned on.
    """
    errors = {}
    for n in (500, 3000):
        spreads = [
            corwin_schultz_spread(
                *simulate_bars(n_days=n, spread=0.001, seed=s)[["high", "low"]]
                .to_dict("series").values()
            ).mean()
            for s in range(6)
        ]
        errors[n] = np.std(spreads)
    assert errors[3000] < errors[500]


def test_profile_reports_all_measures():
    bars = simulate_bars(n_days=3000, spread=0.001, seed=9)
    p = profile_liquidity(bars, ticker="TEST")
    assert p.ticker == "TEST"
    assert p.n_obs == 3000
    assert p.spread_bps == pytest.approx(10.0, abs=4.0)
    assert p.spread_is_usable
    assert p.median_dollar_volume > 0
    assert p.median_range_bps > 0
    assert set(p.as_row()) == {
        "ticker", "spread_bps", "spread_usable", "adv_usd_m", "range_bps",
    }


def test_profile_requires_ohlc():
    with pytest.raises(ValueError, match="missing"):
        profile_liquidity(pd.DataFrame({"close": [1.0]}), ticker="TEST")


# --------------------------------------------------------------------------
# Overnight gaps -- why this estimator is NOT used as the liquidity proxy
# --------------------------------------------------------------------------


def bars_with_gaps(
    n_days: int = 3000,
    *,
    spread: float = 0.001,
    daily_vol: float = 0.01,
    overnight_vol: float = 0.0,
    steps_per_day: int = 78,
    seed: int = 0,
    p0: float = 100.0,
) -> pd.DataFrame:
    """Bars that jump while the market is closed, as real ones do."""
    rng = np.random.default_rng(seed)
    step_vol = daily_vol / np.sqrt(steps_per_day)
    highs, lows, closes = [], [], []
    log_p = np.log(p0)
    for _ in range(n_days):
        log_p += rng.normal(0.0, overnight_vol)
        path = log_p + np.cumsum(rng.normal(0.0, step_vol, steps_per_day))
        eff = np.exp(path)
        highs.append(eff.max() * (1 + spread / 2))
        lows.append(eff.min() * (1 - spread / 2))
        closes.append(eff[-1])
        log_p = path[-1]
    idx = pd.bdate_range("2018-01-02", periods=n_days, name="date")
    return pd.DataFrame({"high": highs, "low": lows, "close": closes}, index=idx)


def test_overnight_adjustment_helps_substantially():
    """Aligning day two onto day one's close removes most of the damage.
    On real SPY data it moves the estimate from -33.0bps to -0.8bps."""
    bars = bars_with_gaps(spread=0.001, overnight_vol=0.01, seed=1)
    una = corwin_schultz_spread(bars["high"], bars["low"]).dropna().mean()
    adj = corwin_schultz_spread(bars["high"], bars["low"], bars["close"]).dropna().mean()
    assert adj > una
    assert abs(adj) < 0.5 * abs(una)


@pytest.mark.parametrize("overnight_vol", [0.005, 0.01, 0.02])
def test_estimator_still_fails_with_realistic_gaps(overnight_vol):
    """The negative result this module documents.

    Even adjusted, the estimator returns a NEGATIVE spread once overnight
    moves are realistic -- and a negative effective spread is impossible, not
    merely small. SPY's overnight moves run around 0.32%, inside this range.
    This is why the cross-section uses dollar volume instead.
    """
    bars = bars_with_gaps(spread=0.001, overnight_vol=overnight_vol, seed=1)
    adjusted = corwin_schultz_spread(bars["high"], bars["low"], bars["close"]).dropna().mean()
    assert adjusted < 0.001  # fails to recover the true 10bps


def test_estimator_works_without_gaps():
    """Establishes the failure is caused by gaps specifically, not by a bug in
    the implementation."""
    bars = bars_with_gaps(spread=0.001, overnight_vol=0.0, seed=1)
    adjusted = corwin_schultz_spread(bars["high"], bars["low"], bars["close"]).dropna().mean()
    assert adjusted == pytest.approx(0.001, abs=0.0004)


def test_unusable_spread_is_flagged():
    bars = bars_with_gaps(spread=0.001, overnight_vol=0.02, seed=1)
    bars["volume"] = 1e7
    assert not profile_liquidity(bars, ticker="GAPPY").spread_is_usable
