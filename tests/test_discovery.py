"""Tests for the price-discovery test.

Built around synthetic worlds where the answer is known by construction: one
where the ETF genuinely leads a stale NAV, one where the premium is pure
mispricing, and one where it is nothing but NAV measurement error. The last is
the important one, because it is the artefact that makes the naive version of
this test untrustworthy.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.discovery import measure_price_discovery


def _index(n):
    return pd.bdate_range("2018-01-02", periods=n, name="date")


def world_etf_leads_nav(n=1500, seed=0, fraction=0.8):
    """The ETF sees news first; NAV catches up next session.

    ``fraction`` of today's premium is realised in tomorrow's NAV move, which
    is the slope the estimator should recover.
    """
    rng = np.random.default_rng(seed)
    news = rng.normal(0, 60, n)            # bps of information, seen by the ETF today
    noise = rng.normal(0, 20, n)

    nav_ret = np.zeros(n)
    nav_ret[1:] = fraction * news[:-1] + noise[1:]

    nav = 100 * np.exp(np.cumsum(nav_ret) / 1e4)
    premium = news
    return pd.Series(premium, index=_index(n)), pd.Series(nav, index=_index(n))


def world_pure_mispricing(n=1500, seed=0):
    """The premium is noise in the ETF price. NAV is already correct and has
    no reason to follow it anywhere."""
    rng = np.random.default_rng(seed)
    nav = 100 * np.exp(np.cumsum(rng.normal(0, 50, n)) / 1e4)
    premium = rng.normal(0, 40, n)
    return pd.Series(premium, index=_index(n)), pd.Series(nav, index=_index(n))


def world_nav_measurement_error(n=1500, seed=0, err_bps=60):
    """THE ARTEFACT. No information anywhere -- the ETF price is exactly right
    and NAV is the true value plus independent error.

    Because premium = price/nav and nav_return = nav_next/nav both divide by
    the same noisy nav, a spurious positive relationship appears from the
    shared denominator alone. A test that cannot tell this apart from genuine
    price discovery is not measuring what it claims to.
    """
    rng = np.random.default_rng(seed)
    true_level = 100 * np.exp(np.cumsum(rng.normal(0, 50, n)) / 1e4)
    err = rng.normal(0, err_bps, n) / 1e4
    observed_nav = true_level * (1 + err)
    premium = (true_level / observed_nav - 1) * 1e4
    return pd.Series(premium, index=_index(n)), pd.Series(observed_nav, index=_index(n))


# --------------------------------------------------------------------------
# Genuine price discovery
# --------------------------------------------------------------------------


@pytest.mark.parametrize("fraction", [0.5, 0.8, 1.0])
def test_recovers_the_lead_coefficient(fraction):
    premium, nav = world_etf_leads_nav(fraction=fraction, seed=1)
    r = measure_price_discovery(
        premium, nav, ticker="LEAD", target_name="index", external=True
    )
    assert r.slope == pytest.approx(fraction, rel=0.15)
    assert r.is_price_discovery


def test_placebo_stays_near_zero_when_lead_is_real():
    """A premium cannot predict a move that already happened."""
    premium, nav = world_etf_leads_nav(seed=2)
    r = measure_price_discovery(premium, nav, ticker="LEAD", target_name="index", external=True)
    assert r.placebo_r_squared < 0.02
    assert r.r_squared > 5 * r.placebo_r_squared


# --------------------------------------------------------------------------
# Cases that must NOT read as price discovery
# --------------------------------------------------------------------------


def test_pure_mispricing_shows_no_lead():
    premium, nav = world_pure_mispricing(seed=3)
    r = measure_price_discovery(premium, nav, ticker="MIS", target_name="nav", external=False)
    assert not r.is_price_discovery
    assert r.r_squared < 0.02


def test_nav_measurement_error_creates_a_spurious_lead():
    """Documents the artefact rather than pretending it is absent.

    There is no information in this world at all, yet regressing the next NAV
    return on the premium produces a clearly positive slope purely from the
    shared denominator. This is why the headline result in the write-up is run
    against an external index instead.
    """
    premium, nav = world_nav_measurement_error(seed=4)
    r = measure_price_discovery(premium, nav, ticker="ART", target_name="nav", external=False)

    assert r.slope > 0.3          # looks like a lead
    assert r.r_squared > 0.10     # and a strong one
    assert not r.uses_external_target  # ...but the flag says do not trust it


def test_external_target_is_immune_to_the_artefact():
    """The decisive property. With the target computed independently of the
    fund's NAV, the artefact cannot appear."""
    rng = np.random.default_rng(5)
    n = 1500
    true_level = 100 * np.exp(np.cumsum(rng.normal(0, 50, n)) / 1e4)
    observed_nav = true_level * (1 + rng.normal(0, 60, n) / 1e4)
    premium = pd.Series((true_level / observed_nav - 1) * 1e4, index=_index(n))

    # An index that shares no input with the fund's NAV.
    external = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 50, n)) / 1e4), index=_index(n)
    )
    r = measure_price_discovery(
        premium, external, ticker="ART", target_name="index", external=True
    )
    assert not r.is_price_discovery
    assert r.r_squared < 0.02


# --------------------------------------------------------------------------
# Mechanics
# --------------------------------------------------------------------------


def test_external_flag_is_reported():
    premium, nav = world_etf_leads_nav(seed=6)
    r = measure_price_discovery(premium, nav, ticker="X", target_name="nav", external=False)
    assert not r.uses_external_target
    assert "shared-nav" in r.summary()
    assert r.as_row()["external"] is False


def test_too_few_observations_raises():
    idx = _index(10)
    with pytest.raises(ValueError, match="usable observations"):
        measure_price_discovery(
            pd.Series(np.arange(10.0), index=idx),
            pd.Series(np.arange(100.0, 110.0), index=idx),
            ticker="T", target_name="nav", external=False,
        )
