"""Tests for the intraday premium range diagnostic.

The point of this module is to detect deviations that open and close within a
session. So the tests construct exactly such a day and check it is seen, and
construct a day with no deviation at all and check nothing is claimed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.intraday import add_intraday_premium, profile_intraday


def _frame(high, low, close, nav):
    idx = pd.bdate_range("2024-01-01", periods=len(high), name="date")
    f = pd.DataFrame(
        {"high": high, "low": low, "close": close, "nav": nav}, index=idx, dtype=float
    )
    f["premium_bps"] = (f["close"] / f["nav"] - 1.0) * 1e4
    return f


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_bounds_computed_against_nav():
    out = add_intraday_premium(_frame([101.0], [99.0], [100.0], [100.0]))
    assert out["premium_high_bps"].iloc[0] == pytest.approx(100.0)
    assert out["premium_low_bps"].iloc[0] == pytest.approx(-100.0)
    assert out["premium_range_bps"].iloc[0] == pytest.approx(200.0)


def test_range_is_never_negative():
    out = add_intraday_premium(_frame([101.0, 100.5], [99.0, 100.1], [100.0, 100.2], [100.0] * 2))
    assert (out["premium_range_bps"] >= 0).all()


def test_missing_columns_rejected():
    with pytest.raises(ValueError, match="missing"):
        add_intraday_premium(pd.DataFrame({"close": [1.0]}))


# --------------------------------------------------------------------------
# The behaviour this exists to detect
# --------------------------------------------------------------------------


def test_round_trip_inside_the_session_is_detected():
    """The core case. Price swings well away from NAV during the day and
    returns to it by the close, so the closing series records nothing."""
    f = _frame(high=[102.0], low=[98.0], close=[100.0], nav=[100.0])
    p = profile_intraday(f, ticker="RT")

    assert p.median_abs_close_bps == pytest.approx(0.0)  # invisible at the close
    assert p.median_range_bps == pytest.approx(400.0)    # but 400bps happened
    assert p.pct_straddling_zero == 100.0                # and it changed sign
    assert p.median_unwound_bps == pytest.approx(200.0)  # 200bps unwound


def test_quiet_day_shows_no_excursion():
    """A fund genuinely tracking NAV all day must not be reported as having
    hidden deviation."""
    f = _frame(high=[100.01], low=[99.99], close=[100.0], nav=[100.0])
    p = profile_intraday(f, ticker="CALM")
    assert p.median_range_bps < 5.0
    assert p.pct_straddling_zero == 100.0  # straddles, but trivially


def test_persistent_deviation_does_not_straddle():
    """A fund at a steady premium all day never crosses zero, so the sign-flip
    measure correctly separates 'wandered across NAV' from 'sat above it'."""
    f = _frame(high=[103.0], low=[102.0], close=[102.5], nav=[100.0])
    p = profile_intraday(f, ticker="PREM")
    assert p.pct_straddling_zero == 0.0
    assert p.median_abs_close_bps == pytest.approx(250.0)


def test_ratio_flags_how_much_is_hidden():
    """range_over_close is the headline: how much wider the within-day
    excursion is than the number close-to-close analysis records."""
    hidden = profile_intraday(_frame([102.0], [98.0], [100.02], [100.0]), ticker="H")
    visible = profile_intraday(_frame([102.6], [102.4], [102.5], [100.0]), ticker="V")
    assert hidden.range_to_close_ratio > 100
    assert visible.range_to_close_ratio < 1


def test_unwound_measures_retracement():
    """Furthest excursion back to the close."""
    f = _frame(high=[105.0], low=[100.0], close=[101.0], nav=[100.0])
    p = profile_intraday(f, ticker="U")
    assert p.median_unwound_bps == pytest.approx(400.0)  # 500bps peak -> 100bps close


# --------------------------------------------------------------------------
# The frozen-basket distinction
# --------------------------------------------------------------------------


def test_foreign_session_flag_is_carried():
    """It decides whether the range is a premium measurement or an upper
    bound, so it must travel with the result rather than live in prose."""
    f = _frame([101.0], [99.0], [100.0], [100.0])
    assert profile_intraday(f, ticker="EWJ", foreign_session=True).foreign_session
    assert "frozen basket" in profile_intraday(f, ticker="EWJ", foreign_session=True).summary()
    assert "upper bound" in profile_intraday(f, ticker="IVV").summary()


def test_moving_basket_inflates_the_range_for_domestic_funds():
    """Why the caveat exists. A domestic fund on a day the market falls 2%
    shows a wide range that is basket movement, not deviation -- and this
    measure cannot tell the two apart."""
    # Price tracks NAV perfectly, but NAV itself moved during the day. Only
    # the close is observed, so the range still looks large.
    f = _frame(high=[102.0], low=[98.0], close=[98.0], nav=[98.0])
    p = profile_intraday(f, ticker="IVV", foreign_session=False)
    assert p.median_range_bps > 380  # looks like deviation
    assert not p.foreign_session      # ...but is flagged as an upper bound only


# --------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------


def test_medians_over_many_days():
    rng = np.random.default_rng(0)
    n = 500
    nav = np.full(n, 100.0)
    close = nav * (1 + rng.normal(0, 0.0005, n))
    high = np.maximum(close, nav) * 1.002
    low = np.minimum(close, nav) * 0.998
    p = profile_intraday(_frame(high, low, close, nav), ticker="MANY")
    assert p.n_obs == n
    assert p.median_range_bps > 0
    assert 0 <= p.pct_straddling_zero <= 100


def test_requires_premium_column():
    f = _frame([101.0], [99.0], [100.0], [100.0]).drop(columns=["premium_bps"])
    with pytest.raises(ValueError, match="call add_premium first"):
        profile_intraday(f, ticker="X")
