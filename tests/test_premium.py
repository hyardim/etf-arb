"""Tests for premium/discount construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.premium import add_premium, describe_premium


def _frame(close, nav):
    idx = pd.bdate_range("2024-01-01", periods=len(close), name="date")
    return pd.DataFrame({"close": close, "nav": nav}, index=idx, dtype=float)


# --------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------


def test_premium_is_a_ratio():
    out = add_premium(_frame([101.0], [100.0]))
    assert out["premium"].iloc[0] == pytest.approx(0.01)
    assert out["premium_bps"].iloc[0] == pytest.approx(100.0)


def test_sign_convention():
    """Positive is a premium, negative a discount."""
    out = add_premium(_frame([101.0, 99.0], [100.0, 100.0]))
    assert out["premium_bps"].iloc[0] > 0
    assert out["premium_bps"].iloc[1] < 0


def test_ratio_is_scale_free():
    """The reason a ratio is used rather than a difference: the same
    proportional deviation must give the same number on a $40 and a $400 fund,
    or half-lives across funds are not comparable."""
    cheap = add_premium(_frame([40.4], [40.0]))["premium_bps"].iloc[0]
    dear = add_premium(_frame([404.0], [400.0]))["premium_bps"].iloc[0]
    assert cheap == pytest.approx(dear)
    assert cheap == pytest.approx(100.0)

    # A fixed 10c gap, by contrast, means very different things.
    small = add_premium(_frame([40.1], [40.0]))["premium_bps"].iloc[0]
    large = add_premium(_frame([400.1], [400.0]))["premium_bps"].iloc[0]
    assert small > 9 * large


def test_zero_premium_when_price_equals_nav():
    out = add_premium(_frame([100.0] * 3, [100.0] * 3))
    assert (out["premium_bps"] == 0).all()


def test_original_columns_preserved_and_input_unmutated():
    frame = _frame([101.0], [100.0])
    out = add_premium(frame)
    assert {"close", "nav"} <= set(out.columns)
    assert "premium" not in frame.columns


def test_custom_column_names():
    frame = _frame([101.0], [100.0]).rename(columns={"close": "px", "nav": "navps"})
    out = add_premium(frame, price_col="px", nav_col="navps")
    assert out["premium_bps"].iloc[0] == pytest.approx(100.0)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_missing_column_raises():
    with pytest.raises(ValueError, match="no 'nav'"):
        add_premium(pd.DataFrame({"close": [1.0]}))


def test_non_positive_nav_raises():
    with pytest.raises(ValueError, match="non-positive NAV"):
        add_premium(_frame([100.0, 100.0], [100.0, 0.0]))


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------


def test_describe_basic_statistics():
    rng = np.random.default_rng(0)
    nav = np.full(500, 100.0)
    close = nav * (1 + rng.normal(0, 0.001, 500))
    s = describe_premium(add_premium(_frame(close, nav)), ticker="TEST")

    assert s.n == 500
    assert abs(s.mean_bps) < 5
    assert s.sd_bps > 0
    assert 0.0 <= s.frac_premium <= 1.0
    assert not s.mean_differs_from_zero


def test_describe_flags_non_zero_mean():
    """A persistent offset is a data-convention alarm, not a finding -- the
    shape a dividend-adjusted price series against unadjusted NAV would take."""
    rng = np.random.default_rng(1)
    nav = np.full(500, 100.0)
    close = nav * (1.005 + rng.normal(0, 0.0005, 500))
    s = describe_premium(add_premium(_frame(close, nav)), ticker="DRIFT")

    assert s.mean_differs_from_zero
    assert s.mean_bps == pytest.approx(50, abs=5)
    assert s.frac_premium > 0.95
    assert "MEAN != 0" in s.summary()


def test_describe_requires_premium_column():
    with pytest.raises(ValueError, match="call add_premium first"):
        describe_premium(_frame([100.0], [100.0]), ticker="TEST")
