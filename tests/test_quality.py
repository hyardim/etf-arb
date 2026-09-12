"""Tests for staleness and outlier checks.

The governing property is that these checks never remove data. March 2020
produced the largest bond ETF dislocations on record and they were genuine;
silently filtering them would delete the part of the sample that actually
tests the hypothesis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.quality import add_quality_flags, check_quality


def _frame(close, nav):
    idx = pd.bdate_range("2024-01-01", periods=len(close), name="date")
    f = pd.DataFrame({"close": close, "nav": nav}, index=idx, dtype=float)
    f["premium_bps"] = (f["close"] / f["nav"] - 1.0) * 1e4
    return f


# --------------------------------------------------------------------------
# The no-drop guarantee
# --------------------------------------------------------------------------


def test_checks_never_drop_rows():
    f = _frame([100, 150, 50, 100.0], [100, 100, 100, 100.0])
    flagged, report = check_quality(f, ticker="TEST", band_bps=500)
    assert len(flagged) == len(f)
    assert report.n_obs == len(f)


def test_extremes_are_flagged_not_removed():
    """A -2000bps crisis observation must survive, flagged."""
    f = _frame([100, 100, 80.0], [100, 100, 100.0])
    flagged, report = check_quality(f, ticker="TEST", band_bps=500)

    assert len(flagged) == 3
    assert report.n_extreme_premium == 1
    assert flagged["extreme_premium"].iloc[2]
    assert flagged["premium_bps"].iloc[2] == pytest.approx(-2000.0)


# --------------------------------------------------------------------------
# Stale NAV
# --------------------------------------------------------------------------


def test_unchanged_nav_with_price_move_is_stale():
    f = _frame([100.0, 101.0], [100.0, 100.0])
    flagged, report = check_quality(f, ticker="TEST", band_bps=500)
    assert flagged["stale_nav"].iloc[1]
    assert report.n_stale_nav == 1


def test_unchanged_nav_without_price_move_is_not_stale():
    """A repeated NAV is only suspicious when the ETF actually moved; a quiet
    day in both is unremarkable."""
    f = _frame([100.0, 100.0001], [100.0, 100.0])
    flagged, report = check_quality(f, ticker="TEST", band_bps=500)
    assert not flagged["stale_nav"].iloc[1]
    assert report.n_stale_nav == 0


def test_price_move_threshold_is_respected():
    f = _frame([100.0, 100.1], [100.0, 100.0])  # 10bps move
    _, lenient = check_quality(f, ticker="T", band_bps=500, price_move_bps=25)
    _, strict = check_quality(f, ticker="T", band_bps=500, price_move_bps=5)
    assert lenient.n_stale_nav == 0
    assert strict.n_stale_nav == 1


def test_longest_nav_run_detected():
    """A long run of identical NAVs indicates a dead feed rather than a
    single missed update."""
    f = _frame([100, 101, 102, 103, 104.0], [100, 100, 100, 100, 101.0])
    _, report = check_quality(f, ticker="TEST", band_bps=500)
    assert report.longest_nav_run == 3


def test_no_false_stale_on_first_row():
    f = _frame([100.0, 100.0], [100.0, 100.0])
    flagged, _ = check_quality(f, ticker="TEST", band_bps=500)
    assert not flagged["stale_nav"].iloc[0]


# --------------------------------------------------------------------------
# Per-sleeve bands
# --------------------------------------------------------------------------


def test_band_is_configurable_per_sleeve():
    """Credit and EM get wider bands than equity because the underlying is
    genuinely harder to value."""
    f = _frame([100.0, 106.0], [100.0, 100.0])  # +600bps
    _, equity = check_quality(f, ticker="EQ", band_bps=500)
    _, credit = check_quality(f, ticker="CR", band_bps=800)
    assert equity.n_extreme_premium == 1
    assert credit.n_extreme_premium == 0


def test_band_applies_to_absolute_value():
    f = _frame([94.0, 106.0], [100.0, 100.0])
    _, report = check_quality(f, ticker="TEST", band_bps=500)
    assert report.n_extreme_premium == 2


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def test_report_exposes_counts_and_dates():
    f = _frame([100, 100, 80.0], [100, 100, 100.0])
    _, report = check_quality(f, ticker="HYG", band_bps=500)
    row = report.as_row()
    assert row["ticker"] == "HYG"
    assert row["n_obs"] == 3
    assert row["n_extreme_premium"] == 1
    assert len(report.extreme_dates) == 1
    assert "HYG" in report.summary()


def test_percentages_computed():
    f = _frame([100.0] * 9 + [80.0], [100.0] * 10)
    _, report = check_quality(f, ticker="TEST", band_bps=500)
    assert report.pct_extreme == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Refusals
# --------------------------------------------------------------------------


def test_missing_column_raises():
    with pytest.raises(ValueError, match="premium_bps"):
        add_quality_flags(pd.DataFrame({"close": [1.0], "nav": [1.0]}), band_bps=500)


def test_non_positive_band_raises():
    with pytest.raises(ValueError, match="band_bps must be positive"):
        add_quality_flags(_frame([100.0], [100.0]), band_bps=0)


# --------------------------------------------------------------------------
# NAV-move tolerance
# --------------------------------------------------------------------------


def test_near_static_nav_is_flagged_stale():
    """Exact equality is too literal. iShares publishes six decimal places, so
    an exactly repeated NAV essentially never occurs -- on the real HYG sample
    an equality test flags nothing. A barely-moving NAV against a moving price
    is the real signal."""
    f = _frame([100.0, 101.0], [100.0, 100.0005])  # NAV +0.05bps, price +100bps
    flagged, report = check_quality(f, ticker="TEST", band_bps=500)
    assert flagged["stale_nav"].iloc[1]
    assert report.n_stale_nav == 1


def test_exact_equality_recoverable():
    """nav_move_bps=0 restores the strict test."""
    f = _frame([100.0, 101.0], [100.0, 100.0005])
    _, report = check_quality(f, ticker="TEST", band_bps=500, nav_move_bps=0.0)
    assert report.n_stale_nav == 0


def test_normally_moving_nav_not_flagged():
    f = _frame([100.0, 101.0], [100.0, 100.9])
    _, report = check_quality(f, ticker="TEST", band_bps=500)
    assert report.n_stale_nav == 0


def test_nav_tolerance_recorded_in_report():
    f = _frame([100.0, 101.0], [100.0, 100.5])
    _, report = check_quality(f, ticker="TEST", band_bps=500, nav_move_bps=2.5)
    assert report.nav_move_bps == 2.5
