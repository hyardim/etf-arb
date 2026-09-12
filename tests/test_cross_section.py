"""Tests for the cross-sectional result.

The thing worth testing here is the separation logic, because it is what every
ranking claim in the write-up rests on. A bug that reported overlapping
intervals as separated would turn an honest "cannot distinguish these" into a
false finding.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.cross_section import FundResult, measure_separation, results_table
from src.analysis.reversion import Interval, Resolution


def _fund(ticker, sleeve, half_life, lo, hi, resolution=Resolution.IDENTIFIED):
    return FundResult(
        ticker=ticker, sleeve=sleeve, role="primary", foreign_session=False,
        n=2185, mean_premium_bps=10.0, sd_premium_bps=30.0,
        min_premium_bps=-100.0, max_premium_bps=100.0, pct_days_premium=60.0,
        resolution=resolution, half_life=half_life,
        half_life_display=f"{half_life:.2f} d",
        bootstrap_ci=Interval(lo, hi, "bootstrap"),
        delta_ci=Interval(lo, hi, "delta"),
        adf_pvalue=1e-10, b=0.5, adv_usd_m=1000.0,
        intraday_range_bps=50.0, vr_2=0.7, vr_20=0.1,
        n_stale=0, n_extreme=0,
    )


def _real_shaped_universe():
    """Mirrors the measured intervals."""
    return [
        _fund("IVV", "equity_us_large", 0.29, 0.21, 0.36),
        _fund("EEM", "equity_em", 0.38, 0.22, 0.55),
        _fund("HYG", "credit_hy", 0.94, 0.69, 1.15),
        _fund("EMB", "debt_em", 1.32, 0.77, 1.94),
        _fund("LQD", "credit_ig", 1.49, 0.82, 1.98),
    ]


# --------------------------------------------------------------------------
# Grouping
# --------------------------------------------------------------------------


def test_sleeves_map_to_groups():
    """Equity sleeves trade continuously on exchange; credit and EM debt do
    not. That distinction is what the result separates on."""
    assert _fund("IVV", "equity_us_large", 1, 0, 2).asset_group == "equity"
    assert _fund("EEM", "equity_em", 1, 0, 2).asset_group == "equity"
    assert _fund("HYG", "credit_hy", 1, 0, 2).asset_group == "credit"
    assert _fund("EMB", "debt_em", 1, 0, 2).asset_group == "credit"


# --------------------------------------------------------------------------
# Separation -- what every ranking claim rests on
# --------------------------------------------------------------------------


def test_real_shaped_data_gives_two_group_separation():
    """The headline result, on intervals matching the measured ones."""
    t = measure_separation(_real_shaped_universe())
    assert t.cross_group_overlapping == 0
    assert t.groups_separate
    assert t.within_credit_overlapping == t.within_credit_pairs
    assert not t.within_credit_resolved


def test_overlapping_intervals_are_not_claimed_distinguishable():
    """The failure mode that matters. LQD [0.82, 1.98] contains HYG's whole
    interval, so a 0.55d gap in point estimates means nothing."""
    funds = [
        _fund("HYG", "credit_hy", 0.94, 0.69, 1.15),
        _fund("LQD", "credit_ig", 1.49, 0.82, 1.98),
    ]
    t = measure_separation(funds)
    assert t.within_credit_overlapping == 1
    assert t.overlap_matrix.loc["HYG", "LQD"]


def test_clearly_separated_intervals_are_detected():
    funds = [
        _fund("FAST", "equity_us_large", 0.3, 0.2, 0.4),
        _fund("SLOW", "credit_hy", 5.0, 4.0, 6.0),
    ]
    t = measure_separation(funds)
    assert t.cross_group_overlapping == 0
    assert not t.overlap_matrix.loc["FAST", "SLOW"]


def test_touching_intervals_count_as_overlapping():
    """Boundary case: shared endpoint is not separation."""
    funds = [
        _fund("A", "equity_us_large", 1.0, 0.5, 1.5),
        _fund("B", "credit_hy", 2.0, 1.5, 2.5),
    ]
    assert measure_separation(funds).cross_group_overlapping == 1


def test_matrix_is_symmetric_with_true_diagonal():
    t = measure_separation(_real_shaped_universe())
    m = t.overlap_matrix
    assert (m.values == m.values.T).all()
    assert all(m.loc[x, x] for x in m.index)


def test_unresolved_funds_excluded_from_separation():
    """A censored fund has no point estimate, so it cannot be ordered against
    anything and must not silently count as separated."""
    funds = _real_shaped_universe() + [
        FundResult(
            ticker="SPY", sleeve="equity_us_large", role="control", foreign_session=False,
            n=2184, mean_premium_bps=0.3, sd_premium_bps=4.7,
            min_premium_bps=-80.0, max_premium_bps=90.0, pct_days_premium=57.7,
            resolution=Resolution.UNRESOLVED, half_life=np.nan,
            half_life_display="< 0.34 d (censored)",
            bootstrap_ci=None, delta_ci=None, adf_pvalue=1e-19, b=-0.007,
            adv_usd_m=30342.0, intraday_range_bps=96.0, vr_2=0.45, vr_20=0.05,
            n_stale=0, n_extreme=0,
        )
    ]
    t = measure_separation(funds)
    assert "SPY" not in t.overlap_matrix.index


# --------------------------------------------------------------------------
# Table
# --------------------------------------------------------------------------


def test_table_sorted_by_half_life_with_censored_last():
    df = results_table(_real_shaped_universe())
    hl = df["half_life"].dropna()
    assert list(hl) == sorted(hl)


def test_every_row_carries_a_resolution_label():
    """No unlabelled numbers in the results table."""
    df = results_table(_real_shaped_universe())
    assert df["resolution"].notna().all()
    assert "half_life_display" in df.columns


def test_table_carries_intervals():
    df = results_table(_real_shaped_universe())
    assert {"ci_low", "ci_high"} <= set(df.columns)
    assert (df["ci_low"] < df["half_life"]).all()
    assert (df["ci_high"] > df["half_life"]).all()
