#!/usr/bin/env python
"""Produce the cross-sectional result: table, separation test, and figure.

    python scripts/cross_section.py

Writes data/processed/results.csv and docs/figures/cross_section.png.
"""

from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.analysis.cross_section import (  # noqa: E402
    analyse_universe,
    measure_separation,
    results_table,
)
from src.viz.cross_section_plot import plot_cross_section  # noqa: E402


def main() -> None:
    warnings.filterwarnings("ignore")
    logging.disable(logging.INFO)

    results = analyse_universe(n_boot=2000)
    table = results_table(results)
    separation = measure_separation(results)

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "results.csv", index=False)

    fig_dir = ROOT / "docs" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_cross_section(results)
    plt.tight_layout()
    plt.savefig(fig_dir / "cross_section.png", dpi=140)

    print("=" * 78)
    print("  PREMIUM / DISCOUNT AND REVERSION, 2018-01-02 to present")
    print("=" * 78)
    cols = ["ticker", "group", "n", "mean_bps", "sd_bps", "pct_prem",
            "resolution", "half_life_display", "ci_low", "ci_high"]
    print(table[cols].to_string(index=False))

    print()
    print("-" * 78)
    print("  CAN THE FUNDS BE TOLD APART? (bootstrap interval overlap)")
    print("-" * 78)
    print(f"  equity vs credit : {separation.cross_group_overlapping}"
          f"/{separation.cross_group_pairs} pairs overlap"
          f"   -> {'SEPARATED' if separation.groups_separate else 'not separated'}")
    print(f"  credit vs credit : {separation.within_credit_overlapping}"
          f"/{separation.within_credit_pairs} pairs overlap"
          f"   -> {'resolved' if separation.within_credit_resolved else 'UNRESOLVED'}")
    print()
    print("  The equity/credit split is firm. The ordering inside credit is not,")
    print("  so it is reported as an ordering that the sample cannot resolve.")
    print()
    print(f"  wrote {out / 'results.csv'}")
    print(f"  wrote {fig_dir / 'cross_section.png'}")


if __name__ == "__main__":
    main()
