"""The cross-sectional figure.

Form choice: a dot plot with error bars, not a bar chart. The interval IS the
finding -- it is what shows the within-credit ordering is unresolved -- and
bars would invite reading magnitude from zero while burying the uncertainty
inside them.

Censored funds appear on the same axis as left-pointing arrows at their upper
bound, so that "faster than we can measure" sits in the ordering rather than
being dropped from the figure. Omitting them would make the equity group look
thinner than it is.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.viz.style import GRID, INK, INK_MUTED, INK_SECONDARY, SERIES, use_house_style

# Two groups, two categorical slots in fixed order. Never cycled.
GROUP_COLOR = {"equity": SERIES[0], "credit": SERIES[1]}
GROUP_LABEL = {
    "equity": "Equity (exchange-traded basket)",
    "credit": "Credit & EM debt (OTC basket)",
}


def plot_cross_section(results, *, ax=None, title=None):
    """Half-life with bootstrap intervals, ordered, grouped by basket type."""
    use_house_style()
    if ax is None:
        _, ax = plt.subplots(figsize=(9.2, 5.0))

    estimated = [r for r in results if r.has_estimate and r.bootstrap_ci]
    censored = [r for r in results if not r.has_estimate]

    estimated.sort(key=lambda r: r.half_life, reverse=True)
    censored.sort(key=lambda r: -r.intraday_range_bps)

    ordered = estimated + censored
    positions = np.arange(len(ordered))[::-1]

    for pos, r in zip(positions, ordered):
        colour = GROUP_COLOR[r.asset_group]
        if r.has_estimate:
            lo, hi = r.bootstrap_ci.lower, r.bootstrap_ci.upper
            ax.plot([lo, hi], [pos, pos], color=colour, lw=2.2, solid_capstyle="round",
                    zorder=2)
            # 2px surface ring keeps the marker legible where it meets the bar.
            ax.plot([r.half_life], [pos], "o", color=colour, ms=9,
                    markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3)
            ax.text(hi + 0.06, pos, f"{r.half_life:.2f}", va="center", fontsize=9,
                    color=INK_SECONDARY)
        else:
            bound = _censor_bound(r)
            ax.annotate("", xy=(0.02, pos), xytext=(bound, pos),
                        arrowprops=dict(arrowstyle="->", color=colour, lw=1.8,
                                        shrinkA=0, shrinkB=0), zorder=2)
            ax.plot([bound], [pos], "|", color=colour, ms=13, mew=2.2, zorder=3)
            ax.text(bound + 0.06, pos, f"< {bound:.2f}", va="center", fontsize=9,
                    color=INK_MUTED, style="italic")

    ax.set_yticks(positions)
    ax.set_yticklabels([r.ticker for r in ordered], fontsize=10)

    # One sampling interval: the line below which reversion is partly
    # unobservable, and the boundary the resolution labels use.
    ax.axvline(1.0, color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ax.text(1.0, len(ordered) - 0.35, " one trading day", fontsize=8.5,
            color=INK_MUTED, va="top")

    ax.set_xlabel("half-life of premium reversion (trading days)")
    ax.set_title(title or "Reversion speed separates by basket type, not within it")
    ax.set_xlim(0, max(2.2, max((r.bootstrap_ci.upper for r in estimated), default=2.0) + 0.45))
    ax.set_ylim(-0.8, len(ordered) - 0.2)
    ax.grid(axis="y", visible=False)

    handles = [
        plt.Line2D([], [], color=GROUP_COLOR[g], lw=2.4, marker="o", ms=8,
                   markeredgecolor="#fcfcfb", markeredgewidth=1.6, label=GROUP_LABEL[g])
        for g in ("equity", "credit")
    ]
    handles.append(
        plt.Line2D([], [], color=INK_MUTED, lw=1.6, marker="|", ms=10,
                   label="censored: upper bound only")
    )
    ax.legend(handles=handles, loc="lower right", fontsize=8.5)
    return ax


def _censor_bound(result) -> float:
    """Upper bound parsed from the display string."""
    try:
        return float(result.half_life_display.split("<")[1].split("d")[0].strip())
    except (IndexError, ValueError):
        return 1.0
