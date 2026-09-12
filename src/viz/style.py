"""Shared plotting style.

Colours come from a validated palette: the categorical slots and the
blue<->red diverging pair both clear colourblind-separation, normal-vision
separation and contrast checks against the light chart surface.

Premium/discount is inherently DIVERGING data -- it has a meaningful zero and
two opposite directions -- so it uses the warm/cool pair with a neutral
midpoint rather than two arbitrary categorical hues.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
GRID = "#e5e4e0"

# Categorical slots, in fixed order. Never cycled, never reordered.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

# Diverging: premium (cool) vs discount (warm), neutral midpoint.
PREMIUM = "#2a78d6"
DISCOUNT = "#e34948"
NEUTRAL = "#f0efec"

# Status, reserved -- never reused as a series colour.
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"


def use_house_style() -> None:
    """Apply the project's matplotlib defaults: recessive axes, thin marks."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 110,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": 10,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "lines.linewidth": 1.6,
            "lines.solid_capstyle": "round",
            "legend.frameon": False,
            "legend.fontsize": 9,
        }
    )


def annotate(ax, x, y, text, *, dx=10, dy=10, color=INK_SECONDARY):
    """Direct label with a hairline leader, used selectively."""
    ax.annotate(
        text,
        xy=(x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        fontsize=8.5,
        color=color,
        arrowprops=dict(arrowstyle="-", color=INK_MUTED, lw=0.7, shrinkA=0, shrinkB=2),
    )
