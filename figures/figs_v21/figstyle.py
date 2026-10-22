# -*- coding: utf-8 -*-
"""Shared design module for the v2.1 figures (matplotlib).

Design contract (distilled from the reference-figure review, see 참고Figure_수집_v21/*/_review.md):
  * every figure is built at its FINAL printed width, so point sizes are true sizes
  * minimum text size 8.5 pt (tick labels), 9.5-10 pt for axis labels and annotations
  * Okabe-Ito palette, identity never by colour alone (marker/line style/direct labels)
  * direct labelling in preference to legends; legends are placed inside free axes space
  * no figure title (the caption lives in the manuscript), panel letters (a) (b) in bold
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

# ---- sizes (inches) ---------------------------------------------------------
W_FULL = 6.3        # full text width in the Word manuscript
W_HALF = 3.1        # half width (two figures side by side are not used; kept for reference)
DPI = 600

# ---- palette (Okabe-Ito, grayscale-safe pairs chosen by luminance) ---------
BLUE = "#0072B2"     # positive voting mass / main method
ORANGE = "#E69F00"   # negative voting mass
VERM = "#D55E00"     # K-persistent / unreached
GREEN = "#009E73"    # stable correct / early stopping
SKY = "#56B4E9"
PURPLE = "#CC79A7"
GRAY = "#6B6B6B"
LGRAY = "#BDBDBD"
GRID = "#E6E6E6"
INK = "#1A1A1A"
MUTED = "#4D4D4D"
BLUE_D, ORANGE_D, VERM_D, GREEN_D = "#004C78", "#8A5E00", "#9B3F00", "#00654C"   # text-on-white variants

# ---- type sizes (pt, at final size) -----------------------------------------
FS_TICK = 9.0
FS_LABEL = 10.0
FS_ANNOT = 9.2
FS_PANEL = 12.0
FS_SMALL = 8.6      # absolute floor for any text


def _pick_font():
    names = {f.name for f in fm.fontManager.ttflist}
    for cand in ("Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"):
        if cand in names:
            return cand
    return "sans-serif"


FONT = _pick_font()


def apply():
    plt.rcParams.update({
        "font.family": FONT, "font.size": FS_ANNOT,
        "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
        "mathtext.default": "it",
        "axes.labelsize": FS_LABEL, "axes.labelcolor": INK, "axes.edgecolor": "#444444", "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": FS_ANNOT, "lines.solid_capstyle": "round",
        "savefig.dpi": DPI, "figure.dpi": 120, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


apply()


def new_fig(h, w=W_FULL, **kw):
    return plt.figure(figsize=(w, h), **kw)


def panel_letter(ax, s, dx=-0.0, dy=1.0, **kw):
    """Bold panel letter placed above-left of the axes, in axes-fraction coordinates."""
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=FS_PANEL, fontweight="bold", va="bottom", ha="left", color=INK, **kw)


def hgrid(ax, axis="y"):
    ax.grid(True, axis=axis, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name, check=True, pad=0.04):
    """Save PNG (600 dpi) + vector PDF into out/, running the layout check first."""
    from layout_check import check_layout
    issues = check_layout(fig) if check else []
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", pad_inches=pad, facecolor="white")
    print(f"saved {name}: {len(issues)} layout issue(s)")
    for i in issues:
        print("   !!", i)
    plt.close(fig)
    return issues


def stripe_lum(color):
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(color)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
