# -*- coding: utf-8 -*-
"""Figure 7 (v2.4): outcomes after candidate-order swapping and constraint-focused prompting on the 37 unanimous
K-persistent errors, with the plain re-judgment comparator added to panel (b).

Panel (a) is unchanged from v2.1 (figure_data_v21.json -> interventions). Panel (b) adds the same 74 items re-judged
once with the original prompt and order at T=0 on 22 July (clean_singlecot.jsonl), from revision_numbers_v24.json.
Run: python -X utf8 fig7_interventions_v24.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "figs_v21"))
import figstyle as fs
from matplotlib.patches import Rectangle, Polygon

fs.OUT = os.path.join(HERE, "out")
os.makedirs(fs.OUT, exist_ok=True)
I = json.load(open(os.path.join(HERE, "..", "figs_v21", "figure_data_v21.json"), encoding="utf-8"))["interventions"]
V5 = json.load(open(os.path.join(HERE, "..", "..", "code", "revision_numbers_v24.json"), encoding="utf-8"))["settings"]["v5"]
C = I["categories"]
LB = I["length_by_category"]
assert I["n"] == 37 and C == {"order only": 7, "constraint only": 7, "both": 6, "none": 17}
assert V5["sets"]["swap"] == I["order"] == 13 and V5["sets"]["constr"] == I["constraint"] == 13
assert V5["ctrl_flip"]["swap5"] == 4 and V5["ctrl_flip"]["constr"] == 4
RERUN, RERUN_CTRL = V5["sets"]["rerun"], V5["ctrl_flip"]["rerun1"]
GROUPS = [("order only", C["order only"]), ("both", C["both"]), ("constraint only", C["constraint only"]), ("none", C["none"])]

fig = fs.new_fig(4.05)
ax = fig.add_axes([0.02, 0.575, 0.96, 0.37])
bx = fig.add_axes([0.225, 0.115, 0.75, 0.36])
S, G = 1.0, 0.18
x = 0.0
span = {}
for name, n in GROUPS:
    x0 = x
    for k in range(n):
        xx = x + k * (S + G)
        if name == "order only":
            ax.add_patch(Rectangle((xx, 0), S, S, fc=fs.BLUE, ec="white", lw=0.6, zorder=3))
        elif name == "constraint only":
            ax.add_patch(Rectangle((xx, 0), S, S, fc=fs.ORANGE, ec="white", lw=0.6, zorder=3))
        elif name == "both":
            ax.add_patch(Polygon([(xx, 0), (xx + S, 0), (xx, S)], closed=True, fc=fs.BLUE, ec="none", zorder=3))
            ax.add_patch(Polygon([(xx + S, 0), (xx + S, S), (xx, S)], closed=True, fc=fs.ORANGE, ec="none", zorder=3))
            ax.add_patch(Rectangle((xx, 0), S, S, fc="none", ec="white", lw=0.6, zorder=4))
        else:
            ax.add_patch(Rectangle((xx, 0), S, S, fc="#EDEDED", ec="#9A9A9A", lw=0.7, zorder=3))
        if name != "none" and k < LB.get(name, 0):
            ax.plot([xx + S / 2], [S / 2], ls="none", marker="o", ms=4.2, mfc="white", mec=fs.INK, mew=0.8, zorder=6)
    span[name] = (x0, x + (n - 1) * (S + G) + S)
    x += n * (S + G) + 0.55
XMAX = x - 0.55
ax.set_xlim(-0.2, XMAX + 0.2)
ax.set_ylim(-3.3, 5.6)
ax.axis("off")


def bracket(x0, x1, y, text, col):
    ax.plot([x0, x0, x1, x1], [y - 0.35, y, y, y - 0.35], color=col, lw=1.4, solid_joinstyle="miter", zorder=5, clip_on=False)
    ax.text((x0 + x1) / 2, y + 0.3, text, ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=col, fontweight="bold")


bracket(span["order only"][0], span["both"][1], 1.75, f"Correct after order swap: {I['order']}", fs.BLUE_D)
bracket(span["both"][0], span["constraint only"][1], 3.65, f"Correct after constraint prompt: {I['constraint']}", fs.ORANGE_D)
for name, lab in (("order only", "order only"), ("both", "both"), ("constraint only", "constraint only"), ("none", "correct after neither")):
    x0, x1 = span[name]
    n = dict(GROUPS)[name]
    ax.text((x0 + x1) / 2, -0.45, f"{n}", ha="center", va="top", fontsize=fs.FS_ANNOT, color=fs.INK, fontweight="bold")
    ax.text((x0 + x1) / 2, -1.55, lab, ha="center", va="top", fontsize=fs.FS_ANNOT - 0.4, color=fs.MUTED)
fs.panel_letter(ax, "(a)", dx=0.0, dy=0.985)
ux0, ux1 = span["order only"][0], span["constraint only"][1]
ax.plot([ux0, ux1], [-2.45, -2.45], color=fs.INK, lw=1.0, zorder=5)
ax.text((ux0 + ux1) / 2, -2.6, f"Union: {I['union']} of 37", ha="center", va="top", fontsize=fs.FS_ANNOT - 0.2, color=fs.INK, fontweight="bold")
kx = span["none"][0] + 0.2
ax.plot([kx + 0.5], [3.8], ls="none", marker="o", ms=5, mfc="white", mec=fs.INK, mew=0.9, zorder=6)
ax.text(kx + 1.4, 3.8, f"also correct after length rewriting{chr(10)}({I['length']} of {I['length_tested']} tested)", ha="left", va="center",
        fontsize=fs.FS_ANNOT - 0.4, color=fs.MUTED, linespacing=1.1)

rows = [("Order swap", I["order"], V5["ctrl_flip"]["swap5"], fs.BLUE, fs.BLUE_D),
        ("Constraint prompt", I["constraint"], V5["ctrl_flip"]["constr"], fs.ORANGE, fs.ORANGE_D),
        ("Plain re-judgment", RERUN, RERUN_CTRL, "#8C8C8C", fs.MUTED)]
yy = np.array([2.0, 1.0, 0.0])
h = 0.31
for (nm, rec, ctl, col, cold), y0 in zip(rows, yy):
    bx.barh(y0 + 0.18, rec, height=h, color=col, edgecolor="white", lw=0.6, zorder=3)
    bx.barh(y0 - 0.18, ctl, height=h, color="#D0D0D0", edgecolor="#8C8C8C", lw=0.6, zorder=3, hatch="///")
    bx.text(rec + 0.3, y0 + 0.18, f"{rec} of 37 errors correct afterwards", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.INK)
    bx.text(ctl + 0.3, y0 - 0.18, f"{ctl} of 37 matched controls incorrect afterwards", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.MUTED)
bx.set_yticks(yy)
bx.set_yticklabels(["Order swap" + chr(10) + "$K$=5", "Constraint" + chr(10) + "prompt, $T$=0", "Plain re-" + chr(10) + "judgment, $T$=0"],
                   fontsize=fs.FS_ANNOT - 0.2, fontweight="bold", linespacing=1.05)
for tl, c in zip(bx.get_yticklabels(), (fs.BLUE_D, fs.ORANGE_D, fs.MUTED)):
    tl.set_color(c)
bx.set_ylim(-0.55, 2.55)
bx.set_xlim(0, 30)
bx.set_xticks([0, 5, 10, 15])
bx.tick_params(axis="y", length=0)
bx.spines["left"].set_visible(False)
fs.hgrid(bx, "x")
bx.set_xlabel("Instances (of 37)", loc="left")
fs.panel_letter(bx, "(b)", dx=-0.275, dy=1.06)
fs.save(fig, "Fig7_interventions_v24")
