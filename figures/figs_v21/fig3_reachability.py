# -*- coding: utf-8 -*-
"""Figure 3 (v2.1): (a) distribution of correct votes c_i (K=5, N=295, broken y axis), (b) stacked diverging voting mass.
Reads figure_data_v21.json (build_data_v21.py). Run: python -X utf8 fig3_reachability.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

D = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))["llmbar"]
cnt = D["counts"]; N = sum(cnt); K = 5
assert cnt == [39, 6, 6, 9, 14, 221] and N == 295
contrib = {c: cnt[c] * ((1 - c / K) if c > K / 2 else -c / K) for c in range(1, 5)}      # signed voting mass per c
pos, neg = sum(v for v in contrib.values() if v > 0), -sum(v for v in contrib.values() if v < 0)
assert abs(pos - D["pos_cases"]) < 1e-6 and abs(neg - D["neg_cases"]) < 1e-6
COL = {0: fs.VERM, 1: fs.ORANGE, 2: fs.ORANGE, 3: fs.BLUE, 4: fs.BLUE, 5: fs.LGRAY}
HAT = {1: None, 2: None, 3: None, 4: None}

fig = fs.new_fig(3.15)
outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], left=0.105, right=0.985, top=0.885, bottom=0.225, wspace=0.20)
inner = outer[0].subgridspec(2, 1, height_ratios=[1, 3.3], hspace=0.07)
ax_t, ax_b = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
axr = fig.add_subplot(outer[1])

# ---------------- (a) counts, broken axis ----------------
xs = np.arange(6)
BH_ = 52
ax_b.bar(xs, [min(n, BH_) for n in cnt], width=0.72, color=[COL[c] for c in range(6)], edgecolor=fs.INK, linewidth=0.7, zorder=3)
ax_t.bar([5], [cnt[5]], width=0.72, color=COL[5], edgecolor=fs.INK, linewidth=0.7, zorder=3)
for ax in (ax_t, ax_b):
    ax.set_xlim(-0.6, 5.6)
ax_b.set_ylim(0, BH_); ax_t.set_ylim(190, 236)
ax_b.set_yticks([0, 10, 20, 30, 40]); ax_t.set_yticks([200, 220])
ax_t.spines["bottom"].set_visible(False); ax_t.tick_params(bottom=False, labelbottom=False)
ax_b.set_xticks(xs); ax_b.set_xticklabels(["0"+chr(10)+"unreached", "1", "2", "3", "4", "5"+chr(10)+"stable"])
ax_b.set_xlabel("Correct votes $c_i$ out of $K$ = 5")
ax_b.set_ylabel("Instances", labelpad=2); ax_b.yaxis.set_label_coords(-0.14, 0.62)
for ax in (ax_t, ax_b):
    ax.grid(True, axis="y", color=fs.GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
# axis-break marks on the left spine
fig.canvas.draw()
bb_b, bb_t = ax_b.get_position(), ax_t.get_position()
ym = (bb_b.y1 + bb_t.y0) / 2; dx_ = 0.011; dy_ = 0.018
for off in (-0.012, 0.012):
    fig.add_artist(plt.Line2D([bb_b.x0 - dx_, bb_b.x0 + dx_], [ym + off - dy_ / 2, ym + off + dy_ / 2], transform=fig.transFigure, color=fs.INK, lw=1.0, clip_on=False))
# value labels (count and share), two lines
for c in range(6):
    n = cnt[c]; share = f"{100 * n / N:.1f}%"
    if c == 5:
        t = ax_t.text(c, n + 3, f"{n}\n{share}", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=fs.INK, linespacing=1.15)
    else:
        t = ax_b.text(c, n + 1.6, f"{n}\n{share}", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=fs.INK, linespacing=1.15)
# gold-reachable bracket over c = 1..4
yb = 27.5
ax_b.plot([0.68, 0.68, 4.32, 4.32], [yb - 1.4, yb, yb, yb - 1.4], color=fs.INK, lw=1.2, solid_joinstyle="miter", zorder=4)
ax_b.text(2.5, yb + 1.8, f"Gold-reachable, $0<c_i<K$:\n{sum(cnt[1:5])} instances ({100 * sum(cnt[1:5]) / N:.1f}%)", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=fs.INK, linespacing=1.2)
fs.panel_letter(ax_t, "(a)", dx=-0.17, dy=1.12)

# ---------------- (b) stacked diverging mass ----------------
rows = {"Voting mass": 1.0, "Net": 0.0}
H = 0.52
left = 0.0
for c in (2, 1):                       # negative, outward from zero
    w = -contrib[c]
    axr.barh(1.0, -w, left=left, height=H, color=COL[c], edgecolor=fs.INK, linewidth=0.7, zorder=3)
    t = axr.text(left - w / 2, 1.0, f"$c$={c}", ha="center", va="center", fontsize=fs.FS_SMALL + 0.2, color=fs.INK); t.set_gid("allow-overlap")
    left -= w
left = 0.0
for c in (3, 4):
    w = contrib[c]
    axr.barh(1.0, w, left=left, height=H, color=COL[c], edgecolor=fs.INK, linewidth=0.7, zorder=3)
    t = axr.text(left + w / 2, 1.0, f"$c$={c}", ha="center", va="center", fontsize=fs.FS_SMALL + 0.2, color="white"); t.set_gid("allow-overlap")
    left += w
net = pos - neg
axr.barh(0.0, net, height=H, color=fs.INK, edgecolor=fs.INK, zorder=3, hatch=None)
axr.axvline(0, color=fs.INK, lw=1.0, zorder=4)
axr.text(-neg, 1.0 + H / 2 + 0.07, "\u2212" + f"{neg:.2f}", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=fs.ORANGE_D, fontweight="bold")
axr.text(pos, 1.0 + H / 2 + 0.07, f"+{pos:.2f}", ha="center", va="bottom", fontsize=fs.FS_ANNOT, color=fs.BLUE_D, fontweight="bold")
axr.text(net + 0.25, 0.0, f"+{net:.2f}", ha="left", va="center", fontsize=fs.FS_ANNOT, color=fs.INK, fontweight="bold")
axr.text(-neg - 0.0, 1.0 - H / 2 - 0.09, "Negative", ha="center", va="top", fontsize=fs.FS_ANNOT, color=fs.ORANGE_D)
axr.text(pos, 1.0 - H / 2 - 0.09, "Positive", ha="center", va="top", fontsize=fs.FS_ANNOT, color=fs.BLUE_D)
axr.set_yticks([0.0]); axr.set_yticklabels(["Net"], fontsize=fs.FS_TICK + 0.4)
axr.set_ylim(-0.5, 1.75); axr.set_xlim(-4.4, 7.2)
axr.set_xticks([-4, -2, 0, 2, 4, 6]); axr.set_xticklabels(["\u22124", "\u22122", "0", "2", "4", "6"])
axr.set_xlabel("Voting mass (instance equivalents)")
axr.spines["left"].set_visible(False); axr.tick_params(axis="y", length=0); fs.hgrid(axr, "x")
fs.panel_letter(axr, "(b)", dx=-0.05, dy=1.04)
fs.save(fig, "Figure_3_reachability_v6")
