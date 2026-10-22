# -*- coding: utf-8 -*-
"""New figure (Sec. 6.2): correct votes (0-5) of the 15 instances that were K-persistent in at least one of three independent K=5 pools.
Data: figure_data_v21.json -> persistence_matrix / persistence_summary. Run: python -X utf8 fig_pools_persistence.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figstyle as fs

D = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))
rows = D["persistence_matrix"]; S = D["persistence_summary"]
rows = sorted(rows, key=lambda r: (sum(r["c"]), r["id"]))               # persistent-in-all block first, then ascending total
M = np.array([r["c"] for r in rows]); n = len(rows); assert n == 15 and (M.sum(1) == 0).sum() == S["n_all_three"] == 7
npers = (M == 0).sum(0); assert list(npers) == [13, 9, 11], npers          # equals Table 11
# single-hue sequential ramp for counts 0..5 (light -> dark)
RAMP = ["#F3F6F9", "#CFE2F1", "#9CC6E3", "#5BA3CF", "#1F78B4", "#0B4F7F"]
TXT = lambda v: "white" if v >= 4 else fs.INK

fig = fs.new_fig(3.95, w=5.9)
ax = fig.add_axes([0.2, 0.155, 0.45, 0.69])
ax.set_xlim(0, 3); ax.set_ylim(n, 0); ax.set_aspect("auto")
for i in range(n):
    for j in range(3):
        v = M[i, j]
        ax.add_patch(Rectangle((j + 0.05, i + 0.08), 0.90, 0.84, fc=RAMP[v], ec="none", zorder=2))
        if v == 0:
            ax.add_patch(Rectangle((j + 0.07, i + 0.12), 0.86, 0.76, fc="none", ec=fs.VERM, lw=1.1, zorder=3))
        ax.text(j + 0.5, i + 0.52, str(v), ha="center", va="center", fontsize=fs.FS_ANNOT, color=TXT(v), fontweight="bold", zorder=4).set_gid("allow-overlap")
ax.set_yticks(np.arange(n) + 0.5); ax.set_yticklabels([r["id"].replace("_", " ") for r in rows], fontsize=fs.FS_SMALL + 0.2)
ax.set_xticks([0.5, 1.5, 2.5]); ax.set_xticklabels(["Pool 1\n(canonical)", "Pool 2", "Pool 3"], fontsize=fs.FS_TICK)
ax.xaxis.tick_top(); ax.tick_params(length=0, pad=3)
for sp in ax.spines.values(): sp.set_visible(False)
# block separator after the persistent-in-all-pools rows
k = int((M.sum(1) == 0).sum())
ax.plot([-0.02, 3.02], [k, k], color=fs.INK, lw=0.9, ls=(0, (3, 2)), zorder=5, clip_on=False)
# right marginal: total correct votes of 15
for i in range(n):
    ax.text(3.18, i + 0.52, f"{M[i].sum()}", ha="left", va="center", fontsize=fs.FS_ANNOT, color=fs.INK).set_gid("allow-overlap")
ax.text(3.18, -0.95, "Total of\n15 votes", ha="left", va="center", fontsize=fs.FS_TICK, color=fs.MUTED, linespacing=1.05)
# bottom marginal: persistent instances per pool
ax.text(-0.02, n + 0.62, "K-persistent", ha="right", va="center", fontsize=fs.FS_ANNOT, color=fs.VERM_D, fontweight="bold")
for j in range(3):
    ax.text(j + 0.5, n + 0.62, str(npers[j]), ha="center", va="center", fontsize=fs.FS_ANNOT, color=fs.VERM_D, fontweight="bold")
# group labels at right
ax.text(3.78, k / 2, f"No correct vote{chr(10)}in any pool: {k}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.2, color=fs.VERM_D, linespacing=1.15)
ax.text(3.78, k + (n - k) / 2, f"At least one{chr(10)}correct vote: {n - k}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.2, color=fs.INK, linespacing=1.15)
# swatch legend (one row, below)
lx0 = 0.0; ly = n + 1.65
for v in range(6):
    xx = -0.02 + v * 0.52 + 0.0
    ax.add_patch(Rectangle((xx + 0.35, ly - 0.32), 0.36, 0.64, fc=RAMP[v], ec=(fs.VERM if v == 0 else "#888888"), lw=(1.3 if v == 0 else 0.6), clip_on=False, zorder=3))
    ax.text(xx + 0.53, ly + 0.0, str(v), ha="center", va="center", fontsize=fs.FS_SMALL + 0.2, color=TXT(v), fontweight="bold", clip_on=False, zorder=4)
ax.text(-0.02 + 6 * 0.52 + 0.45, ly, "correct votes of $K$ = 5", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.2, color=fs.MUTED, clip_on=False)
fs.save(fig, "Figure_pools_persistence_v6")
