# -*- coding: utf-8 -*-
"""Figure 5 (v2.1): accuracy vs mean calls per decision (a) + distribution of calls under unanimity-gated early stopping (b).
Data: figure_data_v21.json -> shared_pool, early_stop; forced-choice Direct from clean_direct_fc.jsonl. Run: python -X utf8 fig5_pareto.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

D = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))
SP, ES = D["shared_pool"], D["early_stop"]
CODE = os.path.normpath(os.path.join(fs.HERE, "..", "..", "code"))
fc = [json.loads(l) for l in open(os.path.join(CODE, "clean_direct_fc.jsonl"), encoding="utf-8")]
acc_fc = float(np.mean([d["verdict"] == d["gold"] for d in fc])); assert abs(acc_fc - 0.810) < 5e-4 and len(fc) == 295
sc = [(k, SP[str(k)]["acc"]) for k in (1, 3, 5)]
lo1, hi1 = SP["1"]["ci"]
calls_es, acc_es = ES["mean_calls"], SP["5"]["acc"]          # early stopping reproduces the K=5 verdict by construction
assert abs(calls_es - 3.20) < 0.005 and ES["n3"] + ES["n5"] == 295

fig = fs.new_fig(2.9)
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 0.8], left=0.105, right=0.985, top=0.9, bottom=0.18, wspace=0.28)
ax, bx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

# ---------- (a) ----------
ax.axhspan(lo1, hi1, color=fs.BLUE, alpha=0.10, lw=0, zorder=1)
ax.text(5.5, hi1 - 0.003, "95% interval of one call", ha="right", va="top", fontsize=fs.FS_ANNOT - 0.6, color=fs.BLUE_D, style="italic")
ax.plot([k for k, _ in sc], [a for _, a in sc], color=fs.BLUE, lw=1.6, zorder=3)
ax.plot([k for k, _ in sc], [a for _, a in sc], ls="none", marker="o", ms=7.5, mfc=fs.BLUE, mec="white", mew=0.9, zorder=4)
ax.plot([1], [acc_fc], ls="none", marker="D", ms=7.5, mfc="white", mec=fs.GRAY, mew=1.6, zorder=4)
ax.plot([calls_es], [acc_es], ls="none", marker="*", ms=13, mfc=fs.GREEN, mec="white", mew=0.8, zorder=5)
# same-accuracy link between early stopping and K=5
ax.annotate("", xy=(5.0 - 0.14, acc_es), xytext=(calls_es + 0.2, acc_es), arrowprops=dict(arrowstyle="<->", color=fs.GREEN_D, lw=1.2, shrinkA=0, shrinkB=0), zorder=4)
ax.text(4.42, acc_es + 0.005, f"same verdict,{chr(10)}{100 * (1 - calls_es / 5):.0f}% fewer calls", ha="center", va="bottom", fontsize=fs.FS_ANNOT - 0.5, color=fs.GREEN_D, linespacing=1.1)
# direct labels
ax.text(1.0, acc_fc - 0.006, "Direct,\nforced choice", ha="left", va="top", fontsize=fs.FS_ANNOT - 0.3, color=fs.MUTED, linespacing=1.1).set_gid("allow-overlap")
for k, a in sc:
    dy, va = (0.007, "bottom") if k == 1 else (-0.0075, "top")
    ax.text(k, a + dy, f"$K$={k}", ha="center", va=va, fontsize=fs.FS_ANNOT - 0.3, color=fs.BLUE_D, fontweight="bold")
ax.text(calls_es + 0.12, acc_es + 0.010, f"Early stopping\n{calls_es:.2f} calls", ha="right", va="bottom", fontsize=fs.FS_ANNOT - 0.3, color=fs.GREEN_D, linespacing=1.1, fontweight="bold")
ax.set_xlim(0.4, 5.6); ax.set_ylim(0.762, 0.875)
ax.set_xticks([1, 2, 3, 4, 5]); ax.set_yticks([0.77, 0.79, 0.81, 0.83, 0.85, 0.87]); ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
fs.hgrid(ax, "y"); ax.set_xlabel("Mean inference calls per decision"); ax.set_ylabel("Accuracy ($N$=295)")
fs.panel_letter(ax, "(a)", dx=-0.17, dy=1.03)
# truncated-axis mark
ax.text(0.01, 0.015, "y axis truncated", transform=ax.transAxes, fontsize=fs.FS_ANNOT - 0.7, color=fs.MUTED, style="italic", ha="left", va="bottom")

# ---------- (b) ----------
vals = [ES["n3"], ES["n5"]]; labs = ["3 calls", "5 calls"]
bx.bar([0, 1], vals, width=0.62, color=[fs.GREEN, fs.LGRAY], edgecolor=fs.INK, linewidth=0.7, zorder=3)
for x, v in zip((0, 1), vals):
    bx.text(x, v + 8, f"{v}\n{100 * v / 295:.1f}%", ha="center", va="bottom", fontsize=fs.FS_ANNOT - 0.3, color=fs.INK, linespacing=1.1)
bx.set_xticks([0, 1]); bx.set_xticklabels(labs); bx.set_ylim(0, 340); bx.set_yticks([0, 100, 200, 300]); bx.set_xlim(-0.6, 1.6)
fs.hgrid(bx, "y"); bx.set_ylabel("Instances"); bx.set_xlabel("Calls used")
fs.panel_letter(bx, "(b)", dx=-0.3, dy=1.03)
fs.save(fig, "Figure_5_pareto_v6")
