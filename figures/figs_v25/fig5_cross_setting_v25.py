# -*- coding: utf-8 -*-
"""Figure 5 (v2.5): as v2.4, plus a row for gpt-4o-mini on the LLMBar test split (order-randomized campaign of
13 July 2026, first five of nine draws; numbers from analyze_v25_revisions.py, W4).
Run: python -X utf8 fig5_cross_setting_v25.py

v2.4 docstring follows.
Figure 5 (v2.4): voting opportunity and benefit across settings, one aggregation rule per marker.

Changes from v2.1: (1) GPT-4o on the common 100-instance sample is its own row, so open-weight judges are
compared with GPT-4o on identical items; (2) every interval comes from the shared routine in
analyze_v24_revisions.py; (3) the modal rule is shown separately for first-occurrence and random tie-breaking.
Data: ../../code/revision_numbers_v24.json. Run: python -X utf8 fig5_cross_setting_v24.py
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "figs_v21"))
import figstyle as fs
from matplotlib.lines import Line2D

fs.OUT = os.path.join(HERE, "out")
os.makedirs(fs.OUT, exist_ok=True)
S_ = json.load(open(os.path.join(HERE, "..", "..", "code", "revision_numbers_v24.json"), encoding="utf-8"))["settings"]
_m = json.load(open(os.path.join(HERE, "..", "..", "code", "revision_numbers_v25.json"), encoding="utf-8"))["W4"]["gpt-4o-mini"]
S_["mini292"] = {"R_gold": _m["R_gold"], "gain_binary": _m["gain5"], "pos_binary": _m["pos"], "neg_binary": _m["neg"],
                 "n_modal_ties": _m["modal_ties"]}

ROWS = [("llmbar295", "GPT-4o, LLMBar", "full test split, $N$=295", "A"),
        ("gpt4o_c100", "GPT-4o, LLMBar", "common sample, $N$=100", "A"),
        ("gemma", "Gemma 3 12B", "common sample, $N$=100", "B"),
        ("qwen", "Qwen2.5 7B", "common sample, $N$=100", "B"),
        ("judgebench", "GPT-4o, JudgeBench", "$N$=120", "A"),
        ("phi4", "Phi-4 14B", "common sample, $N$=100", "B"),
        ("llama", "Llama 3.1 8B", "common sample, $N$=100", "B"),
        ("mini292", "gpt-4o-mini, LLMBar", "random order, $N$=292", "A")]
S = []
for key, lab, sub, fam in ROWS:
    d = S_[key]
    g = d["gain_binary"]
    mf = d["gain_modal_first"][0] if d["n_modal_ties"] else None
    mr = d["gain_modal_random"][0] if d["n_modal_ties"] else None
    S.append((lab, sub, d["R_gold"], g[0], (g[1], g[2]), d["pos_binary"], d["neg_binary"], fam, mf, mr))
S.sort(key=lambda r: r[2])
assert all(abs((r[5] - r[6]) - r[3]) < 1e-9 for r in S)

fig = fs.new_fig(4.2)
gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.32, 1.1], left=0.262, right=0.985, top=0.895, bottom=0.18, wspace=0.12)
ax, bx, cx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
n_ = len(S)
y = np.arange(n_)[::-1].astype(float)


def style(a_, title):
    a_.set_ylim(-0.6, n_ - 0.4)
    a_.tick_params(axis="y", length=0)
    a_.spines["left"].set_visible(False)
    fs.hgrid(a_, "x")
    a_.set_title(title, fontsize=fs.FS_ANNOT, color=fs.INK, fontweight="bold", loc="center", pad=7)


for yy, r in zip(y, S):
    ax.barh(yy, r[2], height=0.58, color="#8FB4D3" if r[7] == "A" else "#B8B8B8", edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(r[2] + 0.02, yy, f"{r[2]:.2f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.INK)
ax.set_xlim(0, 0.88)
ax.set_xticks([0, 0.3, 0.6])
ax.set_xticklabels(["0", "0.3", "0.6"])
ax.set_yticks(y)
ax.set_yticklabels([f"{r[0]}" + chr(10) + f"{r[1]}" for r in S], fontsize=fs.FS_SMALL + 0.1, linespacing=1.05)
style(ax, "Disagreement $R_{gold}$")

for yy, r in zip(y, S):
    pos, neg = r[5], r[6]
    bx.barh(yy, pos * 100, height=0.58, color=fs.BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    bx.barh(yy, -neg * 100, height=0.58, color=fs.ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
    bx.text(pos * 100 + 0.4, yy, f"{pos * 100:.1f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.BLUE_D)
    bx.text(-neg * 100 - 0.4, yy, f"{neg * 100:.1f}", ha="right", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.ORANGE_D)
bx.axvline(0, color=fs.INK, lw=0.9, zorder=4)
bx.set_xlim(-13.5, 13.5)
bx.set_xticks([-10, -5, 0, 5, 10])
bx.set_xticklabels(["10", "5", "0", "5", "10"])
bx.set_yticks([])
style(bx, "Voting mass (points)")

cx.axvline(0, color=fs.INK, lw=0.9, zorder=2)
for yy, r in zip(y, S):
    g, ci, mf, mr = r[3], r[4], r[8], r[9]
    cx.plot([ci[0], ci[1]], [yy, yy], color=fs.INK, lw=1.4, zorder=3)
    for e in ci:
        cx.plot([e, e], [yy - 0.14, yy + 0.14], color=fs.INK, lw=1.3, zorder=3)
    cx.plot([g], [yy], ls="none", marker="D", ms=6.0, mfc=fs.INK, mec="white", mew=0.8, zorder=5)
    if mf is not None and abs(mf - g) > 1e-9:
        cx.plot([mf], [yy], ls="none", marker="^", ms=7.0, mfc="white", mec=fs.INK, mew=1.3, zorder=6)
    if mr is not None and abs(mr - g) > 1e-9:
        cx.plot([mr], [yy], ls="none", marker="o", ms=5.6, mfc="white", mec=fs.VERM_D, mew=1.3, zorder=7)
cx.set_xlim(-0.066, 0.08)
cx.set_xticks([-0.05, 0, 0.05])
cx.set_xticklabels(["−0.05", "0", "+0.05"])
cx.set_yticks([])
style(cx, "Net gain [95% CI]")
fig.legend([Line2D([0], [0], ls="none", marker="D", ms=6, mfc=fs.INK, mec="white"),
            Line2D([0], [0], ls="none", marker="^", ms=7, mfc="white", mec=fs.INK, mew=1.3),
            Line2D([0], [0], ls="none", marker="o", ms=5.6, mfc="white", mec=fs.VERM_D, mew=1.3)],
           ["binary majority", "modal, first-occurrence ties", "modal, random ties"],
           loc="lower right", bbox_to_anchor=(0.99, -0.005), ncol=3, fontsize=fs.FS_SMALL,
           handletextpad=0.1, columnspacing=0.8, frameon=False)
fs.save(fig, "Fig5_cross_setting_v25")
