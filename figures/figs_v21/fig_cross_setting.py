# -*- coding: utf-8 -*-
"""New figure (Secs. 6.3-6.4): (a) gold-reachable disagreement vs net voting gain; (b) positive vs negative voting mass per setting.
Data: figure_data_v21.json (binary-majority gains; modal-rule gains where they differ). Run: python -X utf8 fig_cross_setting.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

D = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))
L, J, O = D["llmbar"], D["judgebench"], D["open"]
S = [  # label, R_gold, gain, ci, pos, neg, N, family, modal gain
    ("GPT-4o, LLMBar", L["R_gold"], L["gain"], (-0.003, 0.022), L["pos"], L["neg"], 295, "A", None),   # CI as in Table 12 / 13
    ("Gemma 3 12B", O["Gemma 3 12B"]["R_gold"], O["Gemma 3 12B"]["gain"], O["Gemma 3 12B"]["ci"], O["Gemma 3 12B"]["pos"], O["Gemma 3 12B"]["neg"], 100, "B", None),
    ("Qwen2.5 7B", O["Qwen2.5 7B"]["R_gold"], O["Qwen2.5 7B"]["gain"], O["Qwen2.5 7B"]["ci"], O["Qwen2.5 7B"]["pos"], O["Qwen2.5 7B"]["neg"], 100, "B", O["Qwen2.5 7B"]["modal"]["gain"]),
    ("GPT-4o, JudgeBench", J["R_gold"], J["gain"], J["ci"], J["pos"], J["neg"], 120, "A", J["modal"]["gain"]),
    ("Phi-4 14B", O["Phi-4 14B"]["R_gold"], O["Phi-4 14B"]["gain"], O["Phi-4 14B"]["ci"], O["Phi-4 14B"]["pos"], O["Phi-4 14B"]["neg"], 100, "B", None),
    ("Llama 3.1 8B", O["Llama 3.1 8B"]["R_gold"], O["Llama 3.1 8B"]["gain"], O["Llama 3.1 8B"]["ci"], O["Llama 3.1 8B"]["pos"], O["Llama 3.1 8B"]["neg"], 100, "B", None),
]
S.sort(key=lambda r: r[1])
assert abs(S[0][2] - 0.0095) < 6e-4 and all(abs((r[4] - r[5]) - r[2]) < 1e-9 for r in S)
FAM = {"A": (fs.BLUE, "o"), "B": (fs.GRAY, "s")}

fig = fs.new_fig(3.0)
gs = fig.add_gridspec(1, 3, width_ratios=[0.78, 1.32, 1.1], left=0.235, right=0.985, top=0.855, bottom=0.185, wspace=0.12)
ax, bx, cx = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2], sharey=None)
n_ = len(S); y = np.arange(n_)[::-1].astype(float)

def style(a_, title):
    a_.set_ylim(-0.6, n_ - 0.4); a_.tick_params(axis="y", length=0); a_.spines["left"].set_visible(False); fs.hgrid(a_, "x")
    a_.set_title(title, fontsize=fs.FS_ANNOT, color=fs.INK, fontweight="bold", loc="center", pad=7)

# ---------------- (a) opportunity ----------------
for yy, r in zip(y, S):
    ax.barh(yy, r[1], height=0.58, color="#8FB4D3" if r[7] == "A" else "#B8B8B8", edgecolor="white", linewidth=0.6, zorder=3)
    ax.text(r[1] + 0.02, yy, f"{r[1]:.2f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.INK)
ax.set_xlim(0, 0.88); ax.set_xticks([0, 0.3, 0.6]); ax.set_xticklabels(["0", "0.3", "0.6"])
ax.set_yticks(y); ax.set_yticklabels([f"{r[0]}"+chr(10)+f"$N$={r[6]}" for r in S], fontsize=fs.FS_SMALL + 0.1, linespacing=1.05)
style(ax, "Disagreement $R_{gold}$"); ax.set_xlabel("")

# ---------------- (b) diverging mass ----------------
for yy, r in zip(y, S):
    lab_, rg, g, ci, pos, neg, n, fam, mg = r
    bx.barh(yy, pos * 100, height=0.58, color=fs.BLUE, edgecolor="white", linewidth=0.6, zorder=3)
    bx.barh(yy, -neg * 100, height=0.58, color=fs.ORANGE, edgecolor="white", linewidth=0.6, zorder=3)
    bx.text(pos * 100 + 0.4, yy, f"{pos * 100:.1f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.BLUE_D)
    bx.text(-neg * 100 - 0.4, yy, f"{neg * 100:.1f}", ha="right", va="center", fontsize=fs.FS_ANNOT - 0.6, color=fs.ORANGE_D)
bx.axvline(0, color=fs.INK, lw=0.9, zorder=4)
bx.set_xlim(-13.5, 13.5); bx.set_xticks([-10, -5, 0, 5, 10]); bx.set_xticklabels(["10", "5", "0", "5", "10"]); bx.set_yticks([])
style(bx, "Voting mass (points)")
bx.text(-13.2, -0.55, "negative", ha="left", va="bottom", fontsize=fs.FS_SMALL + 0.2, color=fs.ORANGE_D, style="italic").set_gid("allow-overlap") if False else None

# ---------------- (c) net gain with CI ----------------
cx.axvline(0, color=fs.INK, lw=0.9, zorder=2)
for yy, r in zip(y, S):
    lab_, rg, g, ci, pos, neg, n, fam, mg = r
    cx.plot([ci[0], ci[1]], [yy, yy], color=fs.INK, lw=1.4, zorder=3)
    for e in ci: cx.plot([e, e], [yy - 0.14, yy + 0.14], color=fs.INK, lw=1.3, zorder=3)
    cx.plot([g], [yy], ls="none", marker="D", ms=6.2, mfc=fs.INK, mec="white", mew=0.8, zorder=5)
    if mg is not None:
        cx.plot([mg], [yy - 0.0], ls="none", marker="^", ms=7.2, mfc="white", mec=fs.INK, mew=1.3, zorder=6)
cx.set_xlim(-0.064, 0.078); cx.set_xticks([-0.05, 0, 0.05]); cx.set_xticklabels(["−0.05", "0", "+0.05"]); cx.set_yticks([])
style(cx, "Net gain [95% CI]")
from matplotlib.lines import Line2D
fig.legend([Line2D([0], [0], ls="none", marker="D", ms=6, mfc=fs.INK, mec="white"), Line2D([0], [0], ls="none", marker="^", ms=7, mfc="white", mec=fs.INK, mew=1.3)],
           ["binary majority", "deployed modal rule"], loc="lower right", bbox_to_anchor=(0.985, -0.012), ncol=2, fontsize=fs.FS_ANNOT - 0.6, handletextpad=0.2, columnspacing=1.0, frameon=False)
fs.save(fig, "Figure_cross_setting_v6")
