# -*- coding: utf-8 -*-
"""Figure 4 (v2.1): accuracy vs committee size K (independent executions), two backbones, shared y-span.
Data: figure_data_v21.json -> ksweep (ksweep_gpt-4o*.json; Wilson 95% CI, N=295). Run: python -X utf8 fig4_ksweep.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

D = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))["ksweep"]
SPAN = 0.175
fig = fs.new_fig(2.75)
gs = fig.add_gridspec(1, 2, left=0.10, right=0.985, top=0.86, bottom=0.2, wspace=0.22)
for j, (name, lo_y, col, mk) in enumerate((("GPT-4o", 0.722, fs.BLUE, "o"), ("gpt-4o-mini", 0.562, fs.GRAY, "s"))):
    ax = fig.add_subplot(gs[j]); rows = D[name]
    K = np.array([r["K"] for r in rows]); acc = np.array([r["acc"] for r in rows]); lo = np.array([r["lo"] for r in rows]); hi = np.array([r["hi"] for r in rows])
    assert list(K) == [1, 3, 5, 7, 9]
    a1, l1, h1 = acc[0], lo[0], hi[0]
    ax.axhspan(l1, h1, color=col, alpha=0.13, lw=0, zorder=1)
    ax.axhline(a1, color=col, lw=1.0, ls=(0, (3, 3)), zorder=2)
    ax.errorbar(K, acc, yerr=[acc - lo, hi - acc], fmt="none", ecolor=col, elinewidth=1.4, capsize=3.2, capthick=1.3, zorder=3)
    ax.plot(K, acc, color=col, lw=1.2, alpha=0.55, zorder=3)
    ax.plot(K, acc, ls="none", marker=mk, ms=7, mfc=col, mec="white", mew=0.9, zorder=4)
    for k, a in zip(K, acc):
        ax.text(k, lo_y + 0.014, f"{a:.3f}", ha="center", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.INK, fontweight="bold")
    ax.text(0.2, lo_y + 0.014, "", fontsize=1)
    ax.set_xlim(0.2, 12.1); ax.set_ylim(lo_y, lo_y + SPAN)
    ax.set_xticks(K); ax.set_yticks(np.round(np.arange(lo_y + 0.005, lo_y + SPAN, 0.04), 3) if False else np.arange(np.ceil((lo_y + 0.03) * 25) / 25, lo_y + SPAN, 0.04))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    fs.hgrid(ax, "y"); ax.set_xlabel("Committee size $K$ (calls per decision)")
    ax.set_title(name, fontsize=fs.FS_LABEL, color=fs.INK, fontweight="bold", loc="left", pad=6)
    ax.text(10.35, a1, "$K$=1" + chr(10) + "level", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.BLUE_D if j == 0 else fs.MUTED, style="italic", linespacing=1.1)
    if j == 0: ax.set_ylabel("Accuracy ($N$=295)")
fs.save(fig, "Figure_4_ksweep_v6")
