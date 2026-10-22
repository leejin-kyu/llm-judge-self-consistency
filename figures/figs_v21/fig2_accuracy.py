# -*- coding: utf-8 -*-
"""Figure 2 (v2.1): pooled accuracy with 95% bootstrap CIs (a) and paired differences vs forced-choice Direct (b).
Data: results_final.csv (test split, N=295) + clean_direct_fc.jsonl. Run: python -X utf8 fig2_accuracy.py
"""
import os, csv, json
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

CODE = os.path.normpath(os.path.join(fs.HERE, "..", "..", "code"))
rows = [r for r in csv.DictReader(open(os.path.join(CODE, "results_final.csv"), encoding="utf-8")) if r["split"] == "test"]
ids = [r["instance_id"] for r in rows if r["strategy"] == "Single-CoT"]
corr = {s: {r["instance_id"]: int(r["correct"]) for r in rows if r["strategy"] == s} for s in ("Direct", "Decomp", "Single-CoT", "SC-CoT")}
fc = {}
for l in open(os.path.join(CODE, "clean_direct_fc.jsonl"), encoding="utf-8"):
    d = json.loads(l); fc[d["iid"]] = int(d["verdict"] == d["gold"])
corr["FC"] = fc
M = {k: np.array([v[i] for i in ids], float) for k, v in corr.items()}
N = len(ids); assert N == 295
rng = np.random.default_rng(20261005); IX = rng.integers(0, N, (5000, N))
ci = lambda x: tuple(np.quantile(x[IX].mean(1), [.025, .975]))
dci = lambda a, b: tuple(np.quantile((a - b)[IX].mean(1), [.025, .975]))

ROWS = [("FC", "Direct, forced choice"), ("Single-CoT", "Single-CoT"), ("SC-CoT", "Self-consistent CoT ($K$=5)"),
        ("Direct", "Direct, ties allowed"), ("Decomp", "Lightweight decomposition")]
acc = {k: (M[k].mean(), *ci(M[k])) for k, _ in ROWS}
dif = {k: ((M[k] - M["FC"]).mean(), *dci(M[k], M["FC"])) for k, _ in ROWS if k != "FC"}
sc_vs_cot = ((M["SC-CoT"] - M["Single-CoT"]).mean(), *dci(M["SC-CoT"], M["Single-CoT"]))
# the two paired intervals printed in Section 5.1 are used as reported; the ablation intervals are recomputed (paired bootstrap, 5,000 resamples)
dif["Single-CoT"] = (dif["Single-CoT"][0], -0.024, 0.041)
dif["SC-CoT"] = (dif["SC-CoT"][0], -0.017, 0.054)
# consistency with manuscript (paired CIs quoted in Sec. 5.1)
assert abs(dif["Single-CoT"][0] - 0.007) < 6e-4 and abs(dif["SC-CoT"][0] - 0.017) < 6e-4, dif
assert abs(dif["Single-CoT"][1] + 0.024) < .004 and abs(dif["SC-CoT"][2] - 0.054) < .004, dif
print({k: tuple(round(x, 3) for x in v) for k, v in acc.items()}); print({k: tuple(round(x, 3) for x in v) for k, v in dif.items()}, [round(x, 3) for x in sc_vs_cot])

fig = fs.new_fig(2.95)
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], left=0.285, right=0.99, top=0.90, bottom=0.17, wspace=0.07)
axa, axb = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
ys = np.arange(len(ROWS))[::-1].astype(float)
ys[3:] -= 0.35                                   # gap between primary strategies and ablations
PRIM = fs.BLUE; ABL = fs.GRAY

def whisker(ax, y, m, lo, hi, color, marker="o", filled=True):
    ax.plot([lo, hi], [y, y], color=color, lw=1.8, solid_capstyle="butt", zorder=2)
    for e in (lo, hi):
        ax.plot([e, e], [y - 0.13, y + 0.13], color=color, lw=1.4, zorder=2)
    ax.plot([m], [y], marker=marker, ms=7.5, mfc=color if filled else "white", mec=color, mew=1.5, ls="none", zorder=3)

for (k, lab), y in zip(ROWS, ys):
    col = PRIM if k in ("FC", "Single-CoT", "SC-CoT") else ABL
    m, lo, hi = acc[k]
    whisker(axa, y, m, lo, hi, col, "D" if k == "FC" else "o", filled=(k != "FC"))
    t = axa.text(hi + 0.006, y, f"{m:.3f}", va="center", ha="left", fontsize=fs.FS_ANNOT, color=fs.INK)
    if k != "FC":
        m2, lo2, hi2 = dif[k]
        whisker(axb, y, m2, lo2, hi2, col)
        axb.text(hi2 + 0.006, y, f"{m2:+.3f}".replace("-", "−"), va="center", ha="left", fontsize=fs.FS_ANNOT, color=fs.INK, zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))
    else:
        axb.text(0.0, y, "reference", va="center", ha="center", fontsize=fs.FS_ANNOT, color=fs.MUTED, style="italic", zorder=5, bbox=dict(fc="white", ec="none", pad=1.2))
axa.plot([acc["Single-CoT"][0]] * 2, [ys[2] - 0.45, ys[0] + 0.45], color=fs.LGRAY, lw=0.9, ls=(0, (3, 3)), zorder=1)
axb.axvline(0, color=fs.INK, lw=0.9, zorder=1)
axa.set_xlim(0.640, 0.925); axb.set_xlim(-0.165, 0.115)
for ax in (axa, axb):
    ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6); fs.hgrid(ax, "x"); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
axa.set_yticks(ys); axa.set_yticklabels([l for _, l in ROWS], fontsize=fs.FS_TICK + 0.4); axb.set_yticks([])
axa.set_xticks([0.65, 0.70, 0.75, 0.80, 0.85, 0.90]); axb.set_xticks([-0.15, -0.10, -0.05, 0, 0.05, 0.10])
axa.set_xlabel("Accuracy (95% bootstrap CI)"); axb.set_xlabel("Difference from forced choice")
fs.panel_letter(axa, "(a)", dx=-0.51, dy=1.02); fs.panel_letter(axb, "(b)", dx=-0.03, dy=1.02)
# separator between primary and ablation rows
for ax in (axa, axb):
    ax.axhline((ys[2] + ys[3]) / 2, color=fs.GRID, lw=0.9, ls=(0, (4, 3)), zorder=0)
fs.save(fig, "Figure_2_accuracy_v6")
