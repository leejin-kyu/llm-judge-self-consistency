# -*- coding: utf-8 -*-
"""Figure B1 (v2.1): risk-coverage. SC vote margin = 3 attainable operating points; decomposition score = curve with bootstrap band.
Data: figure_data_v21.json -> risk_coverage (N=295). AURC computed exactly as in the manuscript (stable sort, all coverage levels).
Run: python -X utf8 figB1_risk_coverage.py
"""
import json, os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs

RC = json.load(open(os.path.join(fs.HERE, "figure_data_v21.json"), encoding="utf-8"))["risk_coverage"]
N = len(RC); assert N == 295
U_SC = np.array([r["sc_u"] for r in RC]); E_SC = np.array([0 if r["sc_ok"] else 1 for r in RC], float)
U_MP = np.array([r["mp_u"] for r in RC]); E_MP = np.array([0 if r["mp_ok"] else 1 for r in RC], float)

def curve(u, e):                      # stable sort on uncertainty only, as in the manuscript
    o = np.argsort(u, kind="stable"); ee = e[o]
    cov = np.arange(1, len(u) + 1) / len(u); risk = np.cumsum(ee) / np.arange(1, len(u) + 1)
    return cov, risk, float(np.trapezoid(risk, cov))
c_sc, r_sc, a_sc = curve(U_SC, E_SC); c_mp, r_mp, a_mp = curve(U_MP, E_MP)
assert abs(a_sc - 0.118) < 6e-4 and abs(a_mp - 0.092) < 6e-4, (a_sc, a_mp)      # dataset-order values (kept for reference)
# tie-averaged AURC (random tie-breaking): the vote margin has only 3 levels, so dataset order is arbitrary
def aurc_rand(u, e, rng, B=5000):
    out = []
    for _ in range(B):
        idx = rng.permutation(len(u)); uu, ee = u[idx], e[idx]
        o = np.argsort(uu, kind="stable"); cs = np.cumsum(ee[o]) / np.arange(1, len(u) + 1)
        out.append(float(np.trapezoid(cs, np.arange(1, len(u) + 1) / len(u))))
    return float(np.mean(out))
_r = np.random.default_rng(1)
a_sc_t, a_mp_t = aurc_rand(U_SC, E_SC, _r), aurc_rand(U_MP, E_MP, _r)
assert abs(a_sc_t - 0.144) < 2e-3 and abs(a_mp_t - 0.100) < 2e-3, (a_sc_t, a_mp_t)
print("AURC", round(a_sc, 4), round(a_mp, 4), "| overall error", E_SC.mean(), E_MP.mean())

# attainable operating points of the discrete vote margin (cut at tie-group boundaries)
lv = np.unique(U_SC); pts = []
for t in lv:
    m = U_SC <= t; pts.append((m.mean(), E_SC[m].mean()))
pts = np.array(pts)
# bootstrap band for the continuous score (1000 resamples), coverage grid from 0.2
grid = np.linspace(0.2, 1.0, 81); rng = np.random.default_rng(7); B = []
for _ in range(5000):
    ix = rng.integers(0, N, N); u, e = U_MP[ix], E_MP[ix]
    o = np.argsort(u, kind="stable"); cs = np.cumsum(e[o]); n = np.ceil(grid * N).astype(int)
    B.append(cs[n - 1] / n)
lo, hi = np.quantile(B, [.025, .975], axis=0)

def oracle(err):
    g = np.linspace(0.2, 1.0, 400); return g, np.clip(g - (1 - err), 0, None) / g

fig = fs.new_fig(3.1)
ax = fig.add_axes([0.105, 0.17, 0.78, 0.77])
m = c_mp >= 0.2
ax.fill_between(grid, lo, hi, color=fs.ORANGE, alpha=0.22, lw=0, zorder=1)
ax.plot(c_mp[m], r_mp[m], color=fs.ORANGE, lw=1.7, zorder=3)
gx, gy = oracle(E_SC.mean()); ax.plot(gx, gy, color=fs.BLUE, lw=0.9, ls=(0, (1.5, 2.2)), alpha=0.9, zorder=2)
gx, gy = oracle(E_MP.mean()); ax.plot(gx, gy, color=fs.ORANGE_D, lw=0.9, ls=(0, (1.5, 2.2)), alpha=0.9, zorder=2)
ax.plot(pts[:, 0], pts[:, 1], color=fs.BLUE, lw=1.7, zorder=3)
ax.plot(pts[:, 0], pts[:, 1], ls="none", marker="o", ms=7.5, mfc=fs.BLUE, mec="white", mew=0.9, zorder=4)
ax.plot([1], [E_MP.mean()], ls="none", marker="s", ms=7, mfc=fs.ORANGE, mec="white", mew=0.9, zorder=4, clip_on=False)
# unselected error levels
for err, col in ((E_SC.mean(), fs.BLUE), (E_MP.mean(), fs.ORANGE_D)):
    ax.plot([0.62, 1.0], [err, err], color=col, lw=0.8, ls=(0, (5, 3)), alpha=0.55, zorder=1)
# legend in the empty upper-left corner (no data there)
from matplotlib.lines import Line2D
H_ = [Line2D([0], [0], color=fs.ORANGE, lw=1.8, marker="s", ms=6, mfc=fs.ORANGE, mec="white"),
      Line2D([0], [0], color=fs.BLUE, lw=1.8, marker="o", ms=6.5, mfc=fs.BLUE, mec="white"),
      Line2D([0], [0], color=fs.GRAY, lw=1.0, ls=(0, (1.5, 2.2))),
      Line2D([0], [0], color=fs.GRAY, lw=1.0, ls=(0, (5, 3)))]
L_ = [f"Decomposition score, AURC = {a_mp_t:.3f}", f"Vote margin ($K$=5), AURC = {a_sc_t:.3f}", "Oracle ranking", "No abstention"]
ax.legend(H_, L_, loc="upper left", bbox_to_anchor=(0.0, 1.0), fontsize=fs.FS_ANNOT - 0.2, handlelength=2.4, labelspacing=0.45, borderaxespad=0.4, frameon=False)
ax.text(1.012, E_SC.mean(), f"{E_SC.mean():.3f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.BLUE_D)
ax.text(1.012, E_MP.mean(), f"{E_MP.mean():.3f}", ha="left", va="center", fontsize=fs.FS_ANNOT - 0.4, color=fs.ORANGE_D)
ax.set_xlim(0.2, 1.0); ax.set_ylim(0, 0.30)
ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0]); ax.set_yticks([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]); ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
fs.hgrid(ax, "y"); ax.set_xlabel("Coverage (fraction of judgments accepted)"); ax.set_ylabel("Selective risk")
fs.save(fig, "Figure_B1_risk_coverage_v6")
