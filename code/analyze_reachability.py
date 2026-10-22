# -*- coding: utf-8 -*-
"""5th-review item 2: formalize the error-reachability decomposition.

For instance i with c_i of K recorded votes correct:
    A_i^(1)  = c_i / K                       (expected single-sample accuracy)
    A_i^(MV) = I(c_i > K/2)                  (majority-vote accuracy, odd K)
    Delta_i  = A_i^(MV) - A_i^(1)            (per-instance voting effect)
    Delta_MV = mean_i Delta_i                (aggregate voting effect)

Proposition 1 (Reachability of Majority Voting): Delta_i = 0 whenever c_i = 0
or c_i = K; majority voting can change expected accuracy only on instances with
0 < c_i < K. Verified by enumeration over all recorded instances.

Reported masses (all on the held-out test split, N=295):
    R_dis         = mean_i I(0 < c_i < K)        (sample-reachable disagreement mass)
    R_correctable = (1/N) sum_i max(Delta_i, 0)  (expected corrected mass)
    R_harmful     = (1/N) sum_i max(-Delta_i, 0) (expected harmed mass)
    R_persistent  = mean_i I(c_i = 0)            (persistent-error mass)

Also emits a single canonical bootstrap CI table (one implementation, one seed,
fresh default_rng(0) per statistic) so that identical quantities carry identical
intervals in every manuscript table (5th review, item 3-4).

Zero additional API calls. Output: reachability_analysis.md
"""
from __future__ import annotations
import csv, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")
OUT = os.path.join(HERE, "reachability_analysis.md")
B = 5000
K = 5

rows = list(csv.DictReader(open(CSV, encoding="utf-8")))
sc = {r["instance_id"]: r for r in rows if r["strategy"] == "SC-CoT"}
cot = {r["instance_id"]: r for r in rows if r["strategy"] == "Single-CoT"}
ids = sorted(i for i, r in sc.items() if r["split"] == "test")
N = len(ids)


def boot_ci(vals):
    """Canonical CI: fresh rng(0) per statistic -> identical values everywhere."""
    vals = np.asarray(vals, float)
    rng = np.random.default_rng(0)
    boots = rng.choice(vals, (B, len(vals))).mean(axis=1)
    return vals.mean(), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


c = {}          # c_i: number of correct votes
unan_wrong = set()
split_wrong = set()
for i in ids:
    vs, g = sc[i]["votes"].split("|"), sc[i]["gold_label"]
    c[i] = sum(v == g for v in vs)
    if c[i] == 0:
        (unan_wrong if len(set(vs)) == 1 else split_wrong).add(i)

A1 = np.array([c[i] / K for i in ids])
AMV = np.array([float(c[i] > K / 2) for i in ids])
delta = AMV - A1

# ---- Proposition 1: verified by enumeration ----
assert all(abs(delta[j]) < 1e-12 for j, i in enumerate(ids) if c[i] in (0, K)), \
    "Proposition 1 violated: Delta_i != 0 on a c_i in {0, K} instance"
assert abs(delta.mean() - (AMV.mean() - A1.mean())) < 1e-12

# ---- masses ----
reach = np.array([0 < c[i] < K for i in ids])
R_dis = reach.mean()
R_corr = delta.clip(min=0).sum() / N
R_harm = (-delta).clip(min=0).sum() / N
R_pers = np.mean([c[i] == 0 for i in ids])
n_reach = int(reach.sum())

# upper bound: even if voting resolved every reachable instance correctly
ub = np.array([1.0 if 0 < c[i] < K else AMV[j] for j, i in enumerate(ids)]).mean()

# ---- Delta_i by c_i (K=5 exact values) ----
by_c = Counter(c.values())
DELTA_BY_C = {5: 0.0, 4: +0.2, 3: +0.4, 2: -0.4, 1: -0.2, 0: 0.0}

# ---- canonical CIs for the quantities quoted in more than one table ----
k5_acc, k5_lo, k5_hi = boot_ci(AMV)
k1_acc, k1_lo, k1_hi = boot_ci(A1)
d_m, d_lo, d_hi = boot_ci(delta)
cot_t0 = np.array([float(int(cot[i]["correct"])) for i in ids])
cot_acc, cot_lo, cot_hi = boot_ci(cot_t0)

L = [f"# Error-reachability decomposition (canonical K=5 logs, N_test={N})", "",
     "> analyze_reachability.py — formal framework quantities; zero additional API calls.",
     "> Proposition 1 (voting affects only instances with 0 < c_i < K) verified by",
     "> enumeration over all test instances.", "",
     "## Per-instance voting effect Delta_i as a function of c_i (K=5)", "",
     "| c_i (correct votes of 5) | Vote outcome | Delta_i | n instances |", "|---|---|---|---|"]
for ci_val in range(5, -1, -1):
    outc = {5: "always correct", 4: "majority correct", 3: "narrow majority correct",
            2: "minority correct", 1: "one correct draw", 0: "never correct"}[ci_val]
    L.append(f"| {ci_val} | {outc} | {DELTA_BY_C[ci_val]:+.1f} | {by_c.get(ci_val, 0)} |")
L += ["",
      "## Reachability masses (test split)", "",
      "| Mass | Definition | Value |", "|---|---|---|",
      f"| R_dis (sample-reachable disagreement) | share with 0 < c_i < K | "
      f"{R_dis:.3f} ({n_reach}/{N}) |",
      f"| R_correctable (expected corrected mass) | (1/N) sum max(Delta_i, 0) | "
      f"{R_corr:.4f} ({delta.clip(min=0).sum():+.2f} cases) |",
      f"| R_harmful (expected harmed mass) | (1/N) sum max(-Delta_i, 0) | "
      f"{R_harm:.4f} ({(-delta).clip(min=0).sum():.2f} cases) |",
      f"| R_persistent (persistent-error mass) | share with c_i = 0 | "
      f"{R_pers:.3f} ({int(R_pers*N)}/{N}: {len(unan_wrong)} unanimous-wrong + "
      f"{len(split_wrong)} split-among-wrong) |", "",
      f"- Net voting effect Delta_MV = {d_m:+.4f} [95% CI {d_lo:+.4f}, {d_hi:+.4f}] "
      f"(= {delta.sum():+.2f} of {N} cases)",
      f"- Empirical upper bound: if voting resolved EVERY reachable instance correctly, "
      f"accuracy would be {ub:.3f} (vs. K=5 actual {k5_acc:.3f}) — the reachable "
      f"disagreement mass caps what same-judge voting can change.",
      f"- Note: R_dis ({n_reach}) differs from the split-vote count (37) because "
      f"{len(split_wrong)} split-vote instances have c_i = 0 (disagreement among wrong "
      f"labels only) — they show disagreement but are not reachable toward the gold label.",
      "",
      "## Canonical bootstrap CIs (single implementation, fresh rng(0) per statistic)", "",
      "Use these values verbatim wherever the quantity appears (resolves the",
      "Table 1 / Table 3 interval mismatch — one bootstrap, one seed).", "",
      "| Quantity | Value [95% CI] |", "|---|---|",
      f"| SC-CoT = K=5 majority accuracy | {k5_acc:.3f} [{k5_lo:.3f}, {k5_hi:.3f}] |",
      f"| K=1 expected accuracy (T=0.7) | {k1_acc:.3f} [{k1_lo:.3f}, {k1_hi:.3f}] |",
      f"| Single-CoT (T=0) accuracy | {cot_acc:.3f} [{cot_lo:.3f}, {cot_hi:.3f}] |",
      f"| Delta_MV (K=5 - K=1, paired) | {d_m:+.4f} [{d_lo:+.4f}, {d_hi:+.4f}] |", ""]

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
print(f"R_dis={R_dis:.3f} R_corr={R_corr:.4f} R_harm={R_harm:.4f} R_pers={R_pers:.3f} "
      f"ub={ub:.3f} DeltaMV={d_m:+.4f}")
print("c_i histogram:", dict(sorted(by_c.items(), reverse=True)))
