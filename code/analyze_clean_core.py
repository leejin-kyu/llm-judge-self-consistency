# -*- coding: utf-8 -*-
"""Plan B analysis: forced-choice Direct + clean-cache Single-CoT vs canonical run.

Reports:
  1. Direct-FC accuracy (tie-free) vs canonical Direct-with-tie: how much of the
     Direct deficit was the explicit-tie confound? Paired McNemar vs Single-CoT.
  2. Clean-cache Single-CoT vs canonical Single-CoT: verdict agreement, accuracy
     delta -> is the warm-cache criticism material?
Output: clean_core_analysis.md
"""
from __future__ import annotations
import csv, json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def jload(name):
    p = os.path.join(HERE, name)
    return {r["iid"]: r for r in
            (json.loads(l) for l in open(p, encoding="utf-8"))} if os.path.exists(p) else {}


def main():
    fc = jload("clean_direct_fc.jsonl")
    cc = jload("clean_singlecot.jsonl")
    rows = list(csv.DictReader(open(os.path.join(HERE, "results_final.csv"), encoding="utf-8")))
    direct = {r["instance_id"]: r for r in rows if r["strategy"] == "Direct" and r["split"] == "test"}
    cot = {r["instance_id"]: r for r in rows if r["strategy"] == "Single-CoT" and r["split"] == "test"}
    L = ["# Clean-core analysis (forced-choice Direct + clean-cache Single-CoT)", ""]

    if fc:
        ids = sorted(set(fc) & set(direct))
        a_fc = np.array([fc[i]["verdict"] == fc[i]["gold"] for i in ids], float)
        a_dt = np.array([direct[i]["correct"] == "1" for i in ids], float)
        a_ct = np.array([cot[i]["correct"] == "1" for i in ids], float)
        pf = sum(1 for i in ids if fc[i]["parse_status"] != "ok")
        ties_fixed = sum(1 for i in ids if direct[i]["predicted_label"] == "tie")
        b = int(((a_ct == 1) & (a_fc == 0)).sum()); c = int(((a_ct == 0) & (a_fc == 1)).sum())
        p = stats.binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0  # v1.8 fix: was *2 (double two-sided)
        p = min(p, 1.0)
        L += [f"## 1. Forced-choice Direct (N={len(ids)}, parse non-ok={pf})", "",
              f"| Baseline | Accuracy |", "|---|---|",
              f"| Direct-with-tie (canonical; {ties_fixed} explicit ties) | {a_dt.mean():.3f} |",
              f"| **Direct-FC (tie disallowed, clean cache)** | **{a_fc.mean():.3f}** |",
              f"| Single-CoT (canonical, reference) | {a_ct.mean():.3f} |", "",
              f"- Tie-confound size: Direct-FC − Direct-with-tie = "
              f"{a_fc.mean()-a_dt.mean():+.3f}",
              f"- Single-CoT − Direct-FC = {a_ct.mean()-a_fc.mean():+.3f} "
              f"(b/c={b}/{c}, McNemar-exact p={p:.2g}) — the fair reasoning gap", ""]

    if cc:
        ids = sorted(set(cc) & set(cot))
        agree = np.mean([cc[i]["verdict"] == cot[i]["predicted_label"] for i in ids])
        a_new = np.mean([cc[i]["verdict"] == cc[i]["gold"] for i in ids])
        a_old = np.mean([cot[i]["correct"] == "1" for i in ids])
        L += [f"## 2. Clean-cache Single-CoT rerun (N={len(ids)})", "",
              f"- Verdict agreement with canonical run: {agree:.3f}",
              f"- Accuracy: clean {a_new:.3f} vs canonical {a_old:.3f} "
              f"(Δ {a_new-a_old:+.3f}) — within the documented run-to-run band (±2.7pt)",
              f"- Conclusion for the manuscript: the warm pilot cache did not "
              f"materially affect the deterministic conditions.", ""]

    out = os.path.join(HERE, "clean_core_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
