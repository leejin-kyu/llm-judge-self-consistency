# -*- coding: utf-8 -*-
"""Plan A analysis: order consistency + extended persistent-error taxonomy.

Inputs: order_swap_stageA.jsonl (295 Single-CoT swapped, T=0),
        order_swap_stageB.jsonl (37 persistent-wrong + 37 controls, K=5 swapped),
        results_final.csv (original-order canonical verdicts).

Reports:
  1. Stage A — order consistency of Single-CoT: agreement rate between the
     original-order verdict and the swapped-order verdict (mapped back), flip
     rate, position-consistency by subset.
  2. Stage B — extended taxonomy of the 37 persistent errors:
       order-correctable persistent  (swapped K=5 majority correct)
       order-robust persistent       (swapped K=5 majority still wrong)
     plus the matched-control comparison (controls should mostly stay correct;
     if controls also flip, order sensitivity is general, not error-specific).
Output: order_swap_analysis.md
"""
from __future__ import annotations
import csv, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")


def jload(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return []
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main(mock=False):
    suf = "_mock" if mock else ""
    A = {r["iid"]: r for r in jload(f"order_swap_stageA{suf}.jsonl")}
    B = jload(f"order_swap_stageB{suf}.jsonl")
    orig = {r["instance_id"]: r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if r["strategy"] == "Single-CoT" and r["split"] == "test"}
    L = [f"# Order-swap analysis{' (MOCK — pipeline validation only)' if mock else ''}", ""]

    if A:
        both = [i for i in A if i in orig]
        agree = [orig[i]["predicted_label"] == A[i]["verdicts_original_frame"][0] for i in both]
        acc_o = [orig[i]["correct"] == "1" for i in both]
        acc_s = [A[i]["verdicts_original_frame"][0] == A[i]["gold"] for i in both]
        L += [f"## Stage A — Single-CoT order consistency (N={len(both)})", "",
              f"- Verdict agreement original vs swapped: {np.mean(agree):.3f} "
              f"(flip rate {1-np.mean(agree):.3f})",
              f"- Accuracy original order {np.mean(acc_o):.3f} vs swapped order "
              f"{np.mean(acc_s):.3f}", ""]
        L += ["| Subset | n | agreement | acc(orig) | acc(swap) |", "|---|---|---|---|---|"]
        for sub in sorted({A[i]["subset"] for i in both}):
            ids = [i for i in both if A[i]["subset"] == sub]
            L.append(f"| {sub} | {len(ids)} | "
                     f"{np.mean([orig[i]['predicted_label']==A[i]['verdicts_original_frame'][0] for i in ids]):.2f} | "
                     f"{np.mean([orig[i]['correct']=='1' for i in ids]):.2f} | "
                     f"{np.mean([A[i]['verdicts_original_frame'][0]==A[i]['gold'] for i in ids]):.2f} |")
        L.append("")

    if B:
        def maj(vs): return Counter(vs).most_common(1)[0][0]
        pers = [r for r in B if r["role"] == "persistent_wrong"]
        ctrl = [r for r in B if r["role"] == "matched_control"]
        p_corr = [maj(r["verdicts_original_frame"]) == r["gold"] for r in pers]
        c_corr = [maj(r["verdicts_original_frame"]) == r["gold"] for r in ctrl]
        p_unan = [len(set(r["verdicts_original_frame"])) == 1 for r in pers]
        L += [f"## Stage B — extended taxonomy (persistent n={len(pers)}, "
              f"controls n={len(ctrl)}; K=5 swapped order)", "",
              f"- **Order-correctable persistent** (swapped majority correct): "
              f"{sum(p_corr)}/{len(pers)} ({np.mean(p_corr):.0%})",
              f"- **Order-robust persistent** (swapped majority still wrong): "
              f"{len(pers)-sum(p_corr)}/{len(pers)} ({1-np.mean(p_corr):.0%})",
              f"- Persistent cases still unanimous under swap: {sum(p_unan)}/{len(pers)}",
              f"- Matched controls remaining correct under swap: "
              f"{sum(c_corr)}/{len(ctrl)} ({np.mean(c_corr):.0%}) — control flip rate "
              f"{1-np.mean(c_corr):.0%} is the general order-sensitivity baseline", "",
              "Reading: if many persistent errors flip, position sensitivity drives them "
              "(order-sensitive persistent); if most survive both orders, they are "
              "order-robust systematic errors — either way the taxonomy sharpens: "
              "Stable correct / Sampling-correctable / Sampling-harmful / "
              "Order-sensitive persistent / Order-and-prompt persistent.", ""]

    out = os.path.join(HERE, f"order_swap_analysis{suf}.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    import sys
    main(mock="--mock" in sys.argv)
