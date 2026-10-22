# -*- coding: utf-8 -*-
"""Plan C analysis: reachability decomposition on JudgeBench (external benchmark).

Primary endpoints (pre-specified; no external registry): K5−K1, disagreement mass, persistent mass —
NOT absolute accuracy. Compares the decomposition structure against LLMBar.
Output: external_bench_analysis.md
"""
from __future__ import annotations
import itertools, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
B = 5000


def main():
    path = os.path.join(HERE, "external_judgebench.jsonl")
    recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    N = len(recs)
    def maj(vs): return Counter(vs).most_common(1)[0][0]
    c = np.array([sum(v == r["gold"] for v in r["pool"]) for r in recs])
    p1 = c / 5
    k5 = np.array([float(maj(r["pool"]) == r["gold"]) for r in recs])
    k3 = np.array([np.mean([float(maj([r["pool"][a], r["pool"][b_], r["pool"][cc]]) == r["gold"])
                            for a, b_, cc in itertools.combinations(range(5), 3)]) for r in recs])
    greedy = np.array([float(r["greedy"] == r["gold"]) for r in recs])
    delta = k5 - p1
    rng = np.random.default_rng(0)
    boots = delta[rng.integers(0, N, (B, N))].mean(axis=1)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    reach = (0 < c) & (c < 5)
    pers = c == 0
    unan = np.array([len(set(r["pool"])) == 1 for r in recs])
    unan_wrong = int(sum(1 for r in recs if len(set(r["pool"])) == 1 and r["pool"][0] != r["gold"]))
    stop3 = np.array([len(set(r["pool"][:3])) == 1 for r in recs])
    ub = np.where(reach, 1.0, k5).mean()

    LLMBAR = {"R_dis": 0.119, "R_corr": 0.0217, "R_harm": 0.0122, "R_pers": 0.132,
              "gain": 0.0095, "k5": 0.827, "unan": 0.87, "ub": 0.868}
    L = [f"# External-benchmark reachability decomposition — JudgeBench (N={N}, GPT-4o)", "",
         "> Pre-specified 120-instance stratified sample (seed 42; frozen before execution, no external registry); no prompt tuning.",
         "> Primary endpoints: K5−K1, disagreement mass, persistent mass (not accuracy).", "",
         "| Quantity | LLMBar (N=295) | JudgeBench (N=120) |", "|---|---|---|",
         f"| Greedy Single-CoT | 0.817 | {greedy.mean():.3f} |",
         f"| K=1 expected (T=0.7) | 0.818 | {p1.mean():.3f} |",
         f"| K=3 expected | 0.825 | {k3.mean():.3f} |",
         f"| K=5 majority | {LLMBAR['k5']:.3f} | {k5.mean():.3f} |",
         f"| Voting gain K5−K1 [95% CI] | +0.0095 [−0.003, +0.022] | "
         f"{delta.mean():+.4f} [{lo:+.4f}, {hi:+.4f}] |",
         f"| R_dis (reachable disagreement) | {LLMBAR['R_dis']:.3f} | {reach.mean():.3f} |",
         f"| R_correctable | {LLMBAR['R_corr']:.4f} | {delta.clip(min=0).mean():.4f} |",
         f"| R_harmful | {LLMBAR['R_harm']:.4f} | {(-delta).clip(min=0).mean():.4f} |",
         f"| R_persistent (c_i=0) | {LLMBAR['R_pers']:.3f} | {pers.mean():.3f} |",
         f"| Unanimity rate (5-0) | 87% | {unan.mean():.0%} |",
         f"| Unanimous-and-wrong | 37/295 (13%) | {unan_wrong}/{N} ({unan_wrong/N:.0%}) |",
         f"| Oracle-on-reachable upper bound | 0.868 | {ub:.3f} |",
         f"| Early-stop mean calls (3→5) | 3.20 | {np.mean(np.where(stop3, 3, 5)):.2f} |", "",
         "Reading: the absolute voting gain varies by benchmark, but the decomposition "
         "consistently shows that majority voting is limited by the amount of "
         "sample-reachable disagreement.", ""]
    out = os.path.join(HERE, "external_bench_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)
    print(f"gain={delta.mean():+.4f} R_dis={reach.mean():.3f} R_pers={pers.mean():.3f} "
          f"unan={unan.mean():.0%} k5={k5.mean():.3f}")


if __name__ == "__main__":
    main()
