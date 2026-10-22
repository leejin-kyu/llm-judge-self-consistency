# -*- coding: utf-8 -*-
"""Cross-family analysis: gemma3 (or other Ollama judge) vs GPT-4o vote structure.

Computes for the non-OpenAI judge: K=1 expected accuracy, K=3/K=5 majority accuracy,
unanimity rate, unanimous-and-wrong count, early-stopping mean calls, expected net
voting gain — and contrasts them with the GPT-4o canonical values to test the
prediction: voting gain scales with vote disagreement.
"""
from __future__ import annotations
import argparse, itertools, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(0)
B = 5000

def maj(vs): return Counter(vs).most_common(1)[0][0]

def analyze(recs, label):
    ids = list(range(len(recs)))
    p1 = np.array([sum(v == r["gold"] for v in r["pool"]) / 5 for r in recs])
    k5 = np.array([int(maj(r["pool"]) == r["gold"]) for r in recs], float)
    k3 = np.array([np.mean([int(maj([r["pool"][a], r["pool"][b], r["pool"][c]]) == r["gold"])
                            for a, b, c in itertools.combinations(range(5), 3)]) for r in recs])
    greedy = np.array([int(r["greedy"] == r["gold"]) for r in recs], float)
    unan = np.array([len(set(r["pool"])) == 1 for r in recs])
    unan_wrong = int(sum(1 for r in recs if len(set(r["pool"])) == 1 and r["pool"][0] != r["gold"]))
    delta = k5 - p1
    idx = RNG.integers(0, len(delta), (B, len(delta)))
    boots = delta[idx].mean(axis=1)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    stop3 = np.array([len(set(r["pool"][:3])) == 1 for r in recs])
    mean_calls = np.mean(np.where(stop3, 3, 5))
    return {
        "label": label, "N": len(recs),
        "greedy": greedy.mean(), "k1": p1.mean(), "k3": k3.mean(), "k5": k5.mean(),
        "gain": delta.mean(), "gain_lo": lo, "gain_hi": hi,
        "unanimity": unan.mean(), "unan_wrong": unan_wrong,
        "disagree": 1 - unan.mean(), "mean_calls": mean_calls,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--journal", default="ollama_gemma3_12b.jsonl")
    ap.add_argument("--label", default="gemma3:12b")
    args = ap.parse_args()
    recs = [json.loads(l) for l in open(os.path.join(HERE, args.journal), encoding="utf-8")]
    r = analyze(recs, args.label)
    # GPT-4o canonical reference (from voting_effect_analysis.md, N=295)
    ref = {"label": "gpt-4o (canonical, N=295)", "greedy": 0.817, "k1": 0.818,
           "k3": 0.825, "k5": 0.827, "gain": 0.0095, "unanimity": 258/295,
           "unan_wrong": 37, "disagree": 1 - 258/295, "mean_calls": 3.20, "N": 295}
    L = [f"# Cross-family validation: {args.label} (local, open-weight) vs GPT-4o", "",
         "| Quantity | " + ref["label"] + f" | {r['label']} (N={r['N']}) |", "|---|---|---|",
         f"| Greedy Single-CoT (T=0) | {ref['greedy']:.3f} | {r['greedy']:.3f} |",
         f"| K=1 expected (T=0.7) | {ref['k1']:.3f} | {r['k1']:.3f} |",
         f"| K=3 expected majority | {ref['k3']:.3f} | {r['k3']:.3f} |",
         f"| K=5 majority | {ref['k5']:.3f} | {r['k5']:.3f} |",
         f"| Voting gain (K=5 − K=1) | +{ref['gain']:.4f} | {r['gain']:+.4f} "
         f"[{r['gain_lo']:+.4f}, {r['gain_hi']:+.4f}] |",
         f"| Unanimity rate (5-0) | {ref['unanimity']:.0%} | {r['unanimity']:.0%} |",
         f"| Vote disagreement rate | {ref['disagree']:.0%} | {r['disagree']:.0%} |",
         f"| Unanimous-and-wrong | {ref['unan_wrong']} | {r['unan_wrong']} |",
         f"| Early-stopping mean calls | {ref['mean_calls']:.2f} | {r['mean_calls']:.2f} |", "",
         "Prediction under test: voting gain scales with vote disagreement; low-entropy "
         "judges mainly reproduce the same bias, higher-variance judges provide more "
         "correctable disagreement.", ""]
    out = os.path.join(HERE, "crossfamily_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)
    for k in ("greedy", "k1", "k5", "gain", "unanimity", "mean_calls"):
        print(k, "=", round(r[k], 4))

if __name__ == "__main__":
    main()
