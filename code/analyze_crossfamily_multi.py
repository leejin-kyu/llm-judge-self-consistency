# -*- coding: utf-8 -*-
"""Multi-judge cross-family analysis: all local Ollama judges + GPT-4o canonical.

Extends analyze_ollama.py (single-journal) to every ollama_*.jsonl journal found,
producing one table sorted by vote-disagreement rate, plus a descriptive
disagreement <-> voting-gain trend check (the paper's prediction: voting gain
scales with vote disagreement; persistent unanimous errors bound what voting
can fix).

Usage: python analyze_crossfamily_multi.py
Output: crossfamily_multi_analysis.md
"""
from __future__ import annotations
import glob, json, os
import numpy as np
from analyze_ollama import analyze

HERE = os.path.dirname(os.path.abspath(__file__))

# GPT-4o canonical reference (voting_effect_analysis.md, records.json, N=295)
GPT4O_REF = {"label": "gpt-4o (canonical)", "N": 295, "greedy": 0.817, "k1": 0.818,
             "k3": 0.825, "k5": 0.827, "gain": 0.0095, "gain_lo": -0.003,
             "gain_hi": 0.022, "unanimity": 258 / 295, "unan_wrong": 37,
             "disagree": 1 - 258 / 295, "mean_calls": 3.20}


def journal_label(path):
    base = os.path.basename(path)[len("ollama_"):-len(".jsonl")]
    # gemma3_12b -> gemma3:12b ; qwen2_5_7b -> qwen2.5:7b (best-effort)
    parts = base.split("_")
    if len(parts) >= 2 and parts[-1].endswith("b"):
        return "_".join(parts[:-1]).replace("_", ".") + ":" + parts[-1]
    return base


def main():
    rows = [GPT4O_REF]
    for path in sorted(glob.glob(os.path.join(HERE, "ollama_*.jsonl"))):
        recs = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if not recs:
            continue
        rows.append(analyze(recs, journal_label(path)))
    rows.sort(key=lambda r: r["disagree"])

    L = ["# Cross-family validation (multi-judge): local open-weight judges vs GPT-4o",
         "",
         "All local judges: 100 stratified LLMBar test instances, 1 greedy CoT (T=0) + "
         "5 CoT samples (T=0.7), identical prompt/protocol (run_ollama_judge.py). "
         "GPT-4o row = canonical run (records.json, N=295). Sorted by disagreement rate.",
         "",
         "| Judge | N | Greedy | K=1 exp. | K=3 | K=5 | Voting gain (K5-K1) [95% CI] | "
         "Unanimity | Disagreement | Unanimous-wrong | Early-stop calls |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ci = (f" [{r['gain_lo']:+.3f}, {r['gain_hi']:+.3f}]"
              if "gain_lo" in r else "")
        L.append(f"| {r['label']} | {r['N']} | {r['greedy']:.3f} | {r['k1']:.3f} | "
                 f"{r['k3']:.3f} | {r['k5']:.3f} | {r['gain']:+.4f}{ci} | "
                 f"{r['unanimity']:.0%} | {r['disagree']:.0%} | {r['unan_wrong']} | "
                 f"{r['mean_calls']:.2f} |")

    # descriptive trend: disagreement vs gain across judges
    d = np.array([r["disagree"] for r in rows])
    g = np.array([r["gain"] for r in rows])
    n = len(rows)
    rank_d, rank_g = d.argsort().argsort(), g.argsort().argsort()
    rho = np.corrcoef(rank_d, rank_g)[0, 1] if n >= 3 else float("nan")
    L += ["",
          f"Descriptive trend across {n} judges: rank correlation between disagreement "
          f"rate and voting gain = {rho:.2f} (n={n}; descriptive only, no inferential "
          "claim). Honest framing: directional consistency with the prediction, not a law.",
          "",
          "Reading: judges with more vote disagreement have more *correctable* errors "
          "available to majority voting, but unanimous-wrong counts show the persistent, "
          "systematically biased error mass that voting cannot address dominates in "
          "weaker judges.", ""]
    out = os.path.join(HERE, "crossfamily_multi_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)
    for r in rows:
        print(f"{r['label']:>24}  disagree={r['disagree']:.2f}  gain={r['gain']:+.4f}  "
              f"k5={r['k5']:.3f}  unan_wrong={r['unan_wrong']}")


if __name__ == "__main__":
    main()
