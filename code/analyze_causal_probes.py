# -*- coding: utf-8 -*-
"""Plan E analysis: rubric-first re-judging + verbosity counterfactual.

Output: causal_probes_analysis.md
"""
from __future__ import annotations
import json, os
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))


def jload(name):
    p = os.path.join(HERE, name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def main():
    L = ["# Causal probes on persistent errors (rubric-first + verbosity counterfactual)", ""]

    rub = jload("causal_rubric.jsonl")
    if rub:
        pers = [r for r in rub if r["role"] == "persistent_wrong"]
        ctrl = [r for r in rub if r["role"] == "matched_control"]
        p_corr = sum(r["verdict"] == r["gold"] for r in pers)
        c_corr = sum(r["verdict"] == r["gold"] for r in ctrl)
        fisher_p = stats.fisher_exact([[p_corr, len(pers)-p_corr],
                                       [37-33, 33]])[1]  # vs holistic baseline flips n/a
        L += [f"## Probe 1 — rubric-first re-judging (constraint-focused prompt, T=0, "
              f"original order)", "",
              f"| Group | n | correct under rubric-first | rate |", "|---|---|---|---|",
              f"| Persistent-wrong (holistic: 0% correct by definition) | {len(pers)} | "
              f"{p_corr} | {p_corr/len(pers):.0%} |",
              f"| Matched controls (holistic: 100% correct by definition) | {len(ctrl)} | "
              f"{c_corr} | {c_corr/len(ctrl):.0%} |", "",
              f"- Persistent errors corrected by constraint-focused prompting: "
              f"{p_corr}/{len(pers)} ({p_corr/len(pers):.0%}); control degradation "
              f"{len(ctrl)-c_corr}/{len(ctrl)} ({1-c_corr/len(ctrl):.0%}).", ""]

    cf = jload("causal_counterfactual.jsonl")
    if cf:
        n = len(cf)
        comp_flip = sum(r["verdict_after_compress"] == r["gold"] for r in cf)
        exp_flip = sum(r["verdict_after_expand"] == r["gold"] for r in cf)
        both = sum(r["verdict_after_compress"] == r["gold"] and
                   r["verdict_after_expand"] == r["gold"] for r in cf)
        ratio_c = np.mean([r["len_wrong_comp"]/r["len_wrong"] for r in cf])
        ratio_e = np.mean([r["len_gold_exp"]/r["len_gold"] for r in cf])
        # binomial vs 0 baseline (holistic verdicts were 100% wrong on these)
        L += [f"## Probe 2 — verbosity counterfactual (N={n} unanimous-and-longer errors; "
              f"baseline: judge wrong on 100% by construction)", "",
              f"| Intervention | Judge now picks gold | rate |", "|---|---|---|",
              f"| (a) Compress the longer WRONG response (mean ratio {ratio_c:.2f}) | "
              f"{comp_flip}/{n} | {comp_flip/n:.0%} |",
              f"| (b) Expand the shorter GOLD response (mean ratio {ratio_e:.2f}) | "
              f"{exp_flip}/{n} | {exp_flip/n:.0%} |",
              f"| Both directions flip | {both}/{n} | {both/n:.0%} |", "",
              "Caveat: compression/expansion are LLM rewrites; despite content-"
              "preservation instructions, unverified semantic drift is possible — "
              "spot-check before quoting individual cases. Flips under BOTH "
              "directions are the strongest evidence consistent with a causal "
              "length effect; residual non-flips indicate content-driven "
              "(length-independent) failure.", ""]

    out = os.path.join(HERE, "causal_probes_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
