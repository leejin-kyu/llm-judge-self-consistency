# -*- coding: utf-8 -*-
"""P3-21: Verbosity-bias regression on the full test split (no API needed).

Model A (per strategy pooled, instance-clustered logit):
    correct ~ length_gap_z + longer_is_gold + subset + strategy + strategy:length_gap_z
Model B (selection model, per strategy):
    chose_longer ~ longer_is_gold + length_gap_z + subset

Also reports descriptive tables: longer-selection rate by strategy x correctness,
share of instances where the longer candidate is gold, accuracy by length-gap tercile.
"""
from __future__ import annotations
import csv, os
from collections import defaultdict
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")
OUT = os.path.join(HERE, "verbosity_regression.md")
STRATS = ["Direct", "Decomp", "Single-CoT", "SC-CoT"]  # DTT-Sym excluded: tie verdicts break chose_longer

def main():
    df = pd.read_csv(CSV)
    df = df[(df["split"] == "test") & (df["strategy"].isin(STRATS))].copy()
    la, lb = df["candidate_A_length"], df["candidate_B_length"]
    df["length_gap"] = (la - lb).abs()
    df["length_gap_z"] = (df["length_gap"] - df["length_gap"].mean()) / df["length_gap"].std()
    df["longer"] = np.where(la > lb, "A", np.where(lb > la, "B", "eq"))
    df["longer_is_gold"] = (df["longer"] == df["gold_label"]).astype(int)
    df["chose_longer"] = (df["predicted_label"] == df["longer"]).astype(int)
    df = df[df["longer"] != "eq"]  # drop exact-equal lengths

    L = [f"# Verbosity-bias regression (test split, N={df['instance_id'].nunique()} instances x {len(STRATS)} strategies)", ""]

    # Descriptives
    L += ["## Longer-selection rate by strategy and correctness", "",
          "| Strategy | overall | on correct | on errors | longer_is_gold base rate |", "|---|---|---|---|---|"]
    base = df.groupby("strategy")["longer_is_gold"].mean()
    for s in STRATS:
        d = df[df.strategy == s]
        L.append(f"| {s} | {d.chose_longer.mean():.2f} | {d[d.correct==1].chose_longer.mean():.2f} | "
                 f"{d[d.correct==0].chose_longer.mean():.2f} | {base[s]:.2f} |")

    L += ["", "## Accuracy by length-gap tercile", "", "| Strategy | small gap | medium | large |", "|---|---|---|---|"]
    df["gap_ter"] = pd.qcut(df["length_gap"], 3, labels=["small", "medium", "large"])
    for s in STRATS:
        d = df[df.strategy == s]
        accs = d.groupby("gap_ter", observed=True)["correct"].mean()
        L.append(f"| {s} | {accs['small']:.3f} | {accs['medium']:.3f} | {accs['large']:.3f} |")

    # Model A: pooled logit with cluster-robust SE by instance
    mA = smf.logit("correct ~ length_gap_z + longer_is_gold + C(subset) + C(strategy, Treatment('Single-CoT')) "
                   "+ C(strategy, Treatment('Single-CoT')):length_gap_z", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["instance_id"]}, disp=0)
    tab = mA.summary2().tables[1].round(3)
    L += ["", "## Model A: correct ~ length_gap + longer_is_gold + subset + strategy (+ interaction), cluster(instance)",
          "", "```", tab.to_string(max_cols=None, line_width=200), "```"]

    # Model B: per-strategy selection model
    L += ["", "## Model B: chose_longer ~ longer_is_gold + length_gap_z + subset (per strategy)", "",
          "| Strategy | b(longer_is_gold) | p | b(length_gap_z) | p |", "|---|---|---|---|---|"]
    for s in STRATS:
        d = df[df.strategy == s]
        m = smf.logit("chose_longer ~ longer_is_gold + length_gap_z + C(subset)", data=d).fit(disp=0)
        L.append(f"| {s} | {m.params['longer_is_gold']:+.2f} | {m.pvalues['longer_is_gold']:.3g} "
                 f"| {m.params['length_gap_z']:+.2f} | {m.pvalues['length_gap_z']:.3g} |")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)

if __name__ == "__main__":
    main()
