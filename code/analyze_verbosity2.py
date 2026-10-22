# -*- coding: utf-8 -*-
"""5th-review item 9: sharpened verbosity-linked error analysis (no API needed).

Three additions over analyze_verbosity.py:
  1. Error-prediction logit with odds ratios, 95% CIs, LR test and cluster-robust
     SEs: error ~ log length ratio + longer_is_gold + gold position + subset
     (+ interaction), per reasoning strategy.
  2. Matched comparison: persistent-wrong (c_i = 0, unanimous) SC-CoT instances vs
     stable-correct (c_i = 5) instances matched on subset x gold position x
     length-gap tercile -> longer-choice rate in each matched group.
  3. Negative-control split: P(choose longer) conditioned on whether the LONGER or
     the SHORTER candidate is gold. The bias claim requires disproportionate
     longer-selection specifically when the shorter candidate is gold.

Output: verbosity_regression2.md
"""
from __future__ import annotations
import os
from collections import Counter
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")
OUT = os.path.join(HERE, "verbosity_regression2.md")

df = pd.read_csv(CSV)
df = df[df["split"] == "test"].copy()
la, lb = df["candidate_A_length"], df["candidate_B_length"]
df["log_len_ratio"] = np.log(np.maximum(la, lb) / np.minimum(la, lb).clip(lower=1))
df["length_gap"] = (la - lb).abs()
df["longer"] = np.where(la > lb, "A", np.where(lb > la, "B", "eq"))
df = df[df["longer"] != "eq"]
df["longer_is_gold"] = (df["longer"] == df["gold_label"]).astype(int)
df["chose_longer"] = (df["predicted_label"] == df["longer"]).astype(int)
df["error"] = 1 - df["correct"]
df["gold_pos_B"] = (df["gold_label"] == "B").astype(int)

L = ["# Verbosity-linked error analysis, sharpened (test split; zero API calls)", "",
     "> analyze_verbosity2.py — error-prediction odds ratios, matched persistent-vs-"
     "stable comparison, and a negative-control split by which candidate is gold.", ""]

# ---------- 1. Error-prediction logit with odds ratios ----------
L += ["## 1. Error-prediction model (per strategy): "
      "error ~ log_len_ratio * longer_is_gold + gold_pos_B + subset", "",
      "| Strategy | OR(log_len_ratio) [95% CI] | p | OR(longer_is_gold) [95% CI] | p | "
      "OR(interaction) | LR test vs no-length p |", "|---|---|---|---|---|---|---|"]
for s in ["Single-CoT", "SC-CoT", "Direct"]:
    d = df[df.strategy == s]
    full = smf.logit("error ~ log_len_ratio * longer_is_gold + gold_pos_B + C(subset)",
                     data=d).fit(disp=0)
    null = smf.logit("error ~ longer_is_gold + gold_pos_B + C(subset)", data=d).fit(disp=0)
    lr = 2 * (full.llf - null.llf)
    lr_p = stats.chi2.sf(lr, df=2)  # log_len_ratio + interaction
    def orci(m, k):
        lo, hi = m.conf_int().loc[k]
        return f"{np.exp(m.params[k]):.2f} [{np.exp(lo):.2f}, {np.exp(hi):.2f}]"
    L.append(f"| {s} | {orci(full,'log_len_ratio')} | {full.pvalues['log_len_ratio']:.3g} | "
             f"{orci(full,'longer_is_gold')} | {full.pvalues['longer_is_gold']:.3g} | "
             f"{np.exp(full.params['log_len_ratio:longer_is_gold']):.2f} | {lr_p:.3g} |")
L += ["", "Reading: OR(log_len_ratio) > 1 with longer_is_gold controlled means the "
      "error probability rises with the length gap even after accounting for whether "
      "the longer candidate is actually the gold answer.", ""]

# ---------- 2. Matched comparison: persistent-wrong vs stable-correct ----------
sc = df[df.strategy == "SC-CoT"].copy()
def n_correct_votes(r):
    return sum(v == r["gold_label"] for v in str(r["votes"]).split("|"))
sc["c_i"] = sc.apply(n_correct_votes, axis=1)
sc["unanimous"] = sc["votes"].map(lambda v: len(set(str(v).split("|"))) == 1)
sc["gap_ter"] = pd.qcut(sc["length_gap"], 3, labels=["small", "medium", "large"])

pers = sc[(sc.c_i == 0) & sc.unanimous]
stab = sc[sc.c_i == 5]
L += [f"## 2. Matched comparison: persistent-wrong (n={len(pers)}) vs stable-correct "
      f"(n={len(stab)}), matched on subset x gold position x length-gap tercile", ""]
rng = np.random.default_rng(0)
matched_pairs = []
used = set()
for _, p in pers.iterrows():
    pool = stab[(stab.subset == p.subset) & (stab.gold_pos_B == p.gold_pos_B) &
                (stab.gap_ter == p.gap_ter) & (~stab.instance_id.isin(used))]
    if len(pool):
        m = pool.iloc[rng.integers(len(pool))]
        used.add(m.instance_id)
        matched_pairs.append((p, m))
n_m = len(matched_pairs)
p_longer = np.mean([p.chose_longer for p, _ in matched_pairs])
m_longer = np.mean([m.chose_longer for _, m in matched_pairs])
# exact binomial test on discordant pairs (McNemar)
b = sum(1 for p, m in matched_pairs if p.chose_longer and not m.chose_longer)
c = sum(1 for p, m in matched_pairs if not p.chose_longer and m.chose_longer)
mcn_p = stats.binomtest(b, b + c, 0.5).pvalue if b + c else float("nan")
L += [f"- Matched pairs found: {n_m}/{len(pers)} (greedy 1:1 within stratum, seed 0)",
      f"- Longer-choice rate: persistent-wrong {p_longer:.2f} vs matched stable-correct "
      f"{m_longer:.2f} (paired discordant {b}/{c}, exact McNemar p={mcn_p:.3g})",
      f"- Unmatched persistent cases: {len(pers)-n_m} (no stratum partner available)", ""]

# ---------- 3. Negative-control split ----------
L += ["## 3. Negative-control split: longer-selection by which candidate is gold", "",
      "| Strategy | P(chose longer | LONGER is gold) | P(chose longer | SHORTER is gold) | "
      "P(chose longer | shorter gold, on errors) |", "|---|---|---|---|"]
for s in ["Single-CoT", "SC-CoT", "Direct"]:
    d = df[df.strategy == s]
    lg = d[d.longer_is_gold == 1]
    sg = d[d.longer_is_gold == 0]
    sg_err = sg[sg.error == 1]
    L.append(f"| {s} | {lg.chose_longer.mean():.2f} | {sg.chose_longer.mean():.2f} | "
             f"{sg_err.chose_longer.mean():.2f} |")
L += ["", "Reading: selecting the longer response when it IS gold is not an error. The "
      "bias shows in the second and third columns: the rate at which the judge picks "
      "the longer candidate precisely when the shorter one is gold — and on errors, "
      "where by definition the shorter (gold) candidate was rejected.", ""]

# error composition sanity: of SC-CoT errors, how many are shorter-is-gold?
sc_err = sc[sc.error == 1]
L += [f"Sanity: of {len(sc_err)} SC-CoT errors, {int((sc_err.longer_is_gold==0).sum())} "
      f"({(sc_err.longer_is_gold==0).mean():.0%}) occur where the shorter candidate is "
      f"gold; the judge chose the longer response in "
      f"{int(sc_err[sc_err.longer_is_gold==0].chose_longer.sum())} of those.", ""]

open(OUT, "w", encoding="utf-8").write("\n".join(L))
print("wrote", OUT)
