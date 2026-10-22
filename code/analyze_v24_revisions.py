# -*- coding: utf-8 -*-
"""v2.4 revision analyses (offline; no API calls, no local model calls).

One shared routine computes every voting-gain interval reported in v2.4, so that
intervals no longer differ through per-script random-number streams.

  V1  Aggregation-rule and tie-handling sensitivity in one format for every setting:
      binary-correctness majority / modal rule with first-occurrence tie-breaking /
      modal rule with uniform random tie-breaking (expected value)
  V2  Cross-family comparison on the common 100-instance sample (seed 99), GPT-4o included
  V3  Repeated pools: equal-allocation mean vs test-split-weighted mean, stratified
      bootstrap that keeps the three pools of an item together
  V4  Fixed-item vote-generation variation vs instance-resampling variation
  V5  No-intervention re-judgment comparators assembled from existing same-session logs
  V6  Ties, parser routes, and the 256-token cap: what the stored logs can and cannot show
  V7  Per-setting precision: SD of Delta_i and N needed for a target half-width
Writes revision_numbers_v24.md and revision_numbers_v24.json next to this script.
"""
from __future__ import annotations
import csv, hashlib, json, os, re
from collections import Counter
from itertools import combinations
import numpy as np
from scipy.stats import beta, binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_MD = os.path.join(HERE, "revision_numbers_v24.md")
OUT_JS = os.path.join(HERE, "revision_numbers_v24.json")
K = 5
B = 100_000
SEED = 20261005
L: list[str] = []
J: dict = {"B": B, "seed": SEED}


def P(s=""):
    L.append(s)


def jl(n):
    return [json.loads(l) for l in open(os.path.join(HERE, n), encoding="utf-8") if l.strip()]


def votes_of(s):
    return [v.strip(" '\"") for v in s.strip("[]").split(",")] if isinstance(s, str) else list(s)


def cp(x, n, a=0.05):
    lo = 0.0 if x == 0 else beta.ppf(a / 2, x, n - x + 1)
    hi = 1.0 if x == n else beta.ppf(1 - a / 2, x + 1, n - x)
    return float(lo), float(hi)


# ---------------------------------------------------------------- shared bootstrap
def boot(values, strata=None, weights=None, q=(0.025, 0.975)):
    """Percentile bootstrap of a (weighted) mean with one shared seed.

    values : (N,) or (N, M) array; columns are resampled jointly (same items).
    strata : optional length-N labels -> resample within stratum (design-preserving).
    weights: optional {stratum: W_h}; estimate = sum_h W_h * mean_h. Without weights
             the estimate is the plain mean of the resampled items.
    Returns (point, lo, hi) per column.
    """
    v = np.asarray(values, float)
    one = v.ndim == 1
    if one:
        v = v[:, None]
    n = v.shape[0]
    rng = np.random.default_rng(SEED)
    chunk = 10_000
    reps = []
    if strata is None:
        point = v.mean(0)
        for _ in range(B // chunk):
            ix = rng.integers(0, n, (chunk, n))
            reps.append(v[ix].mean(1))
    else:
        strata = np.asarray(strata)
        labs = sorted(set(strata.tolist()))
        idx = {h: np.where(strata == h)[0] for h in labs}
        if weights is None:
            w = {h: len(idx[h]) / n for h in labs}
        else:
            tot = sum(weights[h] for h in labs)
            w = {h: weights[h] / tot for h in labs}
        point = sum(w[h] * v[idx[h]].mean(0) for h in labs)
        for _ in range(B // chunk):
            acc = 0
            for h in labs:
                ih = idx[h]
                ix = ih[rng.integers(0, len(ih), (chunk, len(ih)))]
                acc = acc + w[h] * v[ix].mean(1)
            reps.append(acc)
    R = np.concatenate(reps, 0)
    lo, hi = np.quantile(R, q[0], axis=0), np.quantile(R, q[1], axis=0)
    se = R.std(0, ddof=1)
    if one:
        return float(point[0]), float(lo[0]), float(hi[0]), float(se[0])
    return point, lo, hi, se


def f3(x):
    return f"{x:+.3f}"


def ci(t, d=3):
    return f"{t[0]:+.{d+1}f} [{t[1]:+.{d}f}, {t[2]:+.{d}f}]"


# ---------------------------------------------------------------- aggregation rules
def modal_first(v):
    return Counter(v).most_common(1)[0][0]


def modal_tied(v):
    c = Counter(v)
    top = max(c.values())
    return [k for k in c if c[k] == top]


def rule_scores(v, g):
    """Correctness of the K-vote aggregate under the three rules."""
    c = sum(x == g for x in v)
    tied = modal_tied(v)
    return {
        "binary": float(c > len(v) / 2),
        "modal_first": float(modal_first(v) == g),
        "modal_random": (1.0 / len(tied)) if g in tied else 0.0,
    }


def setting_block(name, rows, greedy=None, strata=None):
    """rows: list of (votes, gold). Returns dict of statistics under the shared routine."""
    V = [r[0] for r in rows]
    G = [r[1] for r in rows]
    n = len(V)
    c = np.array([sum(x == g for x in v) for v, g in zip(V, G)])
    p1 = c / K
    sc = {k: np.array([rule_scores(v, g)[k] for v, g in zip(V, G)]) for k in ("binary", "modal_first", "modal_random")}
    ties = [(v, g) for v, g in zip(V, G) if len(modal_tied(v)) > 1]
    tie_gold_in = sum(g in modal_tied(v) for v, g in ties)
    tie_first_gold = sum(modal_first(v) == g for v, g in ties)
    out = {
        "N": n,
        "acc_k1": float(p1.mean()),
        "greedy": (None if greedy is None else float(np.mean(greedy))),
        "R_gold": float(((c > 0) & (c < K)).mean()),
        "n_R_gold": int(((c > 0) & (c < K)).sum()),
        "D_vote": float(np.mean([len(set(v)) > 1 for v in V])),
        "n_pers": int((c == 0).sum()),
        "n_unan_wrong": int(sum(len(set(v)) == 1 and v[0] != g for v, g in zip(V, G))),
        "n_modal_ties": len(ties),
        "ties_gold_among_tied": int(tie_gold_in),
        "ties_first_occurrence_gold": int(tie_first_gold),
        "tie_votes": int(sum(x == "tie" for v in V for x in v)),
        "inst_with_tie_vote": int(sum(any(x == "tie" for x in v) for v in V)),
        "n_votes": n * K,
        "early_stop_calls": float(np.mean([3 if len(set(v[:3])) == 1 else 5 for v in V])),
        "sd_delta_binary": float((sc["binary"] - p1).std(ddof=1)),
    }
    for k in sc:
        d = sc[k] - p1
        out[f"acc5_{k}"] = float(sc[k].mean())
        out[f"gain_{k}"] = boot(d)
        out[f"pos_{k}"] = float(np.clip(d, 0, None).mean())
        out[f"neg_{k}"] = float(np.clip(-d, 0, None).mean())
        if strata is not None:
            out[f"gain_{k}_strat"] = boot(d, strata=strata)
    P(f"\n### {name} (N={n})")
    if greedy is not None:
        P(f"- Greedy CoT accuracy {out['greedy']:.3f}; expected K=1 accuracy {out['acc_k1']:.4f}")
    else:
        P(f"- Expected K=1 accuracy {out['acc_k1']:.4f}")
    P(f"- R_gold {out['R_gold']:.3f} ({out['n_R_gold']}/{n}); D_vote {out['D_vote']:.3f}; c_i=0: {out['n_pers']}; "
      f"unanimous wrong: {out['n_unan_wrong']}; early-stopping mean calls {out['early_stop_calls']:.2f}")
    P(f"- Tie verdicts among individual votes: {out['tie_votes']}/{n*K} "
      f"(instances with >=1 tie vote: {out['inst_with_tie_vote']})")
    P(f"- Modal ties: {out['n_modal_ties']}; gold among tied labels: {tie_gold_in}; "
      f"first-occurrence resolved to gold: {tie_first_gold}")
    P("| Rule | K=5 accuracy | Gain vs K=1 [95% CI] | Positive mass | Negative mass |")
    P("|---|---|---|---|---|")
    for k, lab in (("binary", "Binary-correctness majority"), ("modal_first", "Modal, first-occurrence tie-break"),
                   ("modal_random", "Modal, random tie-break (expected)")):
        P(f"| {lab} | {out[f'acc5_{k}']:.4f} | {ci(out[f'gain_{k}'])} | {out[f'pos_{k}']:.4f} | {out[f'neg_{k}']:.4f} |")
    if strata is not None:
        P(f"- Stratified-bootstrap sensitivity (binary): {ci(out['gain_binary_strat'])}")
    return out


# ---------------------------------------------------------------- load canonical records
SUB = {"full_natural.jsonl": "Natural", "full_neighbor.jsonl": "Neighbor", "full_gptinst.jsonl": "GPTInst",
       "full_gptout.jsonl": "GPTOut", "full_manual.jsonl": "Manual"}
rec = json.load(open(os.path.join(HERE, "records.json"), encoding="utf-8"))
byid, T, ALL = {}, [], []
for ds, items in rec["datasets"].items():
    for i, x in enumerate(items):
        x = dict(x)
        x["iid"] = f"{SUB[ds]}_{i:03d}"
        x["subset"] = SUB[ds]
        byid[x["iid"]] = x
        ALL.append(x)
        if x["split"] == "test":
            T.append(x)
N = len(T)
assert N == 295
SCV = lambda x: x["methods"]["SelfConsistency"]["votes"]
n_test = Counter(x["subset"] for x in T)
W_TEST = {h: n_test[h] / N for h in n_test}

P("# v2.4 revision numbers (generated by analyze_v24_revisions.py)")
P(f"\nShared bootstrap: percentile, B = {B:,}, seed {SEED}, one routine for every interval below. "
  "Offline only; no API or local-model calls.")
P(f"Test-split subset sizes: {dict(n_test)}")

# ================================================================= V1
P("\n## V1. Aggregation rule and tie handling, one format for every setting")
S = {}
S["llmbar295"] = setting_block("GPT-4o, LLMBar test split", [(SCV(x), x["gold"]) for x in T],
                               greedy=[x["methods"]["CoT"]["correct"] for x in T],
                               strata=[x["subset"] for x in T])
jb = jl("external_judgebench.jsonl")
S["judgebench"] = setting_block("GPT-4o, JudgeBench sample", [(votes_of(r["pool"]), r["gold"]) for r in jb],
                                greedy=[r["greedy"] == r["gold"] for r in jb],
                                strata=[r["subset"] for r in jb])
OW = (("phi4", "ollama_phi4_14b.jsonl", "Phi-4 14B"), ("qwen", "ollama_qwen2_5_7b.jsonl", "Qwen2.5 7B"),
      ("gemma", "ollama_gemma3_12b.jsonl", "Gemma 3 12B"), ("llama", "ollama_llama3_1_8b.jsonl", "Llama 3.1 8B"))
ow_rows = {}
for key, f, name in OW:
    R = jl(f)
    ow_rows[key] = R
    S[key] = setting_block(f"{name}, common 100-instance sample (seed 99)", [(r["pool"], r["gold"]) for r in R],
                           greedy=[r["greedy"] == r["gold"] for r in R], strata=[r["subset"] for r in R])
ids99 = [r["iid"] for r in ow_rows["phi4"]]
for key in ow_rows:
    assert sorted(r["iid"] for r in ow_rows[key]) == sorted(ids99), "open-weight judges used different items"
assert all(byid[i]["split"] == "test" for i in ids99)

# ================================================================= V2
P("\n## V2. Cross-family comparison on the common sample")
P(f"- Sample composition (seed 99): {dict(Counter(byid[i]['subset'] for i in ids99))}")
S["gpt4o_c100"] = setting_block("GPT-4o canonical pool restricted to the common 100-instance sample",
                                [(SCV(byid[i]), byid[i]["gold"]) for i in ids99],
                                greedy=[byid[i]["methods"]["CoT"]["correct"] for i in ids99],
                                strata=[byid[i]["subset"] for i in ids99])
six = ["llmbar295", "judgebench", "phi4", "qwen", "gemma", "llama"]
g6 = [S[k]["gain_binary"][0] for k in six]
P(f"\n- Six settings, binary-correctness majority only: gain range {min(g6):+.4f} to {max(g6):+.4f}; "
  f"R_gold range {min(S[k]['R_gold'] for k in six):.3f} to {max(S[k]['R_gold'] for k in six):.3f}")
g6m = [S[k]["gain_modal_first"][0] for k in six]
g6r = [S[k]["gain_modal_random"][0] for k in six]
P(f"- Same six settings, modal first-occurrence: {min(g6m):+.4f} to {max(g6m):+.4f}; "
  f"modal random tie-break: {min(g6r):+.4f} to {max(g6r):+.4f}")
P("- Lower / upper 95% bounds by setting (binary): " + "; ".join(
    f"{k} [{S[k]['gain_binary'][1]:+.3f}, {S[k]['gain_binary'][2]:+.3f}]" for k in six + ["gpt4o_c100"]))

# K=3 nested comparison on the canonical pool (shared routine)
P("\n### K=1 -> K=3 on the canonical pool (ten three-vote subsets per instance)")
d3 = {k: [] for k in ("binary", "modal_first", "modal_random")}
nt3 = 0
for x in T:
    v, g = SCV(x), x["gold"]
    acc = {k: [] for k in d3}
    for comb in combinations(range(K), 3):
        sub = [v[j] for j in comb]
        nt3 += len(modal_tied(sub)) > 1
        rs = rule_scores(sub, g)
        for k in d3:
            acc[k].append(rs[k])
    c = sum(z == g for z in v)
    for k in d3:
        d3[k].append(np.mean(acc[k]) - c / K)
S["k3"] = {}
for k in d3:
    arr = np.array(d3[k])
    S["k3"][k] = boot(arr)
    P(f"- {k}: expected K=3 accuracy {arr.mean() + S['llmbar295']['acc_k1']:.4f}, gain {ci(S['k3'][k])}")
P(f"- Three-vote subsets with a modal tie: {nt3} of {N*10}")

# ================================================================= V3
P("\n## V3. Repeated pools: sample design and target average")
rp = json.load(open(os.path.join(HERE, "splits", "repeat_pool_100_ids.json"), encoding="utf-8"))
ids7 = rp["ids"]
r2 = {r["iid"]: r for r in jl("repeat_pool_run2.jsonl")}
r3 = {r["iid"]: r for r in jl("repeat_pool_run3.jsonl")}
gold7 = {i: byid[i]["gold"] for i in ids7}
assert all(r2[i]["gold"] == gold7[i] and r3[i]["gold"] == gold7[i] for i in ids7)
strata7 = [byid[i]["subset"] for i in ids7]
P(f"- Sample: {dict(Counter(strata7))} (equal allocation, seed {rp['seed']}); "
  f"test-split shares: " + ", ".join(f"{h} {W_TEST[h]:.3f}" for h in W_TEST))
pools7 = [{i: SCV(byid[i]) for i in ids7}, {i: votes_of(r2[i]["pool"]) for i in ids7},
          {i: votes_of(r3[i]["pool"]) for i in ids7}]
cc = np.array([[sum(v == gold7[i] for v in p[i]) for i in ids7] for p in pools7])
for pi, p in enumerate(pools7):
    nt = sum(len(modal_tied(p[i])) > 1 for i in ids7)
    P(f"- Pool {pi+1}: modal ties {nt}; tie votes {sum(v == 'tie' for i in ids7 for v in p[i])}/500")
D7 = ((cc > K / 2).astype(float) - cc / K).T          # 100 x 3
A1 = (cc / K).T
A5 = (cc > K / 2).astype(float).T
RG = ((cc > 0) & (cc < K)).astype(float).T
PE = (cc == 0).astype(float).T
S["rep"] = {}
P("\n| Pool | Quantity | Equal-allocation mean [95% CI] | Test-split-weighted mean [95% CI] |")
P("|---|---|---|---|")
for nm, M_, dig in (("Delta_MV", D7, 3), ("K=1 accuracy", A1, 3), ("K=5 accuracy", A5, 3), ("R_gold", RG, 3),
                    ("K-persistent share", PE, 3)):
    u = boot(M_, strata=strata7)
    w = boot(M_, strata=strata7, weights=W_TEST)
    S["rep"][nm] = {"unweighted": [list(map(float, z)) for z in u[:3]], "weighted": [list(map(float, z)) for z in w[:3]]}
    for pi in range(3):
        fmt = (lambda t, j: f"{t[0][j]:+.4f} [{t[1][j]:+.3f}, {t[2][j]:+.3f}]") if nm == "Delta_MV" else \
              (lambda t, j: f"{t[0][j]:.4f} [{t[1][j]:.3f}, {t[2][j]:.3f}]")
        P(f"| {pi+1} | {nm} | {fmt(u, pi)} | {fmt(w, pi)} |")
mean3 = D7.mean(1)
u = boot(mean3, strata=strata7)
w = boot(mean3, strata=strata7, weights=W_TEST)
S["rep"]["mean3"] = {"unweighted": u, "weighted": w}
P(f"- Mean of the three pools (items resampled jointly): equal allocation {ci(u)}; weighted {ci(w)}")
gains = D7.mean(0)
P(f"- Between-pool SD of the equal-allocation Delta_MV: {np.std(gains, ddof=1):.4f} (three pools); "
  f"pools 2 and 3 only (same day): {gains[1]:+.4f} vs {gains[2]:+.4f}")
gw = [sum(W_TEST[h] * D7[np.array(strata7) == h, j].mean() for h in W_TEST) for j in range(3)]
P(f"- Weighted per-pool Delta_MV: {', '.join(f'{g:+.4f}' for g in gw)}; mean {np.mean(gw):+.4f}, SD {np.std(gw, ddof=1):.4f}")
P(f"- Reference, full test split canonical pool: Delta_MV {S['llmbar295']['gain_binary'][0]:+.4f}, "
  f"R_gold {S['llmbar295']['R_gold']:.3f}, K-persistent share {S['llmbar295']['n_pers']/N:.3f}")
P("- Per-subset canonical Delta_MV on the full test split: " + ", ".join(
    f"{h} {np.mean([(sum(v == x['gold'] for v in SCV(x)) > 2.5) - sum(v == x['gold'] for v in SCV(x)) / 5 for x in T if x['subset'] == h]):+.4f} (n={n_test[h]})"
    for h in n_test))
S["rep"]["gains_unweighted"] = [float(g) for g in gains]
S["rep"]["gains_weighted"] = [float(g) for g in gw]
S["rep"]["pers_counts"] = [int(z) for z in (cc == 0).sum(1)]

# ================================================================= V4
P("\n## V4. Fixed-item vote-generation variation vs instance resampling (repeat sample, N=100)")
se_boot = [boot(D7[:, j])[3] for j in range(3)]
P(f"- Instance-level bootstrap SE of Delta_MV, one pool at N=100: {', '.join(f'{s:.4f}' for s in se_boot)}")
s2_23 = ((D7[:, 1] - D7[:, 2]) ** 2) / 2.0
se_vote_23 = float(np.sqrt(s2_23.sum()) / len(ids7))
s2_3 = D7.var(1, ddof=1)
se_vote_3 = float(np.sqrt(s2_3.sum()) / len(ids7))
P(f"- Vote-generation SE of Delta_MV for the fixed 100 items, from replicate pools: "
  f"{se_vote_23:.4f} (pools 2-3, same day), {se_vote_3:.4f} (three pools; includes any 37-day model change)")
P(f"- Items whose Delta_i differs between pools 2 and 3: {int((D7[:, 1] != D7[:, 2]).sum())}/100")
S["v4"] = {"se_boot": se_boot, "se_vote_23": se_vote_23, "se_vote_3": se_vote_3}
x15 = cp(0, 15)
P(f"- 0 correct of 15 pooled votes: 95% Clopper-Pearson interval for p_i [{x15[0]:.3f}, {x15[1]:.3f}] "
  "(requires the 15 votes to be i.i.d. across the two execution dates)")
x10 = cp(0, 10)
z23 = int(((cc[1] == 0) & (cc[2] == 0)).sum())
P(f"- 0 correct of 10 votes in pools 2 and 3 (same day): {z23} items; 95% CP interval [{x10[0]:.3f}, {x10[1]:.3f}]")
S["v4"]["cp10_hi"] = x10[1]
S["v4"]["n_zero_23"] = z23

# ================================================================= V5
P("\n## V5. No-intervention re-judgment comparators from existing logs")
rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results_final.csv"), encoding="utf-8"))
        if r["strategy"] == "SC-CoT" and r["split"] == "test"]
for r in rows:
    vs = r["votes"].split("|")
    r["unan"] = len(set(vs)) == 1
    r["c_i"] = sum(v == r["gold_label"] for v in vs)
    r["gap"] = abs(int(r["candidate_A_length"]) - int(r["candidate_B_length"]))
pers_r = [r for r in rows if r["unan"] and r["c_i"] == 0]
stab = [r for r in rows if r["c_i"] == 5]
gaps = sorted(r["gap"] for r in rows)
t1, t2 = gaps[len(gaps) // 3], gaps[2 * len(gaps) // 3]
ter = lambda g: 0 if g <= t1 else (1 if g <= t2 else 2)
rr = np.random.default_rng(0)
used, pairs = set(), []
for p in pers_r:
    pool = [s for s in stab if s["subset"] == p["subset"] and s["gold_label"] == p["gold_label"]
            and ter(s["gap"]) == ter(p["gap"]) and s["instance_id"] not in used]
    if not pool:
        pool = [s for s in stab if s["subset"] == p["subset"] and s["gold_label"] == p["gold_label"]
                and s["instance_id"] not in used]
    m = pool[int(rr.integers(len(pool)))]
    used.add(m["instance_id"])
    pairs.append((p["instance_id"], m["instance_id"]))
sb = {r["iid"]: r for r in jl("order_swap_stageB.jsonl")}
assert len(pairs) == 37 and all(p in sb and m in sb for p, m in pairs)
assert all(sb[p]["role"] == "persistent_wrong" and sb[m]["role"] == "matched_control" for p, m in pairs)
PERS = [p for p, _ in pairs]
CTRL = [m for _, m in pairs]
sa = {r["iid"]: r for r in jl("order_swap_stageA.jsonl")}
cs = {r["iid"]: r for r in jl("clean_singlecot.jsonl")}
rb = {r["iid"]: r for r in jl("causal_rubric.jsonl")}
ts = lambda d, ids: (min(d[i]["ts"] for i in ids), max(d[i]["ts"] for i in ids))
P(f"- Timestamps on the 74 items: swapped T=0 single call {ts(sa, PERS + CTRL)}; "
  f"swapped K=5 {ts(sb, PERS + CTRL)}; original-order T=0 rerun {ts(cs, PERS + CTRL)}; "
  f"constraint-focused {ts(rb, PERS + CTRL)}")
P(f"- Pools 2/3 timestamps: {min(r['ts'] for r in r2.values())} to {max(r['ts'] for r in r3.values())}")

ok_swap5 = lambda i: modal_first(votes_of(sb[i]["verdicts_original_frame"])) == sb[i]["gold"]
ok_swap1 = lambda i: votes_of(sa[i]["verdicts_original_frame"])[0] == sa[i]["gold"]
ok_rerun1 = lambda i: cs[i]["verdict"] == cs[i]["gold"]
ok_constr = lambda i: rb[i]["verdict"] == rb[i]["gold"]
ok_canon1 = lambda i: bool(byid[i]["methods"]["CoT"]["correct"])


def mcn(b, c):
    return binomtest(b, b + c, 0.5).pvalue if b + c else 1.0


def compare(name, f_int, f_ref, ids, what):
    a = np.array([f_int(i) for i in ids])
    r = np.array([f_ref(i) for i in ids])
    b, c = int((a & ~r).sum()), int((~a & r).sum())
    P(f"- {name} [{what}, n={len(ids)}]: intervention correct {int(a.sum())}, re-judgment correct {int(r.sum())}; "
      f"both {int((a & r).sum())}, intervention only {b}, re-judgment only {c}, neither {int((~a & ~r).sum())}; "
      f"exact McNemar p = {mcn(b, c):.3f}")
    return {"n": len(ids), "int": int(a.sum()), "ref": int(r.sum()), "both": int((a & r).sum()),
            "int_only": b, "ref_only": c, "neither": int((~a & ~r).sum()), "p": float(mcn(b, c))}


S["v5"] = {}
P("\n### State of the 74 items under the canonical T=0 Single-CoT (15 Jun)")
P(f"- 37 unanimous K-persistent errors: canonical Single-CoT correct on {sum(ok_canon1(i) for i in PERS)}; "
  f"37 matched controls: canonical Single-CoT correct on {sum(ok_canon1(i) for i in CTRL)}")
S["v5"]["canon_pers_ok"] = int(sum(ok_canon1(i) for i in PERS))
S["v5"]["canon_ctrl_ok"] = int(sum(ok_canon1(i) for i in CTRL))

P("\n### (a) Constraint-focused prompt (T=0, one call, 23 Jul) vs original prompt re-judged (T=0, one call, 22 Jul)")
S["v5"]["constr_pers"] = compare("Constraint vs plain rerun", ok_constr, ok_rerun1, PERS, "K-persistent errors")
S["v5"]["constr_ctrl"] = compare("Constraint vs plain rerun", ok_constr, ok_rerun1, CTRL, "matched controls")
P("\n### (b) Candidate-order swap, one call at T=0 (22 Jul 23:14) vs original order re-judged (22 Jul 23:21)")
S["v5"]["swap1_pers"] = compare("Swap T=0 vs plain rerun", ok_swap1, ok_rerun1, PERS, "K-persistent errors")
S["v5"]["swap1_ctrl"] = compare("Swap T=0 vs plain rerun", ok_swap1, ok_rerun1, CTRL, "matched controls")
allids = [x["iid"] for x in T]
S["v5"]["swap1_all"] = compare("Swap T=0 vs plain rerun", ok_swap1, ok_rerun1, allids, "all test instances")

P("\n### (c) Candidate-order swap K=5 (22 Jul) vs fresh original-order K=5 pools 2 and 3 (22 Jul), overlapping items only")
set7 = set(ids7)
ov_p = [i for i in PERS if i in set7]
ov_c = [i for i in CTRL if i in set7]
P(f"- Overlap with the repeated-pool sample: {len(ov_p)} of 37 errors, {len(ov_c)} of 37 controls")
S["v5"]["overlap"] = {"pers": len(ov_p), "ctrl": len(ov_c)}
for pi, rpool in ((2, r2), (3, r3)):
    okp = lambda i, rp_=rpool: modal_first(votes_of(rp_[i]["pool"])) == rp_[i]["gold"]
    S["v5"][f"swap5_pers_pool{pi}"] = compare(f"Swap K=5 vs fresh pool {pi}", ok_swap5, okp, ov_p, "K-persistent errors")
    S["v5"][f"swap5_ctrl_pool{pi}"] = compare(f"Swap K=5 vs fresh pool {pi}", ok_swap5, okp, ov_c, "matched controls")
    anyc = sum(any(v == rpool[i]["gold"] for v in votes_of(rpool[i]["pool"])) for i in ov_p)
    P(f"  - fresh pool {pi}: errors with >=1 correct vote {anyc}/{len(ov_p)}")
rec_swap, rec_con = {i for i in PERS if ok_swap5(i)}, {i for i in PERS if ok_constr(i)}
rec_rerun = {i for i in PERS if ok_rerun1(i)}
P(f"\n- Of the 37 errors: swap K=5 recovered {len(rec_swap)}, constraint {len(rec_con)}, plain T=0 rerun {len(rec_rerun)}; "
  f"recovered by swap or constraint {len(rec_swap | rec_con)}; of those also correct under the plain rerun "
  f"{len((rec_swap | rec_con) & rec_rerun)}; recovered by swap or constraint but not by the plain rerun "
  f"{len((rec_swap | rec_con) - rec_rerun)}; plain rerun only {len(rec_rerun - (rec_swap | rec_con))}")
S["v5"]["sets"] = {"swap": len(rec_swap), "constr": len(rec_con), "rerun": len(rec_rerun),
                   "either": len(rec_swap | rec_con), "either_and_rerun": len((rec_swap | rec_con) & rec_rerun),
                   "either_not_rerun": len((rec_swap | rec_con) - rec_rerun),
                   "rerun_only": len(rec_rerun - (rec_swap | rec_con)),
                   "swap_not_rerun": len(rec_swap - rec_rerun), "constr_not_rerun": len(rec_con - rec_rerun)}
ctrl_flip = {"swap5": sum(not ok_swap5(i) for i in CTRL), "constr": sum(not ok_constr(i) for i in CTRL),
             "rerun1": sum(not ok_rerun1(i) for i in CTRL), "swap1": sum(not ok_swap1(i) for i in CTRL)}
P(f"- Controls incorrect: swap K=5 {ctrl_flip['swap5']}, constraint {ctrl_flip['constr']}, "
  f"plain T=0 rerun {ctrl_flip['rerun1']}, swap T=0 {ctrl_flip['swap1']} (of 37)")
S["v5"]["ctrl_flip"] = {k: int(v) for k, v in ctrl_flip.items()}

# ================================================================= V6
P("\n## V6. Ties, parser routes, and the output cap")
from judges import COT_SYS, COT_USER, _parse_verdict_status
raw_items = {}
for ds in SUB:
    for i, l in enumerate(open(os.path.join(HERE, ds), encoding="utf-8")):
        raw_items[f"{SUB[ds]}_{i:03d}"] = json.loads(l)
cache = json.load(open(os.path.join(HERE, "judge_cache.json"), encoding="utf-8"))
try:
    import tiktoken
    enc = tiktoken.get_encoding("o200k_base")
    ntok = lambda s: len(enc.encode(s))
except Exception:
    ntok = None
found, status, toks, mism = 0, Counter(), [], 0
for x in ALL:
    it = raw_items[x["iid"]]
    key = hashlib.sha1(f"openai|gpt-4o|0.0|{COT_SYS}|{COT_USER.format(x=it['x'], a=it['a'], b=it['b'])}".encode("utf-8")).hexdigest()
    if key in cache:
        found += 1
        raw = cache[key]
        v, st = _parse_verdict_status(raw)
        status[st] += 1
        mism += v != x["methods"]["CoT"]["verdict"]
        if ntok:
            toks.append(ntok(raw))
P(f"- Canonical Single-CoT (T=0) raw outputs recovered from the deterministic cache: {found} of {len(ALL)}; "
  f"parser route {dict(status)}; re-parsed verdict differs from the stored verdict in {mism}")
if toks:
    toks = np.array(toks)
    P(f"- Output length of those raw outputs (o200k_base tokens): median {int(np.median(toks))}, "
      f"95th percentile {int(np.quantile(toks, .95))}, max {int(toks.max())}; at or above 250 tokens: {int((toks >= 250).sum())}")
S["v6"] = {"cot_found": found, "cot_status": dict(status), "cot_mismatch": int(mism),
           "tok_max": (int(toks.max()) if len(toks) else None),
           "tok_ge250": (int((toks >= 250).sum()) if len(toks) else None)}
for cf_ in ("judge_cache.json", "cache_gpt-4o.json", "cache_gpt-4o-mini.json", "cache_gpt-4_1.json"):
    c_ = json.load(open(os.path.join(HERE, cf_), encoding="utf-8"))
    longv = [v for v in c_.values() if isinstance(v, str) and len(v) > 40]
    nov = [v for v in longv if not re.search(r"verdict:\s*([ab]|tie)", v.lower())]
    tk = [ntok(v) for v in longv] if ntok else []
    P(f"- {cf_}: {len(c_)} deterministic entries, {len(longv)} rationale-style outputs; without a final verdict line: "
      f"{len(nov)}" + (f"; at or above 250 tokens: {sum(t >= 250 for t in tk)} (max {max(tk) if tk else 0})" if tk else ""))
    S["v6"][cf_] = {"entries": len(c_), "long": len(longv), "no_verdict": len(nov),
                    "ge250": (int(sum(t >= 250 for t in tk)) if tk else None)}
P("- Stochastic (T=0.7) pool outputs: raw text was not stored in any run (cache bypassed; journals keep parsed verdicts only). "
  "Explicit ties, parser-fallback ties, token-route verdicts, and truncated outputs therefore cannot be separated for the vote pools.")
P("- Runs with parse_status stored: " + "; ".join(
    f"{f}: {dict(Counter(r.get('parse_status') for r in jl(f)))}" for f in ("clean_direct_fc.jsonl", "clean_singlecot.jsonl")))
P("- Tie verdicts by run (parsed label = tie; route unknown): " + "; ".join(
    f"{k} {S[k]['tie_votes']}/{S[k]['n_votes']}" for k in ("llmbar295", "judgebench", "phi4", "qwen", "gemma", "llama")))
P(f"- Greedy tie verdicts: JudgeBench {sum(r['greedy'] == 'tie' for r in jb)}/120; " + "; ".join(
    f"{name} {sum(r['greedy'] == 'tie' for r in ow_rows[key])}/100" for key, _, name in OW))
P(f"- Swapped-order K=5 pools (74 items): tie votes "
  f"{sum(v == 'tie' for i in PERS + CTRL for v in votes_of(sb[i]['verdicts_original_frame']))}/370; "
  f"constraint-focused verdicts = tie: {sum(rb[i]['verdict'] == 'tie' for i in PERS + CTRL)}/74; "
  f"plain T=0 rerun verdicts = tie (all 295): {sum(r['verdict'] == 'tie' for r in cs.values())}")
P("- Max output tokens in the run scripts: holistic CoT 256; terse verdict 8; constraint-focused prompt 512 "
  "(run_causal_probes.py); rewriting calls 1024, re-judged with the 256-token CoT prompt.")
P("- Model strings stored in journals: " + "; ".join(
    f"{f}: {dict(Counter(r.get('model') for r in jl(f)))}" for f in (
        "external_judgebench.jsonl", "repeat_pool_run2.jsonl", "order_swap_stageB.jsonl", "causal_rubric.jsonl")))
P("- Open-weight journals store iid, subset, gold, lengths, greedy verdict, pool, and call count only: "
  "no model digest, quantization tag, context window, timestamp, or raw output.")

# ================================================================= V7
P("\n## V7. Precision by setting (binary-correctness majority)")
P("| Setting | N | Gain | 95% CI | SD of Delta_i | N for half-width 0.02 | N for half-width 0.01 |")
P("|---|---|---|---|---|---|---|")
S["v7"] = {}
for k in six + ["gpt4o_c100"]:
    sd = S[k]["sd_delta_binary"]
    n2, n1 = int(np.ceil((1.96 * sd / 0.02) ** 2)), int(np.ceil((1.96 * sd / 0.01) ** 2))
    g = S[k]["gain_binary"]
    S["v7"][k] = {"sd": sd, "n_hw02": n2, "n_hw01": n1}
    P(f"| {k} | {S[k]['N']} | {g[0]:+.4f} | [{g[1]:+.3f}, {g[2]:+.3f}] | {sd:.3f} | {n2} | {n1} |")
g = S["llmbar295"]["gain_binary"]
P(f"\n- Calls per accuracy-equivalent decision at K=5 vs K=1 (4 extra calls / gain): point {4 / g[0]:.0f}; "
  f"at the upper 95% bound {4 / g[2]:.0f}; undefined or negative at the lower bound ({g[1]:+.4f})")
P(f"- Early stopping 3->5 on the canonical pool: mean calls {S['llmbar295']['early_stop_calls']:.3f}, "
  f"reduction vs 5 = {100 * (1 - S['llmbar295']['early_stop_calls'] / 5):.1f}%, vs K=1 = "
  f"{S['llmbar295']['early_stop_calls']:.2f}x")


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return [clean(v) for v in o.tolist()]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


J["settings"] = clean(S)
open(OUT_MD, "w", encoding="utf-8").write("\n".join(L) + "\n")
json.dump(J, open(OUT_JS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("\n".join(L))
