# -*- coding: utf-8 -*-
"""Compute every number used by the v2.1 figures from the raw per-instance logs.

Output: figure_data_v21.json (read by the figNN_*.py scripts).
No API calls. Run:  python -X utf8 build_data_v21.py
All values are cross-checked against the manuscript tables by check_against_manuscript().
"""
import csv, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.normpath(os.path.join(HERE, "..", "..", "code"))
OUT = os.path.join(HERE, "figure_data_v21.json")
SUB = {"full_natural.jsonl": "Natural", "full_neighbor.jsonl": "Neighbor", "full_gptinst.jsonl": "GPTInst",
       "full_gptout.jsonl": "GPTOut", "full_manual.jsonl": "Manual"}
K = 5
B = 5000

def jl(name):
    return [json.loads(l) for l in open(os.path.join(CODE, name), encoding="utf-8") if l.strip()]
def votes_of(s):
    return [v.strip(" '\"") for v in s.strip("[]").split(",")] if isinstance(s, str) else list(s)
def boot_ci(x, seed=0):
    x = np.asarray(x, float); rng = np.random.default_rng(seed)
    ix = rng.integers(0, len(x), (B, len(x))); m = x[ix].mean(1)
    return float(np.quantile(m, .025)), float(np.quantile(m, .975))

rec = json.load(open(os.path.join(CODE, "records.json"), encoding="utf-8"))
by, T = {}, []
for ds, items in rec["datasets"].items():
    for i, x in enumerate(items):
        iid = f"{SUB[ds]}_{i:03d}"; x = dict(x); x["iid"] = iid; by[iid] = x
        if x["split"] == "test": T.append(x)
N = len(T)

def decompose(c, K=K, mode="binary", pools=None):
    """c: array of correct-vote counts. Returns dict of masses (binary majority)."""
    c = np.asarray(c); d = (c > K / 2).astype(float) - c / K
    return {"n": int(len(c)), "R_gold": float(((c > 0) & (c < K)).mean()), "gain": float(d.mean()),
            "pos": float(np.clip(d, 0, None).mean()), "neg": float(np.clip(-d, 0, None).mean()),
            "R_pers": float((c == 0).mean()), "ci": boot_ci(d)}

D = {}
# ---------- LLMBar canonical test split ----------
c_llmbar = np.array([sum(v == x["gold"] for v in x["methods"]["SelfConsistency"]["votes"]) for x in T])
D["llmbar"] = decompose(c_llmbar)
D["llmbar"]["counts"] = [int((c_llmbar == k).sum()) for k in range(K + 1)]
D["llmbar"]["pos_cases"] = D["llmbar"]["pos"] * N; D["llmbar"]["neg_cases"] = D["llmbar"]["neg"] * N

# ---------- JudgeBench (binary and deployed modal) ----------
jb = jl("external_judgebench.jsonl")
poolj = lambda r: votes_of(r["pool"])
cj = np.array([sum(v == r["gold"] for v in poolj(r)) for r in jb])
D["judgebench"] = decompose(cj)
modal = np.array([float(Counter(poolj(r)).most_common(1)[0][0] == r["gold"]) for r in jb])
dm = modal - cj / K
D["judgebench"]["modal"] = {"gain": float(dm.mean()), "pos": float(np.clip(dm, 0, None).mean()), "neg": float(np.clip(-dm, 0, None).mean()), "ci": boot_ci(dm)}

# ---------- open-weight judges ----------
D["open"] = {}
for f, name in (("ollama_phi4_14b.jsonl", "Phi-4 14B"), ("ollama_qwen2_5_7b.jsonl", "Qwen2.5 7B"),
                ("ollama_gemma3_12b.jsonl", "Gemma 3 12B"), ("ollama_llama3_1_8b.jsonl", "Llama 3.1 8B")):
    R = jl(f); c = np.array([sum(v == r["gold"] for v in r["pool"]) for r in R]); d = decompose(c)
    mm = np.array([float(Counter(r["pool"]).most_common(1)[0][0] == r["gold"]) for r in R]) - c / K
    d["modal"] = {"gain": float(mm.mean()), "pos": float(np.clip(mm, 0, None).mean()), "neg": float(np.clip(-mm, 0, None).mean()), "ci": boot_ci(mm)}
    D["open"][name] = d

# ---------- independent pools on the 100-instance sample ----------
ids = json.load(open(os.path.join(CODE, "splits", "repeat_pool_100_ids.json"), encoding="utf-8"))["ids"]
r2 = {r["iid"]: r for r in jl("repeat_pool_run2.jsonl")}; r3 = {r["iid"]: r for r in jl("repeat_pool_run3.jsonl")}
pools = [[by[i]["methods"]["SelfConsistency"]["votes"] for i in ids], [votes_of(r2[i]["pool"]) for i in ids], [votes_of(r3[i]["pool"]) for i in ids]]
gold = [by[i]["gold"] for i in ids]
cc = np.array([[sum(v == g for v in p[j]) for j, g in enumerate(gold)] for p in pools])  # 3 x 100
D["pools"] = [decompose(cc[k]) for k in range(3)]
anyp = np.where((cc == 0).any(axis=0))[0]
rows = [{"id": ids[j], "c": [int(cc[k, j]) for k in range(3)]} for j in anyp]
rows.sort(key=lambda r: (-sum(r["c"]), -max(r["c"]), r["id"]))
D["persistence_matrix"] = rows
D["persistence_summary"] = {"n_any": int(len(anyp)), "n_all_three": int(((cc == 0).all(axis=0)).sum()),
                            "canon_persistent": int((cc[0] == 0).sum()),
                            "canon_got_any": int(((cc[0] == 0) & ((cc[1] > 0) | (cc[2] > 0))).sum()),
                            "canon_got_majority": int(((cc[0] == 0) & ((cc[1] >= 3) | (cc[2] >= 3))).sum())}

# ---------- interventions on the 37 unanimous K-persistent errors ----------
rows_csv = [r for r in csv.DictReader(open(os.path.join(CODE, "results_final.csv"), encoding="utf-8")) if r["strategy"] == "SC-CoT" and r["split"] == "test"]
for r in rows_csv:
    vs = r["votes"].split("|"); r["unan"] = len(set(vs)) == 1; r["c_i"] = sum(v == r["gold_label"] for v in vs)
    r["gap"] = abs(int(r["candidate_A_length"]) - int(r["candidate_B_length"]))
pers = [r for r in rows_csv if r["unan"] and r["c_i"] == 0]; stab = [r for r in rows_csv if r["c_i"] == 5]
gaps = sorted(r["gap"] for r in rows_csv); t1, t2 = gaps[len(gaps) // 3], gaps[2 * len(gaps) // 3]
ter = lambda g: 0 if g <= t1 else (1 if g <= t2 else 2)
rr = np.random.default_rng(0); used, pairs = set(), []
for p in pers:
    pool = [s for s in stab if s["subset"] == p["subset"] and s["gold_label"] == p["gold_label"] and ter(s["gap"]) == ter(p["gap"]) and s["instance_id"] not in used]
    if not pool:
        pool = [s for s in stab if s["subset"] == p["subset"] and s["gold_label"] == p["gold_label"] and s["instance_id"] not in used]
    if pool:
        m = pool[int(rr.integers(len(pool)))]; used.add(m["instance_id"]); pairs.append((p["instance_id"], m["instance_id"]))
sb = {r["iid"]: r for r in jl("order_swap_stageB.jsonl")}
rb = {r["iid"]: r for r in jl("causal_rubric.jsonl")}
cf = {r["iid"]: r for r in jl("causal_counterfactual.jsonl")}
maj = lambda r: Counter(votes_of(r["verdicts_original_frame"])).most_common(1)[0][0] == r["gold"]
order_rec = {p for p, _ in pairs if maj(sb[p])}; cons_rec = {p for p, _ in pairs if rb[p]["verdict"] == rb[p]["gold"]}
len_rec = {i for i, r in cf.items() if r["verdict_after_compress"] == r["gold"] or r["verdict_after_expand"] == r["gold"]}
len_tested = set(cf)
cats = {"order only": 0, "constraint only": 0, "both": 0, "none": 0}
for p, _ in pairs:
    o, c_ = p in order_rec, p in cons_rec
    cats["both" if o and c_ else "order only" if o else "constraint only" if c_ else "none"] += 1
len_by = Counter(("both" if (p in order_rec and p in cons_rec) else "order only" if p in order_rec else "constraint only" if p in cons_rec else "none") for p in len_rec)
D["interventions"] = {"n": len(pairs), "order": len(order_rec), "constraint": len(cons_rec), "length": len(len_rec), "length_tested": len(len_tested),
                      "categories": cats, "length_by_category": dict(len_by),
                      "control_flip_order": int(sum(1 for _, m in pairs if not maj(sb[m]))),
                      "control_flip_constraint": int(sum(1 for _, m in pairs if rb[m]["verdict"] != rb[m]["gold"])),
                      "union": len(order_rec | cons_rec), "none": 37 - len(order_rec | cons_rec | len_rec)}

# ---------- K-sweep (independent executions) ----------
ks = {}
for f, name in (("ksweep_gpt-4o.json", "GPT-4o"), ("ksweep_gpt-4o-mini.json", "gpt-4o-mini")):
    rows_k = json.load(open(os.path.join(CODE, f), encoding="utf-8"))["rows"]
    ks[name] = [{"K": r["K"], "acc": r["accuracy"], "calls": r["calls_per_decision"]} for r in rows_k]
def wilson(p, n, z=1.96):
    den = 1 + z * z / n; ctr = (p + z * z / (2 * n)) / den; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(ctr - h), float(ctr + h)
for name in ks:
    for r in ks[name]: r["lo"], r["hi"] = wilson(r["acc"], N)
D["ksweep"] = ks
c_mat = np.array([[sum(v == x["gold"] for v in x["methods"]["SelfConsistency"]["votes"][:0] or x["methods"]["SelfConsistency"]["votes"])] for x in T])
# shared-pool expectations K=1,3,5 (binary majority over all subsets)
from itertools import combinations
sh = {1: [], 3: [], 5: []}
for x in T:
    v = x["methods"]["SelfConsistency"]["votes"]; g = x["gold"]
    for k in (1, 3, 5):
        sh[k].append(np.mean([sum(v[j] == g for j in s) > k / 2 for s in combinations(range(5), k)]))
D["shared_pool"] = {str(k): {"acc": float(np.mean(a)), "ci": boot_ci(a)} for k, a in sh.items()}

# ---------- early stopping ----------
pref3 = np.array([len(set(x["methods"]["SelfConsistency"]["votes"][:3])) == 1 for x in T])
D["early_stop"] = {"n3": int(pref3.sum()), "n5": int((~pref3).sum()), "mean_calls": float(np.where(pref3, 3, 5).mean())}

# ---------- risk-coverage inputs ----------
rc = []
for x in T:
    sc = x["methods"]["SelfConsistency"]; mp = x["methods"]["MPCJ-CoT"]
    rc.append({"sc_u": float(sc["u"]), "sc_margin": int(sc["margin"]), "sc_ok": bool(sc["correct"]), "mp_u": float(mp["u"]), "mp_ok": bool(mp["correct"])})
D["risk_coverage"] = rc

json.dump(D, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------- cross-check against manuscript numbers ----------
def chk(name, got, want, tol=6e-4):
    ok = abs(got - want) <= tol
    print(("OK   " if ok else "FAIL ") + f"{name}: {got:.4f} (manuscript {want})")
    return ok
ok = True
ok &= chk("LLMBar R_gold", D["llmbar"]["R_gold"], 0.119); ok &= chk("LLMBar gain", D["llmbar"]["gain"], 0.0095)
ok &= chk("LLMBar pos cases", D["llmbar"]["pos_cases"], 6.40, 5e-3); ok &= chk("LLMBar neg cases", D["llmbar"]["neg_cases"], 3.60, 5e-3)
ok &= chk("JudgeBench R_gold", D["judgebench"]["R_gold"], 0.408); ok &= chk("JB binary gain", D["judgebench"]["gain"], -0.002, 1e-3)
ok &= chk("JB modal gain", D["judgebench"]["modal"]["gain"], 0.007, 1e-3)
for nm, rg, gb, gm in (("Phi-4 14B", .41, .012, .012), ("Qwen2.5 7B", .40, -.012, .028), ("Gemma 3 12B", .21, .016, .016), ("Llama 3.1 8B", .59, -.004, -.004)):
    ok &= chk(nm + " R_gold", D["open"][nm]["R_gold"], rg, 6e-3); ok &= chk(nm + " gain binary", D["open"][nm]["gain"], gb, 1e-3); ok &= chk(nm + " gain modal", D["open"][nm]["modal"]["gain"], gm, 1e-3)
for k, (rg, gn) in enumerate(((.14, .010), (.20, .026), (.19, .010))):
    ok &= chk(f"pool {k+1} R_gold", D["pools"][k]["R_gold"], rg, 6e-3); ok &= chk(f"pool {k+1} gain", D["pools"][k]["gain"], gn, 1e-3)
S = D["persistence_summary"]; print("persistence:", S)
assert (S["n_any"], S["n_all_three"], S["canon_persistent"], S["canon_got_any"], S["canon_got_majority"]) == (15, 7, 13, 6, 2)
I = D["interventions"]; print("interventions:", I)
assert (I["n"], I["order"], I["constraint"], I["length"], I["union"], I["none"]) == (37, 13, 13, 8, 20, 17), I
assert (I["control_flip_order"], I["control_flip_constraint"]) == (4, 4)
for nm in ks: print("ksweep", nm, [(r["K"], r["acc"]) for r in ks[nm]])
print("shared pool:", {k: round(v["acc"], 4) for k, v in D["shared_pool"].items()}, "| early stop:", D["early_stop"])
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
