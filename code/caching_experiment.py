# -*- coding: utf-8 -*-
"""#12 Caching pitfall as a reproducibility experiment.
Compare self-consistency (K) under two cache policies on N items:
  - cache_stochastic=True  : temp>0 calls are cached -> collapse to 1 unique output
  - cache_stochastic=False : temp>0 calls bypass cache -> genuine K samples
Metrics: mean unique outputs among K, mean vote entropy, accuracy.
Usage: python caching_experiment.py --judge gpt-4o --n 40
"""
from __future__ import annotations
import argparse, json, math
from collections import Counter
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
from judges import Judge, JudgeConfig
from mpcj import self_committee

def entropy(votes):
    c = Counter(votes); n = len(votes); h = 0.0
    for k in c:
        p = c[k]/n; h -= p*math.log(p, 2)
    return h

def run(items, cache_stochastic, K, model):
    # separate cache file so the two policies don't interfere
    cp = "cache_stoch.json" if cache_stochastic else "cache_bypass.json"
    j = Judge(JudgeConfig(model=model, cache_path=cp, cache_stochastic=cache_stochastic))
    uniq, ent, corr = [], [], 0
    for it in items:
        r = self_committee(j, it["x"], it["a"], it["b"], K=K)
        uniq.append(len(set(r["votes"]))); ent.append(entropy(r["votes"]))
        corr += (r["verdict"] == it["gold"])
    return {"policy": ("cached-stochastic (buggy)" if cache_stochastic else "cache-bypassed (fixed)"),
            "mean_unique_outputs": round(sum(uniq)/len(uniq), 3),
            "mean_vote_entropy": round(sum(ent)/len(ent), 3),
            "accuracy": round(corr/len(items), 4), "billed_calls": j.api_calls}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o"); ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--K", type=int, default=5); ap.add_argument("--data", default="full_neighbor.jsonl")
    args = ap.parse_args()
    items = [json.loads(l) for l in open(args.data, encoding="utf-8") if l.strip()][:args.n]
    res = [run(items, True, args.K, args.judge), run(items, False, args.K, args.judge)]
    print("\n# #12 Caching pitfall experiment (N=%d, K=%d, %s)" % (len(items), args.K, args.judge))
    print("| Cache policy | mean unique outputs / K | mean vote entropy | accuracy | billed |")
    print("|---|---|---|---|---|")
    for r in res:
        print(f"| {r['policy']} | {r['mean_unique_outputs']}/{args.K} | {r['mean_vote_entropy']} | {r['accuracy']} | {r['billed_calls']} |")
    json.dump(res, open("caching_experiment.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("\nsaved caching_experiment.json")

if __name__ == "__main__":
    main()
