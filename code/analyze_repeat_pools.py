# -*- coding: utf-8 -*-
"""Plan D analysis: API-nondeterminism across three independent K=5 pools.

Run 1 = canonical records.json pools (subset of 100 pre-registered ids);
runs 2, 3 = repeat_pool_run2/3.jsonl. Key question: does the persistent-error
set stay stable across independent pools?  Output: repeat_pools_analysis.md
"""
from __future__ import annotations
import csv, json, os
from collections import Counter
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    keep = set(json.load(open(os.path.join(HERE, "splits", "repeat_pool_100_ids.json"),
                              encoding="utf-8"))["ids"])
    runs = {}
    # run 1: canonical
    rows = [r for r in csv.DictReader(open(os.path.join(HERE, "results_final.csv"), encoding="utf-8"))
            if r["strategy"] == "SC-CoT" and r["instance_id"] in keep]
    runs[1] = {r["instance_id"]: {"pool": r["votes"].split("|"), "gold": r["gold_label"]} for r in rows}
    for rid in (2, 3):
        p = os.path.join(HERE, f"repeat_pool_run{rid}.jsonl")
        if os.path.exists(p):
            runs[rid] = {r["iid"]: {"pool": r["pool"], "gold": r["gold"]}
                         for r in (json.loads(l) for l in open(p, encoding="utf-8"))}
    ids = sorted(set.intersection(*[set(v) for v in runs.values()]))
    def maj(vs): return Counter(vs).most_common(1)[0][0]

    L = [f"# Repeated-pool nondeterminism analysis (N={len(ids)}, {len(runs)} independent K=5 pools)", ""]
    stats_rows, pers_sets, gains = [], {}, []
    for rid in sorted(runs):
        R = runs[rid]
        c = np.array([sum(v == R[i]["gold"] for v in R[i]["pool"]) for i in ids])
        k5 = np.array([float(maj(R[i]["pool"]) == R[i]["gold"]) for i in ids])
        gain = (k5 - c / 5).mean()
        unan = np.mean([len(set(R[i]["pool"])) == 1 for i in ids])
        pers_sets[rid] = {i for i in ids if sum(v == R[i]["gold"] for v in R[i]["pool"]) == 0}
        gains.append(gain)
        stats_rows.append((rid, (c/5).mean(), k5.mean(), gain, unan, len(pers_sets[rid])))
    L += ["| Run | K=1 exp. | K=5 | Voting gain | Unanimity | Persistent (c_i=0) |",
          "|---|---|---|---|---|---|"]
    for rid, k1, k5m, g, u, np_ in stats_rows:
        L.append(f"| {'1 (canonical)' if rid==1 else rid} | {k1:.3f} | {k5m:.3f} | "
                 f"{g:+.4f} | {u:.0%} | {np_} |")
    L.append("")

    # Jaccard of persistent sets
    pairs = [(a, b) for a in sorted(pers_sets) for b in sorted(pers_sets) if a < b]
    L += ["## Persistent-set stability", "",
          "| Pair | Jaccard | intersection | union |", "|---|---|---|---|"]
    for a, b in pairs:
        inter = pers_sets[a] & pers_sets[b]; uni = pers_sets[a] | pers_sets[b]
        L.append(f"| run{a} vs run{b} | {len(inter)/max(len(uni),1):.2f} | {len(inter)} | {len(uni)} |")
    core = set.intersection(*pers_sets.values())
    any_ = set.union(*pers_sets.values())
    L += ["",
          f"- Persistent in ALL {len(runs)} runs (core set): {len(core)}",
          f"- Persistent in at least one run: {len(any_)}",
          f"- Core / union = {len(core)/max(len(any_),1):.0%} — the manuscript may say "
          f"'systematically persistent' for the core set; cases persistent in only one "
          f"run are 'sample-persistent within one pool'.",
          f"- Voting gain across runs: mean {np.mean(gains):+.4f}, range "
          f"[{min(gains):+.4f}, {max(gains):+.4f}]", ""]
    out = os.path.join(HERE, "repeat_pools_analysis.md")
    open(out, "w", encoding="utf-8").write("\n".join(L))
    print("wrote", out)


if __name__ == "__main__":
    main()
