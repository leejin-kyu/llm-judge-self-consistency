# -*- coding: utf-8 -*-
"""P0-1: Flatten records.json (canonical GPT-4o main run, 2026-06-15) into results_final.csv.

One row per (instance, strategy). All manuscript tables/numbers must be regenerated
from this file via analyze_final.py — never hand-copied.

Canonical strategy names (P4-32):
  Direct          <- Vanilla
  DTT-Sym         <- Swap-BPC  (disagreement-to-tie order symmetrization)
  Decomp          <- MPCJ-CoT  (lightweight perspective decomposition)
  Single-CoT      <- CoT
  SC-CoT          <- SelfConsistency (K=5, T=0.7)
"""
from __future__ import annotations
import csv, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RECORDS = os.path.join(HERE, "records.json")
OUT = os.path.join(HERE, "results_final.csv")

NAME_MAP = {
    "Vanilla": "Direct",
    "Swap-BPC": "DTT-Sym",
    "MPCJ-CoT": "Decomp",
    "CoT": "Single-CoT",
    "SelfConsistency": "SC-CoT",
}
SUBSET_MAP = {
    "full_natural.jsonl": "Natural",
    "full_neighbor.jsonl": "Neighbor",
    "full_gptinst.jsonl": "GPTInst",
    "full_gptout.jsonl": "GPTOut",
    "full_manual.jsonl": "Manual",
}

def main():
    r = json.load(open(RECORDS, encoding="utf-8"))
    judge, K = r["judge"], r["K"]
    rows = []
    for fname, items in r["datasets"].items():
        subset = SUBSET_MAP[fname]
        for i, it in enumerate(items):
            iid = f"{subset}_{i:03d}"
            gold, la, lb, split = it["gold"], it["len_a"], it["len_b"], it["split"]
            for old, m in it["methods"].items():
                strat = NAME_MAP[old]
                votes = m.get("votes")
                rows.append({
                    "instance_id": iid, "subset": subset, "split": split,
                    "gold_label": gold, "backbone": judge, "strategy": strat,
                    "temperature": 0.7 if strat == "SC-CoT" else 0.0,
                    "K": K if strat == "SC-CoT" else 1,
                    "run_id": "main_20260615",
                    "predicted_label": m["verdict"],
                    "correct": int(bool(m["correct"])),
                    "candidate_A_length": la, "candidate_B_length": lb,
                    "vote_margin": m.get("margin", ""),
                    "uncertainty": m.get("u", ""),
                    "votes": "|".join(votes) if votes else "",
                })
    cols = list(rows[0].keys())
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    n_inst = len({x["instance_id"] for x in rows})
    n_test = len({x["instance_id"] for x in rows if x["split"] == "test"})
    print(f"wrote {OUT}: {len(rows)} rows, {n_inst} instances ({n_test} test), cost={r['cost']}")

if __name__ == "__main__":
    main()
