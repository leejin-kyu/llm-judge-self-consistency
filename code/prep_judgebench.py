# -*- coding: utf-8 -*-
"""Plan C (5th review, priority 6): prepare the external-benchmark sample.

Downloads JudgeBench (ScalerLab/JudgeBench, 'gpt' split — GPT-4o-generated
response pairs with objective correctness labels) via the HuggingFace
datasets-server rows API, then draws a stratified 120-instance sample by source
category with a PRE-REGISTERED seed (42), saved BEFORE any judging runs.

No prompt tuning is permitted on this benchmark (analysis protocol). Primary
endpoints are K5−K1, disagreement mass, and persistent-error mass — not accuracy.

Outputs: judgebench_full.jsonl (all rows), judgebench_120.jsonl (the sample),
splits/judgebench_120_ids.json (pre-registered id list).
Usage: python prep_judgebench.py
"""
from __future__ import annotations
import json, os, random, urllib.request
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
FULL = os.path.join(HERE, "judgebench_full.jsonl")
OUT = os.path.join(HERE, "judgebench_120.jsonl")
IDS = os.path.join(HERE, "splits", "judgebench_120_ids.json")
SEED = 42
N = 120
API = ("https://datasets-server.huggingface.co/rows?dataset=ScalerLab%2FJudgeBench"
       "&config=default&split=gpt&offset={off}&length=100")


def fetch_all():
    rows, off = [], 0
    while True:
        with urllib.request.urlopen(API.format(off=off), timeout=60) as r:
            data = json.loads(r.read())
        batch = [x["row"] for x in data.get("rows", [])]
        if not batch:
            break
        rows += batch
        off += len(batch)
        if off >= data.get("num_rows_total", 0):
            break
    return rows


def main():
    if os.path.exists(FULL):
        rows = [json.loads(l) for l in open(FULL, encoding="utf-8")]
        print(f"cached: {len(rows)} rows")
    else:
        rows = fetch_all()
        with open(FULL, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"downloaded: {len(rows)} rows -> {FULL}")

    insts = []
    for r in rows:
        lab = (r.get("label") or "").strip().upper()
        gold = "A" if lab.startswith("A") else ("B" if lab.startswith("B") else None)
        if gold is None:
            continue
        insts.append({"iid": f"JB_{r['pair_id']}", "subset": r["source"],
                      "x": r["question"], "a": r["response_A"], "b": r["response_B"],
                      "gold": gold})
    by_src = Counter(i["subset"] for i in insts)
    print("sources:", dict(by_src))

    # proportional stratified sample, pre-registered seed
    rng = random.Random(SEED)
    total = len(insts)
    picked = []
    for src in sorted(by_src):
        pool = [i for i in insts if i["subset"] == src]
        k = max(1, round(N * len(pool) / total))
        picked += rng.sample(pool, min(k, len(pool)))
    picked = picked[:N]
    with open(OUT, "w", encoding="utf-8") as f:
        for i in picked:
            f.write(json.dumps(i, ensure_ascii=False) + "\n")
    os.makedirs(os.path.dirname(IDS), exist_ok=True)
    json.dump({"seed": SEED, "n": len(picked), "ids": sorted(i["iid"] for i in picked)},
              open(IDS, "w", encoding="utf-8"), indent=1)
    print(f"sample: {len(picked)} -> {OUT}; ids pre-registered -> {IDS}")
    print("sample by source:", dict(Counter(i['subset'] for i in picked)))


if __name__ == "__main__":
    main()
