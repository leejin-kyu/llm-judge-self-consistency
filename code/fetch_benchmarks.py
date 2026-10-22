# -*- coding: utf-8 -*-
"""Download the benchmark texts that this repository does not redistribute.

LLMBar (Zeng et al., 2024): github.com/princeton-nlp/LLMBar -> full_natural.jsonl, full_neighbor.jsonl,
  full_gptinst.jsonl, full_gptout.jsonl, full_manual.jsonl (instance i of a file is <Subset>_<i:03d>).
JudgeBench (Tan et al., 2025), GPT-4o split: huggingface.co/datasets/ScalerLab/JudgeBench -> judgebench_full.jsonl,
  judgebench_120.jsonl. The 120-pair sample is re-drawn with the pre-specified seed and must reproduce
  splits/judgebench_120_ids.json exactly; the released id file is never overwritten.
Usage (from the code folder): python fetch_benchmarks.py
"""
import json, os, random
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
import prep_data
import prep_judgebench as pj

for subset in ("Natural", "Neighbor", "GPTInst", "GPTOut", "Manual"):
    items = prep_data.fetch_llmbar(subset)
    path = os.path.join(HERE, f"full_{subset.lower()}.jsonl")
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"LLMBar {subset}: {len(items)} -> {os.path.basename(path)}")

rows = pj.fetch_all()
with open(pj.FULL, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
insts = []
for r in rows:
    lab = (r.get("label") or "").strip().upper()
    gold = "A" if lab.startswith("A") else ("B" if lab.startswith("B") else None)
    if gold is not None:
        insts.append({"iid": f"JB_{r['pair_id']}", "subset": r["source"], "x": r["question"],
                      "a": r["response_A"], "b": r["response_B"], "gold": gold})
by_src = Counter(i["subset"] for i in insts)
rng = random.Random(pj.SEED)
picked = []
for src in sorted(by_src):
    pool = [i for i in insts if i["subset"] == src]
    picked += rng.sample(pool, min(max(1, round(pj.N * len(pool) / len(insts))), len(pool)))
picked = picked[:pj.N]
released = json.load(open(pj.IDS, encoding="utf-8"))["ids"]
assert sorted(i["iid"] for i in picked) == sorted(released), "JudgeBench sample differs from splits/judgebench_120_ids.json"
with open(pj.OUT, "w", encoding="utf-8") as f:
    for i in picked:
        f.write(json.dumps(i, ensure_ascii=False) + "\n")
print(f"JudgeBench: {len(rows)} rows; sample of {len(picked)} matches the released identifiers")
