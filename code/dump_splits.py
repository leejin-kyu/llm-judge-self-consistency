# -*- coding: utf-8 -*-
"""P5-33: Dump the exact seed-0 dev/test split instance ids to splits/instance_ids.json.

Reproduces the split logic used by run_records.py / run_backbone.py:
per subset file, random.Random(seed).shuffle(indices); first 30% = dev.
"""
import json, os, random

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["full_natural.jsonl", "full_neighbor.jsonl", "full_gptinst.jsonl",
            "full_gptout.jsonl", "full_manual.jsonl"]
SUBSET = {"full_natural.jsonl": "Natural", "full_neighbor.jsonl": "Neighbor",
          "full_gptinst.jsonl": "GPTInst", "full_gptout.jsonl": "GPTOut",
          "full_manual.jsonl": "Manual"}
SEED, DEV_FRAC = 0, 0.3

out = {"seed": SEED, "dev_frac": DEV_FRAC, "splits": {}}
for path in DATASETS:
    items = [l for l in open(os.path.join(HERE, path), encoding="utf-8") if l.strip()]
    idx = list(range(len(items)))
    random.Random(SEED).shuffle(idx)
    dev = sorted(idx[:max(1, int(len(items) * DEV_FRAC))])
    test = sorted(set(range(len(items))) - set(dev))
    sub = SUBSET[path]
    out["splits"][sub] = {
        "n": len(items),
        "dev": [f"{sub}_{i:03d}" for i in dev],
        "test": [f"{sub}_{i:03d}" for i in test],
    }
os.makedirs(os.path.join(HERE, "splits"), exist_ok=True)
dst = os.path.join(HERE, "splits", "instance_ids.json")
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
n_dev = sum(len(v["dev"]) for v in out["splits"].values())
n_test = sum(len(v["test"]) for v in out["splits"].values())
print(f"wrote {dst}: dev={n_dev}, test={n_test}")
