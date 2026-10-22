# -*- coding: utf-8 -*-
"""Run all judging strategies on full LLMBar and SAVE per-item records (for analyses #8-#13).
Output: records.json  (per dataset: items + per-method per-item verdict/u/correct/votes; dev/test split)
Usage: python run_records.py --judge gpt-4o   |   --mock (no API)
"""
from __future__ import annotations
import argparse, json, random
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
from judges import Judge, JudgeConfig, MockJudge
from mpcj import (mpcj_pairwise, vanilla_pairwise, swap_only, cot_holistic, self_committee)

DATASETS = ["full_natural.jsonl", "full_neighbor.jsonl", "full_gptinst.jsonl",
            "full_gptout.jsonl", "full_manual.jsonl"]

def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    judge = MockJudge() if args.mock else Judge(JudgeConfig(model=args.judge))

    out = {"judge": (args.judge if not args.mock else "mock"), "K": args.K, "datasets": {}}
    for path in DATASETS:
        items = load(path)
        rng = random.Random(args.seed); idx = list(range(len(items))); rng.shuffle(idx)
        n_dev = max(1, int(len(items) * args.dev_frac))
        dev_set = set(idx[:n_dev])
        recs = []
        for i, it in enumerate(items):
            g = it["gold"]; x, a, b = it["x"], it["a"], it["b"]
            r = {"gold": g, "len_a": len(a), "len_b": len(b),
                 "split": ("dev" if i in dev_set else "test")}
            mv = {}
            van = vanilla_pairwise(judge, x, a, b)
            mv["Vanilla"] = {"verdict": van["verdict"]}
            sw = swap_only(judge, x, a, b)
            mv["Swap-BPC"] = {"verdict": sw["verdict"]}
            ct = cot_holistic(judge, x, a, b)
            mv["CoT"] = {"verdict": ct["verdict"]}
            sc = self_committee(judge, x, a, b, K=args.K)
            mv["SelfConsistency"] = {"verdict": sc["verdict"], "u": sc["u"],
                                     "margin": sc["margin"], "votes": sc["votes"]}
            mc = mpcj_pairwise(judge, x, a, b, cot=True, tau=1.0)  # tau=1 -> never abstain here
            mv["MPCJ-CoT"] = {"verdict": mc["verdict"], "u": mc["u"],
                              "U_crit": mc["U_crit"], "U_ord": mc["U_ord"]}
            for m in mv:
                mv[m]["correct"] = (mv[m]["verdict"] == g)
            r["methods"] = mv
            recs.append(r)
        out["datasets"][path] = recs
        print(f"{path}: {len(items)} items done")
    out["cost"] = {"logical": judge.calls, "billed": judge.api_calls, "tokens": judge.tokens}
    json.dump(out, open("records.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("saved records.json | cost:", out["cost"])

if __name__ == "__main__":
    main()
