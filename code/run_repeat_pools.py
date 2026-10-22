# -*- coding: utf-8 -*-
"""Plan D (5th review, priority 8): quantify API nondeterminism with repeated pools.

100 held-out test instances (20 per subset, pre-registered seed 7) x K=5 (T=0.7)
x 2 additional independent runs.                                     ~1,000 calls

Key question: does the persistent-error set stay stable across independent pools?
Reports (offline, analyze side): per-run K=1/K=5/gain/unanimity/persistent rate,
Jaccard similarity of persistent sets across runs, run-clustered bootstrap.
Resume-safe journals: repeat_pool_run2[_mock].jsonl, repeat_pool_run3[_mock].jsonl.
Usage: python run_repeat_pools.py [--mock] [--workers 4] [--model gpt-4o]
"""
from __future__ import annotations
import argparse, json, os, random, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from judges import Judge, JudgeConfig, MockJudge
from run_ollama_judge import load_test

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, "splits", "repeat_pool_100_ids.json")
SEED, PER_SUBSET = 7, 20


def sample_100():
    if os.path.exists(IDS):
        keep = set(json.load(open(IDS, encoding="utf-8"))["ids"])
        return [i for s in load_test() for i in s if i["iid"] in keep]
    rng = random.Random(SEED)
    picked = []
    for s in load_test():
        picked += rng.sample(s, min(PER_SUBSET, len(s)))
    os.makedirs(os.path.dirname(IDS), exist_ok=True)
    json.dump({"seed": SEED, "per_subset": PER_SUBSET,
               "ids": sorted(i["iid"] for i in picked)}, open(IDS, "w", encoding="utf-8"), indent=1)
    print(f"pre-registered {len(picked)} ids -> {IDS}")
    return picked


def run_one_pool(run_id, insts, args):
    path = os.path.join(HERE, f"repeat_pool_run{run_id}{'_mock' if args.mock else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    todo = [i for i in insts if i["iid"] not in done]
    print(f"[run{run_id}] {len(todo)} to run ({len(done)} done)", flush=True)

    def one(inst):
        j = MockJudge() if args.mock else Judge(JudgeConfig(model=args.model, cache_path=None))
        x, a, b = inst["x"], inst["a"], inst["b"]
        t0 = time.time()
        pool = [j.cot_pairwise(x, a, b, temperature=0.7) for _ in range(5)]
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": inst["gold"],
                "run_id": run_id, "pool": pool,
                "model": args.model if not args.mock else "mock",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": round(time.time()-t0, 2)}

    lock = threading.Lock(); t0 = time.time(); ok = err = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex, open(path, "a", encoding="utf-8") as jf:
        futs = {ex.submit(one, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock: jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                ok += 1
            except Exception as e:
                err += 1; print(f"  ERROR {futs[fut]['iid']}: {e}", flush=True)
    print(f"[run{run_id}] done ok={ok} err={err} in {(time.time()-t0)/60:.1f} min -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    insts = sample_100()
    run_one_pool(2, insts, args)
    run_one_pool(3, insts, args)


if __name__ == "__main__":
    main()
