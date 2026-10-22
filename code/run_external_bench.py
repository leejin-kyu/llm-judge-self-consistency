# -*- coding: utf-8 -*-
"""Plan C runner: K=5 CoT pool (T=0.7) + 1 greedy CoT on the pre-registered
JudgeBench 120-sample (prep_judgebench.py must run first).            ~720 calls

Offline analysis (analyze_reachability-style) then computes K=1/3/5 expected
accuracy, R_dis, R_persistent, correctable/harmful mass, early-stopping calls.
Resume-safe journal: external_judgebench[_mock].jsonl.
Usage: python run_external_bench.py [--mock] [--workers 4] [--model gpt-4o]
"""
from __future__ import annotations
import argparse, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from judges import Judge, JudgeConfig, MockJudge

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "judgebench_120.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    insts = [json.loads(l) for l in open(SAMPLE, encoding="utf-8")]
    path = os.path.join(HERE, f"external_judgebench{'_mock' if args.mock else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    todo = [i for i in insts if i["iid"] not in done]
    print(f"[judgebench] {len(todo)} to run ({len(done)} done)", flush=True)

    def one(inst):
        j = MockJudge() if args.mock else Judge(JudgeConfig(model=args.model, cache_path=None))
        x, a, b = inst["x"], inst["a"], inst["b"]
        t0 = time.time()
        greedy = j.cot_pairwise(x, a, b, temperature=0.0)
        pool = [j.cot_pairwise(x, a, b, temperature=0.7) for _ in range(5)]
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": inst["gold"],
                "len_a": len(a), "len_b": len(b), "greedy": greedy, "pool": pool,
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
            if (ok+err) % 20 == 0: print(f"  {ok+err}/{len(todo)} ({err} err)", flush=True)
    print(f"[judgebench] done ok={ok} err={err} in {(time.time()-t0)/60:.1f} min -> {path}")


if __name__ == "__main__":
    main()
