# -*- coding: utf-8 -*-
"""Item 4 (top-tier review): cross-family validation with one non-OpenAI, open-weight
judge via local Ollama (zero API cost).

Design (minimal): 100 stratified test instances x (1 greedy CoT + 5 CoT at T=0.7).
Reports: K=1 expected accuracy, K=5 majority accuracy, unanimity rate,
unanimous-and-wrong rate, early-stopping mean calls, expected net voting gain —
the quantities needed for the vote-variance <-> voting-gain relationship.

Resume-safe journal: ollama_{model}.jsonl. Usage:
  python run_ollama_judge.py --model gemma3:12b --n 100 [--workers 4]
"""
from __future__ import annotations
import argparse, json, os, random, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from judges import Judge, JudgeConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["full_natural.jsonl", "full_neighbor.jsonl", "full_gptinst.jsonl",
            "full_gptout.jsonl", "full_manual.jsonl"]
SUBSET = {"full_natural.jsonl": "Natural", "full_neighbor.jsonl": "Neighbor",
          "full_gptinst.jsonl": "GPTInst", "full_gptout.jsonl": "GPTOut",
          "full_manual.jsonl": "Manual"}
SPLIT_SEED, DEV_FRAC, SAMPLE_SEED = 0, 0.3, 99


def load_test():
    out = []
    for path in DATASETS:
        items = [json.loads(l) for l in open(os.path.join(HERE, path), encoding="utf-8") if l.strip()]
        idx = list(range(len(items)))
        random.Random(SPLIT_SEED).shuffle(idx)
        dev = set(idx[:max(1, int(len(items) * DEV_FRAC))])
        sub = SUBSET[path]
        out.append([{"iid": f"{sub}_{i:03d}", "subset": sub, **it}
                    for i, it in enumerate(items) if i not in dev])
    return out  # list per subset


def stratified(per_subset, n):
    total = sum(len(s) for s in per_subset)
    rng = random.Random(SAMPLE_SEED)
    picked = []
    for s in per_subset:
        k = round(n * len(s) / total)
        picked += rng.sample(s, min(k, len(s)))
    return picked[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemma3:12b")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    journal = os.path.join(HERE, f"ollama_{args.model.replace(':', '_').replace('.', '_')}.jsonl")
    done = set()
    if os.path.exists(journal):
        for l in open(journal, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    insts = [i for i in stratified(load_test(), args.n) if i["iid"] not in done]
    print(f"[{args.model}] {len(insts)} to run ({len(done)} done)", flush=True)

    def run_one(inst):
        judge = Judge(JudgeConfig(model=args.model, cache_path=None,
                                  base_url=os.environ.get("OLLAMA_BASE_URL",
                                                          "http://localhost:11434/v1")))
        x, a, b, g = inst["x"], inst["a"], inst["b"], inst["gold"]
        greedy = judge.cot_pairwise(x, a, b, temperature=0.0)
        pool = [judge.cot_pairwise(x, a, b, temperature=0.7) for _ in range(5)]
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": g,
                "len_a": len(a), "len_b": len(b),
                "greedy": greedy, "pool": pool, "calls": judge.calls}

    t0 = time.time(); n_ok = n_err = 0
    wlock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(journal, "a", encoding="utf-8") as jf:
        futs = {ex.submit(run_one, i): i for i in insts}
        for fut in as_completed(futs):
            inst = futs[fut]
            try:
                rec = fut.result()
                with wlock:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                n_ok += 1
            except Exception as e:
                n_err += 1
                print(f"  ERROR {inst['iid']}: {type(e).__name__}: {e}", flush=True)
            if (n_ok + n_err) % 10 == 0:
                print(f"  {n_ok+n_err}/{len(insts)} ({n_err} err) {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"[{args.model}] done ok={n_ok} err={n_err} in {(time.time()-t0)/60:.1f} min -> {journal}", flush=True)


if __name__ == "__main__":
    main()
