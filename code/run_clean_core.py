# -*- coding: utf-8 -*-
"""Plan B (5th review, priority 4): clean canonical rerun + fair Direct baseline.

1. Forced-choice Direct: tie-free prompt (A or B only), temperature 0,
   all 295 held-out test instances.                                  ~295 calls
2. Single-CoT clean rerun: empty cache, temperature 0, prompt hashes stored,
   all 295 test instances.                                           ~295 calls

Both run with cache_path=None (no warm-cache influence) and store full metadata
(model string, timestamp, decoding params, prompt SHA-256, parse status).
Resume-safe journals: clean_direct_fc[_mock].jsonl, clean_singlecot[_mock].jsonl.
Usage: python run_clean_core.py [--mock] [--workers 4] [--model gpt-4o]
"""
from __future__ import annotations
import argparse, hashlib, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from judges import Judge, JudgeConfig, MockJudge, COT_SYS, COT_USER
from run_ollama_judge import load_test

HERE = os.path.dirname(os.path.abspath(__file__))

FC_SYS = ("You are a careful, impartial evaluator. Ignore response length and "
          "position. You MUST choose one response; 'tie' is not allowed. "
          "Output exactly one token: A or B.")
FC_USER = """[Instruction]
{x}

[Response A]
{a}

[Response B]
{b}

Which response better follows the instruction? You must answer with exactly one token: A or B."""


def parse_fc(text):
    t = (text or "").strip().upper()
    for tok in ("A", "B"):
        if t == tok or t.startswith(tok + ".") or t.startswith(tok + ")"):
            return tok, "ok"
    if "A" in t and "B" not in t: return "A", "loose"
    if "B" in t and "A" not in t: return "B", "loose"
    return "parse_fail", "fail"


def run(name, insts, call, workers, mock):
    path = os.path.join(HERE, f"{name}{'_mock' if mock else ''}.jsonl")
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    todo = [i for i in insts if i["iid"] not in done]
    print(f"[{name}] {len(todo)} to run ({len(done)} done)", flush=True)
    lock = threading.Lock(); t0 = time.time(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(path, "a", encoding="utf-8") as jf:
        futs = {ex.submit(call, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock: jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                ok += 1
            except Exception as e:
                err += 1; print(f"  ERROR {futs[fut]['iid']}: {e}", flush=True)
            if (ok+err) % 25 == 0: print(f"  {ok+err}/{len(todo)} ({err} err)", flush=True)
    print(f"[{name}] done ok={ok} err={err} in {(time.time()-t0)/60:.1f} min -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    insts = [i for s in load_test() for i in s]

    def direct_fc(inst):
        j = MockJudge() if args.mock else Judge(JudgeConfig(model=args.model, cache_path=None))
        x, a, b = inst["x"], inst["a"], inst["b"]
        user = FC_USER.format(x=x, a=a, b=b)
        t0 = time.time()
        if args.mock:
            raw = "A" if len(a) >= len(b) else "B"
        else:
            raw = j._chat(FC_SYS, user, 0.0, max_tokens=8)
        v, status = parse_fc(raw)
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": inst["gold"],
                "strategy": "Direct-FC", "temperature": 0.0, "verdict": v,
                "parse_status": status, "raw": (raw or "")[:40],
                "prompt_sha256": hashlib.sha256((FC_SYS + user).encode()).hexdigest(),
                "model": args.model if not args.mock else "mock", "cache": "disabled",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": round(time.time()-t0, 2)}

    def cot_clean(inst):
        j = MockJudge() if args.mock else Judge(JudgeConfig(model=args.model, cache_path=None))
        x, a, b = inst["x"], inst["a"], inst["b"]
        t0 = time.time()
        v = j.cot_pairwise(x, a, b, temperature=0.0)
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": inst["gold"],
                "strategy": "Single-CoT-clean", "temperature": 0.0, "verdict": v,
                "prompt_sha256": hashlib.sha256(
                    (COT_SYS + COT_USER.format(x=x, a=a, b=b)).encode()).hexdigest(),
                "model": args.model if not args.mock else "mock", "cache": "disabled",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": round(time.time()-t0, 2)}

    run("clean_direct_fc", insts, direct_fc, args.workers, args.mock)
    run("clean_singlecot", insts, cot_clean, args.workers, args.mock)


if __name__ == "__main__":
    main()
