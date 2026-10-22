# -*- coding: utf-8 -*-
"""Batch B: instrumented re-run campaign (one canonical execution per backbone).

Implements, in a single run per backbone:
  - P1-14 order randomization: per-instance A/B order assigned by a fixed seed;
    every strategy sees the same order.
  - P1-13 per-instance records for ALL backbones -> paired stats everywhere.
  - P1-11/12 nested K-sweep: a pool of stochastic CoT samples per instance
    (9 for gpt-4o / gpt-4o-mini, 5 for gpt-4.1); SC-CoT(K) = majority of the
    first K pool samples; permutation analysis is done offline.
  - P1-9 factorial (gpt-4o only): Direct-T0.7 (K=1) and a 5-sample Direct pool
    (voting without reasoning).
  - P0-6 forced-choice reconciliation: on order disagreement, a third
    adjudication call that must choose A or B (no tie).
  - P1-15/P2-19 logging: per-call latency, input/output tokens, parse status.

Resume-safe: appends one JSON line per instance to campaign_{model}.jsonl and
skips instances already present. Usage:
  python run_campaign.py --judge gpt-4o-mini [--workers 8] [--limit 3] [--mock]
"""
from __future__ import annotations
import argparse, json, os, random, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass
from judges import Judge, JudgeConfig, MockJudge, COT_USER, _parse_verdict_status
from mpcj import mpcj_pairwise, _flip

HERE = os.path.dirname(os.path.abspath(__file__))
DATASETS = ["full_natural.jsonl", "full_neighbor.jsonl", "full_gptinst.jsonl",
            "full_gptout.jsonl", "full_manual.jsonl"]
SUBSET = {"full_natural.jsonl": "Natural", "full_neighbor.jsonl": "Neighbor",
          "full_gptinst.jsonl": "GPTInst", "full_gptout.jsonl": "GPTOut",
          "full_manual.jsonl": "Manual"}
ORDER_SEED = 1234
SPLIT_SEED, DEV_FRAC = 0, 0.3

FORCED_SYS = ("You are an impartial evaluator. Think step by step about which response "
              "better follows the instruction; ignore length and position. You MUST choose "
              "one response; 'tie' is not allowed. End with a final line 'Verdict: A' or "
              "'Verdict: B'.")


def load_instances(limit=None):
    out = []
    for path in DATASETS:
        items = [json.loads(l) for l in open(os.path.join(HERE, path), encoding="utf-8") if l.strip()]
        idx = list(range(len(items)))
        random.Random(SPLIT_SEED).shuffle(idx)
        dev = set(idx[:max(1, int(len(items) * DEV_FRAC))])
        sub = SUBSET[path]
        for i, it in enumerate(items):
            out.append({"iid": f"{sub}_{i:03d}", "subset": sub,
                        "split": "dev" if i in dev else "test", **it})
    if limit:
        rng = random.Random(7)
        out = rng.sample(out, limit)
    return out


def metrics_slice(judge, start):
    """Aggregate judge.log entries added since `start`."""
    entries = judge.log[start:]
    return {
        "calls": len(entries),
        "billed": sum(1 for e in entries if not e["cached"]),
        "in_tok": sum(e["in_tok"] for e in entries),
        "out_tok": sum(e["out_tok"] for e in entries),
        "latency": round(sum(e["latency"] for e in entries), 3),
        "fallback_tie": sum(1 for e in entries if e.get("parse") == "fallback_tie"),
    }


def run_instance(inst, model, mock, pool_n, factorial, cache_path):
    judge = MockJudge() if mock else Judge(JudgeConfig(model=model, cache_path=cache_path))
    rng = random.Random(f"order-{ORDER_SEED}-{inst['iid']}")
    swap = rng.random() < 0.5
    a, b = (inst["b"], inst["a"]) if swap else (inst["a"], inst["b"])
    gold = _flip(inst["gold"]) if swap else inst["gold"]
    x = inst["x"]
    rec = {"iid": inst["iid"], "subset": inst["subset"], "split": inst["split"],
           "swap": swap, "gold": gold, "len_a": len(a), "len_b": len(b),
           "strategies": {}}

    def add(name, verdict, extra=None, start=0):
        rec["strategies"][name] = {"verdict": verdict, "correct": verdict == gold,
                                   **(extra or {}), "m": metrics_slice(judge, start)}

    # Direct (T0)
    s = len(judge.log); v = judge.pairwise(x, a, b, "correctness")
    add("Direct", v, start=s)

    # DTT-Sym + forced-choice reconciliation (shares the two order calls)
    s = len(judge.log)
    v_ab = judge.pairwise(x, a, b, "correctness")
    v_ba = judge.pairwise(x, b, a, "correctness")
    agree = (v_ab == _flip(v_ba))
    add("DTT-Sym", v_ab if agree else "tie", {"order_agree": agree}, start=s)
    s = len(judge.log)
    if agree:
        fc = v_ab
    else:
        raw = judge._chat(FORCED_SYS, COT_USER.format(x=x, a=a, b=b),
                          temperature=0.0, max_tokens=256)
        fc, status = _parse_verdict_status(raw)
        if judge.log: judge.log[-1]["parse"] = status
        if fc == "tie":  # refuses to choose: fall back to first-order verdict
            fc = v_ab if v_ab != "tie" else v_ba
    add("DTT-FC", fc, {"order_agree": agree}, start=s)

    # Lightweight decomposition (5 criteria x 2 orders, criterion CoT)
    s = len(judge.log)
    d = mpcj_pairwise(judge, x, a, b, cot=True, tau=1.0)
    add("Decomp", d["verdict"], {"u": round(d["u"], 4)}, start=s)

    # Single-CoT (T0)
    s = len(judge.log); v = judge.cot_pairwise(x, a, b)
    add("Single-CoT", v, start=s)

    # Stochastic CoT pool (T0.7) -> SC-CoT(K=5) + nested K-sweep offline
    s = len(judge.log)
    pool = [judge.cot_pairwise(x, a, b, temperature=0.7) for _ in range(pool_n)]
    sc5 = Counter(pool[:5]).most_common(1)[0][0]
    add("SC-CoT", sc5, {"pool": pool, "margin": Counter(pool[:5]).most_common(1)[0][1]}, start=s)

    if factorial:
        s = len(judge.log)
        v = judge.pairwise(x, a, b, "correctness", temperature=0.7)
        add("Direct-T0.7", v, start=s)
        s = len(judge.log)
        dpool = [judge.pairwise(x, a, b, "correctness", temperature=0.7) for _ in range(5)]
        add("Direct-SC", Counter(dpool).most_common(1)[0][0], {"pool": dpool}, start=s)

    judge.flush_cache()
    rec["billed"] = judge.api_calls
    rec["tokens"] = judge.tokens
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default="gpt-4o-mini")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    model = args.judge
    pool_n = 5 if model == "gpt-4.1" else 9
    factorial = (model == "gpt-4o")
    cache_path = os.path.join(HERE, f"cache_campaign_{model.replace('.', '_')}.json")
    journal = os.path.join(HERE, f"campaign_{model.replace('.', '_')}{'_mock' if args.mock else ''}.jsonl")

    done = set()
    if os.path.exists(journal):
        for l in open(journal, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    insts = [i for i in load_instances(args.limit) if i["iid"] not in done]
    print(f"[{model}] {len(insts)} instances to run ({len(done)} already done); "
          f"pool={pool_n}, factorial={factorial}, workers={args.workers}", flush=True)

    t0 = time.time(); n_ok = n_err = 0; billed = tokens = 0
    wlock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as ex, \
         open(journal, "a", encoding="utf-8") as jf:
        futs = {ex.submit(run_instance, i, model, args.mock, pool_n, factorial, cache_path): i
                for i in insts}
        for fut in as_completed(futs):
            inst = futs[fut]
            try:
                rec = fut.result()
                with wlock:
                    jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                n_ok += 1; billed += rec["billed"]; tokens += rec["tokens"]
            except Exception as e:
                n_err += 1
                print(f"  ERROR {inst['iid']}: {type(e).__name__}: {e}", flush=True)
            total = n_ok + n_err
            if total % 20 == 0 or total == len(insts):
                el = time.time() - t0
                print(f"  {total}/{len(insts)} done ({n_err} err) | billed={billed} "
                      f"tokens={tokens:,} | {el/60:.1f} min", flush=True)
    print(f"[{model}] finished: ok={n_ok} err={n_err} billed={billed} tokens={tokens:,} "
          f"in {(time.time()-t0)/60:.1f} min -> {journal}", flush=True)


if __name__ == "__main__":
    main()
