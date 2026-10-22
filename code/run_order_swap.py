# -*- coding: utf-8 -*-
"""Plan A (5th review, priority 5): candidate-order swap to re-decompose persistent errors.

Stage A: all 295 held-out test instances, Single-CoT with candidates exchanged
         (B/A order), temperature 0  -> order consistency / flip rate.   ~295 calls
Stage B: focused K=5 (T=0.7, swapped order) on the 37 unanimous-and-wrong SC-CoT
         instances + 37 matched unanimous-and-correct controls (matched on subset,
         gold position, length-gap tercile; greedy 1:1, seed 0).         ~370 calls

Verdicts are recorded both raw (in the swapped frame) and mapped back to the
original frame (A<->B). Resume-safe journals: order_swap_stageA[_mock].jsonl,
order_swap_stageB[_mock].jsonl. Usage:
    python run_order_swap.py [--mock] [--workers 4] [--stage A|B|both]
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from judges import Judge, JudgeConfig, MockJudge, COT_SYS, COT_USER
from run_ollama_judge import load_test

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")
SWAP = {"A": "B", "B": "A", "tie": "tie"}


def test_instances():
    return {i["iid"]: i for s in load_test() for i in s}


def stageB_targets():
    """37 unanimous-wrong + 37 matched unanimous-correct SC-CoT instances."""
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if r["strategy"] == "SC-CoT" and r["split"] == "test"]
    for r in rows:
        vs = r["votes"].split("|")
        r["unan"] = len(set(vs)) == 1
        r["c_i"] = sum(v == r["gold_label"] for v in vs)
        la, lb = int(r["candidate_A_length"]), int(r["candidate_B_length"])
        r["gap"] = abs(la - lb)
    pers = [r for r in rows if r["unan"] and r["c_i"] == 0]
    stab = [r for r in rows if r["c_i"] == 5]
    gaps = sorted(r["gap"] for r in rows)
    t1, t2 = gaps[len(gaps)//3], gaps[2*len(gaps)//3]
    def ter(g): return 0 if g <= t1 else (1 if g <= t2 else 2)
    rng = np.random.default_rng(0)
    used, controls = set(), []
    for p in pers:
        pool = [s for s in stab if s["subset"] == p["subset"]
                and s["gold_label"] == p["gold_label"] and ter(s["gap"]) == ter(p["gap"])
                and s["instance_id"] not in used]
        if not pool:  # relax tercile if stratum empty
            pool = [s for s in stab if s["subset"] == p["subset"]
                    and s["gold_label"] == p["gold_label"] and s["instance_id"] not in used]
        if pool:
            m = pool[int(rng.integers(len(pool)))]
            used.add(m["instance_id"])
            controls.append((p["instance_id"], m["instance_id"]))
    targets = [iid for pair in controls for iid in pair]
    roles = {}
    for p_id, m_id in controls:
        roles[p_id] = "persistent_wrong"; roles[m_id] = "matched_control"
    return targets, roles


def journal_path(stage, mock):
    return os.path.join(HERE, f"order_swap_stage{stage}{'_mock' if mock else ''}.jsonl")


def run_stage(stage, insts, roles, judge_factory, k, temp, workers):
    path = journal_path(stage, isinstance(judge_factory(), MockJudge))
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    todo = [i for i in insts if i["iid"] not in done]
    print(f"[stage {stage}] {len(todo)} to run ({len(done)} done)", flush=True)
    lock = threading.Lock()

    def one(inst):
        j = judge_factory()
        x, a, b, g = inst["x"], inst["a"], inst["b"], inst["gold"]
        t0 = time.time()
        # swapped order: candidates presented as (b, a); map verdicts back
        raw = [j.cot_pairwise(x, b, a, temperature=temp) for _ in range(k)]
        mapped = [SWAP.get(v, "tie") for v in raw]
        return {"iid": inst["iid"], "subset": inst["subset"], "gold": g,
                "role": roles.get(inst["iid"], "full_test"),
                "len_a": len(a), "len_b": len(b), "order": "swapped",
                "k": k, "temperature": temp,
                "raw_swapped": raw, "verdicts_original_frame": mapped,
                "prompt_sha256": hashlib.sha256(
                    (COT_SYS + COT_USER.format(x=x, a=b, b=a)).encode()).hexdigest(),
                "model": getattr(j.cfg, "model", "mock") if hasattr(j, "cfg") else "mock",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "latency_s": round(time.time()-t0, 2)}

    t0 = time.time(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(path, "a", encoding="utf-8") as jf:
        futs = {ex.submit(one, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock: jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                ok += 1
            except Exception as e:
                err += 1; print(f"  ERROR {futs[fut]['iid']}: {e}", flush=True)
            if (ok+err) % 20 == 0: print(f"  {ok+err}/{len(todo)} ({err} err)", flush=True)
    print(f"[stage {stage}] done ok={ok} err={err} in {(time.time()-t0)/60:.1f} min -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--stage", default="both", choices=["A", "B", "both"])
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    jf = (lambda: MockJudge()) if args.mock else \
         (lambda: Judge(JudgeConfig(model=args.model, cache_path=None)))
    inst = test_instances()
    if args.stage in ("A", "both"):
        run_stage("A", list(inst.values()), {}, jf, k=1, temp=0.0, workers=args.workers)
    if args.stage in ("B", "both"):
        targets, roles = stageB_targets()
        print(f"stage B targets: {len(targets)} ({Counter(roles.values())})")
        missing = [t for t in targets if t not in inst]
        assert not missing, f"targets missing from test set: {missing[:5]}"
        run_stage("B", [inst[t] for t in targets], roles, jf, k=5, temp=0.7,
                  workers=args.workers)


if __name__ == "__main__":
    main()
