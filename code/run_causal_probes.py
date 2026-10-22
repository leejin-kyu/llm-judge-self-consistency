# -*- coding: utf-8 -*-
"""Plan E (5th review, priority 9 optional): two causal probes on persistent errors.

Probe 1 — rubric-first re-judging (~74 calls):
    The 37 persistent-wrong instances + their 37 matched controls (roles from
    order_swap_stageB.jsonl), re-judged once with a constraint-focused prompt:
    list the instruction's explicit constraints first, then check violations,
    ignore length. T=0, original candidate order.
    If errors flip: "persistent under repeated holistic judging, but correctable
    through constraint-focused prompting".

Probe 2 — verbosity counterfactual (~124 calls):
    The 31 unanimous-and-longer SC-CoT errors (judge unanimously chose the longer
    response while the shorter is gold). Two symmetric interventions with the
    same backbone, then Single-CoT re-judging (T=0):
      (a) compress the chosen longer (wrong) response to ~50%, content-preserving;
      (b) expand the shorter (gold) response to ~2x with filler, no new content.
    Flips toward gold under (a)/(b) are evidence consistent with a causal length
    effect; no-flips keep the claim at association.

Resume-safe journals: causal_rubric[_mock].jsonl, causal_counterfactual[_mock].jsonl.
Usage: python run_causal_probes.py [--mock] [--workers 4] [--model gpt-4o]
"""
from __future__ import annotations
import argparse, csv, json, os, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from judges import Judge, JudgeConfig, MockJudge
from run_ollama_judge import load_test

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "results_final.csv")

RUBRIC_SYS = ("You are an impartial evaluator. First list the explicit constraints "
              "stated in the instruction. Then identify which response violates each "
              "constraint. Ignore response length unless length is explicitly "
              "required. End with a final line 'Verdict: A', 'Verdict: B', or "
              "'Verdict: tie'.")
RUBRIC_USER = """[Instruction]
{x}

[Response A]
{a}

[Response B]
{b}

List the instruction's explicit constraints, check each response against them, then give your verdict."""

COMPRESS_SYS = ("You compress text. Rewrite the given response at roughly half its "
                "length, preserving every claim, step, and piece of content; remove "
                "only redundancy, filler, and verbosity. Do not add or correct "
                "anything. Output only the rewritten response.")
EXPAND_SYS = ("You expand text. Rewrite the given response at roughly twice its "
              "length by adding redundancy, restatement, and elaboration of what is "
              "already said. Do NOT add any new information, claims, or corrections. "
              "Output only the rewritten response.")


def journal(name, mock):
    return os.path.join(HERE, f"{name}{'_mock' if mock else ''}.jsonl")


def resume_run(path, insts, one, workers):
    done = set()
    if os.path.exists(path):
        for l in open(path, encoding="utf-8"):
            try: done.add(json.loads(l)["iid"])
            except Exception: pass
    todo = [i for i in insts if i["iid"] not in done]
    print(f"[{os.path.basename(path)}] {len(todo)} to run ({len(done)} done)", flush=True)
    lock = threading.Lock(); t0 = time.time(); ok = err = 0
    with ThreadPoolExecutor(max_workers=workers) as ex, open(path, "a", encoding="utf-8") as jf:
        futs = {ex.submit(one, i): i for i in todo}
        for fut in as_completed(futs):
            try:
                rec = fut.result()
                with lock: jf.write(json.dumps(rec, ensure_ascii=False) + "\n"); jf.flush()
                ok += 1
            except Exception as e:
                err += 1; print(f"  ERROR {futs[fut]['iid']}: {e}", flush=True)
    print(f"  done ok={ok} err={err} in {(time.time()-t0)/60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--model", default="gpt-4o")
    args = ap.parse_args()
    inst = {i["iid"]: i for s in load_test() for i in s}
    mk = (lambda: MockJudge()) if args.mock else \
         (lambda: Judge(JudgeConfig(model=args.model, cache_path=None)))

    # ---------- Probe 1: rubric-first on persistent + matched controls ----------
    roles = {}
    sb = os.path.join(HERE, "order_swap_stageB.jsonl")
    for l in open(sb, encoding="utf-8"):
        r = json.loads(l); roles[r["iid"]] = r["role"]
    targets1 = [dict(inst[i], role=roles[i]) for i in sorted(roles) if i in inst]

    def rubric_one(t):
        j = mk()
        x, a, b = t["x"], t["a"], t["b"]
        if args.mock:
            v = "A" if len(a) >= len(b) else "B"
        else:
            raw = j._chat(RUBRIC_SYS, RUBRIC_USER.format(x=x, a=a, b=b), 0.0, max_tokens=512)
            v = j._parse_and_log(raw) if hasattr(j, "_parse_and_log") else raw
        return {"iid": t["iid"], "subset": t["subset"], "gold": t["gold"],
                "role": t["role"], "verdict": v, "temperature": 0.0,
                "model": args.model if not args.mock else "mock",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    resume_run(journal("causal_rubric", args.mock), targets1, rubric_one, args.workers)

    # ---------- Probe 2: counterfactual on unanimous-and-longer errors ----------
    rows = [r for r in csv.DictReader(open(CSV, encoding="utf-8"))
            if r["strategy"] == "SC-CoT" and r["split"] == "test" and r["correct"] == "0"]
    targets2 = []
    for r in rows:
        vs = r["votes"].split("|")
        la, lb = int(r["candidate_A_length"]), int(r["candidate_B_length"])
        longer = "A" if la > lb else "B"
        if len(set(vs)) == 1 and r["predicted_label"] == longer and r["gold_label"] != longer:
            targets2.append(dict(inst[r["instance_id"]], longer=longer))
    print(f"counterfactual targets (unanimous & chose-longer & shorter-is-gold): {len(targets2)}")

    def cf_one(t):
        j = mk()
        x, a, b, g, longer = t["x"], t["a"], t["b"], t["gold"], t["longer"]
        wrong_text = a if longer == "A" else b
        gold_text = b if longer == "A" else a
        if args.mock:
            comp, expd = wrong_text[: len(wrong_text)//2], gold_text * 2
        else:
            comp = j._chat(COMPRESS_SYS, wrong_text, 0.0, max_tokens=1024)
            expd = j._chat(EXPAND_SYS, gold_text, 0.0, max_tokens=1024)
        # (a) compressed wrong candidate replaces the original longer one
        ca, cb = (comp, b) if longer == "A" else (a, comp)
        v_comp = mk().cot_pairwise(x, ca, cb, temperature=0.0)
        # (b) expanded gold candidate replaces the original shorter one
        ea, eb = (a, expd) if longer == "A" else (expd, b)
        v_exp = mk().cot_pairwise(x, ea, eb, temperature=0.0)
        return {"iid": t["iid"], "subset": t["subset"], "gold": g, "longer_wrong": longer,
                "len_wrong": len(wrong_text), "len_wrong_comp": len(comp),
                "len_gold": len(gold_text), "len_gold_exp": len(expd),
                "verdict_after_compress": v_comp, "verdict_after_expand": v_exp,
                "model": args.model if not args.mock else "mock",
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}

    resume_run(journal("causal_counterfactual", args.mock), targets2, cf_one, args.workers)


if __name__ == "__main__":
    main()
