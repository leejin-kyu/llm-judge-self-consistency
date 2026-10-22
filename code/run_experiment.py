# -*- coding: utf-8 -*-
"""Full MPCJ experiment harness — §6용 정식 수치 생성.

기능:
  - 여러 데이터셋(LLMBar 5종 + MT-Bench) 자동 다운로드/로드
  - 베이스라인 패널: Vanilla / Swap-only(BPC) / CoT-holistic(G-Eval식) / Self-committee / MPCJ(+ablation)
  - dev/test 분할: MPCJ의 tau를 dev에서 선택(목표 coverage) → test 보고 (training-free: 하이퍼파라미터만)
  - bootstrap 95% CI (accuracy)
  - 디스크 캐시로 비용 절감, judge 다중화 지원
  - 결과: results_full.json + results_table.md (paper §6에 붙일 수 있는 형태)

사용(새 키 발급 후):
  pip install -r requirements.txt
  cp .env.example .env   # OPENAI_API_KEY 입력
  python prep_data.py --bench llmbar --subset Natural  --out d_natural.jsonl
  python prep_data.py --bench llmbar --subset Neighbor --out d_neighbor.jsonl
  python prep_data.py --bench llmbar --subset GPTInst  --out d_gptinst.jsonl
  python prep_data.py --bench llmbar --subset GPTOut   --out d_gptout.jsonl
  python prep_data.py --bench llmbar --subset Manual   --out d_manual.jsonl
  python run_experiment.py --data d_natural.jsonl d_neighbor.jsonl d_gptinst.jsonl d_gptout.jsonl d_manual.jsonl \
                           --judge gpt-4o --target-coverage 0.8 --boot 1000
  # 빠른 점검:  python run_experiment.py --mock --data d_neighbor.jsonl
"""
from __future__ import annotations
import argparse, json, random
import numpy as np

try:
    from dotenv import load_dotenv; load_dotenv()
except Exception:
    pass

from judges import Judge, JudgeConfig, MockJudge
from mpcj import (mpcj_pairwise, vanilla_pairwise, swap_only, cot_holistic, self_committee)
import metrics as M
import data as D

LAB = {"A": 1, "B": -1, "tie": 0}


def boot_ci(preds, golds, n_boot=1000, seed=0):
    rng = random.Random(seed)
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    if not pairs:
        return (float("nan"), float("nan"), float("nan"))
    accs = []
    n = len(pairs)
    for _ in range(n_boot):
        s = [pairs[rng.randrange(n)] for _ in range(n)]
        accs.append(sum(1 for p, g in s if p == g) / n)
    accs.sort()
    lo = accs[int(0.025 * n_boot)]; hi = accs[int(0.975 * n_boot)]
    mean = sum(p == g for p, g in pairs) / n
    return (mean, lo, hi)


def eval_method(items, recs):
    golds = [it["gold"] for it in items]
    preds = [None if r.get("abstain") else r["verdict"] for r in recs]
    len_a = [len(it["a"]) for it in items]; len_b = [len(it["b"]) for it in items]
    acc, lo, hi = boot_ci(preds, golds)
    ncov = sum(1 for p in preds if p is not None)
    return {
        "accuracy": round(acc, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "coverage": round(ncov / len(items), 4), "answered": ncov, "total": len(items),
        "cohen_kappa": round(M.kappa(preds, golds), 4),
        "spearman": round(M.spearman(preds, golds), 4),
        "position_flip_rate": round(M.position_flip_rate(recs), 4),
        "verbosity_bias_slope": round(M.verbosity_bias_slope(preds, len_a, len_b), 5),
    }


def run_all(items, judge, mpcj_kw):
    """Return per-method record-lists (computed once per method)."""
    methods = {
        "Vanilla-pairwise":      lambda it: vanilla_pairwise(judge, it["x"], it["a"], it["b"]),
        "Swap-only (BPC,Wang24)": lambda it: swap_only(judge, it["x"], it["a"], it["b"]),
        "CoT-holistic (G-Eval)": lambda it: cot_holistic(judge, it["x"], it["a"], it["b"]),
        "Self-committee (K=5)":  lambda it: self_committee(judge, it["x"], it["a"], it["b"], K=5),
        "MPCJ (terse)":          lambda it: mpcj_pairwise(judge, it["x"], it["a"], it["b"], **mpcj_kw),
        "MPCJ-CoT (ours)":       lambda it: mpcj_pairwise(judge, it["x"], it["a"], it["b"],
                                                          cot=True, **{k: v for k, v in mpcj_kw.items() if k != "cot"}),
    }
    out = {}
    for name, fn in methods.items():
        out[name] = [fn(it) for it in items]
    return out


def select_tau(dev_items, dev_recs_mpcj, target_coverage):
    """Pick smallest tau achieving >= target coverage on dev (training-free hyperparam)."""
    best = 0.5
    for tau in [round(t, 2) for t in np.linspace(0.1, 0.9, 17)]:
        cov = sum(1 for r in dev_recs_mpcj if r["u"] < tau) / len(dev_recs_mpcj)
        if cov >= target_coverage:
            best = tau; break
    return best


def apply_tau(recs, tau):
    return [dict(r, abstain=(r["u"] >= tau)) for r in recs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="jsonl 경로(여러 개 가능)")
    ap.add_argument("--judge", default=None, help="모델명(예 gpt-4o). 생략 시 .env")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--target-coverage", type=float, default=0.8)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--dev-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    make_judge = (lambda: MockJudge()) if args.mock else \
        (lambda: Judge(JudgeConfig(provider=args.provider, model=args.judge)))
    judge = make_judge()
    mpcj_kw = dict(alpha=0.5, beta=0.5, gamma=0.0, K=1, tau=0.5, symmetrize=True)

    all_results = {}
    pooled = {}  # method -> (items, recs) across datasets (test split)
    for path in args.data:
        items = D.load_jsonl(path)
        rng = random.Random(args.seed)
        idx = list(range(len(items))); rng.shuffle(idx)
        n_dev = max(1, int(len(items) * args.dev_frac))
        dev_i = set(idx[:n_dev])
        dev = [items[i] for i in range(len(items)) if i in dev_i]
        test = [items[i] for i in range(len(items)) if i not in dev_i]

        recs_all = run_all(items, judge, mpcj_kw)
        # ★ Selective contribution: abstention layered on the strongest judge (same records, no extra API)
        recs_all["SelfComm+Abstain (ours)"] = recs_all["Self-committee (K=5)"]
        dev_idx_list = [i for i in range(len(items)) if i in dev_i]

        ds_res = {}; taus = {}
        for name, recs in recs_all.items():
            test_recs = [recs[i] for i in range(len(items)) if i not in dev_i]
            test_items = test
            if name.startswith("MPCJ") or "Abstain" in name:  # tau selected on dev
                dev_recs = [recs[i] for i in dev_idx_list]
                tau = select_tau(dev, dev_recs, args.target_coverage)
                taus[name] = tau
                test_recs = apply_tau(test_recs, tau)
            ds_res[name] = eval_method(test_items, test_recs)
            pooled.setdefault(name, ([], []))
            pooled[name][0].extend(test_items); pooled[name][1].extend(test_recs)
        ds_res["_meta"] = {"tau_selected": taus, "n_dev": len(dev), "n_test": len(test)}
        all_results[path] = ds_res

    # pooled (all datasets, test) summary
    pooled_res = {name: eval_method(it, rc) for name, (it, rc) in pooled.items()}
    all_results["POOLED"] = pooled_res
    all_results["_cost"] = {"logical_calls": judge.calls, "api_calls": judge.api_calls,
                            "tokens": judge.tokens, "judge": (args.judge or "mock")}

    json.dump(all_results, open("results_full.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # markdown table (POOLED)
    cols = ["accuracy", "ci95", "coverage", "cohen_kappa", "spearman",
            "position_flip_rate", "verbosity_bias_slope"]
    lines = ["# Results (POOLED across datasets, test split)\n",
             f"judge = {args.judge or 'mock'} | cost: logical={judge.calls}, api(billed)={judge.api_calls}, tokens={judge.tokens}\n",
             "| Method | " + " | ".join(cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    for name, r in pooled_res.items():
        lines.append("| " + name + " | " + " | ".join(str(r[c]) for c in cols) + " |")
    md = "\n".join(lines)
    open("results_table.md", "w", encoding="utf-8").write(md)
    print(md)
    print("\nsaved results_full.json, results_table.md")
    if args.mock:
        print("[!] --mock: 파이프라인 점검용(가짜 judge). 논문 수치 아님.")


if __name__ == "__main__":
    main()
