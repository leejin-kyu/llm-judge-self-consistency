# -*- coding: utf-8 -*-
"""Download & convert judge benchmarks to MPCJ jsonl: {x,a,b,gold}.

LLMBar (Zeng et al., 2024) — princeton-nlp/LLMBar, label = index(1|2) of the better output.
사용:
  python prep_data.py --bench llmbar --subset Neighbor --n 40 --out data_llmbar_neighbor.jsonl
  python prep_data.py --bench llmbar --subset Natural  --out data_llmbar_natural.jsonl
  python prep_data.py --bench mtbench --out data_mtbench.jsonl     # datasets 필요
subset: Natural | Neighbor | GPTInst | GPTOut | Manual  (Adversarial 4종 + Natural)
"""
from __future__ import annotations
import argparse, json, urllib.request

LLMBAR_BASE = "https://raw.githubusercontent.com/princeton-nlp/LLMBar/main/Dataset/LLMBar"
LLMBAR_PATHS = {
    "Natural":  "Natural/dataset.json",
    "Neighbor": "Adversarial/Neighbor/dataset.json",
    "GPTInst":  "Adversarial/GPTInst/dataset.json",
    "GPTOut":   "Adversarial/GPTOut/dataset.json",
    "Manual":   "Adversarial/Manual/dataset.json",
}


def fetch_llmbar(subset):
    url = f"{LLMBAR_BASE}/{LLMBAR_PATHS[subset]}"
    raw = json.load(urllib.request.urlopen(url, timeout=60))
    out = []
    for it in raw:
        lab = int(it["label"])  # 1 -> output_1 better, 2 -> output_2 better
        out.append({
            "x": it["input"].strip(),
            "a": it["output_1"].strip(),
            "b": it["output_2"].strip(),
            "gold": "A" if lab == 1 else "B",
            "src": f"LLMBar/{subset}",
        })
    return out


def fetch_mtbench():
    """MT-Bench human judgments (Zheng et al., 2023) via HuggingFace datasets.
    pairwise 인간투표만 사용, tie 포함."""
    from datasets import load_dataset
    ds = load_dataset("lmsys/mt_bench_human_judgments", split="human")
    out = []
    for r in ds:
        # 각 레코드: question(turns), model_a, model_b, conversation_a/b, winner(model_a|model_b|tie)
        try:
            q = r["question"]["turns"][0] if isinstance(r["question"], dict) else r["conversation_a"][0]["content"]
            a = r["conversation_a"][-1]["content"]
            b = r["conversation_b"][-1]["content"]
        except Exception:
            continue
        w = r.get("winner", "")
        gold = {"model_a": "A", "model_b": "B", "tie": "tie"}.get(w)
        if gold is None:
            continue
        out.append({"x": q, "a": a, "b": b, "gold": gold, "src": "MT-Bench"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["llmbar", "mtbench"], required=True)
    ap.add_argument("--subset", default="Neighbor")
    ap.add_argument("--n", type=int, default=0, help="0=전체")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items = fetch_llmbar(args.subset) if args.bench == "llmbar" else fetch_mtbench()
    if args.n > 0:
        items = items[:args.n]
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items -> {args.out}")
    g = {}
    for it in items:
        g[it["gold"]] = g.get(it["gold"], 0) + 1
    print("gold 분포:", g)


if __name__ == "__main__":
    main()
