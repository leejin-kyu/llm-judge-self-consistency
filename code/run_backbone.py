# -*- coding: utf-8 -*-
"""#1 Multi-backbone: run the 5-strategy panel on the full LLMBar test split for one judge backbone.
Appends a row to multibackbone.json. Usage: python run_backbone.py --judge gpt-4o
"""
from __future__ import annotations
import argparse, json, os, random
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
from judges import Judge, JudgeConfig, MockJudge
from mpcj import mpcj_pairwise, vanilla_pairwise, swap_only, cot_holistic, self_committee

DATASETS = ["full_natural.jsonl","full_neighbor.jsonl","full_gptinst.jsonl","full_gptout.jsonl","full_manual.jsonl"]
def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--judge",default="gpt-4o")
    ap.add_argument("--mock",action="store_true"); ap.add_argument("--K",type=int,default=5)
    ap.add_argument("--seed",type=int,default=0); ap.add_argument("--dev-frac",type=float,default=0.3)
    args=ap.parse_args()
    # per-backbone cache so models don't collide
    judge=MockJudge() if args.mock else Judge(JudgeConfig(model=args.judge,
                       cache_path=f"cache_{args.judge.replace('.','_')}.json"))
    METHODS={"Vanilla":lambda it:vanilla_pairwise(judge,it["x"],it["a"],it["b"]),
             "PosCalib":lambda it:swap_only(judge,it["x"],it["a"],it["b"]),
             "Decomp":lambda it:mpcj_pairwise(judge,it["x"],it["a"],it["b"],cot=True,tau=1.0),
             "CoT":lambda it:cot_holistic(judge,it["x"],it["a"],it["b"]),
             "SelfConsistency":lambda it:self_committee(judge,it["x"],it["a"],it["b"],K=args.K)}
    acc={m:0 for m in METHODS}; ntest=0
    for path in DATASETS:
        items=load(path); rng=random.Random(args.seed); idx=list(range(len(items))); rng.shuffle(idx)
        dev=set(idx[:max(1,int(len(items)*args.dev_frac))])
        for i,it in enumerate(items):
            if i in dev: continue
            ntest+=1
            for m,fn in METHODS.items():
                if fn(it)["verdict"]==it["gold"]: acc[m]+=1
    row={"judge":(args.judge if not args.mock else "mock"),"N_test":ntest,"K":args.K,
         "acc":{m:round(acc[m]/ntest,4) for m in METHODS},
         "billed":judge.api_calls,"tokens":judge.tokens}
    allres=[]
    if os.path.exists("multibackbone.json"):
        allres=json.load(open("multibackbone.json",encoding="utf-8"))
    allres=[r for r in allres if r["judge"]!=row["judge"]]+[row]
    json.dump(allres,open("multibackbone.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)
    print(row["judge"],"->",row["acc"],"| billed",row["billed"])

if __name__=="__main__": main()
