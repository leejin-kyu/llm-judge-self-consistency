# -*- coding: utf-8 -*-
"""#3/#4 K (and optional T) sweep for self-consistency on one backbone, + accuracy-cost Pareto.
Usage: python run_ksweep.py --judge gpt-4o --Ks 1 3 5 7 9
Outputs ksweep_<judge>.json + Figure_4_pareto.png
"""
from __future__ import annotations
import argparse, json, random
try:
    from dotenv import load_dotenv; load_dotenv()
except Exception: pass
from judges import Judge, JudgeConfig, MockJudge
from mpcj import self_committee, cot_holistic
from collections import Counter
import math

DATASETS=["full_natural.jsonl","full_neighbor.jsonl","full_gptinst.jsonl","full_gptout.jsonl","full_manual.jsonl"]
def load(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]
def ent(v):
    c=Counter(v); n=len(v); return -sum((c[k]/n)*math.log(c[k]/n,2) for k in c)

def test_items(seed=0,dev_frac=0.3):
    out=[]
    for path in DATASETS:
        items=load(path); rng=random.Random(seed); idx=list(range(len(items))); rng.shuffle(idx)
        dev=set(idx[:max(1,int(len(items)*dev_frac))])
        out+=[it for i,it in enumerate(items) if i not in dev]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--judge",default="gpt-4o")
    ap.add_argument("--Ks",type=int,nargs="+",default=[1,3,5,7,9]); ap.add_argument("--T",type=float,default=0.7)
    ap.add_argument("--mock",action="store_true")
    args=ap.parse_args()
    judge=MockJudge() if args.mock else Judge(JudgeConfig(model=args.judge,
                       cache_path=f"cache_{args.judge.replace('.','_')}.json"))
    items=test_items(); rows=[]
    for K in args.Ks:
        corr=0; ents=[]
        for it in items:
            if K==1:
                v=cot_holistic(judge,it["x"],it["a"],it["b"])["verdict"]; ents.append(0.0)
            else:
                r=self_committee(judge,it["x"],it["a"],it["b"],K=K,temperature=args.T)
                v=r["verdict"]; ents.append(ent(r["votes"]))
            corr+=(v==it["gold"])
        rows.append({"K":K,"T":args.T,"accuracy":round(corr/len(items),4),
                     "calls_per_decision":K,"mean_vote_entropy":round(sum(ents)/len(ents),4)})
        print(f"K={K} T={args.T}: acc={rows[-1]['accuracy']} entropy={rows[-1]['mean_vote_entropy']}")
    base=rows[0]["accuracy"]
    for r in rows: r["delta_over_CoT"]=round(r["accuracy"]-base,4); r["acc_per_call"]=round(r["accuracy"]/r["K"],4)
    json.dump({"judge":args.judge,"rows":rows},open(f"ksweep_{args.judge.replace('.','_')}.json","w",encoding="utf-8"),ensure_ascii=False,indent=2)

    # Pareto figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10})
    fig,ax=plt.subplots(figsize=(6.2,4.2))
    xs=[r["K"] for r in rows]; ys=[r["accuracy"] for r in rows]
    ax.plot(xs,ys,"o-",color="#2E7D32",lw=2)
    for r in rows: ax.annotate(f"K={r['K']}",(r["K"],r["accuracy"]),textcoords="offset points",xytext=(6,-2),fontsize=8)
    ax.set_xlabel("Calls per decision (K)"); ax.set_ylabel("Accuracy (test)")
    ax.set_title(f"Accuracy vs. cost: self-consistency K-sweep ({args.judge})",fontsize=11,fontweight="bold")
    ax.grid(alpha=0.3); plt.tight_layout()
    fig.savefig("../../03_그림표/Figure_4_pareto.png",dpi=300,bbox_inches="tight")
    print("saved ksweep + Figure_4_pareto")

if __name__=="__main__": main()
