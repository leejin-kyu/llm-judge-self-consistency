# -*- coding: utf-8 -*-
"""MPCJ core (training-free, API-only) + baselines.

구현은 실험_프로토콜.md §1 식과 일치:
  U_crit = H(P)/log3  (기준 간 합의 엔트로피)
  U_ord  = 순서 swap 시 판정이 뒤집히는 기준 비율
  u = a*U_crit + b*U_ord (+ g*U_sc)   ;  u<tau -> verdict, else ABSTAIN
"""
from __future__ import annotations
import math
from collections import Counter
from judges import CRITERIA

LABELS = ("A", "B", "tie")


def _flip(v: str) -> str:
    return {"A": "B", "B": "A", "tie": "tie"}[v]


def _entropy_norm(dist: dict) -> float:
    tot = sum(dist.values())
    if tot == 0:
        return 1.0
    h = 0.0
    for k in LABELS:
        p = dist.get(k, 0) / tot
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(3)  # normalize to [0,1]


def mpcj_pairwise(judge, x, a, b, criteria=None, weights=None,
                  alpha=0.5, beta=0.5, gamma=0.0, K=1, tau=0.5,
                  symmetrize=True, cot=False):
    """Return dict: verdict, uncertainty u, components, abstain flag.
    cot=False -> terse per-criterion judge; cot=True -> criterion-focused CoT judge (MPCJ-CoT)."""
    criteria = criteria or list(CRITERIA.keys())
    weights = weights or {c: 1.0 for c in criteria}
    jf = judge.cot_criterion if cot else judge.pairwise

    per_crit = {}     # criterion -> reconciled verdict
    order_consistent = []
    sc_consistency = []

    for c in criteria:
        # self-consistency samples (K>=1). temp>0 only if K>1.
        temp = 0.0 if K == 1 else 0.7
        samples_ab = [jf(x, a, b, c, temperature=temp) for _ in range(K)]
        v_ab = Counter(samples_ab).most_common(1)[0][0]
        if K > 1:
            sc_consistency.append(Counter(samples_ab).most_common(1)[0][1] / K)

        if symmetrize:
            samples_ba = [jf(x, b, a, c, temperature=temp) for _ in range(K)]
            v_ba = Counter(samples_ba).most_common(1)[0][0]
            agree = (v_ab == _flip(v_ba))
            order_consistent.append(1.0 if agree else 0.0)
            v_c = v_ab if agree else "tie"   # Reconcile
        else:
            v_c = v_ab
            order_consistent.append(1.0)
        per_crit[c] = v_c

    # weighted verdict distribution
    dist = {l: 0.0 for l in LABELS}
    for c, v in per_crit.items():
        dist[v] += weights[c]
    verdict = max(LABELS, key=lambda l: dist[l])

    U_crit = _entropy_norm(dist)
    U_ord = 1.0 - (sum(order_consistent) / len(order_consistent))
    U_sc = 1.0 - (sum(sc_consistency) / len(sc_consistency)) if sc_consistency else 0.0

    u = alpha * U_crit + beta * U_ord + gamma * U_sc
    return {
        "verdict": verdict, "u": u, "abstain": u >= tau,
        "U_crit": U_crit, "U_ord": U_ord, "U_sc": U_sc,
        "per_criterion": per_crit,
    }


# ---------------- baselines ----------------
def vanilla_pairwise(judge, x, a, b, criterion="correctness"):
    """Single-order, single holistic criterion judge."""
    return {"verdict": judge.pairwise(x, a, b, criterion), "u": 0.0, "abstain": False}


def swap_only(judge, x, a, b, criterion="correctness"):
    """Order symmetrization on a single holistic criterion (=Wang 2024 BPC, our ablation B3)."""
    v_ab = judge.pairwise(x, a, b, criterion)
    v_ba = judge.pairwise(x, b, a, criterion)
    verdict = v_ab if v_ab == _flip(v_ba) else "tie"
    return {"verdict": verdict, "u": 0.0 if v_ab == _flip(v_ba) else 1.0,
            "abstain": False}


def cot_holistic(judge, x, a, b):
    """G-Eval-style chain-of-thought holistic judge (baseline B4)."""
    return {"verdict": judge.cot_pairwise(x, a, b), "u": 0.0, "abstain": False}


def self_committee(judge, x, a, b, K=5, temperature=0.7):
    """Training-free self-consistency: K stochastic CoT votes, majority.
    Returns votes & margin for calibration/risk-coverage analysis."""
    votes = [judge.cot_pairwise(x, a, b, temperature=temperature) for _ in range(K)]
    cnt = Counter(votes)
    verdict, top = cnt.most_common(1)[0]
    u = 1.0 - top / K
    return {"verdict": verdict, "u": u, "abstain": False,
            "votes": votes, "margin": top, "K": K}
