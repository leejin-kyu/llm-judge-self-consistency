# -*- coding: utf-8 -*-
"""Evaluation metrics: alignment, bias-specific, coverage-alignment."""
from __future__ import annotations
import numpy as np

try:
    from sklearn.metrics import cohen_kappa_score
    from scipy.stats import spearmanr
except Exception:
    cohen_kappa_score = None
    spearmanr = None

LAB = {"A": 1, "B": -1, "tie": 0}


def accuracy(preds, golds):
    """Exclude items where pred is ABSTAIN (None)."""
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    if not pairs:
        return float("nan"), 0
    correct = sum(1 for p, g in pairs if p == g)
    return correct / len(pairs), len(pairs)


def kappa(preds, golds):
    pairs = [(p, g) for p, g in zip(preds, golds) if p is not None]
    if not pairs or cohen_kappa_score is None:
        return float("nan")
    P = [LAB[p] for p, _ in pairs]
    G = [LAB[g] for _, g in pairs]
    try:
        return float(cohen_kappa_score(P, G))
    except Exception:
        return float("nan")


def spearman(preds, golds):
    pairs = [(LAB[p], LAB[g]) for p, g in zip(preds, golds) if p is not None]
    if len(pairs) < 3 or spearmanr is None:
        return float("nan")
    P, G = zip(*pairs)
    r = spearmanr(P, G).correlation
    return float(r) if r == r else float("nan")


def position_flip_rate(records):
    """records: list of dicts with 'U_ord' (mean over criteria). Returns mean flip rate."""
    vals = [r["U_ord"] for r in records if "U_ord" in r]
    return float(np.mean(vals)) if vals else float("nan")


def verbosity_bias_slope(preds, len_a, len_b):
    """Regress verdict(+1 if A wins) on (len_a - len_b). Positive slope => length bias."""
    xs, ys = [], []
    for p, la, lb in zip(preds, len_a, len_b):
        if p is None or p == "tie":
            continue
        xs.append(la - lb)
        ys.append(1.0 if p == "A" else 0.0)
    if len(xs) < 3:
        return float("nan")
    xs = np.array(xs, float); ys = np.array(ys, float)
    if xs.std() == 0:
        return float("nan")
    return float(np.polyfit(xs, ys, 1)[0])


def coverage_alignment(records, golds, taus=None):
    """For MPCJ records (each has 'u','verdict'): sweep tau -> (coverage, accuracy)."""
    taus = taus if taus is not None else [round(t, 2) for t in np.linspace(0.1, 0.9, 9)]
    out = []
    n = len(records)
    for tau in taus:
        preds = [r["verdict"] if r["u"] < tau else None for r in records]
        cov = sum(1 for p in preds if p is not None) / n
        acc, _ = accuracy(preds, golds)
        out.append({"tau": tau, "coverage": cov, "accuracy": acc})
    return out


def cost(judge):
    return {"calls": getattr(judge, "calls", None), "tokens": getattr(judge, "tokens", None)}
