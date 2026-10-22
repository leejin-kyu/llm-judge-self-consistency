# -*- coding: utf-8 -*-
"""Batch B analysis: campaign_{model}.jsonl -> campaign_analysis.md

Sections:
 1. Per-backbone panel accuracy (test split, 95% bootstrap CI) + paired
    SC-CoT vs Single-CoT tests with Holm over backbones + TOST equivalence.
 2. Factorial ablation (gpt-4o): reasoning x temperature x voting.
 3. Nested K-sweep from the per-instance sample pool (permutation-averaged).
 4. DTT-Sym vs DTT-FC (forced-choice reconciliation).
 5. Cost/latency table per strategy per backbone.
 6. Parse-status and order (position) diagnostics.
"""
from __future__ import annotations
import json, os
from collections import Counter, defaultdict
import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "campaign_analysis.md")
RNG = np.random.default_rng(0)
B = 5000
EQ_DELTA = 0.02
MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
PRICES = {"gpt-4o": (2.50, 10.00), "gpt-4o-mini": (0.15, 0.60), "gpt-4.1": (2.00, 8.00)}
PANEL = ["Direct", "DTT-Sym", "DTT-FC", "Decomp", "Single-CoT", "SC-CoT"]


def load(model):
    path = os.path.join(HERE, f"campaign_{model.replace('.', '_')}.jsonl")
    if not os.path.exists(path):
        return None
    recs = [json.loads(l) for l in open(path, encoding="utf-8")]
    return {r["iid"]: r for r in recs}


def acc_ci(corr):
    corr = np.asarray(corr, float)
    boots = RNG.choice(corr, (B, len(corr))).mean(axis=1)
    return corr.mean(), np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def mcnemar_exact(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * stats.binom.cdf(min(b, c), n, 0.5))


def paired(ca, cb):
    ca, cb = np.asarray(ca, float), np.asarray(cb, float)
    b = int(((ca == 1) & (cb == 0)).sum()); c = int(((ca == 0) & (cb == 1)).sum())
    d = ca - cb
    idx = RNG.integers(0, len(d), (B, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    lo90, hi90 = np.percentile(boots, 5), np.percentile(boots, 95)
    eq = (lo90 > -EQ_DELTA) and (hi90 < EQ_DELTA)
    return dict(d=d.mean(), lo=lo, hi=hi, b=b, c=c, p=mcnemar_exact(b, c),
                eq=eq, lo90=lo90, hi90=hi90)


def holm(ps):
    order = np.argsort(ps); out = [0.0] * len(ps); prev = 0.0
    for rank, i in enumerate(order):
        adj = max(min(1.0, (len(ps) - rank) * ps[i]), prev)
        out[i] = adj; prev = adj
    return out


def main():
    data = {m: load(m) for m in MODELS}
    L = ["# Batch B campaign analysis (order-randomized instrumented runs)", "",
         f"> analyze_campaign.py; bootstrap B={B}, seed 0; equivalence margin ±{EQ_DELTA}.", ""]
    for m in MODELS:
        n = len(data[m]) if data[m] else 0
        L.append(f"- {m}: {n} instances loaded")
    L.append("")

    # ---------- 1. Panel + paired per backbone ----------
    L += ["## 1. Per-backbone panel (test split)", ""]
    ps_primary, rows_primary = [], []
    for m in MODELS:
        d = data[m]
        if not d: continue
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        L += [f"### {m} (N_test={len(ids)})", "",
              "| Strategy | Accuracy [95% CI] |", "|---|---|"]
        for s in PANEL:
            corr = [int(d[i]["strategies"][s]["correct"]) for i in ids]
            a, lo, hi = acc_ci(corr)
            L.append(f"| {s} | {a:.3f} [{lo:.3f}, {hi:.3f}] |")
        sc = [int(d[i]["strategies"]["SC-CoT"]["correct"]) for i in ids]
        cot = [int(d[i]["strategies"]["Single-CoT"]["correct"]) for i in ids]
        r = paired(sc, cot)
        ps_primary.append(r["p"]); rows_primary.append((m, r, len(ids)))
        L.append("")
    hp = holm(ps_primary) if ps_primary else []
    L += ["### Primary comparison: SC-CoT − Single-CoT (per backbone, Holm over backbones)", "",
          "| Backbone | Δacc | 95% paired CI | b/c | p (Holm) | sig | TOST ±2%p |", "|---|---|---|---|---|---|---|"]
    for (m, r, n), ph in zip(rows_primary, hp):
        L.append(f"| {m} | {r['d']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['b']}/{r['c']} "
                 f"| {ph:.2g} | {'yes' if ph < 0.05 else 'no'} | "
                 f"{'equivalent' if r['eq'] else 'not established'} (90% CI [{r['lo90']:+.3f},{r['hi90']:+.3f}]) |")

    # ---------- 2. Factorial (gpt-4o) ----------
    d = data.get("gpt-4o")
    if d and "Direct-T0.7" in next(iter(d.values()))["strategies"]:
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        def corr(s): return [int(d[i]["strategies"][s]["correct"]) for i in ids]
        def corr_pool1():  # CoT T0.7 K=1 = first pool sample
            return [int(d[i]["strategies"]["SC-CoT"]["pool"][0] == d[i]["gold"]) for i in ids]
        def corr_dpool1():
            return [int(d[i]["strategies"]["Direct-SC"]["pool"][0] == d[i]["gold"]) for i in ids]
        cells = {
            "Direct-T0 (K=1)": corr("Direct"),
            "Direct-T0.7 (K=1)": corr("Direct-T0.7"),
            "Direct-SC (T0.7, K=5)": corr("Direct-SC"),
            "CoT-T0 (K=1)": corr("Single-CoT"),
            "CoT-T0.7 (K=1)": corr_pool1(),
            "CoT-SC (T0.7, K=5)": corr("SC-CoT"),
        }
        L += ["", f"## 2. Factorial ablation (gpt-4o, N_test={len(ids)})", "",
              "| Condition | Accuracy [95% CI] |", "|---|---|"]
        for name, cs in cells.items():
            a, lo, hi = acc_ci(cs)
            L.append(f"| {name} | {a:.3f} [{lo:.3f}, {hi:.3f}] |")
        L += ["", "| Effect | Δacc | 95% paired CI | b/c | p (raw) |", "|---|---|---|---|---|"]
        effects = [
            ("Reasoning (CoT-T0 − Direct-T0)", cells["CoT-T0 (K=1)"], cells["Direct-T0 (K=1)"]),
            ("Temperature, direct (T0.7 − T0)", cells["Direct-T0.7 (K=1)"], cells["Direct-T0 (K=1)"]),
            ("Temperature, CoT (T0.7 − T0)", cells["CoT-T0.7 (K=1)"], cells["CoT-T0 (K=1)"]),
            ("Voting w/o reasoning (Direct-SC − Direct-T0.7)", cells["Direct-SC (T0.7, K=5)"], cells["Direct-T0.7 (K=1)"]),
            ("Voting w/ reasoning (CoT-SC − CoT-T0.7)", cells["CoT-SC (T0.7, K=5)"], cells["CoT-T0.7 (K=1)"]),
        ]
        for name, ca, cb in effects:
            r = paired(ca, cb)
            L.append(f"| {name} | {r['d']:+.3f} | [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['b']}/{r['c']} | {r['p']:.2g} |")

    # ---------- 3. Nested K-sweep from pool ----------
    L += ["", "## 3. Nested K-sweep (permutation-averaged over the shared sample pool)", ""]
    NPERM = 200
    for m in MODELS:
        d = data.get(m)
        if not d: continue
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        pools = [d[i]["strategies"]["SC-CoT"]["pool"] for i in ids]
        golds = [d[i]["gold"] for i in ids]
        pn = min(len(p) for p in pools)
        Ks = [k for k in (1, 3, 5, 7, 9) if k <= pn]
        L += [f"### {m} (pool={pn}, {NPERM} permutations, N_test={len(ids)})", "",
              "| K | Mean acc | SD over perms | Δ vs K=1 (mean) | Mean vote entropy |", "|---|---|---|---|---|"]
        rng = np.random.default_rng(42)
        acc_by_k = {k: [] for k in Ks}; ent_by_k = {k: [] for k in Ks}
        base_k1 = []
        for _ in range(NPERM):
            perms = [rng.permutation(pn) for _ in pools]
            for k in Ks:
                cs, ents = [], []
                for pool, gold, pm in zip(pools, golds, perms):
                    sel = [pool[j] for j in pm[:k]]
                    cnt = Counter(sel); v, top = cnt.most_common(1)[0]
                    cs.append(int(v == gold))
                    probs = np.array([c / k for c in cnt.values()])
                    ents.append(float(-(probs * np.log(probs)).sum()))
                acc_by_k[k].append(np.mean(cs)); ent_by_k[k].append(np.mean(ents))
            base_k1.append(acc_by_k[1][-1])
        for k in Ks:
            accs = np.array(acc_by_k[k])
            dk = accs - np.array(base_k1)
            L.append(f"| {k} | {accs.mean():.3f} | {accs.std():.4f} | {dk.mean():+.3f} | "
                     f"{np.mean(ent_by_k[k]):.3f} |")
        L.append("")

    # ---------- 4. DTT-FC ----------
    L += ["## 4. Forced-choice reconciliation vs disagreement-to-tie", "",
          "| Backbone | Direct | DTT-Sym | DTT-FC | FC−Direct Δ [95% CI] | p | order-agree rate |",
          "|---|---|---|---|---|---|---|"]
    for m in MODELS:
        d = data.get(m)
        if not d: continue
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        cd = [int(d[i]["strategies"]["Direct"]["correct"]) for i in ids]
        ct = [int(d[i]["strategies"]["DTT-Sym"]["correct"]) for i in ids]
        cf = [int(d[i]["strategies"]["DTT-FC"]["correct"]) for i in ids]
        agree = np.mean([d[i]["strategies"]["DTT-Sym"]["order_agree"] for i in ids])
        r = paired(cf, cd)
        L.append(f"| {m} | {np.mean(cd):.3f} | {np.mean(ct):.3f} | {np.mean(cf):.3f} | "
                 f"{r['d']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}] | {r['p']:.2g} | {agree:.2f} |")

    # ---------- 5. Cost table ----------
    L += ["", "## 5. Cost / latency per strategy (test split, per 1,000 decisions)", ""]
    for m in MODELS:
        d = data.get(m)
        if not d: continue
        pin, pout = PRICES[m]
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        strats = list(next(iter(d.values()))["strategies"])
        L += [f"### {m}", "",
              "| Strategy | Accuracy | Calls | In tok | Out tok | Cost (USD/1k) | Latency p50/p95 (s) |",
              "|---|---|---|---|---|---|---|"]
        for s in strats:
            ms = [d[i]["strategies"][s]["m"] for i in ids]
            acc = np.mean([d[i]["strategies"][s]["correct"] for i in ids])
            calls = np.mean([x["calls"] for x in ms])
            it = np.mean([x["in_tok"] for x in ms]); ot = np.mean([x["out_tok"] for x in ms])
            cost1k = (it * pin + ot * pout) / 1e6 * 1000
            lat = np.array([x["latency"] for x in ms])
            L.append(f"| {s} | {acc:.3f} | {calls:.1f} | {it:.0f} | {ot:.0f} | ${cost1k:.2f} | "
                     f"{np.percentile(lat,50):.1f}/{np.percentile(lat,95):.1f} |")
        L.append(f"\n(assumed pricing USD/1M tokens: input {pin}, output {pout})\n")

    # ---------- 6. Diagnostics ----------
    L += ["## 6. Parse-status and position diagnostics", "",
          "| Backbone | Strategy | fallback-tie calls (test total) |", "|---|---|---|"]
    for m in MODELS:
        d = data.get(m)
        if not d: continue
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        strats = list(next(iter(d.values()))["strategies"])
        for s in strats:
            fb = sum(d[i]["strategies"][s]["m"]["fallback_tie"] for i in ids)
            if fb: L.append(f"| {m} | {s} | {fb} |")
    L.append("")
    L += ["| Backbone | Strategy | acc (original order) | acc (swapped) |", "|---|---|---|---|"]
    for m in MODELS:
        d = data.get(m)
        if not d: continue
        ids = sorted(i for i, r in d.items() if r["split"] == "test")
        for s in ["Direct", "Single-CoT", "SC-CoT"]:
            a0 = np.mean([d[i]["strategies"][s]["correct"] for i in ids if not d[i]["swap"]])
            a1 = np.mean([d[i]["strategies"][s]["correct"] for i in ids if d[i]["swap"]])
            L.append(f"| {m} | {s} | {a0:.3f} | {a1:.3f} |")

    open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
