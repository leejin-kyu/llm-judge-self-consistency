# -*- coding: utf-8 -*-
"""Regenerate every number, table, and figure of the paper from the released records. No model calls.

    python reproduce_all_tables.py            # analyses + figures
    python reproduce_all_tables.py --no-figs  # analyses only

Run code/fetch_benchmarks.py once first: some analyses read instance lengths and texts from the benchmark files.
Main outputs: code/revision_numbers_v24.md and code/revision_numbers_v25.md (all interval estimates of the paper),
code/*_analysis.md, and figures/*/out/*.pdf. Figure 1 (framework diagram) needs a headless Chrome and is skipped
unless --fig1 is given.
"""
import os, subprocess, sys

ROOT = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(ROOT, "code")
ANALYSES = [
    "build_final_csv.py",           # records.json -> results_final.csv
    "analyze_final.py",             # pooled accuracy, paired tests
    "analyze_voting_effect.py",     # K=1/3/5 within the shared pool, early stopping
    "analyze_reachability.py",      # decomposition, voting masses
    "analyze_verbosity.py", "analyze_verbosity2.py",   # response-length analyses (Online Resource S11)
    "analyze_ollama.py", "analyze_crossfamily_multi.py",
    "analyze_order_swap.py", "analyze_clean_core.py", "analyze_external_bench.py",
    "analyze_repeat_pools.py", "analyze_causal_probes.py", "analyze_campaign.py",
    "analyze_v18_revisions.py", "analyze_v19_revisions.py",
    "analyze_v24_revisions.py",     # shared bootstrap, tie conventions, comparators, parsing checks
    "analyze_v25_revisions.py",     # precision, pilot-to-test, swap gains/losses, 13 July run
]
FIGURES = [
    ("figs_v21", "build_data_v21.py"), ("figs_v21", "fig2_accuracy.py"), ("figs_v21", "fig3_reachability.py"),
    ("figs_v21", "fig_pools_persistence.py"), ("figs_v25", "fig5_cross_setting_v25.py"),
    ("figs_v24", "fig7_interventions_v24.py"), ("figs_v21", "fig5_pareto.py"),
    ("figs_v21", "fig4_ksweep.py"), ("figs_v21", "figB1_risk_coverage.py"),
]


def run(cwd, script):
    print(f"== {os.path.relpath(os.path.join(cwd, script), ROOT)} ==", flush=True)
    r = subprocess.run([sys.executable, "-X", "utf8", script], cwd=cwd)
    if r.returncode != 0:
        sys.exit(f"FAILED at {script}")


for s in ANALYSES:
    run(CODE, s)
if "--no-figs" not in sys.argv:
    if "--fig1" in sys.argv:
        FIGURES.insert(1, ("figs_v21", "fig1_framework.py"))
    for d, s in FIGURES:
        run(os.path.join(ROOT, "figures", d), s)
print("All tables, numbers, and figures regenerated.")
