# Does Self-Consistency Help LLM Judges? Decomposing Majority Voting in Pairwise Comparison — replication package

Prompts, sample identifiers, vote-level records, run settings, and analysis code for the paper
*Does Self-Consistency Help LLM Judges? Decomposing Majority Voting in Pairwise Comparison* (Lee, Sung, and Park).
Everything in the paper can be regenerated offline from these records; no API key is needed.

## Quick start

```bash
pip install -r requirements.txt
cd code && python fetch_benchmarks.py && cd ..     # benchmark texts (not redistributed here)
python reproduce_all_tables.py                     # every number, table, and figure
python -m pytest code/tests -q                     # cache-bypass test (no key needed)
```

The interval estimates of the paper are written to `code/revision_numbers_v24.md` and `code/revision_numbers_v25.md`
(shared percentile bootstrap, 100,000 resamples, seed 20261005).

## Contents

| Path | What it is |
|---|---|
| `code/prompts/` | system and user templates of every strategy, including the forced-choice, constraint-focused, and rewriting prompts (exported from the runners) |
| `code/splits/` | development/test split (seed 0), JudgeBench 120-pair sample (seed 42), repeated-pool sample (seed 7) |
| `code/records.json`, `code/results_final.csv` | canonical GPT-4o run: every strategy and all five SC-CoT votes per instance |
| `code/*.jsonl` | one journal per follow-up run with parsed verdicts, requested model string, and timestamps where logged |
| `code/judge_cache.json`, `code/cache_*.json` | raw outputs of all cached deterministic (T = 0) calls, keyed by SHA-1 of the request |
| `code/analyze_*.py` | analyses; `analyze_v24_revisions.py` and `analyze_v25_revisions.py` produce the intervals reported in the paper |
| `code/run_*.py` | runners for the API and Ollama experiments; the API runners are resume-safe and, except `caching_experiment.py`, have a zero-cost `--mock` mode |
| `figures/` | figure scripts (Figure 1 requires a headless Chrome) |
| `run_settings.json` | requested model, temperature, K, output cap, candidate order, item set, and date of every run |
| `code/file_timestamps.json` | original modification times of the scripts, records, and split files (evidence for the study chronology) |
| `UNRECOVERABLE.md` | information that was not recorded (model snapshots, raw stochastic outputs, Ollama configuration) |

Instance identifiers: `<Subset>_<i:03d>` is the i-th item (0-based) of the LLMBar subset file; `JB_<pair_id>` is a JudgeBench pair.

## Run journals

| File | Run (2026, KST) |
|---|---|
| `records.json` | canonical GPT-4o run, 15 June |
| `campaign_gpt-4o-mini.jsonl`, `campaign_gpt-4o.jsonl` | instrumented run with random candidate order and nine-draw pools, 13 July (stopped at quota) |
| `ollama_*.jsonl` | open-weight judges, 14–21 July |
| `order_swap_stageA.jsonl`, `order_swap_stageB.jsonl` | swapped order at T = 0 (all test items) and K = 5 (37 errors, 37 controls), 22 July |
| `clean_direct_fc.jsonl`, `clean_singlecot.jsonl` | forced-choice Direct and plain re-judgment, 22 July |
| `external_judgebench.jsonl` | JudgeBench sample, 22 July |
| `repeat_pool_run2.jsonl`, `repeat_pool_run3.jsonl` | repeated pools, 22 July |
| `causal_rubric.jsonl`, `causal_counterfactual.jsonl` | constraint-focused prompt and length rewriting, 23 July |

## Re-running the experiments (optional, needs an API key)

Copy `code/.env.example` to `code/.env` and set `OPENAI_API_KEY`. Request a dated snapshot rather than an alias and
store the returned model identifier and system fingerprint: the original runs did not, which is why those fields are
missing here.

## Licences

Code: MIT (see `LICENSE`). Records and annotations in this repository: CC BY 4.0.
LLMBar and JudgeBench are subject to their own licences and are downloaded from their original sources.
