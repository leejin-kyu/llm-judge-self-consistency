# Information that was not recorded and cannot be recovered

These items are reported as unknown in the paper and are not reconstructed or inferred.

| Item | Affected runs | Consequence |
|---|---|---|
| Resolved OpenAI model snapshot and `system_fingerprint` | every OpenAI run (requests used the aliases `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`) | Runs on different dates (15–16 June, 13 July, 22–23 July 2026) may have used different model weights. |
| Raw text of the stochastic (T = 0.7) votes | every vote pool | Explicit ties, parser-fallback ties, and outputs truncated before the verdict line cannot be separated. Only parsed labels exist. |
| Ollama model digests, quantization levels, effective context windows | open-weight judges | Only the tags are known. |
| Request timestamps | open-weight judges, 16 June backbone replications and K-sweeps | Only dates are known. |
| Per-instance records | 16 June backbone replications (`multibackbone.json`), independent K-sweeps (`ksweep_*.json`), stochastic-cache audit (`caching_experiment.json`) | Only aggregate accuracies were saved; paired tests and decompositions are not possible for these runs. |
| Remaining instances of the 13 July campaign | gpt-4o-mini (3 of 419), gpt-4o (213 of 419), gpt-4.1 (all) | The run stopped when the API quota was exhausted. |

Raw outputs **are** available for every deterministic (T = 0) call that was cached (`judge_cache.json`, `cache_*.json`),
including all 419 canonical Single-CoT outputs.
