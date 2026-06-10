# Raw experimental results

This directory holds **all raw outputs** behind the figures and tables in the
paper. Every number reported in the paper can be traced to a JSON here. Files are
small (JSON + a few `.npy`/`.npz` arrays); model weights are **not** included
(load them from HuggingFace at the pinned revisions, see the repo root README).

All runs use 8-bit weights on a single 24 GB GPU (long-context and larger-model
runs use a 40 GB A100); seeds and environment (package versions, GPU, git commit)
are embedded inside each `results.json`.

> **Note on head counts.** The number of detected retrieval heads is *not* fixed:
> it depends on the detection context length and threshold, so it varies modestly
> across runs (e.g. OLMo 81–95, Qwen 58–64, Mistral 96–98). Each file reports its
> own run's count; cross-model patches always cover a fixed *fraction* of that
> run's detected set.

---

## Folder map (→ where it appears in the paper)

| Path | Contents | Paper |
|------|----------|-------|
| `layer_a/{llama2,llama3,qwen,olmo}/results.json` | Per-model static detection: retrieval heads, per-head scores, dimension-utility stats (Cohen's d, clustered permutation p, partial Spearman). | Layer A, Table 2 |
| `layer_a/*/norms.npy` | Per-(layer,head) L1 query-projection norm grid (the utility signal). | — (raw) |
| `layer_a/*/retrieval_scores.npy` | Per-(layer,head) mean argmax retrieval score. | — (raw) |
| `layer_a/olmo/head_masking.json` | Head-masking knockout on OLMo (baseline → retrieval-masked → random-masked). | Table 3 (OLMo row) |
| `layer_a/olmo/niah_*.npy` | Per-sample NIAH correctness for the knockout conditions. | Table 3 |
| `layer_a/fdr_summary.json` | Cross-model Benjamini–Hochberg FDR over the four clustered-permutation p-values (2/4 rejected: Qwen, OLMo). | Section 3.6 / Table 2 bold |
| `layer_a/threshold_sweep.json` | Detection-threshold robustness (τ ∈ [0.05, 0.3]) per model. | Section 4 "Robustness" |
| `roperetrivalresults_final/layer_a/*/seed_aggregate.json` | 3-seed (42/123/2024) aggregates: head count, d, clustered p, per seed. | Table 2 (mean±SD), Table 10 |
| `layer_b/checkpoint_*/results.json` | Full Layer-A pipeline re-run at 45 OLMo-2 pretraining checkpoints (84–3859 B tokens). | Layer B, Figure 2 |
| `layer_c/comparison.json` | θ comparison LLaMA-2 (θ=10k) vs LLaMA-3.1 (θ=500k): head counts, the H1 assessment, confound note. | H1 (Section 4) |
| `layer_d/olmo_new/population_patch.json` | Representative OLMo population patch (n=200, k=16): all conditions, McNemar, CI, specificity. | Table 4, Table 11 (seed 42) |
| `layer_d/olmo/...` | Earlier OLMo population patch (n=150 seed-42 config — see caveat below). | superseded |
| `layer_d_seeds/olmo/seed{42,123,2024}/population_patch.json` | OLMo 3-seed population patch (all n=200). | Table 11 |
| `roperetrivalresults_review/dose_response/olmo/dose_response.json` | OLMo dose-response: NIAH acc vs k∈{8,16,32,48,64} low-freq vs random dims. | Table 5, Figure 3 |
| `roperetrivalresults_review/knockout/qwen/knockout.json` | Head-masking knockout on Qwen (1.00 → 0.58). | Table 3 (Qwen row) |
| `roperetrivalresults_review3/layer_d_qwen/qwen/seed*/population_patch.json` | Qwen 3-seed population patch. | Table 12 |
| `roperetrivalresults_review3/layer_d_longctx/qwen_ctx8192.json` | Qwen patch at 8192 tokens (fixed top-30 coverage). | Table 13 |
| `roperetrivalresults_review3/layer_d_bigmodel/mistral7b_ctx4096.json` | Mistral top-30 @4096 (the under-coverage near-null). | Section 6 (Mistral null) |
| `roperetrivalresults_review3/gqa_kv_group_distribution.json` | Retrieval-head spread across KV groups (Qwen, LLaMA-3.1). | Table 14 |
| `bigmodels_results/coverage_sweep/{qwen2.5-14b,gemma2-9b}_top*.json` | Head-coverage dose-response (30/50/100% of detected heads). | Table 6 (Qwen-14B), Section 6 (Gemma) |
| `roperetrivalresults_mistralcov/mistral_coverage/mistral_top{30,60,97}.json` | Mistral head-coverage dose-response. | Table 6 (Mistral) |
| `bigmodels_results/matched_fraction/*_frac50.json`, `roperetrivalresults_mistralcov/matched_fraction/*_frac50.json` | Frequency patch at matched 50% coverage, all five models. | Table 7 |
| `ablations/quant_ablation_olmo.json` | 8-bit vs fp16 detection on OLMo (head Jaccard, score ρ, Cohen's d both precisions). | Table 9 |
| `proxy_validation/proxy_robustness.json`, `roperetrivalresults_final/proxy_validation/*` | Argmax proxy vs teacher-forced copy score (ρ, top-N Jaccard, d under each detector). | Table 15 |
| `diagnostics/gqa_rope_diagnostic.json` | GQA / rotate-half RoPE boundary diagnostic. | Section 3.4 |
| `*/figures/*.pdf` | Source PDFs for the paper figures. | Figures 1–6 |

---

## JSON field glossary (population-patch files)

| Field | Meaning |
|-------|---------|
| `baseline` | NIAH accuracy with no patch (should be ~1.00 for the test to be interpretable). |
| `low_freq` / `high_freq` | Accuracy after zeroing the k lowest- / highest-frequency RoPE dims **in the retrieval heads**. |
| `low_utility` / `high_utility` / `random` | Same, but zeroing low-/high-L1-norm dims, or random dims (controls for the utility axis). |
| `frequency_effect` | `low_freq − baseline` (negative = low-freq removal hurts recall). The headline causal number. |
| `low_freq_control_heads` (= **ctrl** in tables) | Accuracy after zeroing the same low-freq dims in **layer-matched non-retrieval heads**. A clean, head-specific effect leaves this near **1.00**; a low value means the effect "leaks" to non-retrieval heads. |
| `retrieval_head_specificity` | Gap between the retrieval-head drop and the control drop. |
| `frequency_mcnemar` | Exact McNemar test on paired (low- vs high-freq) per-sample correctness: `b`, `c`, `n_discordant`, `p_value`. |
| `frequency_effect_ci95` | Bootstrap 95% CI on the paired accuracy difference. |
| `perplexity_ratio`, `specificity_ratio`, `specificity_verdict` | Task-specificity control: plain-text perplexity change vs NIAH drop; retrieval-specific iff ratio < 0.33. |
| `n_detected`, `top_k`, `coverage` | This run's detected head count, the number patched, and the patched fraction. |

`d_h/8` doses (k=32 for the 256-dim Gemma heads, k=16 for 128-dim heads) keep the
patched fraction of each head constant across models.

---

## Caveat: superseded aggregate file

`layer_d_seeds/olmo/population_aggregate.json` averages an **early seed-42 run
collected at n=150** (frequency effect −0.053), giving a mean of ≈ −0.094. The
paper does **not** use this file: it reports the seed-42 run at the same **n=200**
as the other seeds (`layer_d/olmo_new/population_patch.json`, −0.115), so the
3-seed OLMo frequency effect is **−0.115 ± 0.025** (see `seed42/`, `seed123/`,
`seed2024/`). Use the per-seed files, not the stale aggregate.
