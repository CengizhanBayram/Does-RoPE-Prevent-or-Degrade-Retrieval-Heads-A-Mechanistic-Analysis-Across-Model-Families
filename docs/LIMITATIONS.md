# Limitations & Threats to Validity

This document collects the caveats that must appear in the paper's *Limitations*
section (or be addressed before submission). Each item maps to the
publication-readiness checklist.

## Methodological

- **A1 — Retrieval-head metric is an adapted proxy.** We do *not* re-implement
  the full copy-paste retrieval score of Wu et al. (ICLR 2025). We use a
  single forward pass and flag a head when its attention argmax at the final
  input position falls inside the needle span, aggregated as a hit rate. This is
  coarser and should be described as *"an adapted, single-pass attention-argmax
  variant of Wu et al. (2025)"*. It is adequate for relative comparisons across
  heads/models but is not directly comparable to numbers reported with the
  original method.

- **A2 — Quantization.** All models are loaded in 8-bit by default. Quantization
  perturbs attention distributions and can shift retrieval-head detection. Run
  the fp16 ablation (`--no_8bit`) on at least one model and report the delta.

- **A3 — Confounded cross-family comparison.** "Higher RoPE θ → fewer retrieval
  heads" (H1) is confounded across model families by training data, scale, and
  architecture. Causal language should be reserved for the controlled
  comparisons: Layer B (same OLMo-2 model across training checkpoints) and, more
  weakly, Layer C (LLaMA-2 vs LLaMA-3.1, same family, different θ). Treat the
  family-level comparison as correlational/exploratory.

- **A5 — Context length is nominal.** Haystacks are built to an approximate
  character budget; the actual tokenized length is recorded per sample
  (`actual_token_length`) and summarised in each results JSON. Report measured
  lengths, not the nominal `context_lengths` targets.

- **A6 — Needle distinctiveness.** The needle ("The secret passphrase is
  XXXXX.") is lexically very distinct from the shuffled PG-19 haystack, making
  retrieval comparatively easy. Results may not transfer to harder,
  distractor-rich retrieval settings.

- **A7 — Threshold choice.** `score_threshold` (default 0.1) and `k_dims`
  (default 16) are not derived from first principles. `scripts/run_threshold_sweep.py`
  reports headline metrics across thresholds; include it to show robustness.

## Statistical

- **B1 — Pseudoreplication.** Heads within a layer are not independent, so the
  Welch t-test over all heads overstates significance. We additionally report a
  **layer-clustered permutation test** (`clustered_permutation_p`), which is the
  statistic to emphasise.
- **B2 — Multiple comparisons.** Cross-model tests are FDR-corrected
  (Benjamini-Hochberg, `results/layer_a/fdr_summary.json`).
- **B3 — Seed variance.** Run multiple seeds (`--seeds` / `config niah.seeds`);
  `seed_aggregate.json` reports mean ± SD of the headline metrics.
- **B4 — Effect sizes.** Cohen's d and bootstrap 95% CIs accompany every mean
  comparison; the Layer-D causal effect also carries a bootstrap CI.
- **B5 — Correlation choice.** Retrieval scores are zero-inflated; Spearman's
  rho is the primary correlation statistic, Pearson is reported for comparison
  only.

## Reproducibility

- **C1 — Pin model revisions.** Replace `revision: "main"` in `configs/config.yaml`
  with exact commit SHAs before the reportable run; the loader warns otherwise.
- **C4 — Determinism.** Set `reproducibility.strict_determinism: true` for the
  final run. Even so, some CUDA kernels may remain non-deterministic; the
  multi-seed variance (B3) bounds residual noise.

## Data, licensing, compute

- **E1 — Data.** PG-19 is streamed; we keep the first N short sentences (see
  `src/corpus.py`). Report dataset version and access date.
- **G2 — Model licenses.** LLaMA-2/3.1 are gated with use restrictions; Qwen and
  OLMo are Apache-2.0. Cite each model/dataset license (see `LICENSE`).
- **F1 — Compute budget.** Report GPU type (L4 24 GB), total GPU-hours, and
  per-layer runtime.
