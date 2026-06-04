# Limitations & Threats to Validity

This document collects the caveats that must appear in the paper's *Limitations*
section (or be addressed before submission). Each item maps to the
publication-readiness checklist.

## Core contribution (paper §6) — frequency vs. utility causal test

Layer D patches each retrieval head along **two independent axes** and measures
the drop in NIAH accuracy:

- **Utility axis:** zero the k lowest- vs highest-L1-norm dimensions
  (`low_utility` / `high_utility`), with a `random` control.
  `causal_effect = random − low_utility`.
- **Frequency axis (the novel test):** zero the k lowest- vs highest-RoPE-
  *frequency* dimensions (`low_freq` / `high_freq`).
  `frequency_effect = low_freq − high_freq`.

Because low-frequency and high-utility dimensions are correlated but not
identical, the frequency axis isolates the question *"is retrieval degraded by
RoPE frequency specifically, or merely by general dimension utility?"* — a
distinction prior work does not make. The bootstrap CI on `frequency_effect`
(`mean_frequency_effect_ci95` in the Layer-D results JSON) determines whether
the effect excludes zero. **The finding is publishable in either direction**:
if the CI excludes zero, retrieval is frequency-specific; if it includes zero,
RoPE frequency is *not* the causal driver beyond general utility. Report it
plainly either way.

## Methodological

- **A1 — Retrieval-head metric is an adapted proxy.** We do *not* re-implement
  the full copy-paste retrieval score of Wu et al. (ICLR 2025). We use a
  single forward pass and flag a head when its attention argmax at the final
  input position falls inside the needle span, aggregated as a hit rate. This is
  coarser and should be described as *"an adapted, single-pass attention-argmax
  variant of Wu et al. (2025)"*. It is adequate for relative comparisons across
  heads/models but is not directly comparable to numbers reported with the
  original method.

- **A2 — Quantization.** All models load in 8-bit by default. Because detection
  is argmax-based (discrete), a tiny continuous rounding shift can flip a head's
  argmax when its top-2 positions are close — quantization can move detection
  more than "slightly". `scripts/run_quant_ablation.py` (8-bit vs fp16, one
  representative model, seq=4096) reports THREE levels for BOTH the argmax score
  and the quantization-robust **mass score** (needle-span attention mass, via
  `score_heads(return_mass=True)`): (1) head-set Jaccard overlap, (2) per-head
  score correlation, (3) whether the retrieval-vs-non-retrieval utility gap
  keeps the same sign/significance. Defend stability of the *finding direction*,
  not byte-identical head sets. One model suffices (architecture-independent
  numerical artifact); the verdict is written to `quant_ablation_<model>.json`.

- **P10 — Layer C (θ) is confounded, not a clean θ manipulation.** LLaMA-2 vs
  LLaMA-3.1 differ in RoPE θ (10K vs 500K) **and** in pretraining data, token
  count, tokenizer, and attention scheme (MHA vs GQA). So a Layer-C difference
  (or null) cannot be attributed to θ alone. State this explicitly: causal θ
  isolation rests on (a) Layer D's within-model frequency patching (§6) and
  (b) a future single-model θ-sweep (retrain/finetune one model at several θ).
  Report Layer C as *suggestive*, with multi-seed CIs (e.g. "31±4 vs 28±3,
  difference not significant").

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
  `seed_aggregate.json` reports mean ± SD of the headline metrics. Report e.g.
  "47 ± 3 retrieval heads (5 seeds)", not a single number.
- **§2.3 — Paired-seed control.** Seeds are *NIAH-sampling* seeds (weights are
  fixed). For cross-model comparison the SAME seed must yield identical data for
  every model. We enforce this in two stages: `generate_niah_specs(seed)`
  produces tokenizer-independent text specs (RNG consumption never depends on a
  tokenizer), and `paired_spec_subset` keeps only specs that tokenize in EVERY
  model (intersection-drop) so all models are scored on an equal n. Dropped
  specs are logged. Layer B is single-seed per checkpoint (full sweep) with a
  5-seed crystallization-step validation (`run_layer_b.py --validate_steps`).
  Regression-guarded by `tests/test_paired_seed.py`.
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
