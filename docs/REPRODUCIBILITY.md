# Reproducibility Guide

Everything needed to reproduce the results, in the order a reviewer would check.

## 1. Environment

```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt   # for tests
```

Python 3.10+ and CUDA 12.x. Each results JSON embeds an `environment` block
(Python, package versions, GPU/CUDA, git commit + dirty flag) captured by
`src/repro.py` — check it matches when comparing runs.

## 2. Pin everything before the reportable run

- **Model weights (item C1):** edit `configs/config.yaml` and replace each
  `revision: "main"` with the exact commit SHA from the model's HuggingFace
  "Files and versions" page. The loader logs a warning while a revision is
  unpinned.
- **Determinism (item C4):** set `reproducibility.strict_determinism: true`.
- **Seeds (item B3):** `niah.seeds` lists the seeds used; the defaults are
  `[42, 123, 2024]`.

## 3. Run

```bash
# Layer A — static multi-model analysis, all models, all seeds
python scripts/run_layer_a.py --model all

# fp16 quantization ablation (item A2)
python scripts/run_layer_a.py --model llama3 --no_8bit

# Threshold sensitivity (item A7) — reuses saved scores, no model reload
python scripts/run_threshold_sweep.py --model all

# Layer B — OLMo training dynamics (revision-based checkpoints)
python scripts/run_layer_b.py --step_stride 20000 --resume

# Layer C — θ comparison (LLaMA-2 vs LLaMA-3.1)
python scripts/run_layer_c.py

# Layer D — activation patching (causal test)
python scripts/run_layer_d.py --model llama3
```

## 4. What each run records

- `results/layer_a/<model>/results.json` — retrieval heads, scores, dimension
  utility, and `statistical_tests` including `cohens_d`, bootstrap CI
  (`ci_low`/`ci_high`), and the layer-clustered permutation p-value.
- `results/layer_a/<model>/seed_aggregate.json` — mean ± SD across seeds.
- `results/layer_a/fdr_summary.json` — Benjamini-Hochberg correction across
  models (item B2).
- `results/layer_a/threshold_sweep.json` — robustness to `score_threshold`.
- `results/layer_d/<model>/results.json` — causal effects with a bootstrap CI.

## 5. Tests

```bash
pytest tests/ -q
```

Pure-logic tests (stats, corpus) run anywhere; model-dependent tests are skipped
automatically when torch is unavailable (e.g. in CI).

## 6. Artifact sharing

`results/` and `*.npy` are git-ignored because they are large outputs. Share the
final result JSONs and figures via an archival host (Zenodo/OSF) and cite the
DOI in the paper (item E2).
