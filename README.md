# Does RoPE Prevent or Degrade Retrieval Heads?
## A Mechanistic Analysis Across Model Families

This repository contains all code for the paper:

> **"Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families"**

---

## Research Question

RoPE's dimension inefficiency at high frequencies (Chiang & Yogatama, ACL Findings 2025) raises two competing hypotheses:

- **H1 — Prevention**: Dimension inefficiency suppresses the *formation* of retrieval heads during training.
- **H2 — Degradation**: Retrieval heads form normally, but RoPE *degrades* their effectiveness at inference.

---

## Methodology at a Glance

| Layer | What | Models | Runtime |
|-------|------|--------|---------|
| A | Static multi-model analysis | LLaMA-3.1-8B, Qwen2.5-7B, OLMo-2-7B | ~6–9 h |
| B | OLMo-2 training dynamics (500+ checkpoints) | OLMo-2-7B | ~12–25 h |
| C | Natural θ experiment | LLaMA-2-7B vs LLaMA-3.1-8B | ~4–6 h |
| D | Activation patching (causality test) | Best model from Layer A | ~3–4 h |

All experiments run on a single **NVIDIA L4 24 GB** GPU using 8-bit quantization.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Python 3.10+ and CUDA 12.x recommended.

### HuggingFace Access

LLaMA-2 and LLaMA-3.1 are gated models. Request access at HuggingFace then
authenticate with any of:

```bash
huggingface-cli login                 # interactive
export HF_TOKEN=hf_xxx                 # env var (also read from a .env file)
```

`src/auth_utils.py` resolves the token from argument → env var → `.env`
(`HF_TOKEN`/`HUGGING_FACE_HUB_TOKEN`) and also exposes a minimal GitHub client
for pushing artifacts. **Never commit tokens** — `.env` is git-ignored.

### Pin model revisions before the reportable run

`configs/config.yaml` ships with `revision: "main"` per model. For a
reproducible run, replace each with the exact commit SHA (the loader warns while
a revision is unpinned). See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Usage

### Layer A — Static Multi-Model Analysis

```bash
# Single model
python scripts/run_layer_a.py --model llama3 --n_samples 100

# All models sequentially
python scripts/run_layer_a.py --model all

# Multiple seeds for variance estimation (item B3)
python scripts/run_layer_a.py --model all --seeds 42 123 2024

# fp16 quantization ablation (item A2)
python scripts/run_layer_a.py --model llama3 --no_8bit

# Threshold sensitivity — reuses saved scores, no model reload (item A7)
python scripts/run_threshold_sweep.py --model all
```

Outputs: `results/layer_a/{model}/results.json`, `seed_aggregate.json` (multi-seed),
`results/layer_a/fdr_summary.json` (cross-model FDR), figures under `results/layer_a/figures/`.

### Layer B — OLMo-2 Training Dynamics

```bash
# Faster run (~12 h, 25 checkpoints)
python scripts/run_layer_b.py --step_stride 20000 --resume

# Full run (~25 h, 50 checkpoints)
python scripts/run_layer_b.py --step_stride 10000 --resume
```

Outputs: `results/layer_b/checkpoint_{step}/results.json`, `results/layer_b/figures/figure3_training_dynamics.pdf`

### Layer C — θ Comparison

```bash
python scripts/run_layer_c.py
```

Outputs: `results/layer_c/{llama2,llama3}/results.json`, `results/layer_c/figures/figure4_theta_comparison.pdf`

### Layer D — Activation Patching

```bash
# Auto-selects model with most retrieval heads
python scripts/run_layer_d.py --k_dims 16

# Explicit model
python scripts/run_layer_d.py --model llama3 --k_dims 16
```

Outputs: `results/layer_d/{model}/results.json`, `results/layer_d/{model}/figures/figure5_patching.pdf`

### Google Colab

Open `notebooks/full_analysis_colab.ipynb` in Colab, select **L4 GPU** runtime, and run cells top to bottom. Results are saved to Google Drive automatically.

---

## Project Structure

```
.
├── configs/
│   └── config.yaml              # All hyperparameters and model paths
├── src/
│   ├── retrieval_head_detector.py   # NIAH scoring (adapted Wu et al. proxy — see A1)
│   ├── dimension_utility.py         # Chiang & Yogatama utility + inferential stats
│   ├── olmo_checkpoint_loader.py    # OLMo-2 checkpoint management
│   ├── activation_patching.py       # Hook-based dimension zeroing
│   ├── niah_evaluator.py            # Generation-based NIAH harness
│   ├── corpus.py                    # Shared haystack corpus (single source)
│   ├── stats_utils.py               # Cohen's d, bootstrap CI, BH-FDR, perm test
│   ├── repro.py                     # Determinism + environment/version capture
│   ├── model_loader.py              # Revision-pinned, quantization-aware loading
│   ├── auth_utils.py                # HF + GitHub token handling
│   └── visualization.py            # All figure functions
├── scripts/
│   ├── run_layer_a.py
│   ├── run_layer_b.py
│   ├── run_layer_c.py
│   ├── run_layer_d.py
│   └── run_threshold_sweep.py
├── tests/                           # pytest unit tests (CI-checked)
├── docs/
│   ├── LIMITATIONS.md               # Threats to validity (for the paper)
│   └── REPRODUCIBILITY.md           # Step-by-step reproduction guide
├── notebooks/
│   └── full_analysis_colab.ipynb   # Colab L4 notebook (all layers)
└── results/                         # Auto-created; one subdir per layer
    ├── layer_a/
    ├── layer_b/
    ├── layer_c/
    └── layer_d/
```

---

## Key Design Decisions

**No weight modification** — all patches are applied through PyTorch forward hooks and removed immediately after each condition. Model weights are never altered on disk.

**Resume safety** — every checkpoint result is written to disk before the model is unloaded. Interrupted runs can be resumed with `--resume`.

**8-bit quantization** — all 7B models are loaded with `bitsandbytes` int8 quantization (~8 GB VRAM each). Falls back to fp16 if bitsandbytes is unavailable. Quantization affects attention; run the `--no_8bit` ablation (item A2).

**Reproducibility** — seeds are fixed via `src/repro.set_determinism` (optionally
strict CUDA determinism); multi-seed runs report mean ± SD; every result JSON
embeds an `environment` block (package versions, GPU, git commit). See
[docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

**Statistics** — beyond the Welch t-test we report Cohen's d, bootstrap 95% CIs,
a layer-clustered permutation test (to avoid pseudoreplication), and
Benjamini-Hochberg FDR across models. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

**Method honesty** — the retrieval-head metric is an *adapted single-pass proxy*
of Wu et al. (2025), not their full copy-paste score (item A1).

---

## Result Format

Each `results.json` follows this schema:

```json
{
  "model": "llama3",
  "timestamp": "...",
  "seed": 42,
  "load_in_8bit": true,
  "actual_token_length": { "mean": 0, "min": 0, "max": 0 },
  "environment": {
    "python": "3.10.x",
    "packages": { "torch": "...", "transformers": "..." },
    "hardware": { "gpu": "NVIDIA L4", "vram_total_gb": 24.0 },
    "git_commit": "…", "git_dirty": false
  },
  "retrieval_heads": [[layer, head], ...],
  "retrieval_scores": [[...], ...],
  "dimension_utility": {
    "per_head_scalar": [[...], ...],
    "frequency_profile": { "retrieval_mean": [...], ... }
  },
  "statistical_tests": {
    "retrieval_mean_utility": 0.0,
    "non_retrieval_mean_utility": 0.0,
    "t_statistic": 0.0,
    "p_value": 0.0,
    "cohens_d": 0.0,
    "ci_low": 0.0,
    "ci_high": 0.0,
    "clustered_permutation_p": 0.0,
    "pearson_r": 0.0,
    "pearson_p": 0.0,
    "spearman_rho": 0.0,
    "spearman_p": 0.0
  },
  "summary": {
    "n_retrieval_heads": 0,
    "n_total_heads": 1024,
    "retrieval_fraction": 0.0,
    "hypothesis_supported": "H1 | H2 | inconclusive | anomaly"
  }
}
```

`hypothesis_supported` logic (a *screening* heuristic — report the
cluster-aware permutation test and FDR-corrected results as primary):
- `p < 0.05` and `retrieval_mean_utility < non_retrieval_mean_utility` → **H2** (degradation)
- `p < 0.05` and `retrieval_mean_utility > non_retrieval_mean_utility` → **anomaly**
- `p ≥ 0.05` → **inconclusive**

---

## Limitations & Reproducibility

Before submitting, read [docs/LIMITATIONS.md](docs/LIMITATIONS.md) (threats to
validity, mapped to the checklist) and [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md)
(exact reproduction steps). Key caveats: the retrieval metric is an adapted
proxy (A1); cross-family θ comparison is confounded (A3); quantization needs an
ablation (A2); pin model revisions and run multiple seeds before the final run.

## Testing

```bash
pip install -r requirements-dev.txt
pytest tests/ -q
```

CI (`.github/workflows/ci.yml`) runs the pure-logic tests on every push.

## License

Code: **MIT** (see [LICENSE](LICENSE)). Models and datasets keep their own
licenses — LLaMA-2/3.1 are gated with use restrictions; Qwen2.5 and OLMo-2 are
Apache-2.0; PG-19 per its dataset card. Cite each in any publication (item G2).

---

## Related Work

| Paper | Relevance |
|-------|-----------|
| Wu et al., ICLR 2025 | Retrieval heads: universal, sparse, intrinsic — we use an *adapted single-pass proxy* of their score (see [docs/LIMITATIONS.md](docs/LIMITATIONS.md) A1) |
| Chiang & Yogatama, ACL Findings 2025 | RoPE dimension inefficiency — defines our utility proxy |
| Yang et al., NeurIPS 2025 | RoPE+NoPE hybrid outperforms pure RoPE |
| LLaMA-4 iRoPE (Meta) | NoPE every 4 layers — natural ablation |
| OLMo-2 (Ai2) | 500+ public training checkpoints — enables Layer B |

---

## Citation

```bibtex
@article{bayram2025rope_retrieval,
  title   = {Does {RoPE} Prevent or Degrade Retrieval Heads?
             A Mechanistic Analysis Across Model Families},
  author  = {Bayram, Cengizhan},
  year    = {2025}
}
```
