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

LLaMA-2 and LLaMA-3.1 are gated models. Request access at HuggingFace then authenticate:

```bash
huggingface-cli login
```

---

## Usage

### Layer A — Static Multi-Model Analysis

```bash
# Single model
python scripts/run_layer_a.py --model llama3 --n_samples 100

# All models sequentially
python scripts/run_layer_a.py --model all

# Custom config
python scripts/run_layer_a.py --model qwen --config configs/config.yaml
```

Outputs: `results/layer_a/{model}/results.json`, `results/layer_a/figures/figure1_scatter.pdf`, `figure2_frequency_profile.pdf`

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
│   ├── retrieval_head_detector.py   # Wu et al. NIAH scoring
│   ├── dimension_utility.py         # Chiang & Yogatama utility measurement
│   ├── olmo_checkpoint_loader.py    # OLMo-2 checkpoint management
│   ├── activation_patching.py       # Hook-based dimension zeroing
│   ├── niah_evaluator.py            # Generation-based NIAH harness
│   └── visualization.py            # All figure functions
├── scripts/
│   ├── run_layer_a.py
│   ├── run_layer_b.py
│   ├── run_layer_c.py
│   └── run_layer_d.py
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

**8-bit quantization** — all 7B models are loaded with `bitsandbytes` int8 quantization (~8 GB VRAM each). Falls back to fp16 if bitsandbytes is unavailable.

**Reproducibility** — `torch`, `numpy`, and `random` seeds are fixed at the start of every run. Seed value is stored in every result JSON.

---

## Result Format

Each `results.json` follows this schema:

```json
{
  "model": "llama3",
  "timestamp": "...",
  "seed": 42,
  "hardware": { "gpu": "NVIDIA L4", "vram_total_gb": 24.0 },
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

`hypothesis_supported` logic:
- `p < 0.05` and `retrieval_mean_utility < non_retrieval_mean_utility` → **H2** (degradation)
- `p < 0.05` and `retrieval_mean_utility > non_retrieval_mean_utility` → **anomaly**
- `p ≥ 0.05` → **inconclusive**

---

## Related Work

| Paper | Relevance |
|-------|-----------|
| Wu et al., ICLR 2025 | Retrieval heads: universal, sparse, intrinsic — defines our scoring metric |
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
