# Does RoPE Prevent or Degrade Retrieval Heads?
### A Mechanistic Analysis Across Model Families

Code, data, and a reproducibility harness for the paper:

> **Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families**
> Cengizhan Bayram (Independent Researcher).
> arXiv preprint, 2026. *(arXiv ID added on announcement.)*

This repository is **self-contained and auditable**: every number in the paper traces to a
raw JSON under [`rope_retrieval_results/`](rope_retrieval_results/) (see that folder's
[README](rope_retrieval_results/README.md) for the file→table/figure map).

---

## TL;DR — what we find

We ask whether rotary position embeddings (RoPE), and their base $\theta$, *prevent* or
*degrade* the retrieval heads that long-context recall depends on. Across four open-weight
7–8B models (two attention regimes, a 100× range of $\theta$), plus three larger/extra-family
models for the causal test, we find that **neither intuitive story holds**:

1. **Retrieval heads are causally necessary.** Masking the detected heads collapses
   needle-in-a-haystack recall (OLMo-2 1.00→0.00; Qwen2.5 1.00→0.58), while masking an equal
   number of random heads does nothing — a double dissociation in two attention families.
2. **Higher $\theta$ does not reduce retrieval-head count** (H1 *prevention* not supported;
   a directional, confounded refutation).
3. **The norm–utility link is family-specific**, significant in *opposite* directions
   (Qwen $d=-0.49$, OLMo $d=+0.50$; LLaMA null). Because OLMo and LLaMA-3.1 share
   $\theta=500\mathrm{k}$ yet differ, the effect is **not $\theta$-driven** — there is no single
   universal law.
4. **The causal axis is RoPE *frequency*, not norm-utility.** Zeroing the lowest-frequency
   (long-wavelength) dimensions of retrieval heads degrades recall dose-dependently
   (1.00→0.18 at 32/128 dims, vs 0.98 for random dims); the effect is head-specific and
   task-specific. The *direction* holds in all five models patched (OLMo-2, Qwen2.5-7B/14B,
   Gemma-2-9B, Mistral-7B; four lineages). We **do not** claim cross-model magnitude
   (confounded by coverage and detector localization).

Built on Chiang & Yogatama (ACL Findings 2025), whose causal frequency result we replicate
with matched controls, a frequency-aware ordering, a dose-response curve, and multi-seed
significance.

---

## Models

| Role | Model | Attn. | $\theta$ | License |
|------|-------|-------|------|---------|
| Core panel | LLaMA-2-7B | MHA | 10,000 | Llama-2 |
| Core panel | LLaMA-3.1-8B | GQA | 500,000 | Llama-3.1 |
| Core panel | Qwen2.5-7B | GQA | 1,000,000 | Apache-2.0 |
| Core panel | OLMo-2-7B | MHA | 500,000 | Apache-2.0 |
| Causal test (extra) | Qwen2.5-14B, Gemma-2-9B, Mistral-7B | GQA | — | (resp.) |

Exact HuggingFace commit hashes are pinned in [`configs/config.yaml`](configs/config.yaml)
(and listed in the paper's appendix).

---

## Pipeline

The study has three analyses, labelled by pipeline stage (there is **no separate Layer C** —
an early position-conditioned analysis was folded into A/B):

| Stage | What | Models | Approx. runtime |
|-------|------|--------|-----------------|
| **A** | Static multi-model detection + dimension-utility test | four-model panel | ~6–9 h |
| **B** | OLMo-2 training dynamics (retrieval heads emerge over pretraining) | OLMo-2-7B | ~12–25 h |
| **D** | Causal validation: head-masking knockout + RoPE-frequency patch | OLMo-2, Qwen2.5, + extras | ~3–6 h |

The within-family $\theta$ comparison (LLaMA-2 vs LLaMA-3.1) used for the H1 test in Section 4
is produced by `scripts/run_layer_c.py`. The core panel and 4096-token patches fit a single
**NVIDIA L4 24 GB** (8-bit); the 8192-token patch and the larger models (14B/9B) use a
**Colab A100 40 GB**.

---

## Setup

```bash
pip install -r requirements.txt          # Python 3.10+, CUDA 12.x recommended
```

**Gated models.** LLaMA-2 / LLaMA-3.1 require HuggingFace access. Authenticate with:

```bash
huggingface-cli login          # interactive, OR
export HF_TOKEN=hf_xxx          # env var (also read from a .env file)
```

`src/auth_utils.py` resolves the token from argument → env var → `.env`.
**Never commit tokens** — `.env`, `*.token`, `hf_token*` are git-ignored.

**Pin revisions before a reportable run.** `configs/config.yaml` ships `revision: "main"`
per model; replace each with the exact commit SHA for reproducibility (the loader warns while
a revision is unpinned). See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

---

## Reproduce

### Command line

```bash
# Layer A — static multi-model detection (+ multi-seed, fp16 ablation, threshold sweep)
python scripts/run_layer_a.py --model all --seeds 42 123 2024
python scripts/run_layer_a.py --model olmo --no_8bit          # 8-bit vs fp16 ablation
python scripts/run_threshold_sweep.py --model all             # threshold robustness

# Layer B — OLMo-2 training dynamics (resume-safe)
python scripts/run_layer_b.py --step_stride 20000 --resume

# H1 theta comparison (LLaMA-2 vs LLaMA-3.1)
python scripts/run_layer_c.py

# Layer D — knockout + frequency patch
python scripts/run_layer_d.py --k_dims 16
python scripts/run_quant_ablation.py                          # 8-bit vs fp16 detection
python scripts/diagnose_gqa.py                                # GQA / rotate-half diagnostic
```

### Google Colab (no local GPU)

Notebooks in [`notebooks/`](notebooks/) mirror the paper's runs; open in Colab, pick the GPU
runtime in each notebook's header, run top to bottom:

| Notebook | Covers |
|----------|--------|
| `full_analysis_colab.ipynb` | Layers A/B/D end-to-end (L4) |
| `multiseed_analysis_colab.ipynb` | 3-seed Layer A (Table 2/10) |
| `layer_d_only_colab.ipynb` | Knockout + frequency patch (Tables 3–5) |
| `quant_ablation_colab.ipynb` | 8-bit vs fp16 (Table 9) |
| `proxy_robustness_colab.ipynb` | argmax vs copy-score detector (Tables 8/15) |
| `reviewer_bigmodels_a100_colab.ipynb` | Qwen-14B / Gemma-9B coverage (A100; Tables 6/7) |
| `reviewer_mistral_coverage_colab.ipynb` | Mistral coverage sweep (Table 6) |
| `reviewer_copyhead_freq_colab.ipynb` | copy-score frequency patch (Table 8) |
| `review_runs_colab.ipynb`, `reviewer_experiments*_colab.ipynb` | long-context / extra runs |

> Notebook tokens are **placeholders** — set your own `HF_TOKEN` before running.

---

## Repository layout

```
├── configs/config.yaml          # hyperparameters + pinned model revisions
├── src/                         # library (hook-based, no weight modification)
│   ├── retrieval_head_detector.py   # single-pass attention-argmax proxy of Wu et al. (2025)
│   ├── dimension_utility.py         # Chiang & Yogatama utility + frequency ordering
│   ├── activation_patching.py       # forward-hook dimension zeroing / head masking
│   ├── olmo_checkpoint_loader.py    # OLMo-2 pretraining-checkpoint management (Layer B)
│   ├── niah_evaluator.py            # generation-based NIAH harness (exact-match scoring)
│   ├── corpus.py                    # shared PG-19 haystack (single source)
│   ├── stats_utils.py               # Cohen's d, bootstrap CI, layer-clustered perm test, BH-FDR, McNemar
│   ├── repro.py / model_loader.py / auth_utils.py / visualization.py
├── scripts/                     # run_layer_{a,b,c,d}.py, run_quant_ablation.py, run_threshold_sweep.py, diagnose_gqa.py
├── notebooks/                   # Colab notebooks (table above)
├── tests/                       # pytest unit tests (pure-logic, CI-checked)
├── docs/                        # LIMITATIONS.md, REPRODUCIBILITY.md
├── rope_retrieval_results/      # ALL raw results (JSON) — every paper number traces here
│   └── README.md                #   file → table/figure map + JSON field glossary
└── results/                     # auto-created when you re-run (git-ignored)
```

---

## Design decisions (why you can trust the numbers)

- **No weight modification** — all interventions are PyTorch forward hooks, removed after each
  condition; weights are never altered on disk.
- **Cluster-aware statistics** — heads within a layer are not independent, so the primary test
  is a **layer-clustered permutation test** (not a naive per-head t-test, which would
  pseudoreplicate); we also report Cohen's d, bootstrap 95% CIs, exact McNemar (paired patch),
  and Benjamini–Hochberg FDR across models.
- **Validate the claim, not the metric** — the detector is an *adapted single-pass proxy* of
  Wu et al. (2025); conclusions are checked against a stricter teacher-forced copy score and
  an 8-bit-vs-fp16 ablation.
- **Coverage-fair cross-model patches** — cross-model comparisons always patch a fixed
  *fraction* of each run's detected heads (head counts vary with detection context/threshold;
  every run's count is mapped in the paper's provenance table).
- **Resume safety** — each result is written to disk before the model is unloaded; runs resume
  with `--resume`, so the full pipeline fits short single-GPU sessions.
- **Reproducibility** — seeds fixed via `src/repro.set_determinism`; every result JSON embeds an
  `environment` block (package versions, GPU, git commit). See
  [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) and [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

---

## Versions

This single repository tracks every version of the project:

- **`arxiv-v1`** (git tag + GitHub Release) — the exact code and data matching the arXiv v1
  preprint. Check this out to reproduce the paper as published.
- **`main`** — ongoing work toward the extended (journal) version; see the Roadmap below.

```bash
git checkout arxiv-v1     # reproduce the preprint exactly
git checkout main         # latest, including new experiments
```

Each subsequent paper version gets its own tag (`v2`, …); the arXiv link does not change.

## Roadmap (extended version)

Planned additions that strengthen the work toward a peer-reviewed venue:

- **Real long-context benchmarks** (RULER / LongBench): test whether the same heads and the
  frequency dependence transfer beyond synthetic NIAH.
- **Within-model $\theta$ control**: vary the RoPE base at inference on a *fixed* model to
  isolate $\theta$ from data/tokenizer/architecture (a clean H1 test).
- **Copy-score sweep across all five models** to settle cross-model magnitude.
- **More seeds** for the currently single-seed cross-family / long-context / copy-score runs.
- **Larger, size-varied model panel** to turn the heterogeneity refutation into a positive account.

Contributions and independent replications are welcome — open an issue or PR.

---

## License

- **Code:** MIT (see [LICENSE](LICENSE)).
- **Released results** (`rope_retrieval_results/`): free to reuse with attribution.
- **Models / datasets** keep their own licenses: LLaMA-2 / LLaMA-3.1 are gated with use
  restrictions; Qwen2.5, OLMo-2 are Apache-2.0; Gemma-2 per its terms; PG-19 per its dataset
  card. Cite each in any publication.

## Citation

```bibtex
@misc{bayram2026doesropepreventdegrade,
      title={Does RoPE Prevent or Degrade Retrieval Heads? A Mechanistic Analysis Across Model Families}, 
      author={Cengizhan Bayram},
      year={2026},
      eprint={2606.21249},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2606.21249}, 
}
```

*(Replace `arXiv:XXXX.XXXXX` with the assigned identifier once the preprint is announced.)*

## Acknowledgements

This work builds directly on **Wu et al. (ICLR 2025)** (retrieval heads) and
**Chiang & Yogatama (ACL Findings 2025)** (RoPE dimension inefficiency), and uses the
publicly released intermediate pretraining checkpoints of **OLMo-2 (Ai2)**.
