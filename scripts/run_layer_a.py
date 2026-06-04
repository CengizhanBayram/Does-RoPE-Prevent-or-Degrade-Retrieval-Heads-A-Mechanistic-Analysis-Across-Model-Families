"""
Layer A: Static multi-model analysis.

For each model: detect retrieval heads via NIAH, compute dimension utility,
run statistical tests, save results, generate figures.

Estimated runtime: ~2–3 hours per model on L4 24 GB.

Usage:
    python scripts/run_layer_a.py --model llama3 --n_samples 100
    python scripts/run_layer_a.py --model all --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimension_utility import DimensionUtilityAnalyzer, diagnose_dimension_norms
from src.model_loader import load_model as _load_model_shared
from src.repro import capture_environment, set_determinism
from src.retrieval_head_detector import (
    RetrievalHeadDetector,
    generate_niah_specs,
    paired_spec_subset,
)
from src.stats_utils import benjamini_hochberg
from src.visualization import (
    plot_dimension_utility_profile,
    plot_retrieval_vs_utility_scatter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _hardware_info() -> dict:
    info: dict = {}
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 1
        )
        info["cuda_version"] = torch.version.cuda or "unknown"
    else:
        info["gpu"] = "CPU"
        info["vram_total_gb"] = 0
        info["cuda_version"] = "N/A"
    return info


def _determine_hypothesis(stats: dict) -> str:
    p = stats.get("p_value", float("nan"))
    ret_mean = stats.get("retrieval_mean_utility", float("nan"))
    non_ret_mean = stats.get("non_retrieval_mean_utility", float("nan"))
    if p != p:  # NaN
        return "inconclusive"
    if p < 0.05:
        if ret_mean < non_ret_mean:
            return "H2"
        return "anomaly"
    return "inconclusive"


def _load_model(model_name: str, model_cfg: dict, load_in_8bit: bool | None = None):
    """Load a model at its pinned revision (delegates to src.model_loader)."""
    return _load_model_shared(model_cfg, model_name, load_in_8bit=load_in_8bit)


def _unload_model(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Core analysis pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    model_name: str,
    model_cfg: dict,
    config: dict,
    n_samples: int,
    results_dir: Path,
    seed: int = 42,
    load_in_8bit: bool | None = None,
    persist: bool = True,
    specs: list[dict] | None = None,
) -> dict:
    """
    Run the full Layer-A pipeline for a single model.

    Args:
        persist: When True, write results.json and norms.npy to disk. Set False
            for the extra seeds of a multi-seed run (item B3) so the primary
            seed's artifacts (consumed by Layer D) are not overwritten.
        specs: Pre-generated, intersection-filtered NIAH specs for paired-seed
            comparison (§2.3). When None, samples are generated for this model
            alone (no cross-model pairing).

    Returns the result dict (also written to disk when persist=True).
    """
    strict = config.get("reproducibility", {}).get("strict_determinism", False)
    set_determinism(seed, strict=strict)
    out_dir = results_dir / "layer_a" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    niah_cfg = config["niah"]
    context_lengths: list[int] = niah_cfg["context_lengths"]
    needle_positions: list[float] = niah_cfg["needle_positions"]
    score_threshold: float = niah_cfg.get("score_threshold", 0.1)
    fig_fmt: str = config.get("output", {}).get("figure_format", "pdf")
    fig_dpi: int = config.get("output", {}).get("figures_dpi", 300)

    model, tokenizer = _load_model(model_name, model_cfg, load_in_8bit=load_in_8bit)

    # --- Retrieval head detection ---
    logger.info("[%s] Preparing NIAH samples …", model_name)
    detector = RetrievalHeadDetector(
        model, tokenizer, config, score_threshold=score_threshold, seed=seed
    )
    if specs is not None:
        # Paired-seed path: identical, intersection-filtered specs across models.
        samples = detector.prepare_samples(specs)
    else:
        samples = detector.generate_niah_samples(
            n_samples=n_samples,
            context_lengths=context_lengths,
            needle_positions=needle_positions,
        )

    logger.info("[%s] Scoring retrieval heads …", model_name)
    scores = detector.score_heads(samples)
    retrieval_heads = detector.get_retrieval_heads(scores)
    detector.save_scores(scores, out_dir / "retrieval_scores.npy")

    # --- Dimension utility ---
    logger.info("[%s] Computing dimension utility …", model_name)
    analyzer = DimensionUtilityAnalyzer(model, config)
    norms = analyzer.compute_query_projection_norms()
    utility_stats = analyzer.compute_utility_scores(norms, retrieval_heads)
    correlation = analyzer.compute_retrieval_utility_correlation(scores, norms)
    freq_profile = analyzer.dimension_profile_by_frequency(norms, retrieval_heads)

    # --- Build result dict ---
    n_layers = model_cfg.get("n_layers", scores.shape[0])
    n_heads = model_cfg.get("n_heads", scores.shape[1])

    stat_tests = {
        "retrieval_mean_utility": utility_stats["retrieval_mean"],
        "non_retrieval_mean_utility": utility_stats["non_retrieval_mean"],
        "retrieval_std_utility": utility_stats["retrieval_std"],
        "non_retrieval_std_utility": utility_stats["non_retrieval_std"],
        "t_statistic": utility_stats["t_statistic"],
        "p_value": utility_stats["p_value"],
        "cohens_d": utility_stats["cohens_d"],
        "mean_diff": utility_stats["mean_diff"],
        "ci_low": utility_stats["ci_low"],
        "ci_high": utility_stats["ci_high"],
        "clustered_permutation_p": utility_stats["clustered_permutation_p"],
        "pearson_r": correlation["pearson_r"],
        "pearson_p": correlation["pearson_p"],
        "spearman_rho": correlation["spearman_rho"],
        "spearman_p": correlation["spearman_p"],
        "partial_spearman_r": correlation.get("partial_spearman_r"),
        "partial_p": correlation.get("partial_p"),
        "partial_ci_low": correlation.get("partial_ci_low"),
        "partial_ci_high": correlation.get("partial_ci_high"),
    }

    actual_lengths = [s["actual_token_length"] for s in samples] if samples else []

    result = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "load_in_8bit": (model_cfg.get("load_in_8bit", True)
                         if load_in_8bit is None else load_in_8bit),
        "n_samples_generated": len(samples),
        "actual_token_length": {
            "mean": float(np.mean(actual_lengths)) if actual_lengths else None,
            "min": int(np.min(actual_lengths)) if actual_lengths else None,
            "max": int(np.max(actual_lengths)) if actual_lengths else None,
        },
        "environment": capture_environment(),
        "retrieval_heads": retrieval_heads,
        "retrieval_scores": scores.tolist(),
        "dimension_utility": {
            "per_head_scalar": utility_stats["per_head_scalar"],
            "per_head_scalar_zscore": utility_stats.get("per_head_scalar_zscore"),
            "frequency_profile": freq_profile,
            # Figure-2 spike diagnostic: is the low-norm dim genuine or artifact?
            "raw_dim_diagnostic": diagnose_dimension_norms(
                norms, analyzer.head_dim, n_heads=analyzer.n_heads,
                n_kv_heads=getattr(model.config, "num_key_value_heads", analyzer.n_heads),
            ),
        },
        "statistical_tests": stat_tests,
        "summary": {
            "n_retrieval_heads": len(retrieval_heads),
            "n_total_heads": n_layers * n_heads,
            "retrieval_fraction": len(retrieval_heads) / (n_layers * n_heads),
            "hypothesis_supported": _determine_hypothesis(stat_tests),
        },
    }

    if persist:
        result_path = out_dir / "results.json"
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("[%s] Results saved to %s", model_name, result_path)
        # Save norms for activation patching (Layer D)
        np.save(out_dir / "norms.npy", norms)

    _unload_model(model)

    return result


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def generate_figures(
    all_results: dict[str, dict],
    results_dir: Path,
    dpi: int = 300,
    fmt: str = "pdf",
) -> None:
    """Generate Figure 1 and Figure 2 from all model results."""
    fig_dir = results_dir / "layer_a" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_retrieval_vs_utility_scatter(
            all_results,
            fig_dir / f"figure1_scatter.{fmt}",
            dpi=dpi,
        )
        logger.info("Figure 1 saved.")
    except Exception as exc:
        logger.error("Figure 1 failed: %s", exc)

    try:
        plot_dimension_utility_profile(
            all_results,
            fig_dir / f"figure2_frequency_profile.{fmt}",
            dpi=dpi,
        )
        logger.info("Figure 2 saved.")
    except Exception as exc:
        logger.error("Figure 2 failed: %s", exc)


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(all_results: dict[str, dict]) -> None:
    header = (
        f"{'Model':<10} {'#Ret':>6} {'#Tot':>6} {'Frac':>6} "
        f"{'RetU':>8} {'NonRetU':>8} {'p-val':>10} {'Pearson r':>10} {'Hyp':>12}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for name, res in all_results.items():
        s = res["summary"]
        t = res["statistical_tests"]
        print(
            f"{name:<10} {s['n_retrieval_heads']:>6} {s['n_total_heads']:>6} "
            f"{s['retrieval_fraction']:>6.3f} "
            f"{t['retrieval_mean_utility']:>8.4f} {t['non_retrieval_mean_utility']:>8.4f} "
            f"{t['p_value']:>10.4e} {t['pearson_r']:>10.4f} "
            f"{s['hypothesis_supported']:>12}"
        )
    print("=" * len(header) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer A: Static multi-model analysis")
    parser.add_argument(
        "--model",
        choices=["llama3", "llama2", "qwen", "olmo", "all"],
        default="llama3",
        help="Which model(s) to analyse.",
    )
    parser.add_argument("--n_samples", type=int, default=None, help="Override n_samples from config.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml.")
    parser.add_argument("--seed", type=int, default=42, help="Single seed (ignored if --seeds given).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Multiple seeds for variance estimation (item B3). "
             "Defaults to config niah.seeds, else [--seed].",
    )
    parser.add_argument(
        "--no_8bit", action="store_true",
        help="Load models in fp16 instead of 8-bit (quantization ablation, item A2).",
    )
    return parser.parse_args()


def _aggregate_seed_summaries(per_seed: list[dict]) -> dict:
    """Mean ± SD of key metrics across seeds (item B3)."""
    def _col(path: tuple[str, ...]) -> list[float]:
        out = []
        for r in per_seed:
            v: Any = r
            for k in path:
                v = v[k]
            if v is not None and v == v:  # skip None/NaN
                out.append(float(v))
        return out

    metrics = {
        "n_retrieval_heads": ("summary", "n_retrieval_heads"),
        "retrieval_fraction": ("summary", "retrieval_fraction"),
        "cohens_d": ("statistical_tests", "cohens_d"),
        "spearman_rho": ("statistical_tests", "spearman_rho"),
        "clustered_permutation_p": ("statistical_tests", "clustered_permutation_p"),
    }
    agg: dict = {"seeds": [r["seed"] for r in per_seed], "n_seeds": len(per_seed)}
    for name, path in metrics.items():
        vals = _col(path)
        agg[name] = {
            "mean": float(np.mean(vals)) if vals else None,
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "values": vals,
        }
    return agg


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    n_samples = args.n_samples or config["niah"]["n_samples"]
    fig_dpi = config.get("output", {}).get("figures_dpi", 300)
    fig_fmt = config.get("output", {}).get("figure_format", "pdf")
    load_in_8bit = False if args.no_8bit else None  # None → use per-model config

    seeds = args.seeds or config["niah"].get("seeds") or [args.seed]
    logger.info("Running with seeds: %s", seeds)

    model_keys = [
        m for m in (
            list(config["models"].keys()) if args.model == "all" else [args.model]
        )
        if m in config["models"]
    ]
    if not model_keys:
        logger.error("No valid models selected.")
        return

    context_lengths = config["niah"]["context_lengths"]
    needle_positions = config["niah"]["needle_positions"]
    paired = len(model_keys) > 1

    # Pre-load tokenizers once for the paired-seed intersection (§2.3). Cheap —
    # tokenizers are tiny and model weights are not loaded here.
    tokenizers: dict[str, object] = {}
    if paired:
        from transformers import AutoTokenizer
        for m in model_keys:
            cfg = config["models"][m]
            tokenizers[m] = AutoTokenizer.from_pretrained(
                cfg["hf_id"], revision=cfg.get("revision") or "main",
                trust_remote_code=True,
            )
        logger.info("Paired-seed control ON across models: %s", model_keys)

    # seed-OUTER so every model sees the identical spec subset for a given seed.
    per_seed_by_model: dict[str, list[dict]] = {m: [] for m in model_keys}
    for idx, seed in enumerate(seeds):
        specs = generate_niah_specs(seed, n_samples, context_lengths, needle_positions)
        if paired:
            specs, dropped = paired_spec_subset(specs, tokenizers)
            logger.info(
                "seed=%d: %d paired specs kept (dropped %d): %s",
                seed, len(specs), len(dropped), dropped,
            )
        for model_name in model_keys:
            logger.info("=" * 60)
            logger.info("Analysis: %s  seed=%d", model_name, seed)
            logger.info("=" * 60)
            try:
                res = run_analysis(
                    model_name=model_name,
                    model_cfg=config["models"][model_name],
                    config=config,
                    n_samples=n_samples,
                    results_dir=results_dir,
                    seed=seed,
                    load_in_8bit=load_in_8bit,
                    persist=(idx == 0),  # only primary seed writes canonical artifacts
                    specs=specs,
                )
                per_seed_by_model[model_name].append(res)
            except Exception as exc:
                logger.error(
                    "Analysis failed for %s seed=%d: %s", model_name, seed, exc,
                    exc_info=True,
                )

    all_results: dict[str, dict] = {}  # primary-seed result per model (for figures)
    for model_name in model_keys:
        per_seed = per_seed_by_model[model_name]
        if not per_seed:
            continue
        all_results[model_name] = per_seed[0]
        if len(per_seed) > 1:
            agg = _aggregate_seed_summaries(per_seed)
            agg_path = results_dir / "layer_a" / model_name / "seed_aggregate.json"
            with open(agg_path, "w") as f:
                json.dump(agg, f, indent=2)
            logger.info(
                "[%s] %d retrieval heads: mean=%.1f ± %.1f over %d seeds.",
                model_name,
                agg["n_retrieval_heads"]["mean"],
                agg["n_retrieval_heads"]["std"] or 0.0,
                len(per_seed),
            )

    if all_results:
        # FDR across models on the (cluster-aware) utility test (item B2).
        names = list(all_results.keys())
        pvals = [all_results[m]["statistical_tests"]["clustered_permutation_p"] for m in names]
        fdr = benjamini_hochberg(pvals)
        fdr_summary = {
            "models": names,
            "raw_p_clustered_permutation": pvals,
            "p_adjusted_bh": fdr["p_adjusted"],
            "rejected_at_0.05": fdr["rejected"],
            "n_significant": fdr["n_significant"],
        }
        fdr_path = results_dir / "layer_a" / "fdr_summary.json"
        fdr_path.parent.mkdir(parents=True, exist_ok=True)
        with open(fdr_path, "w") as f:
            json.dump(fdr_summary, f, indent=2)
        logger.info("Cross-model FDR summary saved to %s", fdr_path)

        generate_figures(all_results, results_dir, dpi=fig_dpi, fmt=fig_fmt)
        print_summary_table(all_results)
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
