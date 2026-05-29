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

import numpy as np
import torch
import yaml

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimension_utility import DimensionUtilityAnalyzer
from src.retrieval_head_detector import RetrievalHeadDetector
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


def _load_model(model_name: str, model_cfg: dict):
    """Load a model with optional 8-bit quantization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = model_cfg["hf_id"]
    load_8bit = model_cfg.get("load_in_8bit", True)

    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except ImportError:
        has_bnb = False
        if load_8bit:
            logger.warning("bitsandbytes not found; loading in fp16.")

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    if load_8bit and has_bnb:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    logger.info("Loading %s (%s) …", model_name, hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()

    if torch.cuda.is_available():
        logger.info(
            "VRAM after load: %.1f / %.1f GB",
            torch.cuda.memory_allocated() / 1e9,
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    return model, tokenizer


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
) -> dict:
    """
    Run the full Layer-A pipeline for a single model.

    Returns the result dict (also written to disk).
    """
    _set_seeds(seed)
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

    model, tokenizer = _load_model(model_name, model_cfg)

    # --- Retrieval head detection ---
    logger.info("[%s] Generating NIAH samples …", model_name)
    detector = RetrievalHeadDetector(
        model, tokenizer, config, score_threshold=score_threshold, seed=seed
    )
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
        "t_statistic": utility_stats["t_statistic"],
        "p_value": utility_stats["p_value"],
        "pearson_r": correlation["pearson_r"],
        "pearson_p": correlation["pearson_p"],
        "spearman_rho": correlation["spearman_rho"],
        "spearman_p": correlation["spearman_p"],
    }

    result = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "hardware": _hardware_info(),
        "retrieval_heads": retrieval_heads,
        "retrieval_scores": scores.tolist(),
        "dimension_utility": {
            "per_head_scalar": utility_stats["per_head_scalar"],
            "frequency_profile": freq_profile,
        },
        "statistical_tests": stat_tests,
        "summary": {
            "n_retrieval_heads": len(retrieval_heads),
            "n_total_heads": n_layers * n_heads,
            "retrieval_fraction": len(retrieval_heads) / (n_layers * n_heads),
            "hypothesis_supported": _determine_hypothesis(stat_tests),
        },
    }

    # Save result JSON
    result_path = out_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("[%s] Results saved to %s", model_name, result_path)

    # Save norms for activation patching
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
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    n_samples = args.n_samples or config["niah"]["n_samples"]
    fig_dpi = config.get("output", {}).get("figures_dpi", 300)
    fig_fmt = config.get("output", {}).get("figure_format", "pdf")

    model_keys = (
        list(config["models"].keys())
        if args.model == "all"
        else [args.model]
    )

    all_results: dict[str, dict] = {}

    for model_name in model_keys:
        if model_name not in config["models"]:
            logger.error("Model '%s' not found in config.", model_name)
            continue
        logger.info("=" * 60)
        logger.info("Starting analysis: %s", model_name)
        logger.info("=" * 60)
        try:
            res = run_analysis(
                model_name=model_name,
                model_cfg=config["models"][model_name],
                config=config,
                n_samples=n_samples,
                results_dir=results_dir,
                seed=args.seed,
            )
            all_results[model_name] = res
        except Exception as exc:
            logger.error("Analysis failed for %s: %s", model_name, exc, exc_info=True)

    if all_results:
        generate_figures(all_results, results_dir, dpi=fig_dpi, fmt=fig_fmt)
        print_summary_table(all_results)
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    main()
