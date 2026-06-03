"""
Layer C: Natural θ experiment — LLaMA-2 (θ=10K) vs. LLaMA-3.1 (θ=500K).

Same architectural family, different RoPE base frequencies.
Compares retrieval head counts and dimension utility profiles.

Estimated runtime: ~4–6 hours on L4 24 GB.

Usage:
    python scripts/run_layer_c.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimension_utility import DimensionUtilityAnalyzer
from src.model_loader import load_model as _load_model_shared
from src.repro import capture_environment, set_determinism
from src.retrieval_head_detector import (
    RetrievalHeadDetector,
    generate_niah_specs,
    paired_spec_subset,
)
from src.visualization import (
    plot_dimension_utility_profile,
    plot_theta_comparison_heatmaps,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


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


def _load_model(model_name: str, model_cfg: dict, load_in_8bit: bool | None = None):
    """Load a model at its pinned revision (delegates to src.model_loader)."""
    return _load_model_shared(model_cfg, model_name, load_in_8bit=load_in_8bit)


def _unload(model) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _determine_hypothesis(stats: dict) -> str:
    p = stats.get("p_value", float("nan"))
    ret_mean = stats.get("retrieval_mean_utility", float("nan"))
    non_ret_mean = stats.get("non_retrieval_mean_utility", float("nan"))
    if p != p:
        return "inconclusive"
    if p < 0.05:
        return "H2" if ret_mean < non_ret_mean else "anomaly"
    return "inconclusive"


def analyse_model(
    model_name: str,
    model_cfg: dict,
    config: dict,
    results_dir: Path,
    seed: int = 42,
    specs: list[dict] | None = None,
    persist: bool = True,
) -> dict:
    """Run the Layer-A pipeline for one model and return the result dict.

    Args:
        specs: Pre-generated, intersection-filtered NIAH specs for paired-seed
            comparison (§2.3). When None, samples are generated for this model.
        persist: Write results.json/norms.npy only when True (primary seed).
    """
    strict = config.get("reproducibility", {}).get("strict_determinism", False)
    set_determinism(seed, strict=strict)
    niah_cfg = config["niah"]
    n_samples = niah_cfg["n_samples"]

    out_dir = results_dir / "layer_c" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"

    model, tokenizer = _load_model(model_name, model_cfg)

    detector = RetrievalHeadDetector(
        model, tokenizer, config,
        score_threshold=niah_cfg.get("score_threshold", 0.1),
        seed=seed,
    )
    if specs is not None:
        samples = detector.prepare_samples(specs)
    else:
        samples = detector.generate_niah_samples(
            n_samples=n_samples,
            context_lengths=niah_cfg["context_lengths"],
            needle_positions=niah_cfg["needle_positions"],
        )
    scores = detector.score_heads(samples)
    retrieval_heads = detector.get_retrieval_heads(scores)

    analyzer = DimensionUtilityAnalyzer(model, config)
    norms = analyzer.compute_query_projection_norms()
    utility_stats = analyzer.compute_utility_scores(norms, retrieval_heads)
    correlation = analyzer.compute_retrieval_utility_correlation(scores, norms)
    freq_profile = analyzer.dimension_profile_by_frequency(norms, retrieval_heads)

    if persist:
        np.save(out_dir / "norms.npy", norms)

    n_layers, n_heads = scores.shape
    stat_tests = {
        "retrieval_mean_utility": utility_stats["retrieval_mean"],
        "non_retrieval_mean_utility": utility_stats["non_retrieval_mean"],
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
    }

    result = {
        "model": model_name,
        "theta": model_cfg.get("theta"),
        "timestamp": datetime.utcnow().isoformat(),
        "seed": seed,
        "environment": capture_environment(),
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

    if persist:
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)

    _unload(model)
    return result


def h1_from_theta_comparison(llama2_res: dict, llama3_res: dict) -> str:
    """
    Assess H1 (Prevention): high θ → more dimension inefficiency → fewer retrieval heads.

    Returns a short assessment string.
    """
    n2 = llama2_res["summary"]["n_retrieval_heads"]
    n3 = llama3_res["summary"]["n_retrieval_heads"]
    theta2 = llama2_res.get("theta", 10000)
    theta3 = llama3_res.get("theta", 500000)

    if n3 < n2:
        return f"H1 supported: θ={theta3:.0f} has fewer retrieval heads ({n3}) than θ={theta2:.0f} ({n2})"
    elif n3 > n2:
        return f"H1 not supported: θ={theta3:.0f} has MORE retrieval heads ({n3}) than θ={theta2:.0f} ({n2})"
    return f"Inconclusive: equal retrieval head counts ({n2}={n3}) across θ"


def print_comparison_table(llama2_res: dict, llama3_res: dict) -> None:
    print("\n" + "=" * 70)
    print(f"{'Metric':<35} {'LLaMA-2 (θ=10K)':>16} {'LLaMA-3 (θ=500K)':>16}")
    print("=" * 70)
    for label, key1, key2 in [
        ("# Retrieval heads", ("summary", "n_retrieval_heads"), ("summary", "n_retrieval_heads")),
        ("Retrieval fraction", ("summary", "retrieval_fraction"), ("summary", "retrieval_fraction")),
        ("Retrieval mean utility", ("statistical_tests", "retrieval_mean_utility"), ("statistical_tests", "retrieval_mean_utility")),
        ("Non-retrieval mean utility", ("statistical_tests", "non_retrieval_mean_utility"), ("statistical_tests", "non_retrieval_mean_utility")),
        ("p-value (t-test)", ("statistical_tests", "p_value"), ("statistical_tests", "p_value")),
        ("Pearson r", ("statistical_tests", "pearson_r"), ("statistical_tests", "pearson_r")),
    ]:
        v2 = llama2_res[key1[0]][key1[1]]
        v3 = llama3_res[key2[0]][key2[1]]
        if isinstance(v2, float):
            print(f"{label:<35} {v2:>16.4f} {v3:>16.4f}")
        else:
            print(f"{label:<35} {v2:>16} {v3:>16}")

    print("=" * 70)
    print(h1_from_theta_comparison(llama2_res, llama3_res))
    print("=" * 70 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer C: θ comparison")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Multiple PAIRED NIAH seeds (item B3 / §2.3). Defaults to config "
             "niah.seeds, else [--seed].",
    )
    return parser.parse_args()


def _run_paired(config, results_dir, seeds):
    """Run LLaMA-2 vs LLaMA-3.1 across PAIRED seeds; return primary results + aggregates."""
    from transformers import AutoTokenizer

    models = ["llama2", "llama3"]
    niah = config["niah"]
    tokenizers = {
        m: AutoTokenizer.from_pretrained(
            config["models"][m]["hf_id"],
            revision=config["models"][m].get("revision") or "main",
            trust_remote_code=True,
        )
        for m in models
    }

    per_seed = {m: [] for m in models}
    for idx, seed in enumerate(seeds):
        specs = generate_niah_specs(seed, niah["n_samples"],
                                    niah["context_lengths"], niah["needle_positions"])
        specs, dropped = paired_spec_subset(specs, tokenizers)
        logger.info("seed=%d: %d paired specs (dropped %d)", seed, len(specs), len(dropped))
        for m in models:
            res = analyse_model(m, config["models"][m], config, results_dir,
                                seed=seed, specs=specs, persist=(idx == 0))
            per_seed[m].append(res)

    primary = {m: per_seed[m][0] for m in models}
    if len(seeds) > 1:
        for m in models:
            heads = [r["summary"]["n_retrieval_heads"] for r in per_seed[m]]
            agg = {"seeds": seeds, "n_retrieval_heads_mean": float(np.mean(heads)),
                   "n_retrieval_heads_std": float(np.std(heads, ddof=1)),
                   "n_retrieval_heads_values": heads}
            with open(results_dir / "layer_c" / m / "seed_aggregate.json", "w") as f:
                json.dump(agg, f, indent=2)
            logger.info("[%s] retrieval heads = %.1f ± %.1f over %d seeds",
                        m, agg["n_retrieval_heads_mean"], agg["n_retrieval_heads_std"], len(seeds))
    return primary["llama2"], primary["llama3"]


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    fig_dpi = config.get("output", {}).get("figures_dpi", 300)
    fig_fmt = config.get("output", {}).get("figure_format", "pdf")
    seeds = args.seeds or config["niah"].get("seeds") or [args.seed]
    logger.info("Layer C paired seeds: %s", seeds)

    llama2_res, llama3_res = _run_paired(config, results_dir, seeds)

    # Save comparison JSON
    comparison_path = results_dir / "layer_c" / "comparison.json"
    with open(comparison_path, "w") as f:
        json.dump(
            {
                "h1_assessment": h1_from_theta_comparison(llama2_res, llama3_res),
                "llama2_summary": llama2_res["summary"],
                "llama3_summary": llama3_res["summary"],
                "llama2_stats": llama2_res["statistical_tests"],
                "llama3_stats": llama3_res["statistical_tests"],
            },
            f,
            indent=2,
        )

    # Figures
    fig_dir = results_dir / "layer_c" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    try:
        plot_theta_comparison_heatmaps(
            llama2_res, llama3_res,
            fig_dir / f"figure4_theta_comparison.{fig_fmt}",
            dpi=fig_dpi,
        )
        logger.info("Figure 4 saved.")
    except Exception as exc:
        logger.error("Figure 4 failed: %s", exc)

    try:
        plot_dimension_utility_profile(
            {"llama2": llama2_res, "llama3": llama3_res},
            fig_dir / f"figure2c_frequency_profiles.{fig_fmt}",
            dpi=fig_dpi,
        )
        logger.info("Frequency profile figure saved.")
    except Exception as exc:
        logger.error("Frequency profile figure failed: %s", exc)

    print_comparison_table(llama2_res, llama3_res)


if __name__ == "__main__":
    main()
