"""
Layer B: OLMo-2 training dynamics across 500+ checkpoints.

For each checkpoint (filtered by step_stride): load with 8-bit quantization,
score retrieval heads on 50 NIAH samples, compute dimension utility norms,
save partial results, unload model. Resume-safe.

Estimated runtime: ~25 hours with step_stride=10000 (~50 checkpoints).
                   ~12 hours with step_stride=20000 (~25 checkpoints).

Usage:
    python scripts/run_layer_b.py --step_stride 10000 --resume
    python scripts/run_layer_b.py --step_stride 20000 --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import gc
import glob  # FIX #19: module-level import (was inside a function body)
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
from src.olmo_checkpoint_loader import OLMoCheckpointLoader
from src.retrieval_head_detector import RetrievalHeadDetector
from src.visualization import plot_olmo_training_dynamics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_N_NIAH_SAMPLES = 50  # fixed for all checkpoints


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


def process_checkpoint(
    revision: str,
    step: int,
    loader: OLMoCheckpointLoader,
    config: dict,
    results_dir: Path,
    seed: int = 42,
) -> dict | None:
    """
    Load, analyse, and unload a single OLMo-2 checkpoint.

    Returns the result dict, or None on failure.
    """
    out_dir = results_dir / "layer_b" / f"checkpoint_{step}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / "results.json"

    _set_seeds(seed)
    niah_cfg = config["niah"]

    try:
        model, tokenizer = loader.load_checkpoint(
            revision=revision,
            load_in_8bit=config["models"]["olmo"].get("load_in_8bit", True),
        )
    except Exception as exc:
        logger.error("Failed to load revision '%s': %s", revision, exc)
        return None

    try:
        # NIAH scoring
        detector = RetrievalHeadDetector(
            model,
            tokenizer,
            config,
            score_threshold=niah_cfg.get("score_threshold", 0.1),
            seed=seed,
        )
        samples = detector.generate_niah_samples(
            n_samples=_N_NIAH_SAMPLES,
            context_lengths=niah_cfg["context_lengths"],
            needle_positions=niah_cfg["needle_positions"],
        )
        scores = detector.score_heads(samples)
        retrieval_heads = detector.get_retrieval_heads(scores)

        # Dimension utility
        analyzer = DimensionUtilityAnalyzer(model, config)
        norms = analyzer.compute_query_projection_norms()
        utility_stats = analyzer.compute_utility_scores(norms, retrieval_heads)
        freq_profile = analyzer.dimension_profile_by_frequency(norms, retrieval_heads)

        n_layers, n_heads = scores.shape
        result = {
            "step": step,
            "revision": revision,
            "timestamp": datetime.utcnow().isoformat(),
            "seed": seed,
            "hardware": _hardware_info(),
            "retrieval_heads": retrieval_heads,
            "retrieval_scores": scores.tolist(),
            "dimension_utility": {
                "per_head_scalar": utility_stats["per_head_scalar"],
                "retrieval_mean": utility_stats["retrieval_mean"],
                "non_retrieval_mean": utility_stats["non_retrieval_mean"],
                "t_statistic": utility_stats["t_statistic"],
                "p_value": utility_stats["p_value"],
                "frequency_profile": freq_profile,
            },
            "summary": {
                "n_retrieval_heads": len(retrieval_heads),
                "n_total_heads": n_layers * n_heads,
                "retrieval_fraction": len(retrieval_heads) / (n_layers * n_heads),
            },
        }

        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Checkpoint step=%d saved to %s", step, result_path)
        return result

    except Exception as exc:
        logger.error("Analysis failed for step=%d: %s", step, exc, exc_info=True)
        return None
    finally:
        loader.clear_checkpoint(model)


def load_all_checkpoint_results(results_dir: Path) -> list[dict]:
    """Load all completed checkpoint result JSONs, sorted by step."""
    pattern = results_dir / "layer_b" / "checkpoint_*" / "results.json"
    paths = sorted(
        glob.glob(str(pattern)),
        key=lambda p: int(Path(p).parent.name.replace("checkpoint_", "")),
    )
    results = []
    for p in paths:
        try:
            with open(p) as f:
                results.append(json.load(f))
        except Exception as exc:
            logger.warning("Could not load %s: %s", p, exc)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer B: OLMo-2 training dynamics")
    parser.add_argument("--step_stride", type=int, default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip checkpoints that already have results.",
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    ckpt_cfg = config.get("olmo_checkpoints", {})
    step_stride = args.step_stride or ckpt_cfg.get("step_stride", 10000)
    resume = args.resume or ckpt_cfg.get("resume", True)
    fig_dpi = config.get("output", {}).get("figures_dpi", 300)
    fig_fmt = config.get("output", {}).get("figure_format", "pdf")

    loader = OLMoCheckpointLoader(config)

    logger.info("Listing available checkpoints …")
    filtered = loader.get_filtered_checkpoints(step_stride=step_stride)
    if not filtered:
        logger.error("No checkpoints found. Check HuggingFace access.")
        return

    logger.info("Processing %d checkpoints (stride=%d).", len(filtered), step_stride)

    processed = 0
    for revision in filtered:
        step = loader.get_step_from_revision(revision)
        if step is None:
            logger.warning("Could not parse step from revision '%s'; skipping.", revision)
            continue

        if resume and loader.is_computed(revision, str(results_dir)):
            logger.info("Skipping step=%d (already computed).", step)
            continue

        logger.info("Processing checkpoint step=%d (%s) …", step, revision)
        result = process_checkpoint(
            revision=revision,
            step=step,
            loader=loader,
            config=config,
            results_dir=results_dir,
            seed=args.seed,
        )
        if result is not None:
            processed += 1

    logger.info("Processed %d checkpoints.", processed)

    # Generate Figure 3
    checkpoint_results = load_all_checkpoint_results(results_dir)
    if len(checkpoint_results) >= 2:
        fig_dir = results_dir / "layer_b" / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_olmo_training_dynamics(
                checkpoint_results,
                fig_dir / f"figure3_training_dynamics.{fig_fmt}",
                dpi=fig_dpi,
            )
            logger.info("Figure 3 saved.")
        except Exception as exc:
            logger.error("Figure 3 failed: %s", exc)
    else:
        logger.warning("Not enough checkpoint results for Figure 3.")


if __name__ == "__main__":
    main()
