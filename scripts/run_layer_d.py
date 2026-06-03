"""
Layer D: Activation patching — causal analysis of dimension utility.

Loads Layer-A results for the best model (most retrieval heads), then runs
three-condition dimension-zeroing experiments to test whether low-utility
dimensions are causally irrelevant (H2) or load-bearing.

Estimated runtime: ~3–4 hours on L4 24 GB.

Usage:
    python scripts/run_layer_d.py --model llama3 --k_dims 16
    python scripts/run_layer_d.py --model llama3 --config configs/config.yaml
"""

from __future__ import annotations

import argparse
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

from src.activation_patching import ActivationPatcher
from src.dimension_utility import DimensionUtilityAnalyzer
from src.model_loader import load_model as _load_model_shared
from src.repro import capture_environment, set_determinism
from src.retrieval_head_detector import RetrievalHeadDetector
from src.visualization import plot_activation_patching_results

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


def _load_layer_a_results(model_name: str, results_dir: Path) -> dict:
    """Load Layer-A results for a model."""
    result_path = results_dir / "layer_a" / model_name / "results.json"
    if not result_path.exists():
        raise FileNotFoundError(
            f"Layer-A results not found at {result_path}. "
            "Run run_layer_a.py first."
        )
    with open(result_path) as f:
        return json.load(f)


def _load_norms(model_name: str, results_dir: Path) -> np.ndarray | None:
    """Load the saved norms matrix from Layer A."""
    norms_path = results_dir / "layer_a" / model_name / "norms.npy"
    if norms_path.exists():
        return np.load(norms_path)
    return None


def _pick_best_model(config: dict, results_dir: Path) -> str:
    """Select the model with the most retrieval heads from completed Layer-A runs."""
    best_model, best_count = None, -1
    for model_name in config["models"]:
        result_path = results_dir / "layer_a" / model_name / "results.json"
        if not result_path.exists():
            continue
        try:
            with open(result_path) as f:
                res = json.load(f)
            n = res["summary"]["n_retrieval_heads"]
            if n > best_count:
                best_count, best_model = n, model_name
        except Exception:
            pass
    return best_model or list(config["models"].keys())[0]


def _load_model(model_name: str, model_cfg: dict, load_in_8bit: bool | None = None):
    """Load a model at its pinned revision (delegates to src.model_loader)."""
    return _load_model_shared(model_cfg, model_name, load_in_8bit=load_in_8bit)


def _results_to_serializable(results: dict) -> dict:
    """Convert (layer, head) tuple keys to strings for JSON serialisation."""
    return {
        f"layer{l}_head{h}": v
        for (l, h), v in results.items()
    }


def print_causal_effect_table(df) -> None:
    has_freq = "frequency_effect" in df.columns
    width = 96 if has_freq else 75
    print("\n" + "=" * width)
    header = (f"{'Layer':>6} {'Head':>6} {'Baseline':>10} {'Low-util':>10} "
             f"{'Random':>10} {'High-util':>10} {'CE':>10}")
    if has_freq:
        header += f" {'Low-freq':>10} {'High-freq':>10}"
    print(header)
    print("=" * width)
    for _, row in df.iterrows():
        line = (
            f"{int(row['layer']):>6} {int(row['head']):>6} "
            f"{row['baseline']:>10.4f} {row['low_utility']:>10.4f} "
            f"{row['random']:>10.4f} {row['high_utility']:>10.4f} "
            f"{row['causal_effect']:>10.4f}"
        )
        if has_freq:
            line += f" {row['low_freq']:>10.4f} {row['high_freq']:>10.4f}"
        print(line)
    print("=" * width)
    mean_ce = df["causal_effect"].mean()
    direction = "H2 (low-util dims causally irrelevant)" if mean_ce < 0 else "unexpected (low-util load-bearing)"
    print(f"Mean causal effect (utility): {mean_ce:.4f}  →  {direction}")
    if has_freq:
        mean_fe = df["frequency_effect"].mean()
        fdir = ("frequency-specific: high-freq RoPE dims load-bearing"
                if mean_fe > 0 else
                "frequency-specific: low-freq dims load-bearing"
                if mean_fe < 0 else "no frequency specificity")
        print(f"Mean frequency effect (§6):  {mean_fe:.4f}  →  {fdir}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Layer D: Activation patching")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model key. If omitted, the model with most retrieval heads from Layer A is used.",
    )
    parser.add_argument("--k_dims", type=int, default=None, help="Dimensions to zero per condition.")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    fig_dpi = config.get("output", {}).get("figures_dpi", 300)
    fig_fmt = config.get("output", {}).get("figure_format", "pdf")
    k_dims = args.k_dims or config.get("activation_patching", {}).get("k_dims", 16)

    model_name = args.model or _pick_best_model(config, results_dir)
    logger.info("Using model: %s", model_name)

    # Load Layer-A results
    layer_a = _load_layer_a_results(model_name, results_dir)
    retrieval_heads = [tuple(h) for h in layer_a["retrieval_heads"]]
    utility_scores = layer_a["dimension_utility"]

    # Attach raw norms matrix for per-dim utility
    norms = _load_norms(model_name, results_dir)
    if norms is not None:
        utility_scores["_norms"] = norms
    else:
        logger.warning("Norms matrix not found; using coarse per-head scalar for dim selection.")

    if not retrieval_heads:
        logger.error("No retrieval heads found in Layer-A results for %s.", model_name)
        return

    logger.info("Found %d retrieval heads to patch.", len(retrieval_heads))

    strict = config.get("reproducibility", {}).get("strict_determinism", False)
    set_determinism(args.seed, strict=strict)
    model, tokenizer = _load_model(model_name, config["models"][model_name])

    # Generate fresh NIAH samples for patching evaluation
    detector = RetrievalHeadDetector(
        model, tokenizer, config,
        score_threshold=config["niah"].get("score_threshold", 0.1),
        seed=args.seed,
    )
    samples = detector.generate_niah_samples(
        n_samples=args.n_samples,
        context_lengths=config["niah"]["context_lengths"],
        needle_positions=config["niah"]["needle_positions"],
    )

    # RoPE frequency ordering for the frequency-axis causal test (paper §6).
    analyzer = DimensionUtilityAnalyzer(model, config)
    freq_order = analyzer.freq_order

    patcher = ActivationPatcher(model, tokenizer, config)
    patching_results = patcher.run_patching_experiment(
        retrieval_heads=retrieval_heads,
        utility_scores=utility_scores,
        samples=samples,
        k_dims=k_dims,
        n_samples=args.n_samples,
        freq_order=freq_order,
    )
    causal_df = patcher.compute_causal_effect(patching_results)

    # Save results
    out_dir = results_dir / "layer_d" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    def _mean_ci(values: np.ndarray) -> tuple[float, list[float]]:
        """Mean + bootstrap 95% CI (item B4)."""
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        if len(values) < 2:
            return (float(values.mean()) if len(values) else float("nan"),
                    [float("nan"), float("nan")])
        rng = np.random.default_rng(args.seed)
        boot = np.array([
            rng.choice(values, size=len(values), replace=True).mean()
            for _ in range(10000)
        ])
        return float(values.mean()), [float(np.quantile(boot, 0.025)),
                                      float(np.quantile(boot, 0.975))]

    mean_ce, ce_ci = _mean_ci(causal_df["causal_effect"].to_numpy())

    summary = {
        "mean_causal_effect": mean_ce,
        "mean_causal_effect_ci95": ce_ci,
        "n_heads_tested": int(len(causal_df)),
        "hypothesis": "H2" if mean_ce < 0 else "unexpected",
    }

    # Frequency-axis result (paper §6) — only present if conditions were run.
    if "frequency_effect" in causal_df.columns:
        mean_fe, fe_ci = _mean_ci(causal_df["frequency_effect"].to_numpy())
        # CI excludes 0 → frequency matters beyond general utility.
        freq_specific = not (fe_ci[0] <= 0 <= fe_ci[1])
        summary.update({
            "mean_frequency_effect": mean_fe,
            "mean_frequency_effect_ci95": fe_ci,
            "frequency_specific": bool(freq_specific),
            "frequency_finding": (
                "frequency-specific (high-freq RoPE dims are load-bearing)"
                if (freq_specific and mean_fe > 0) else
                "frequency-specific (low-freq dims more load-bearing)"
                if (freq_specific and mean_fe < 0) else
                "no frequency specificity beyond general utility"
            ),
        })
        logger.info(
            "Frequency effect: mean=%.4f CI95=%s → %s",
            mean_fe, fe_ci, summary["frequency_finding"],
        )

    result = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "k_dims": k_dims,
        "n_samples": args.n_samples,
        "environment": capture_environment(),
        "patching_results": _results_to_serializable(patching_results),
        "causal_effects": causal_df.to_dict(orient="records"),
        "summary": summary,
    }

    result_path = out_dir / "results.json"
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info("Patching results saved to %s", result_path)

    # Figure 5
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        plot_activation_patching_results(
            patching_results,
            fig_dir / f"figure5_patching.{fig_fmt}",
            dpi=fig_dpi,
        )
        logger.info("Figure 5 saved.")
    except Exception as exc:
        logger.error("Figure 5 failed: %s", exc)

    print_causal_effect_table(causal_df)


if __name__ == "__main__":
    main()
