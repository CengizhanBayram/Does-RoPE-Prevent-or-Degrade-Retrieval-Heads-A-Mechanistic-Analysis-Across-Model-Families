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


def _load_model(model_name: str, model_cfg: dict):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = model_cfg["hf_id"]
    load_8bit = model_cfg.get("load_in_8bit", True)
    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except ImportError:
        has_bnb = False

    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {"device_map": "auto", "trust_remote_code": True}
    if load_8bit and has_bnb:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    logger.info("Loading model %s …", model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()
    return model, tokenizer


def _results_to_serializable(results: dict) -> dict:
    """Convert (layer, head) tuple keys to strings for JSON serialisation."""
    return {
        f"layer{l}_head{h}": v
        for (l, h), v in results.items()
    }


def print_causal_effect_table(df) -> None:
    import pandas as pd
    print("\n" + "=" * 75)
    print(f"{'Layer':>6} {'Head':>6} {'Baseline':>10} {'Low-util':>10} {'Random':>10} {'High-util':>10} {'CE':>10}")
    print("=" * 75)
    for _, row in df.iterrows():
        print(
            f"{int(row['layer']):>6} {int(row['head']):>6} "
            f"{row['baseline']:>10.4f} {row['low_utility']:>10.4f} "
            f"{row['random']:>10.4f} {row['high_utility']:>10.4f} "
            f"{row['causal_effect']:>10.4f}"
        )
    print("=" * 75)
    mean_ce = df["causal_effect"].mean()
    direction = "H2 (low-util dims causally irrelevant)" if mean_ce < 0 else "unexpected (low-util load-bearing)"
    print(f"Mean causal effect: {mean_ce:.4f}  →  {direction}\n")


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

    _set_seeds(args.seed)
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

    patcher = ActivationPatcher(model, tokenizer, config)
    patching_results = patcher.run_patching_experiment(
        retrieval_heads=retrieval_heads,
        utility_scores=utility_scores,
        samples=samples,
        k_dims=k_dims,
        n_samples=args.n_samples,
    )
    causal_df = patcher.compute_causal_effect(patching_results)

    # Save results
    out_dir = results_dir / "layer_d" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "model": model_name,
        "timestamp": datetime.utcnow().isoformat(),
        "seed": args.seed,
        "k_dims": k_dims,
        "n_samples": args.n_samples,
        "patching_results": _results_to_serializable(patching_results),
        "causal_effects": causal_df.to_dict(orient="records"),
        "summary": {
            "mean_causal_effect": float(causal_df["causal_effect"].mean()),
            "hypothesis": "H2" if causal_df["causal_effect"].mean() < 0 else "unexpected",
        },
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
