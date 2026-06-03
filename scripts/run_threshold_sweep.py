"""
Threshold sensitivity analysis (item A7).

The retrieval-head classification depends on `score_threshold`. A reviewer will
ask whether the conclusions survive a different cutoff. This script reuses the
already-saved retrieval-score matrix from a Layer-A run (no model reload) and
recomputes the headline metrics across a sweep of thresholds.

Usage:
    python scripts/run_threshold_sweep.py --model llama3
    python scripts/run_threshold_sweep.py --model all --thresholds 0.05 0.1 0.2 0.3
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.stats_utils import clustered_permutation_test, cohens_d

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def sweep_model(model_name: str, results_dir: Path, thresholds: list[float]) -> dict | None:
    """Recompute retrieval-head metrics over a threshold sweep for one model."""
    base = results_dir / "layer_a" / model_name
    scores_path = base / "retrieval_scores.npy"
    norms_path = base / "norms.npy"
    if not scores_path.exists() or not norms_path.exists():
        logger.warning("Missing scores/norms for %s (run Layer A first).", model_name)
        return None

    scores = np.load(scores_path)            # (n_layers, n_heads)
    norms = np.load(norms_path)              # (n_layers, n_heads, head_dim)
    per_head_scalar = norms.mean(axis=-1)    # (n_layers, n_heads)
    n_layers, n_heads = scores.shape
    layer_idx = np.repeat(np.arange(n_layers), n_heads)
    flat_util = per_head_scalar.flatten()

    rows = []
    for thr in thresholds:
        labels = (scores.flatten() >= thr).astype(int)
        n_ret = int(labels.sum())
        if n_ret < 2 or (len(labels) - n_ret) < 2:
            rows.append({"threshold": thr, "n_retrieval_heads": n_ret,
                         "retrieval_fraction": n_ret / labels.size,
                         "cohens_d": None, "clustered_permutation_p": None})
            continue
        ret_vals = flat_util[labels == 1]
        non_vals = flat_util[labels == 0]
        d = cohens_d(ret_vals, non_vals)
        perm = clustered_permutation_test(flat_util, labels, layer_idx, n_perm=2000)
        rows.append({
            "threshold": thr,
            "n_retrieval_heads": n_ret,
            "retrieval_fraction": n_ret / labels.size,
            "cohens_d": d,
            "clustered_permutation_p": perm["p_value"],
        })

    logger.info("[%s] threshold sweep:", model_name)
    for r in rows:
        logger.info(
            "  thr=%.2f  n_ret=%d  frac=%.3f  d=%s  perm_p=%s",
            r["threshold"], r["n_retrieval_heads"], r["retrieval_fraction"],
            f"{r['cohens_d']:.3f}" if r["cohens_d"] is not None else "n/a",
            f"{r['clustered_permutation_p']:.4f}" if r["clustered_permutation_p"] is not None else "n/a",
        )
    return {"model": model_name, "sweep": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrieval-head threshold sensitivity")
    parser.add_argument("--model", default="all", help="Model key or 'all'.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.05, 0.1, 0.15, 0.2, 0.3],
        help="Thresholds to sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))

    models = list(config["models"].keys()) if args.model == "all" else [args.model]
    out = {}
    for m in models:
        res = sweep_model(m, results_dir, args.thresholds)
        if res is not None:
            out[m] = res

    if out:
        out_path = results_dir / "layer_a" / "threshold_sweep.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2)
        logger.info("Threshold sweep saved to %s", out_path)
    else:
        logger.warning("No Layer-A artifacts found to sweep.")


if __name__ == "__main__":
    main()
