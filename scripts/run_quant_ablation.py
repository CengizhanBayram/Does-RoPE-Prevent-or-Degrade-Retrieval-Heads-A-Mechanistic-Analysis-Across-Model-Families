"""
A2 — Quantization ablation: 8-bit vs fp16.

The detection metric is argmax-based (discrete), so a tiny continuous shift from
8-bit rounding can flip a head's argmax when its top-2 positions are close. We
therefore do NOT just compare retrieval-head COUNTS; we report three levels, for
BOTH the argmax score and the quantization-robust mass score:

  1. Head identity  — Jaccard overlap of the retrieval-head sets (argmax).
  2. Score agreement — Pearson/Spearman of the per-head score matrices.
  3. Finding direction — does the retrieval-vs-non-retrieval utility gap keep the
     same sign and significance (clustered permutation) in both modes?

The claim defended is the stability of the *finding*, not byte-identical head
sets. Run on ONE model that PRODUCED a significant finding (so the test is
informative -- a null model like LLaMA-3.1, d~0/p~1, would stay null trivially
and tell us nothing). We default to OLMo-2: it has a significant utility effect
(d=+0.50) and is also the Layer-D model, so robustness there matters most.
Quantization perturbs the discrete argmax near the detection threshold, which
does NOT require long context; we use seq=2048 so fp16 (eager attention) stays
within a 24 GB budget.

Usage:
    python scripts/run_quant_ablation.py                      # olmo, seq=2048
    python scripts/run_quant_ablation.py --model qwen --seq 2048
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimension_utility import DimensionUtilityAnalyzer
from src.model_loader import load_model
from src.repro import capture_environment, set_determinism
from src.retrieval_head_detector import (
    RetrievalHeadDetector,
    generate_niah_specs,
)
from src.stats_utils import jaccard

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _score_one_mode(model_name, model_cfg, config, specs, seed, load_in_8bit):
    """Load a model in the given precision, score heads, return per-mode results."""
    set_determinism(seed, strict=config.get("reproducibility", {}).get("strict_determinism", False))
    model, tokenizer = load_model(model_cfg, model_name, load_in_8bit=load_in_8bit)
    try:
        detector = RetrievalHeadDetector(
            model, tokenizer, config,
            score_threshold=config["niah"].get("score_threshold", 0.1), seed=seed,
        )
        samples = detector.prepare_samples(specs)
        scored = detector.score_heads(samples, return_mass=True)  # {"argmax","mass"}
        retrieval_heads = detector.get_retrieval_heads(scored["argmax"])

        analyzer = DimensionUtilityAnalyzer(model, config)
        norms = analyzer.compute_query_projection_norms()
        util = analyzer.compute_utility_scores(norms, retrieval_heads)
        return {
            "argmax": scored["argmax"],
            "mass": scored["mass"],
            "retrieval_heads": retrieval_heads,
            "n_samples": len(samples),
            "utility": util,
        }
    finally:
        del model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass


def _corr(a: np.ndarray, b: np.ndarray) -> dict:
    af, bf = a.flatten(), b.flatten()
    out = {"pearson_r": float("nan"), "spearman_rho": float("nan")}
    if len(np.unique(af)) > 1 and len(np.unique(bf)) > 1:
        out["pearson_r"] = float(stats.pearsonr(af, bf)[0])
        out["spearman_rho"] = float(stats.spearmanr(af, bf)[0])
    return out


def _finding(util: dict) -> dict:
    """Sign + significance of the retrieval-vs-non-retrieval utility gap."""
    gap = util["retrieval_mean"] - util["non_retrieval_mean"]
    return {
        "retrieval_minus_non_retrieval": float(gap),
        "direction": "retrieval_lower" if gap < 0 else "retrieval_higher",
        "clustered_permutation_p": util["clustered_permutation_p"],
        "cohens_d": util["cohens_d"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A2 quantization ablation (8-bit vs fp16)")
    parser.add_argument("--model", default="olmo",
                        help="Model key. Default olmo: a SIGNIFICANT-finding model (d=+0.50) "
                             "+ the Layer-D model. A null model (llama3) would be uninformative.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--seq", type=int, default=2048,
                        help="Single context length. 2048 keeps fp16 within 24 GB; "
                             "quantization's argmax shift does not need long context.")
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    results_dir = Path(config.get("output", {}).get("results_dir", "./results"))
    model_cfg = config["models"][args.model]
    n_samples = args.n_samples or config["niah"]["n_samples"]

    # Identical specs at a single seq length for both precision modes.
    specs = generate_niah_specs(
        args.seed, n_samples, [args.seq], config["niah"]["needle_positions"]
    )
    logger.info("Ablation on %s: %d specs at seq=%d", args.model, len(specs), args.seq)

    logger.info("=== Mode 1/2: 8-bit ===")
    r8 = _score_one_mode(args.model, model_cfg, config, specs, args.seed, load_in_8bit=True)
    # Free the 8-bit model fully before loading fp16 (the per-mode finally del's
    # the model, but detector/analyzer refs only die here, on frame exit).
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    logger.info("=== Mode 2/2: fp16 ===")
    r16 = _score_one_mode(args.model, model_cfg, config, specs, args.seed, load_in_8bit=False)

    thr = config["niah"].get("score_threshold", 0.1)
    report = {
        "model": args.model,
        "seq_length": args.seq,
        "n_samples_8bit": r8["n_samples"],
        "n_samples_fp16": r16["n_samples"],
        "score_threshold": thr,
        "seed": args.seed,
        "timestamp": datetime.utcnow().isoformat(),
        "environment": capture_environment(),
        # Level 1 — head identity
        "head_identity": {
            "n_8bit": len(r8["retrieval_heads"]),
            "n_fp16": len(r16["retrieval_heads"]),
            **jaccard(r8["retrieval_heads"], r16["retrieval_heads"]),
        },
        # Level 2 — score agreement (both metrics)
        "score_agreement": {
            "argmax": _corr(r8["argmax"], r16["argmax"]),
            "mass": _corr(r8["mass"], r16["mass"]),
        },
        # Level 3 — finding direction (both modes)
        "finding": {
            "8bit": _finding(r8["utility"]),
            "fp16": _finding(r16["utility"]),
        },
    }
    # The headline robustness verdict.
    same_dir = report["finding"]["8bit"]["direction"] == report["finding"]["fp16"]["direction"]
    report["finding"]["same_direction"] = same_dir
    report["verdict"] = (
        "robust: finding direction preserved under quantization"
        if same_dir else
        "FRAGILE: finding direction differs between 8-bit and fp16 — report as limitation"
    )

    out_dir = results_dir / "ablations"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"quant_ablation_{args.model}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    hi = report["head_identity"]
    logger.info("Head sets: n=%d (8bit) vs %d (fp16), Jaccard=%.3f",
                hi["n_8bit"], hi["n_fp16"], hi["jaccard"])
    logger.info("Score agreement: argmax rho=%.3f, mass rho=%.3f",
                report["score_agreement"]["argmax"]["spearman_rho"],
                report["score_agreement"]["mass"]["spearman_rho"])
    logger.info("Verdict: %s", report["verdict"])
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    main()
