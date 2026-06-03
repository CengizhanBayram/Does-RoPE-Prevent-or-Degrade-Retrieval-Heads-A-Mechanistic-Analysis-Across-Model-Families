"""
Hook-based activation patching for causal analysis of dimension utility.

Implements Layer D: for each retrieval head, three conditions are tested
(low-utility zeroing, random zeroing, high-utility zeroing) to determine
whether the low-utility dimensions identified by Chiang & Yogatama (2025)
are actually load-bearing for retrieval behaviour.
"""

from __future__ import annotations

import gc
import logging
import random
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import pandas as pd
import torch
from torch import Tensor

logger = logging.getLogger(__name__)


class ActivationPatcher:
    """
    Applies reversible dimension-zeroing patches to Q/K projections via hooks.

    Patches are applied only for the duration of a `with patcher.patch_head(...)`
    block; model weights are never modified.
    """

    def __init__(self, model: Any, tokenizer: Any, config: dict) -> None:
        """
        Args:
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            config: Global config dict (activation_patching section used).
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.k_dims: int = config.get("activation_patching", {}).get("k_dims", 16)

        self.model.eval()

    # ------------------------------------------------------------------
    # Hook context manager
    # ------------------------------------------------------------------

    @contextmanager
    def patch_head(
        self,
        layer_idx: int,
        head_idx: int,
        dims_to_zero: list[int],
        mode: str = "qk",
    ) -> Generator[None, None, None]:
        """
        Context manager that zeroes specified dimensions in Q and/or K output.

        The hook intercepts the output of q_proj (and k_proj if mode includes
        'k') and sets the specified dimension slices for the given head to zero.

        Args:
            layer_idx: Index of the transformer layer.
            head_idx: Index of the attention head within that layer.
            dims_to_zero: List of per-head dimension indices to zero out.
            mode: "q", "k", or "qk" (default).

        Yields:
            Nothing; patches are active within the block.
        """
        layer = self.model.model.layers[layer_idx]
        hooks = []
        head_dim = self._get_head_dim()

        def _make_hook(proj_name: str):
            def hook(module, input_: Any, output: Tensor) -> Tensor:
                # output shape: (batch, seq_len, hidden_dim)
                out = output.clone()
                start = head_idx * head_dim
                for d in dims_to_zero:
                    if 0 <= d < head_dim:
                        out[:, :, start + d] = 0.0
                return out
            return hook

        try:
            if "q" in mode:
                h = layer.self_attn.q_proj.register_forward_hook(_make_hook("q_proj"))
                hooks.append(h)
            if "k" in mode:
                h = layer.self_attn.k_proj.register_forward_hook(_make_hook("k_proj"))
                hooks.append(h)
            yield
        finally:
            for h in hooks:
                h.remove()

    def _get_head_dim(self) -> int:
        """Return per-head dimension, preferring explicit config field."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            # Some models expose head_dim directly
            if hasattr(cfg, "head_dim"):
                return cfg.head_dim
            try:
                return cfg.hidden_size // cfg.num_attention_heads
            except AttributeError:
                pass
        return 128  # safe fallback for 7B models

    # ------------------------------------------------------------------
    # NIAH accuracy under a patch condition
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _evaluate_accuracy(
        self,
        samples: list[dict],
        layer_idx: int | None = None,
        head_idx: int | None = None,
        dims_to_zero: list[int] | None = None,
        mode: str = "qk",
    ) -> float:
        """
        Compute NIAH accuracy (fraction of samples where the code appears in output).

        A NEW context manager is created for EACH sample so the generator is
        never exhausted mid-loop.

        Args:
            samples: NIAH sample dicts.
            layer_idx: Layer to patch (None = no patch, i.e. baseline).
            head_idx: Head to patch.
            dims_to_zero: Dimensions to zero.
            mode: Projection axes to patch.

        Returns:
            Accuracy in [0, 1].
        """
        do_patch = (
            layer_idx is not None
            and head_idx is not None
            and dims_to_zero is not None
            and len(dims_to_zero) > 0
        )
        device = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            next(self.model.parameters()),
        ).device
        correct = 0
        total = 0

        for sample in samples:
            input_ids = torch.tensor(
                [sample["prompt_ids"]], dtype=torch.long, device=device
            )
            try:
                # FIX #3: Create a fresh context manager for every sample.
                # @contextmanager generators are single-use; reusing one causes
                # GeneratorExit on the second iteration.
                if do_patch:
                    ctx = self.patch_head(layer_idx, head_idx, dims_to_zero, mode)
                else:
                    ctx = _null_context()

                with ctx:
                    out = self.model.generate(
                        input_ids,
                        max_new_tokens=20,
                        do_sample=False,
                    )

                generated_ids = out[0, input_ids.shape[1]:]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )
                if sample["code"] in generated_text:
                    correct += 1
                total += 1

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("OOM during patching eval; skipping sample.")
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise
            finally:
                # FIX #17-style: guard del so NameError can't happen
                if "input_ids" in locals():
                    del input_ids
                if "out" in locals():
                    del out
                torch.cuda.empty_cache()

        return correct / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    # Full experiment
    # ------------------------------------------------------------------

    def run_patching_experiment(
        self,
        retrieval_heads: list[tuple[int, int]],
        utility_scores: dict,
        samples: list[dict],
        k_dims: int | None = None,
        n_samples: int = 50,
        random_seeds: list[int] | None = None,
    ) -> dict:
        """
        Run three-condition activation patching for each retrieval head.

        Conditions per head:
          - baseline: no patch  (computed once, shared across all heads)
          - low_utility: zero k_dims dimensions with lowest L1 norm
          - random: zero k_dims randomly chosen dimensions (avg over 5 seeds)
          - high_utility: zero k_dims dimensions with highest L1 norm

        Args:
            retrieval_heads: List of (layer, head) tuples.
            utility_scores: Dict from DimensionUtilityAnalyzer (must contain
                "_norms" ndarray or "per_head_scalar" nested list).
            samples: NIAH samples for evaluation.
            k_dims: Number of dimensions to zero. Defaults to self.k_dims.
            n_samples: Number of samples to use per condition.
            random_seeds: Seeds for random condition averaging (default: 5 seeds).

        Returns:
            Dict keyed by (layer, head) tuples with per-condition accuracies.
        """
        k = k_dims or self.k_dims
        seeds = random_seeds or list(range(5))
        eval_samples = samples[:n_samples]

        norms_matrix: np.ndarray | None = utility_scores.get("_norms")

        # FIX #11: Compute baseline ONCE outside the head loop.
        logger.info("Computing baseline accuracy (no patch) …")
        baseline_acc = self._evaluate_accuracy(eval_samples)
        logger.info("Baseline accuracy: %.3f", baseline_acc)

        results: dict = {}

        for (layer_idx, head_idx) in retrieval_heads:
            logger.info("Patching experiment: layer=%d head=%d", layer_idx, head_idx)

            if norms_matrix is not None:
                head_norms = norms_matrix[layer_idx, head_idx]  # (head_dim,)
            else:
                scalar_grid = np.array(utility_scores["per_head_scalar"])
                head_norms = np.full(
                    self._get_head_dim(),
                    scalar_grid[layer_idx, head_idx],
                    dtype=np.float32,
                )

            dim_indices = np.arange(len(head_norms))
            sorted_by_utility = np.argsort(head_norms)  # ascending = lowest first

            low_dims = sorted_by_utility[:k].tolist()
            high_dims = sorted_by_utility[-k:].tolist()

            low_acc = self._evaluate_accuracy(
                eval_samples, layer_idx, head_idx, low_dims
            )
            high_acc = self._evaluate_accuracy(
                eval_samples, layer_idx, head_idx, high_dims
            )

            random_accs: list[float] = []
            for seed in seeds:
                rng = np.random.RandomState(seed)
                rand_dims = rng.choice(dim_indices, size=k, replace=False).tolist()
                acc = self._evaluate_accuracy(
                    eval_samples, layer_idx, head_idx, rand_dims
                )
                random_accs.append(acc)
            random_acc = float(np.mean(random_accs))

            results[(layer_idx, head_idx)] = {
                "layer": layer_idx,
                "head": head_idx,
                "baseline": baseline_acc,
                "low_utility": low_acc,
                "random": random_acc,
                "high_utility": high_acc,
                "low_dims": low_dims,
                "high_dims": high_dims,
                "k_dims": k,
            }
            logger.info(
                "  baseline=%.3f  low=%.3f  rand=%.3f  high=%.3f",
                baseline_acc, low_acc, random_acc, high_acc,
            )

        return results

    # ------------------------------------------------------------------
    # Causal effect
    # ------------------------------------------------------------------

    def compute_causal_effect(self, results: dict) -> pd.DataFrame:
        """
        Compute the causal effect of low-utility dimension removal.

        causal_effect = (baseline - low_utility) - (baseline - random)
                      = random - low_utility

        Negative → low-utility dims are already unused (H2-consistent).
        Positive → low-utility dims are load-bearing (unexpected).

        Args:
            results: Output of run_patching_experiment().

        Returns:
            DataFrame with columns: layer, head, baseline, low_utility, random,
            high_utility, causal_effect, relative_drop_low, relative_drop_high.
        """
        rows = []
        for (layer, head), data in results.items():
            baseline = data["baseline"]
            low = data["low_utility"]
            rand = data["random"]
            high = data["high_utility"]

            causal_effect = rand - low
            relative_drop_low = (baseline - low) / (baseline + 1e-8)
            relative_drop_high = (baseline - high) / (baseline + 1e-8)

            rows.append(
                {
                    "layer": layer,
                    "head": head,
                    "baseline": baseline,
                    "low_utility": low,
                    "random": rand,
                    "high_utility": high,
                    "causal_effect": causal_effect,
                    "relative_drop_low": relative_drop_low,
                    "relative_drop_high": relative_drop_high,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: null context manager (module-level, importable)
# ---------------------------------------------------------------------------

@contextmanager
def _null_context() -> Generator[None, None, None]:
    """No-op context manager used when no patch is applied."""
    yield
