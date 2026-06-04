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
        n_heads, n_kv_heads = self._get_n_heads_kv()
        group_size = max(1, n_heads // max(1, n_kv_heads))
        # FIX P2 (GQA): q_proj outputs all n_heads query heads, so the query head
        # lives at head_idx*head_dim. But k_proj outputs only n_kv_heads heads;
        # query head_idx maps to KV head (head_idx // group_size). Using
        # head_idx*head_dim into k_proj is out of bounds for GQA (e.g. LLaMA-3:
        # 32 q heads, 8 kv heads → head_idx 31 → 3968 >> 1024).
        kv_head_idx = head_idx // group_size

        def _make_hook(is_kv: bool):
            start = (kv_head_idx if is_kv else head_idx) * head_dim

            def hook(module, input_: Any, output: Tensor) -> Tensor:
                # output shape: (batch, seq_len, n_proj_heads * head_dim)
                out = output.clone()
                width = out.shape[-1]
                for d in dims_to_zero:
                    col = start + d
                    if 0 <= d < head_dim and col < width:
                        out[:, :, col] = 0.0
                return out
            return hook

        try:
            if "q" in mode:
                h = layer.self_attn.q_proj.register_forward_hook(_make_hook(is_kv=False))
                hooks.append(h)
            if "k" in mode:
                h = layer.self_attn.k_proj.register_forward_hook(_make_hook(is_kv=True))
                hooks.append(h)
            yield
        finally:
            for h in hooks:
                h.remove()

    def _get_n_heads_kv(self) -> tuple[int, int]:
        """Return (num_attention_heads, num_key_value_heads). MHA → kv == heads."""
        cfg = getattr(self.model, "config", None)
        n_heads = getattr(cfg, "num_attention_heads", 32) if cfg else 32
        n_kv = getattr(cfg, "num_key_value_heads", None) if cfg else None
        if n_kv is None:
            n_kv = n_heads  # MHA (e.g. LLaMA-2)
        return int(n_heads), int(n_kv)

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
        logger.warning(
            "Could not derive head_dim from model config; falling back to 128. "
            "Patched dimension indices may be wrong for this model (item D4)."
        )
        return 128  # fallback for 7B models

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
        freq_order: "np.ndarray | None" = None,
    ) -> dict:
        """
        Run multi-condition activation patching for each retrieval head.

        Utility-axis conditions (always run):
          - baseline: no patch  (computed once, shared across all heads)
          - low_utility: zero k_dims dimensions with lowest L1 norm
          - random: zero k_dims randomly chosen dimensions (avg over 5 seeds)
          - high_utility: zero k_dims dimensions with highest L1 norm

        Frequency-axis conditions (run only when ``freq_order`` is given):
          - low_freq: zero the k_dims LOWEST-frequency RoPE dimensions
          - high_freq: zero the k_dims HIGHEST-frequency RoPE dimensions

        The frequency conditions are the key causal test (paper §6): they
        disentangle whether retrieval is degraded by *RoPE frequency
        specifically* or merely by *general dimension utility*. Because
        low-frequency and high-utility dimensions are correlated but not
        identical, comparing high_freq vs low_freq isolates the frequency axis.

        Args:
            retrieval_heads: List of (layer, head) tuples.
            utility_scores: Dict from DimensionUtilityAnalyzer (must contain
                "_norms" ndarray or "per_head_scalar" nested list).
            samples: NIAH samples for evaluation.
            k_dims: Number of dimensions to zero. Defaults to self.k_dims.
            n_samples: Number of samples to use per condition.
            random_seeds: Seeds for random condition averaging (default: 5 seeds).
            freq_order: Per-head dimension indices sorted by ASCENDING RoPE
                frequency (``DimensionUtilityAnalyzer.freq_order``). When None,
                the frequency conditions are skipped.

        Returns:
            Dict keyed by (layer, head) tuples with per-condition accuracies.
        """
        k = k_dims or self.k_dims
        seeds = random_seeds or list(range(5))
        eval_samples = samples[:n_samples]

        # Frequency-ranked dimension indices (constant across heads for a model).
        low_freq_dims: list[int] | None = None
        high_freq_dims: list[int] | None = None
        if freq_order is not None:
            fo = np.asarray(freq_order).astype(int)
            low_freq_dims = fo[:k].tolist()      # lowest-frequency dims
            high_freq_dims = fo[-k:].tolist()    # highest-frequency dims
            logger.info(
                "Frequency conditions enabled: low_freq dims=%s, high_freq dims=%s",
                low_freq_dims, high_freq_dims,
            )

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

            entry = {
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

            # Frequency-axis conditions (paper §6 causal test).
            if low_freq_dims is not None and high_freq_dims is not None:
                low_freq_acc = self._evaluate_accuracy(
                    eval_samples, layer_idx, head_idx, low_freq_dims
                )
                high_freq_acc = self._evaluate_accuracy(
                    eval_samples, layer_idx, head_idx, high_freq_dims
                )
                entry.update({
                    "low_freq": low_freq_acc,
                    "high_freq": high_freq_acc,
                    "low_freq_dims": low_freq_dims,
                    "high_freq_dims": high_freq_dims,
                })
                logger.info(
                    "  baseline=%.3f  low=%.3f  rand=%.3f  high=%.3f  "
                    "low_freq=%.3f  high_freq=%.3f",
                    baseline_acc, low_acc, random_acc, high_acc,
                    low_freq_acc, high_freq_acc,
                )
            else:
                logger.info(
                    "  baseline=%.3f  low=%.3f  rand=%.3f  high=%.3f",
                    baseline_acc, low_acc, random_acc, high_acc,
                )

            results[(layer_idx, head_idx)] = entry

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

        Frequency axis (paper §6), present only when the frequency conditions
        were run:
            frequency_effect = (baseline - high_freq) - (baseline - low_freq)
                             = low_freq - high_freq
        Positive → zeroing HIGH-frequency RoPE dims hurts retrieval more than
        zeroing low-frequency dims, i.e. retrieval is *frequency-specific*.
        ~Zero → frequency does not matter beyond general utility.

        Args:
            results: Output of run_patching_experiment().

        Returns:
            DataFrame with columns: layer, head, baseline, low_utility, random,
            high_utility, causal_effect, relative_drop_low, relative_drop_high,
            and (when available) low_freq, high_freq, frequency_effect.
        """
        rows = []
        for (layer, head), data in results.items():
            baseline = data["baseline"]
            low = data["low_utility"]
            rand = data["random"]
            high = data["high_utility"]

            causal_effect = rand - low
            # FIX D3: a relative drop is undefined when the model never solves
            # the task unpatched (baseline == 0). Report NaN instead of a
            # meaningless ratio dominated by the 1e-8 epsilon.
            if baseline <= 0:
                relative_drop_low = float("nan")
                relative_drop_high = float("nan")
                if not getattr(self, "_warned_zero_baseline", False):
                    logger.warning(
                        "Baseline accuracy is 0 for layer=%d head=%d; relative "
                        "drops set to NaN. Causal effects may be uninterpretable "
                        "— check that the model solves NIAH unpatched.",
                        layer, head,
                    )
                    self._warned_zero_baseline = True
            else:
                relative_drop_low = (baseline - low) / baseline
                relative_drop_high = (baseline - high) / baseline

            row = {
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

            # Frequency axis (paper §6) — only when those conditions were run.
            if "low_freq" in data and "high_freq" in data:
                low_freq = data["low_freq"]
                high_freq = data["high_freq"]
                row["low_freq"] = low_freq
                row["high_freq"] = high_freq
                # frequency_effect = low_freq - high_freq; >0 ⇒ zeroing high-freq
                # dims hurts more ⇒ retrieval is frequency-specific.
                row["frequency_effect"] = low_freq - high_freq

            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helper: null context manager (module-level, importable)
# ---------------------------------------------------------------------------

@contextmanager
def _null_context() -> Generator[None, None, None]:
    """No-op context manager used when no patch is applied."""
    yield
