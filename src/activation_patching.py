"""
Hook-based activation patching for causal analysis of dimension utility.

Implements Layer D: for each retrieval head, three conditions are tested
(low-utility zeroing, random zeroing, high-utility zeroing) to determine
whether the low-utility dimensions identified by Chiang & Yogatama (2025)
are actually load-bearing for retrieval behaviour.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import re
import tempfile
from contextlib import contextmanager
from typing import Any, Generator

import numpy as np
import pandas as pd
import torch
from torch import Tensor

try:  # progress bar is optional; degrade to a no-op if tqdm is unavailable
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable=None, **_kwargs):
        return iterable

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

    @contextmanager
    def patch_heads(
        self,
        head_dims: "dict[tuple[int, int], list[int]]",
        mode: str = "qk",
    ) -> Generator[None, None, None]:
        """Zero per-head dims for MANY heads at once (population-level patch).

        Args:
            head_dims: maps (layer, head) -> per-head dim indices to zero.
            mode: "q", "k", or "qk".

        Heads in the same layer share one q_proj/k_proj, so a hook is registered
        ONCE per layer and zeroes all that layer's requested (head, dim) columns.
        GQA: a query head maps to KV head (head // group_size) for k_proj.
        """
        head_dim = self._get_head_dim()
        n_heads, n_kv_heads = self._get_n_heads_kv()
        group_size = max(1, n_heads // max(1, n_kv_heads))

        by_layer: dict[int, list[tuple[int, list[int]]]] = {}
        for (layer_idx, head_idx), dims in head_dims.items():
            by_layer.setdefault(layer_idx, []).append((head_idx, dims))

        def _cols(entries: list, is_kv: bool) -> list[int]:
            cols: set[int] = set()
            for head_idx, dims in entries:
                base = (head_idx // group_size if is_kv else head_idx) * head_dim
                for d in dims:
                    if 0 <= d < head_dim:
                        cols.add(base + d)
            return sorted(cols)

        def _make_hook(cols: list[int]):
            def hook(module, input_: Any, output: Tensor) -> Tensor:
                out = output.clone()
                width = out.shape[-1]
                for c in cols:
                    if c < width:
                        out[:, :, c] = 0.0
                return out
            return hook

        hooks = []
        try:
            for layer_idx, entries in by_layer.items():
                layer = self.model.model.layers[layer_idx]
                if "q" in mode:
                    hooks.append(layer.self_attn.q_proj.register_forward_hook(
                        _make_hook(_cols(entries, is_kv=False))))
                if "k" in mode:
                    hooks.append(layer.self_attn.k_proj.register_forward_hook(
                        _make_hook(_cols(entries, is_kv=True))))
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

    @torch.no_grad()
    def _evaluate_population(
        self,
        samples: list[dict],
        head_dims: "dict[tuple[int, int], list[int]] | None" = None,
        mode: str = "qk",
    ) -> float:
        """NIAH accuracy with MANY heads patched simultaneously (or none)."""
        device = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            next(self.model.parameters()),
        ).device
        correct = total = 0
        for sample in samples:
            input_ids = torch.tensor(
                [sample["prompt_ids"]], dtype=torch.long, device=device
            )
            try:
                ctx = self.patch_heads(head_dims, mode) if head_dims else _null_context()
                with ctx:
                    out = self.model.generate(
                        input_ids, max_new_tokens=20, do_sample=False
                    )
                gen = self.tokenizer.decode(
                    out[0, input_ids.shape[1]:], skip_special_tokens=True
                )
                if sample["code"] in gen:
                    correct += 1
                total += 1
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning("OOM during population eval; skipping sample.")
                    gc.collect(); torch.cuda.empty_cache()
                else:
                    raise
            finally:
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
        checkpoint_path: str | None = None,
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
            checkpoint_path: Optional JSON path. When given, the baseline and
                each head's result are written incrementally (atomically) after
                every head, and an existing file is loaded on start so already
                completed heads are skipped. This makes a long run resumable
                across interrupted sessions WITHOUT changing any computed value
                (identical heads, conditions, and seeds). When None, behaviour is
                exactly as before (single in-memory pass, no persistence).

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

        # Resume from checkpoint if one exists (skips already-computed heads and
        # reuses the saved baseline). Computed values are unchanged.
        results: dict = {}
        baseline_acc: float | None = None
        if checkpoint_path is not None:
            loaded = _load_checkpoint(checkpoint_path)
            if loaded is not None:
                baseline_acc = loaded.get("baseline")
                for key, entry in loaded.get("heads", {}).items():
                    lh = _parse_head_key(key)
                    if lh is not None:
                        results[lh] = entry
                if results:
                    logger.info(
                        "Resuming from checkpoint %s: %d head(s) already done.",
                        checkpoint_path, len(results),
                    )

        # FIX #11: Compute baseline ONCE outside the head loop.
        if baseline_acc is None:
            logger.info("Computing baseline accuracy (no patch) …")
            baseline_acc = self._evaluate_accuracy(eval_samples)
            logger.info("Baseline accuracy: %.3f", baseline_acc)
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path, baseline_acc, results)

        # Only patch heads not already restored from a checkpoint, so the
        # progress bar reflects true remaining work on resume.
        todo = [h for h in retrieval_heads if h not in results]
        done = len(retrieval_heads) - len(todo)
        for (layer_idx, head_idx) in tqdm(
            todo, desc=f"Patching heads ({done} done)", unit="head"
        ):
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
            if checkpoint_path is not None:
                _save_checkpoint(checkpoint_path, baseline_acc, results)

        return results

    def run_population_patching(
        self,
        retrieval_heads: list[tuple[int, int]],
        utility_scores: dict,
        samples: list[dict],
        k_dims: int | None = None,
        n_samples: int = 50,
        random_seeds: list[int] | None = None,
        freq_order: "np.ndarray | None" = None,
    ) -> dict:
        """Patch ALL given heads simultaneously, per condition (breaks ceiling).

        When per-head zeroing leaves NIAH at ceiling (a single head has too
        little leverage), this asks the population question: does zeroing a
        dimension *class* across ALL retrieval heads at once degrade recall?
        Cost is per CONDITION, not per head: ``5 + len(random_seeds)``
        evaluations total.

        Returns a flat dict: n_heads, baseline, low_utility, random,
        high_utility, causal_effect, and (when ``freq_order`` is given) low_freq,
        high_freq, frequency_effect — each a single population accuracy.
        """
        k = k_dims or self.k_dims
        seeds = random_seeds or list(range(5))
        eval_samples = samples[:n_samples]
        norms_matrix: np.ndarray | None = utility_scores.get("_norms")
        head_dim = self._get_head_dim()

        low_map: dict = {}
        high_map: dict = {}
        for (layer_idx, head_idx) in retrieval_heads:
            if norms_matrix is not None:
                head_norms = norms_matrix[layer_idx, head_idx]
            else:
                scalar_grid = np.array(utility_scores["per_head_scalar"])
                head_norms = np.full(
                    head_dim, scalar_grid[layer_idx, head_idx], dtype=np.float32
                )
            order = np.argsort(head_norms)
            low_map[(layer_idx, head_idx)] = order[:k].tolist()
            high_map[(layer_idx, head_idx)] = order[-k:].tolist()

        out: dict = {
            "n_heads": len(retrieval_heads),
            "k_dims": k,
            "baseline": self._evaluate_population(eval_samples, None),
            "low_utility": self._evaluate_population(eval_samples, low_map),
            "high_utility": self._evaluate_population(eval_samples, high_map),
        }
        rand_accs = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            rmap = {
                lh: rng.choice(head_dim, size=k, replace=False).tolist()
                for lh in retrieval_heads
            }
            rand_accs.append(self._evaluate_population(eval_samples, rmap))
        out["random"] = float(np.mean(rand_accs))
        out["causal_effect"] = out["random"] - out["low_utility"]
        logger.info(
            "Population (%d heads): baseline=%.3f low=%.3f rand=%.3f high=%.3f",
            out["n_heads"], out["baseline"], out["low_utility"],
            out["random"], out["high_utility"],
        )

        if freq_order is not None:
            fo = np.asarray(freq_order).astype(int)
            low_freq = fo[:k].tolist()
            high_freq = fo[-k:].tolist()
            out["low_freq"] = self._evaluate_population(
                eval_samples, {lh: low_freq for lh in retrieval_heads}
            )
            out["high_freq"] = self._evaluate_population(
                eval_samples, {lh: high_freq for lh in retrieval_heads}
            )
            out["frequency_effect"] = out["low_freq"] - out["high_freq"]
            logger.info(
                "Population freq: low_freq=%.3f high_freq=%.3f freq_effect=%.4f",
                out["low_freq"], out["high_freq"], out["frequency_effect"],
            )
        return out

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


# ---------------------------------------------------------------------------
# Helpers: resumable checkpointing for run_patching_experiment
# ---------------------------------------------------------------------------

_HEAD_KEY_RE = re.compile(r"layer(\d+)_head(\d+)")


def _parse_head_key(key: str) -> tuple[int, int] | None:
    """Parse a 'layer{L}_head{H}' checkpoint key back into an (L, H) tuple."""
    m = _HEAD_KEY_RE.fullmatch(key)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2))


def _load_checkpoint(path: str) -> dict | None:
    """Load a patching checkpoint, or None if missing/corrupt (start fresh)."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Checkpoint %s unreadable (%s); starting fresh.", path, exc)
        return None


def _save_checkpoint(path: str, baseline: float, results: dict) -> None:
    """Atomically write {baseline, heads} so an interrupt cannot corrupt it.

    Writes to a temp file in the same directory and renames over the target,
    which is atomic on a given filesystem (so a disconnect mid-write leaves the
    previous good checkpoint intact).
    """
    payload = {
        "baseline": baseline,
        "heads": {f"layer{l}_head{h}": v for (l, h), v in results.items()},
    }
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
