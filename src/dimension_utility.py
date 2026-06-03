"""
Dimension utility measurement following Chiang & Yogatama (ACL Findings 2025).

Utility is proxied by the L1 norm of each row of the Q projection matrix.
High-frequency RoPE dimensions rotate rapidly, reducing their utility for
long-range attention; this module quantifies that effect per head.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


class DimensionUtilityAnalyzer:
    """
    Measures per-dimension utility of Q projections across all attention heads.

    Utility proxy: L1 norm of each row in q_proj.weight, reshaped per head.
    Higher norm → the model dedicates more representational capacity to that
    dimension → higher utility.
    """

    def __init__(self, model: Any, config: dict) -> None:
        """
        Args:
            model: HuggingFace causal LM with model.model.layers structure.
            config: Global config dict (models section used for theta/head_dim).
        """
        self.model = model
        self.config = config

        # FIX #14: Robust rope_theta extraction across model families.
        # Qwen2.5 stores it as rope_theta; some models use rope_scaling.rope_theta.
        self.theta: float = self._extract_rope_theta(model)
        logger.info("RoPE theta = %.0f", self.theta)

        try:
            self.n_layers: int = model.config.num_hidden_layers
            self.n_heads: int = model.config.num_attention_heads
            self.hidden_dim: int = model.config.hidden_size
        except AttributeError as exc:
            raise ValueError(
                "Model config missing required fields (num_hidden_layers, "
                "num_attention_heads, hidden_size)."
            ) from exc

        # FIX #5 / #14: Prefer explicit head_dim from config when available.
        # Avoids integer-division errors when hidden_size % n_heads != 0.
        if hasattr(model.config, "head_dim"):
            self.head_dim: int = model.config.head_dim
        else:
            self.head_dim = self.hidden_dim // self.n_heads

        # Pre-compute RoPE frequency order (ascending frequency).
        # RoPE applies frequencies to *pairs* of dimensions: dim 2i and 2i+1
        # share frequency theta^(-2i/head_dim).
        half = self.head_dim // 2
        freqs = np.array(
            [self.theta ** (-2 * i / self.head_dim) for i in range(half)],
            dtype=np.float64,
        )
        # Expand: position 2i and 2i+1 share the same frequency
        freq_per_dim = np.repeat(freqs, 2)  # length = head_dim
        # Argsort ascending → index 0 is the *lowest* frequency dimension
        self.freq_order: np.ndarray = np.argsort(freq_per_dim)
        self.freq_values_sorted: np.ndarray = freq_per_dim[self.freq_order]

    @staticmethod
    def _extract_rope_theta(model: Any) -> float:
        """Extract rope_theta from model config, handling multiple attribute paths."""
        cfg = getattr(model, "config", None)
        if cfg is None:
            return 10000.0

        # Direct attribute (most models: LLaMA, OLMo, Qwen2.5)
        if hasattr(cfg, "rope_theta"):
            return float(cfg.rope_theta)

        # Nested under rope_scaling dict
        rope_scaling = getattr(cfg, "rope_scaling", None)
        if isinstance(rope_scaling, dict):
            for key in ("rope_theta", "base"):
                if key in rope_scaling:
                    return float(rope_scaling[key])

        logger.warning(
            "Could not find rope_theta in model config; defaulting to 10000. "
            "Frequency profile may be inaccurate."
        )
        return 10000.0

    def _get_layers(self):
        """Return model layer list, handling multiple architecture variants."""
        model_inner = getattr(self.model, "model", self.model)
        for attr in ("layers", "transformer.blocks", "transformer.h", "decoder.layers"):
            obj = model_inner
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError("Cannot locate layer list in model architecture.")

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_query_projection_norms(self) -> np.ndarray:
        """
        Compute per-dimension L1 norms of the Q projection for every head.

        For each layer the q_proj weight matrix has shape
        (n_q_heads * head_dim, hidden_dim). We reshape it to
        (n_q_heads, head_dim, hidden_dim) then take the L1 norm along
        the input-feature axis (axis=-1), yielding (n_q_heads, head_dim).

        Returns:
            np.ndarray of shape (n_layers, n_heads, head_dim), float32.
        """
        layers = self._get_layers()
        all_norms: list[np.ndarray] = []

        for layer_idx, layer in enumerate(layers):
            try:
                q_proj = layer.self_attn.q_proj
            except AttributeError:
                logger.warning(
                    "Layer %d has no self_attn.q_proj; inserting zeros.", layer_idx
                )
                all_norms.append(
                    np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
                )
                continue

            weight = q_proj.weight.detach().cpu().float()  # (out_dim, in_dim)
            out_dim = weight.shape[0]

            # FIX #5: q_proj always projects to ALL Q heads, so n_q_heads = out_dim // head_dim.
            # This equals self.n_heads for standard models and remains correct for GQA
            # (where n_kv_heads < n_heads but q_proj still outputs n_heads * head_dim).
            n_q_heads = out_dim // self.head_dim
            if n_q_heads == 0:
                logger.warning(
                    "Layer %d: q_proj out_dim=%d < head_dim=%d; inserting zeros.",
                    layer_idx, out_dim, self.head_dim,
                )
                all_norms.append(
                    np.zeros((self.n_heads, self.head_dim), dtype=np.float32)
                )
                del weight
                continue

            # Reshape to (n_q_heads, head_dim, in_features)
            w = weight.view(n_q_heads, self.head_dim, -1)
            # L1 norm over input features → (n_q_heads, head_dim)
            norms = w.abs().sum(dim=-1).numpy().astype(np.float32)
            del weight, w

            if n_q_heads < self.n_heads:
                # Should not happen for q_proj, but guard anyway.
                # Repeat groups to match n_heads (correct for any GQA structure).
                repeat_factor = self.n_heads // n_q_heads
                norms = np.repeat(norms, repeat_factor, axis=0)
            elif n_q_heads > self.n_heads:
                norms = norms[: self.n_heads]

            all_norms.append(norms)

        result = np.stack(all_norms, axis=0)  # (n_layers, n_heads, head_dim)
        logger.info(
            "Q-projection norms computed: shape=%s, mean=%.4f",
            result.shape,
            result.mean(),
        )
        return result

    # ------------------------------------------------------------------
    # Statistical analysis
    # ------------------------------------------------------------------

    def compute_utility_scores(
        self,
        norms: np.ndarray,
        retrieval_heads: list[tuple[int, int]],
    ) -> dict:
        """
        Compare mean dimension utility between retrieval and non-retrieval heads.

        Utility scalar per head = mean(L1 norms across head_dim).

        Args:
            norms: (n_layers, n_heads, head_dim) from compute_query_projection_norms().
            retrieval_heads: List of (layer, head) tuples from RetrievalHeadDetector.

        Returns:
            Dict with keys: retrieval_mean, retrieval_std, non_retrieval_mean,
            non_retrieval_std, t_statistic, p_value, per_head_scalar,
            n_retrieval, n_non_retrieval.
        """
        per_head_scalar = norms.mean(axis=-1)  # (n_layers, n_heads)
        retrieval_set = set(retrieval_heads)
        n_layers, n_heads = per_head_scalar.shape

        retrieval_vals: list[float] = []
        non_retrieval_vals: list[float] = []

        for layer in range(n_layers):
            for head in range(n_heads):
                val = float(per_head_scalar[layer, head])
                if (layer, head) in retrieval_set:
                    retrieval_vals.append(val)
                else:
                    non_retrieval_vals.append(val)

        if len(retrieval_vals) < 2 or len(non_retrieval_vals) < 2:
            logger.warning(
                "Insufficient samples for t-test: %d retrieval, %d non-retrieval.",
                len(retrieval_vals),
                len(non_retrieval_vals),
            )
            t_stat, p_val = float("nan"), float("nan")
        else:
            t_stat, p_val = stats.ttest_ind(
                retrieval_vals, non_retrieval_vals, equal_var=False
            )

        ret_arr = np.array(retrieval_vals, dtype=np.float32)
        non_ret_arr = np.array(non_retrieval_vals, dtype=np.float32)

        return {
            "retrieval_mean": float(ret_arr.mean()) if len(ret_arr) else float("nan"),
            "retrieval_std": float(ret_arr.std()) if len(ret_arr) else float("nan"),
            "non_retrieval_mean": float(non_ret_arr.mean()) if len(non_ret_arr) else float("nan"),
            "non_retrieval_std": float(non_ret_arr.std()) if len(non_ret_arr) else float("nan"),
            "t_statistic": float(t_stat),
            "p_value": float(p_val),
            "per_head_scalar": per_head_scalar.tolist(),
            "n_retrieval": len(retrieval_vals),
            "n_non_retrieval": len(non_retrieval_vals),
        }

    def compute_retrieval_utility_correlation(
        self,
        retrieval_scores: np.ndarray,
        norms: np.ndarray,
    ) -> dict:
        """
        Compute Pearson and Spearman correlation between retrieval score and
        mean dimension utility across all heads.

        Args:
            retrieval_scores: (n_layers, n_heads) hit-rate matrix.
            norms: (n_layers, n_heads, head_dim) norm matrix.

        Returns:
            Dict with keys: pearson_r, pearson_p, spearman_rho, spearman_p.
        """
        per_head_utility = norms.mean(axis=-1).flatten()
        per_head_retrieval = retrieval_scores.flatten()

        if len(np.unique(per_head_retrieval)) < 2 or len(np.unique(per_head_utility)) < 2:
            logger.warning(
                "Constant array detected; correlation is undefined. Returning NaN."
            )
            return {
                "pearson_r": float("nan"),
                "pearson_p": float("nan"),
                "spearman_rho": float("nan"),
                "spearman_p": float("nan"),
            }

        pearson_r, pearson_p = stats.pearsonr(per_head_retrieval, per_head_utility)
        spearman_rho, spearman_p = stats.spearmanr(per_head_retrieval, per_head_utility)

        logger.info(
            "Correlation — Pearson r=%.3f (p=%.4f), Spearman ρ=%.3f (p=%.4f)",
            pearson_r, pearson_p, spearman_rho, spearman_p,
        )
        return {
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "spearman_rho": float(spearman_rho),
            "spearman_p": float(spearman_p),
        }

    # ------------------------------------------------------------------
    # Frequency profile
    # ------------------------------------------------------------------

    def dimension_profile_by_frequency(
        self,
        norms: np.ndarray,
        retrieval_heads: list[tuple[int, int]],
    ) -> dict:
        """
        Compute per-dimension utility profiles sorted by RoPE frequency.

        For each group (retrieval / non-retrieval) we gather all heads'
        norm vectors, sort dimensions by ascending RoPE frequency, then
        compute mean ± std across heads.

        Args:
            norms: (n_layers, n_heads, head_dim) norm matrix.
            retrieval_heads: List of (layer, head) tuples.

        Returns:
            Dict with keys: retrieval_mean, retrieval_std, non_retrieval_mean,
            non_retrieval_std, freq_order, theta, head_dim.
        """
        retrieval_set = set(retrieval_heads)
        n_layers, n_heads, head_dim = norms.shape

        retrieval_norms: list[np.ndarray] = []
        non_retrieval_norms: list[np.ndarray] = []

        for layer in range(n_layers):
            for head in range(n_heads):
                norm_vec = norms[layer, head, :]
                sorted_norm = norm_vec[self.freq_order]
                if (layer, head) in retrieval_set:
                    retrieval_norms.append(sorted_norm)
                else:
                    non_retrieval_norms.append(sorted_norm)

        # FIX #8: Warn when retrieval_heads is empty — silent zeros would
        # produce a false H1 signal on the frequency profile plot.
        if not retrieval_norms:
            logger.warning(
                "dimension_profile_by_frequency: no retrieval heads found. "
                "Retrieval curve will be all zeros — do not interpret as H1 evidence. "
                "Consider lowering score_threshold."
            )

        def _stats(arr_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            if not arr_list:
                z = np.zeros(head_dim, dtype=np.float32)
                return z, z
            mat = np.stack(arr_list, axis=0)
            return mat.mean(axis=0).astype(np.float32), mat.std(axis=0).astype(np.float32)

        ret_mean, ret_std = _stats(retrieval_norms)
        non_ret_mean, non_ret_std = _stats(non_retrieval_norms)

        # Normalize so both groups are comparable on [0, 1].
        # FIX #8: Use only non_retrieval max when retrieval is empty, to avoid
        # dividing by 0 from an all-zeros ret_mean.
        all_max = max(
            float(ret_mean.max()) if retrieval_norms else 0.0,
            float(non_ret_mean.max()),
            1e-8,
        )
        ret_mean_norm = ret_mean / all_max
        non_ret_mean_norm = non_ret_mean / all_max

        return {
            "retrieval_mean": ret_mean_norm.tolist(),
            "retrieval_std": (ret_std / all_max).tolist(),
            "non_retrieval_mean": non_ret_mean_norm.tolist(),
            "non_retrieval_std": (non_ret_std / all_max).tolist(),
            "freq_order": self.freq_order.tolist(),
            "freq_values_sorted": self.freq_values_sorted.tolist(),
            "theta": self.theta,
            "head_dim": head_dim,
            "n_retrieval_heads": len(retrieval_norms),
            "n_non_retrieval_heads": len(non_retrieval_norms),
        }
