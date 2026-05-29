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

        # Infer theta from model config; fall back to 10000 (LLaMA-2 default)
        self.theta: float = float(
            getattr(getattr(model, "config", None), "rope_theta", 10000.0)
        )
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

        # head_dim = hidden_dim / n_heads (integer)
        self.head_dim: int = self.hidden_dim // self.n_heads

        # Pre-compute RoPE frequency order (ascending frequency)
        # RoPE applies frequencies to *pairs* of dimensions: dim 2i and 2i+1
        # share frequency theta^(-2i/head_dim).
        # We assign each of the head_dim positions its corresponding frequency.
        half = self.head_dim // 2
        freqs = np.array(
            [self.theta ** (-2 * i / self.head_dim) for i in range(half)],
            dtype=np.float64,
        )
        # Expand: position 2i and 2i+1 share the same frequency
        freq_per_dim = np.repeat(freqs, 2)  # length = head_dim
        # Argsort ascending → index 0 is the *lowest* frequency dimension
        self.freq_order: np.ndarray = np.argsort(freq_per_dim)
        # Also store the frequency value for each sorted position
        self.freq_values_sorted: np.ndarray = freq_per_dim[self.freq_order]

    # ------------------------------------------------------------------
    # Core computation
    # ------------------------------------------------------------------

    def compute_query_projection_norms(self) -> np.ndarray:
        """
        Compute per-dimension L1 norms of the Q projection for every head.

        For each layer the q_proj weight matrix has shape
        (hidden_dim, hidden_dim). We reshape it to (n_heads, head_dim, hidden_dim)
        then take the L1 norm along the input-feature axis (axis=-1), yielding
        a (n_heads, head_dim) matrix of norms.

        Returns:
            np.ndarray of shape (n_layers, n_heads, head_dim), float32.
        """
        all_norms: list[np.ndarray] = []

        for layer_idx, layer in enumerate(self.model.model.layers):
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

            weight = q_proj.weight.detach().cpu().float()  # (hidden_dim, hidden_dim)

            # Handle GQA / MQA: q_proj output dim may differ from hidden_dim
            out_dim = weight.shape[0]
            n_q_heads = out_dim // self.head_dim

            # Reshape to (n_q_heads, head_dim, in_features)
            w = weight.view(n_q_heads, self.head_dim, -1)
            # L1 norm over input features: (n_q_heads, head_dim)
            norms = w.abs().sum(dim=-1).numpy().astype(np.float32)

            # If GQA has fewer Q heads than n_heads, tile to match
            if n_q_heads < self.n_heads:
                repeats = self.n_heads // n_q_heads
                norms = np.tile(norms, (repeats, 1))

            all_norms.append(norms[: self.n_heads])

            del weight, w

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
        # per_head_scalar: (n_layers, n_heads) — average utility across dims
        per_head_scalar = norms.mean(axis=-1)  # (n_layers, n_heads)

        retrieval_set = set(retrieval_heads)
        n_layers, n_heads = per_head_scalar.shape

        retrieval_vals: list[float] = []
        non_retrieval_vals: list[float] = []

        for l in range(n_layers):
            for h in range(n_heads):
                val = float(per_head_scalar[l, h])
                if (l, h) in retrieval_set:
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

        pearson_r, pearson_p = stats.pearsonr(per_head_retrieval, per_head_utility)
        spearman_rho, spearman_p = stats.spearmanr(per_head_retrieval, per_head_utility)

        logger.info(
            "Correlation — Pearson r=%.3f (p=%.4f), Spearman ρ=%.3f (p=%.4f)",
            pearson_r,
            pearson_p,
            spearman_rho,
            spearman_p,
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

        for l in range(n_layers):
            for h in range(n_heads):
                norm_vec = norms[l, h, :]  # (head_dim,)
                # Sort by frequency order
                sorted_norm = norm_vec[self.freq_order]
                if (l, h) in retrieval_set:
                    retrieval_norms.append(sorted_norm)
                else:
                    non_retrieval_norms.append(sorted_norm)

        def _stats(arr_list: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
            if not arr_list:
                z = np.zeros(head_dim, dtype=np.float32)
                return z, z
            mat = np.stack(arr_list, axis=0)  # (N, head_dim)
            return mat.mean(axis=0).astype(np.float32), mat.std(axis=0).astype(np.float32)

        ret_mean, ret_std = _stats(retrieval_norms)
        non_ret_mean, non_ret_std = _stats(non_retrieval_norms)

        # Normalize so both groups are comparable on [0, 1]
        all_max = max(ret_mean.max(), non_ret_mean.max(), 1e-8)
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
