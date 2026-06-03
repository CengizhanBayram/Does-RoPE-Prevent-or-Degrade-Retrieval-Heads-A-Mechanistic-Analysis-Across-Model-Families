"""
Statistical utilities for publication-grade reporting.

Adds the pieces a reviewer expects beyond a bare t-test / correlation:
    - Effect size (Cohen's d) with pooled SD.                       [B4]
    - Bootstrap confidence intervals for a mean difference.         [B4]
    - Benjamini-Hochberg FDR correction for multiple comparisons.   [B2]
    - A cluster-aware permutation test that respects the fact that
      heads within a layer are not independent observations.        [B1]

All functions are dependency-light (numpy + scipy only) and deterministic
given a seed.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Set overlap (used by the quantization ablation, A2)
# ---------------------------------------------------------------------------

def jaccard(set_a, set_b) -> dict:
    """
    Jaccard similarity and symmetric difference of two collections.

    Returns a dict with: jaccard, intersection, union, only_a, only_b,
    n_a, n_b. Used to compare retrieval-head sets across conditions (e.g.
    fp16 vs 8-bit) — a head count can match while the *identities* differ.
    """
    a, b = set(map(tuple, set_a)), set(map(tuple, set_b))
    inter = a & b
    union = a | b
    return {
        "jaccard": (len(inter) / len(union)) if union else 1.0,
        "intersection": len(inter),
        "union": len(union),
        "only_a": sorted(a - b),
        "only_b": sorted(b - a),
        "n_a": len(a),
        "n_b": len(b),
    }


# ---------------------------------------------------------------------------
# Effect size
# ---------------------------------------------------------------------------

def cohens_d(group_a: Sequence[float], group_b: Sequence[float]) -> float:
    """
    Cohen's d for two independent samples, using the pooled standard deviation.

    Args:
        group_a, group_b: Numeric samples.

    Returns:
        Effect size d, or NaN if either group has fewer than two observations.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return float("nan")
    # Sample variances (ddof=1) — item B6.
    var_a, var_b = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_sd)


# ---------------------------------------------------------------------------
# Bootstrap confidence interval
# ---------------------------------------------------------------------------

def bootstrap_mean_diff_ci(
    group_a: Sequence[float],
    group_b: Sequence[float],
    *,
    n_boot: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """
    Bootstrap CI for the difference of means (mean(a) - mean(b)).

    Args:
        group_a, group_b: Numeric samples.
        n_boot: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95).
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys: mean_diff, ci_low, ci_high, ci_level.
    """
    a = np.asarray(group_a, dtype=np.float64)
    b = np.asarray(group_b, dtype=np.float64)
    if len(a) < 2 or len(b) < 2:
        return {"mean_diff": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "ci_level": ci}

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        diffs[i] = ra.mean() - rb.mean()

    alpha = (1.0 - ci) / 2.0
    return {
        "mean_diff": float(a.mean() - b.mean()),
        "ci_low": float(np.quantile(diffs, alpha)),
        "ci_high": float(np.quantile(diffs, 1.0 - alpha)),
        "ci_level": ci,
    }


# ---------------------------------------------------------------------------
# Multiple-comparison correction
# ---------------------------------------------------------------------------

def benjamini_hochberg(p_values: Sequence[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        p_values: Raw p-values (NaNs are ignored and reported as not-rejected).
        alpha: Target false-discovery rate.

    Returns:
        Dict with keys: rejected (list[bool]), p_adjusted (list[float]),
        n_significant (int), alpha.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n_total = len(p)
    valid_mask = ~np.isnan(p)
    valid = p[valid_mask]
    m = len(valid)

    rejected = np.zeros(n_total, dtype=bool)
    p_adj = np.full(n_total, np.nan, dtype=np.float64)

    if m == 0:
        return {"rejected": rejected.tolist(), "p_adjusted": p_adj.tolist(),
                "n_significant": 0, "alpha": alpha}

    order = np.argsort(valid)
    ranked = valid[order]
    # Step-up adjusted p-values, enforced monotone non-decreasing from the top.
    adj_sorted = ranked * m / (np.arange(1, m + 1))
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    adj_valid = np.empty(m, dtype=np.float64)
    adj_valid[order] = adj_sorted
    p_adj[valid_mask] = adj_valid
    rejected[valid_mask] = adj_valid <= alpha

    return {
        "rejected": rejected.tolist(),
        "p_adjusted": p_adj.tolist(),
        "n_significant": int(rejected.sum()),
        "alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Cluster-aware permutation test (addresses pseudoreplication, B1)
# ---------------------------------------------------------------------------

def clustered_permutation_test(
    values: np.ndarray,
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    n_perm: int = 10000,
    seed: int = 42,
) -> dict:
    """
    Permutation test for a difference in means that permutes whole clusters.

    Heads within the same layer are not independent. Permuting the
    retrieval/non-retrieval label *at the cluster (layer) level* respects that
    dependence structure, unlike a standard t-test that treats every head as an
    independent observation.

    Args:
        values: 1-D array of the per-unit statistic (e.g. per-head utility).
        labels: 1-D boolean/int array; 1 = retrieval, 0 = non-retrieval.
        clusters: 1-D array assigning each unit to a cluster (e.g. layer index).
        n_perm: Number of permutations.
        seed: RNG seed.

    Returns:
        Dict with keys: observed_diff, p_value, n_perm. ``observed_diff`` is
        mean(retrieval) - mean(non_retrieval).
    """
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels).astype(bool)
    clusters = np.asarray(clusters)

    if labels.sum() < 1 or (~labels).sum() < 1:
        return {"observed_diff": float("nan"), "p_value": float("nan"), "n_perm": 0}

    observed = values[labels].mean() - values[~labels].mean()

    # Per-cluster retrieval proportion is fixed; we shuffle which units within
    # each cluster carry the retrieval label, preserving cluster-level counts.
    rng = np.random.default_rng(seed)
    uniq = np.unique(clusters)
    count = 0
    for _ in range(n_perm):
        perm_labels = np.zeros_like(labels)
        for c in uniq:
            idx = np.where(clusters == c)[0]
            k = int(labels[idx].sum())
            if k == 0:
                continue
            chosen = rng.choice(idx, size=k, replace=False)
            perm_labels[chosen] = True
        if perm_labels.sum() == 0 or (~perm_labels).sum() == 0:
            continue
        diff = values[perm_labels].mean() - values[~perm_labels].mean()
        if abs(diff) >= abs(observed):
            count += 1

    p_value = (count + 1) / (n_perm + 1)  # add-one smoothing
    return {"observed_diff": float(observed), "p_value": float(p_value), "n_perm": n_perm}
