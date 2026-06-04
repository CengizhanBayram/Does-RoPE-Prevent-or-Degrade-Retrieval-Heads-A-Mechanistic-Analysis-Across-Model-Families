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
# Layer-controlled partial correlation (item P4)
# ---------------------------------------------------------------------------

def _residualize_within_group(x: np.ndarray, group: np.ndarray) -> np.ndarray:
    """Subtract each group's mean (categorical control = within-layer demean)."""
    out = np.array(x, dtype=np.float64)
    for c in np.unique(group):
        m = group == c
        out[m] = out[m] - out[m].mean()
    return out


def partial_correlation(
    x: Sequence[float],
    y: Sequence[float],
    control: Sequence,
    *,
    method: str = "spearman",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Correlation of x and y after controlling for a categorical ``control``
    (here: the layer index) — item P4.

    Layer drives BOTH retrieval tendency and the L1-norm scale, so the raw
    correlation is confounded. We remove each variable's per-layer mean and
    correlate the residuals, with a LAYER-CLUSTERED bootstrap CI (resample whole
    layers, not individual heads — respects non-independence, item B1).

    Returns: partial_r, p_value, ci_low, ci_high, method, n_clusters.
    """
    from scipy import stats as _st

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    g = np.asarray(control)

    def _corr(xr, yr):
        if len(np.unique(xr)) < 2 or len(np.unique(yr)) < 2:
            return float("nan")
        if method == "pearson":
            return float(_st.pearsonr(xr, yr)[0])
        return float(_st.spearmanr(xr, yr)[0])

    rx, ry = _residualize_within_group(x, g), _residualize_within_group(y, g)
    if method == "pearson":
        r, p = (float("nan"), float("nan")) if len(np.unique(rx)) < 2 else _st.pearsonr(rx, ry)
    else:
        r, p = (float("nan"), float("nan")) if len(np.unique(rx)) < 2 else _st.spearmanr(rx, ry)

    # Cluster bootstrap over layers.
    clusters = np.unique(g)
    rng = np.random.default_rng(seed)
    boot = []
    idx_by_cluster = {c: np.where(g == c)[0] for c in clusters}
    for _ in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([idx_by_cluster[c] for c in chosen])
        gb = g[idx]
        rxb = _residualize_within_group(x[idx], gb)
        ryb = _residualize_within_group(y[idx], gb)
        boot.append(_corr(rxb, ryb))
    boot = np.array([b for b in boot if b == b])  # drop NaN
    ci = ([float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))]
          if len(boot) else [float("nan"), float("nan")])

    return {
        "partial_r": float(r), "p_value": float(p),
        "ci_low": ci[0], "ci_high": ci[1],
        "method": method, "n_clusters": int(len(clusters)),
    }


def lead_lag(
    series_a: Sequence[float],
    series_b: Sequence[float],
    *,
    max_lag: int | None = None,
    n_perm: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Cross-correlation lead-lag analysis between two training-time series (item P9).

    Positive ``best_lag`` means series_a LEADS series_b (a's changes precede b's)
    — e.g. "retrieval heads form before utility shifts". Significance via a
    permutation test that shuffles one series.

    Args:
        series_a, series_b: Equal-length sequences (e.g. per-checkpoint
            n_retrieval_heads and mean utility), ideally z-scored beforehand.
        max_lag: Maximum lag to scan (default: len//2).

    Returns:
        best_lag, peak_corr, p_value, lags, corrs.
    """
    a = np.asarray(series_a, dtype=np.float64)
    b = np.asarray(series_b, dtype=np.float64)
    n = len(a)
    if n < 4 or len(b) != n:
        return {"best_lag": 0, "peak_corr": float("nan"), "p_value": float("nan"),
                "lags": [], "corrs": []}
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    max_lag = max_lag or (n // 2)

    def _xcorr(x, y):
        lags = list(range(-max_lag, max_lag + 1))
        cs = []
        for lag in lags:
            # Positive lag ⇒ x LEADS y by `lag`: align x[t] with y[t+lag].
            if lag > 0:
                c = np.corrcoef(x[:-lag], y[lag:])[0, 1]
            elif lag < 0:
                c = np.corrcoef(x[-lag:], y[:lag])[0, 1]
            else:
                c = np.corrcoef(x, y)[0, 1]
            cs.append(0.0 if np.isnan(c) else c)
        return lags, cs

    lags, corrs = _xcorr(a, b)
    peak_i = int(np.argmax(np.abs(corrs)))
    best_lag, peak = lags[peak_i], corrs[peak_i]

    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        bp = rng.permutation(b)
        _, cs = _xcorr(a, bp)
        if max(np.abs(cs)) >= abs(peak):
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"best_lag": int(best_lag), "peak_corr": float(peak),
            "p_value": float(p), "lags": lags, "corrs": corrs}


def crystallization_onset(n_heads_series: Sequence[float]) -> int:
    """
    Index of the crystallization ONSET in a retrieval-head-count training series.

    Returns the first checkpoint whose count crosses the midpoint between the
    early baseline (median of the first ~20%) and the peak. This is robust to a
    later dip→recovery spike that fools a naive argmax(first-difference) — item
    P7. Falls back to argmax(diff)+1 if no crossing is found.
    """
    heads = np.asarray(n_heads_series, dtype=np.float64)
    if len(heads) < 3:
        return 0
    k = max(1, len(heads) // 5)
    baseline = float(np.median(heads[:k]))
    peak = float(heads.max())
    crossings = np.where(heads >= (baseline + peak) / 2.0)[0]
    if len(crossings):
        return int(crossings[0])
    return int(np.argmax(np.diff(heads))) + 1


def layer_zscore(values: np.ndarray) -> np.ndarray:
    """
    Z-score a (n_layers, n_heads) matrix WITHIN each layer (item P3).

    Removes the layer-scale effect that causes vertical banding in Figure 1.
    Layers with zero variance map to zeros.
    """
    v = np.asarray(values, dtype=np.float64)
    mean = v.mean(axis=1, keepdims=True)
    std = v.std(axis=1, ddof=0, keepdims=True)
    out = np.zeros_like(v)
    nz = std[:, 0] > 0
    out[nz] = (v[nz] - mean[nz]) / std[nz]
    return out


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
