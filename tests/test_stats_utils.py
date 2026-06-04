"""Unit tests for src/stats_utils.py (item D1)."""

import math

import numpy as np

from src.stats_utils import (
    benjamini_hochberg,
    bootstrap_mean_diff_ci,
    clustered_permutation_test,
    cohens_d,
    crystallization_onset,
    jaccard,
    layer_zscore,
    lead_lag,
    partial_correlation,
)


def test_crystallization_onset_ignores_later_dip_recovery():
    # Low baseline ~10, sharp rise to ~45 at index 5, then a dip→recovery spike
    # near the end (the argmax(diff) trap). Onset must be the rise (~index 5),
    # NOT the late recovery.
    series = [10, 10, 11, 10, 12, 45, 44, 46, 43, 11, 48]
    assert crystallization_onset(series) == 5


def test_crystallization_onset_short_series():
    assert crystallization_onset([1, 2]) == 0


def test_partial_correlation_removes_layer_confound():
    # Construct a pure layer confound: within each layer x and y are unrelated,
    # but layer means rise together → raw corr is high, partial corr ~0.
    rng = np.random.default_rng(0)
    x, y, layer = [], [], []
    for L in range(8):
        base = L * 10.0  # shared layer effect drives both up
        for _ in range(12):
            x.append(base + rng.normal(0, 1))
            y.append(base + rng.normal(0, 1))
            layer.append(L)
    raw = abs(np.corrcoef(x, y)[0, 1])
    res = partial_correlation(x, y, layer, method="pearson", n_boot=500)
    assert raw > 0.8                       # raw correlation inflated by layer
    assert abs(res["partial_r"]) < 0.3     # partial removes the confound
    assert res["n_clusters"] == 8


def test_layer_zscore_zero_mean_unit_var_per_layer():
    v = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    z = layer_zscore(v)
    assert np.allclose(z.mean(axis=1), 0.0, atol=1e-9)
    assert np.allclose(z.std(axis=1), 1.0, atol=1e-9)


def test_lead_lag_detects_shift():
    # b is a lagged copy of a (b[t] = a[t-2]) → a leads b by +2.
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, size=30)
    b = np.concatenate([[0, 0], a[:-2]])
    res = lead_lag(a, b, n_perm=1000)
    assert res["best_lag"] == 2
    assert res["p_value"] < 0.05


def test_jaccard_identical_sets():
    heads = [(0, 1), (2, 3), (5, 7)]
    r = jaccard(heads, heads)
    assert r["jaccard"] == 1.0
    assert r["only_a"] == [] and r["only_b"] == []


def test_jaccard_partial_overlap():
    a = [(0, 0), (1, 1), (2, 2)]
    b = [(1, 1), (2, 2), (3, 3)]
    r = jaccard(a, b)
    assert r["intersection"] == 2 and r["union"] == 4
    assert abs(r["jaccard"] - 0.5) < 1e-9
    assert r["only_a"] == [(0, 0)] and r["only_b"] == [(3, 3)]


def test_jaccard_empty_sets():
    assert jaccard([], [])["jaccard"] == 1.0


def test_cohens_d_sign_and_zero():
    a = [10, 11, 12, 13, 14]
    b = [0, 1, 2, 3, 4]
    d = cohens_d(a, b)
    assert d > 0  # a clearly larger than b
    # symmetric magnitude
    assert math.isclose(cohens_d(b, a), -d, rel_tol=1e-9)


def test_cohens_d_insufficient_data_is_nan():
    assert math.isnan(cohens_d([1.0], [2.0, 3.0]))


def test_cohens_d_zero_variance_is_nan():
    assert math.isnan(cohens_d([5, 5, 5], [5, 5, 5]))


def test_benjamini_hochberg_basic():
    # One tiny p plus large ones: only the tiny should survive.
    res = benjamini_hochberg([0.001, 0.4, 0.6, 0.8], alpha=0.05)
    assert res["rejected"][0] is True
    assert res["n_significant"] == 1
    # adjusted p-values are monotone-bounded in [0, 1]
    adj = [p for p in res["p_adjusted"] if not math.isnan(p)]
    assert all(0.0 <= p <= 1.0 for p in adj)


def test_benjamini_hochberg_handles_nan():
    res = benjamini_hochberg([float("nan"), 0.001])
    assert res["rejected"][0] is False
    assert res["rejected"][1] is True


def test_benjamini_hochberg_all_nan():
    res = benjamini_hochberg([float("nan"), float("nan")])
    assert res["n_significant"] == 0


def test_bootstrap_ci_contains_true_diff():
    rng = np.random.default_rng(0)
    a = rng.normal(5.0, 1.0, size=200)
    b = rng.normal(3.0, 1.0, size=200)
    ci = bootstrap_mean_diff_ci(a, b, n_boot=2000, seed=1)
    assert ci["ci_low"] < ci["mean_diff"] < ci["ci_high"]
    assert ci["ci_low"] > 0  # difference is clearly positive


def test_bootstrap_ci_is_deterministic():
    a, b = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]
    assert bootstrap_mean_diff_ci(a, b, seed=42) == bootstrap_mean_diff_ci(a, b, seed=42)


def test_clustered_permutation_detects_effect():
    # 4 clusters; retrieval heads have systematically higher values.
    rng = np.random.default_rng(0)
    values, labels, clusters = [], [], []
    for c in range(4):
        for _ in range(8):
            values.append(rng.normal(0.0, 0.1)); labels.append(0); clusters.append(c)
        for _ in range(2):
            values.append(rng.normal(1.0, 0.1)); labels.append(1); clusters.append(c)
    res = clustered_permutation_test(
        np.array(values), np.array(labels), np.array(clusters), n_perm=2000
    )
    assert res["observed_diff"] > 0
    assert res["p_value"] < 0.05


def test_clustered_permutation_no_effect_high_p():
    rng = np.random.default_rng(1)
    values = rng.normal(0, 1, size=100)
    labels = np.array([0, 1] * 50)
    clusters = np.repeat(np.arange(10), 10)
    res = clustered_permutation_test(values, labels, clusters, n_perm=2000)
    assert res["p_value"] > 0.05
