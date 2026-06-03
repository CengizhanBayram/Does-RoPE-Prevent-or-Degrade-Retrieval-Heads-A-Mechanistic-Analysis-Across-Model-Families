"""Unit tests for src/stats_utils.py (item D1)."""

import math

import numpy as np

from src.stats_utils import (
    benjamini_hochberg,
    bootstrap_mean_diff_ci,
    clustered_permutation_test,
    cohens_d,
)


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
