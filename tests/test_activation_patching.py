"""Unit tests for causal-effect computation in ActivationPatcher (item D1)."""

import math

import pytest

pytest.importorskip("torch")

from src.activation_patching import ActivationPatcher  # noqa: E402


class _DummyModel:
    def eval(self):
        return self


def _make_patcher():
    return ActivationPatcher(_DummyModel(), tokenizer=None, config={})


def test_causal_effect_is_random_minus_low():
    patcher = _make_patcher()
    results = {
        (5, 3): {"layer": 5, "head": 3, "baseline": 0.8,
                 "low_utility": 0.7, "random": 0.6, "high_utility": 0.2},
    }
    df = patcher.compute_causal_effect(results)
    row = df.iloc[0]
    assert math.isclose(row["causal_effect"], 0.6 - 0.7, rel_tol=1e-9)
    # relative drops use the real baseline (not the epsilon hack)
    assert math.isclose(row["relative_drop_low"], (0.8 - 0.7) / 0.8, rel_tol=1e-9)


def test_zero_baseline_gives_nan_relative_drop():
    patcher = _make_patcher()
    results = {
        (0, 0): {"layer": 0, "head": 0, "baseline": 0.0,
                 "low_utility": 0.0, "random": 0.0, "high_utility": 0.0},
    }
    df = patcher.compute_causal_effect(results)
    assert math.isnan(df.iloc[0]["relative_drop_low"])
    assert math.isnan(df.iloc[0]["relative_drop_high"])
    # causal_effect itself is still defined (0 here)
    assert df.iloc[0]["causal_effect"] == 0.0
