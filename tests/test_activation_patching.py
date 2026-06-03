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


def test_frequency_effect_present_and_signed():
    # §6: zeroing high-freq dims hurts more (high_freq acc lower) ⇒ freq_effect > 0.
    patcher = _make_patcher()
    results = {
        (7, 2): {"layer": 7, "head": 2, "baseline": 0.9,
                 "low_utility": 0.85, "random": 0.8, "high_utility": 0.3,
                 "low_freq": 0.88, "high_freq": 0.40},
    }
    df = patcher.compute_causal_effect(results)
    assert "frequency_effect" in df.columns
    # frequency_effect = low_freq - high_freq = 0.88 - 0.40
    assert math.isclose(df.iloc[0]["frequency_effect"], 0.48, rel_tol=1e-9)


def test_frequency_columns_absent_when_not_run():
    # Backward-compatible: no freq keys ⇒ no frequency_effect column.
    patcher = _make_patcher()
    results = {
        (0, 0): {"layer": 0, "head": 0, "baseline": 0.8,
                 "low_utility": 0.7, "random": 0.6, "high_utility": 0.5},
    }
    df = patcher.compute_causal_effect(results)
    assert "frequency_effect" not in df.columns
