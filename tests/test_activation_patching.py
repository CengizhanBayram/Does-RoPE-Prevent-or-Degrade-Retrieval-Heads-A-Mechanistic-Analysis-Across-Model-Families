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


def _fake_eval_factory(counter):
    def fake_eval(samples, layer_idx=None, head_idx=None, dims_to_zero=None, mode="qk"):
        counter["n"] += 1
        return 1.0 if layer_idx is None else 0.5
    return fake_eval


def test_checkpoint_resume_skips_done_heads(tmp_path, monkeypatch):
    """A resumable run must (a) reproduce the full result, (b) recompute nothing
    on a full resume, and (c) recompute exactly the missing head on a partial
    resume — i.e. identical science, only persistence added."""
    import json
    import numpy as np

    heads = [(0, 0), (0, 1), (1, 2)]
    norms = np.random.RandomState(0).rand(2, 3, 8)   # (layers, heads, head_dim)
    us = {"_norms": norms}
    samples = [{"prompt_ids": [1, 2, 3], "code": "x"}] * 3
    freq = np.arange(8)
    ckpt = str(tmp_path / "ck.json")
    kwargs = dict(k_dims=2, n_samples=3, random_seeds=[0, 1],
                  freq_order=freq, checkpoint_path=ckpt)
    # per head: low + high + 2 random + low_freq + high_freq = 6 eval calls
    per_head = 6

    counter = {"n": 0}
    p1 = _make_patcher()
    monkeypatch.setattr(p1, "_evaluate_accuracy", _fake_eval_factory(counter))
    r1 = p1.run_patching_experiment(heads, us, samples, **kwargs)
    assert set(r1.keys()) == set(heads)
    assert counter["n"] == 1 + per_head * len(heads)   # 1 baseline + heads

    # (b) Full resume: baseline + every head loaded → zero new evaluations.
    counter["n"] = 0
    p2 = _make_patcher()
    monkeypatch.setattr(p2, "_evaluate_accuracy", _fake_eval_factory(counter))
    r2 = p2.run_patching_experiment(heads, us, samples, **kwargs)
    assert counter["n"] == 0
    for h in heads:
        assert r2[h] == r1[h]                          # identical values

    # (c) Partial resume: drop one head from the checkpoint file.
    with open(ckpt, encoding="utf-8") as f:
        data = json.load(f)
    data["heads"].pop("layer1_head2")
    with open(ckpt, "w", encoding="utf-8") as f:
        json.dump(data, f)
    counter["n"] = 0
    p3 = _make_patcher()
    monkeypatch.setattr(p3, "_evaluate_accuracy", _fake_eval_factory(counter))
    r3 = p3.run_patching_experiment(heads, us, samples, **kwargs)
    assert set(r3.keys()) == set(heads)
    assert counter["n"] == per_head                    # only the missing head


# ---- population-level patching (ceiling-breaking) --------------------------

import types  # noqa: E402


def _fake_model_for_hooks(n_layers=2, n_heads=4, n_kv=2, head_dim=4):
    """Fake HF-like model that records the forward hook registered on each
    q_proj / k_proj so the hook function can be invoked directly in a test."""
    torch = pytest.importorskip("torch")

    class _Rec:
        def __init__(self):
            self.hook = None

        def register_forward_hook(self, fn):
            self.hook = fn
            return types.SimpleNamespace(remove=lambda: None)

    class _Layer:
        def __init__(self):
            self.self_attn = types.SimpleNamespace(q_proj=_Rec(), k_proj=_Rec())

    class _Model:
        def __init__(self):
            self.model = types.SimpleNamespace(layers=[_Layer() for _ in range(n_layers)])
            self.config = types.SimpleNamespace(
                num_attention_heads=n_heads, num_key_value_heads=n_kv,
                head_dim=head_dim, hidden_size=n_heads * head_dim)

        def eval(self):
            return self

    return _Model()


def test_patch_heads_zeroes_correct_gqa_columns():
    torch = pytest.importorskip("torch")
    head_dim, n_heads, n_kv = 4, 4, 2          # GQA group_size = 2
    model = _fake_model_for_hooks(2, n_heads, n_kv, head_dim)
    patcher = ActivationPatcher(model, tokenizer=None, config={})

    # layer 0: head 0 zero dim 1; head 3 zero dim 2
    head_dims = {(0, 0): [1], (0, 3): [2]}
    with patcher.patch_heads(head_dims, mode="qk"):
        qhook = model.model.layers[0].self_attn.q_proj.hook
        khook = model.model.layers[0].self_attn.k_proj.hook
        # q_proj width = n_heads*head_dim = 16; cols: head0 dim1 -> 1, head3 dim2 -> 14
        q_out = qhook(None, None, torch.ones(1, 2, n_heads * head_dim))
        # k_proj width = n_kv*head_dim = 8; head0 -> kv0 col 1; head3 -> kv1 col 4+2=6
        k_out = khook(None, None, torch.ones(1, 2, n_kv * head_dim))

    assert q_out[0, 0, 1] == 0 and q_out[0, 0, 14] == 0
    assert q_out[0, 0, 0] == 1 and q_out[0, 0, 2] == 1     # untouched
    assert k_out[0, 0, 1] == 0 and k_out[0, 0, 6] == 0
    assert k_out[0, 0, 0] == 1 and k_out[0, 0, 7] == 1     # untouched


def test_run_population_patching_is_per_condition_not_per_head():
    import numpy as np
    patcher = _make_patcher()
    heads = [(0, 0), (0, 1), (1, 2)]
    norms = np.random.RandomState(0).rand(2, 3, 8)
    us = {"_norms": norms}
    samples = [{"prompt_ids": [1, 2, 3], "code": "x"}] * 4

    calls = {"n": 0}

    def fake_pop(samples, head_dims=None, mode="qk"):
        calls["n"] += 1
        return 1.0 if head_dims is None else 0.7

    # monkeypatch via setattr (no pytest fixture needed here)
    patcher._evaluate_population = fake_pop
    out = patcher.run_population_patching(
        heads, us, samples, k_dims=2, n_samples=4,
        random_seeds=[0, 1, 2], freq_order=np.arange(8))

    # evals = baseline + low + high + 3 random + low_freq + high_freq = 8 (NOT x heads)
    assert calls["n"] == 5 + 3
    assert out["n_heads"] == 3
    assert out["baseline"] == 1.0
    assert "frequency_effect" in out and "causal_effect" in out
