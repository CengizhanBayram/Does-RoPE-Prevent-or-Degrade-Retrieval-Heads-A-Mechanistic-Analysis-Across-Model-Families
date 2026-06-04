"""
Regression tests for P2:
  - RoPE frequency pairing uses the rotate_half (NeoX) convention.
  - GQA-aware k_proj indexing in activation patching.
"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("torch")
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from src.activation_patching import ActivationPatcher  # noqa: E402
from src.dimension_utility import DimensionUtilityAnalyzer  # noqa: E402


# --- RoPE pairing (rotate_half) -------------------------------------------

class _CfgModel:
    def __init__(self, head_dim, n_heads, theta):
        self.config = SimpleNamespace(
            num_hidden_layers=2, num_attention_heads=n_heads,
            hidden_size=n_heads * head_dim, head_dim=head_dim, rope_theta=theta,
        )

    def eval(self):
        return self


def test_rope_pairing_is_rotate_half_not_interleaved():
    head_dim, theta = 8, 10000.0
    an = DimensionUtilityAnalyzer(_CfgModel(head_dim, 4, theta), {})
    half = head_dim // 2
    fpd = an.freq_per_dim
    # rotate_half: dim j and dim j+half share the SAME frequency.
    for j in range(half):
        assert np.isclose(fpd[j], fpd[j + half]), f"dim {j} != dim {j+half}"
    # interleaved (the OLD bug) would instead pair 2j and 2j+1:
    assert not np.isclose(fpd[0], fpd[1]), "looks interleaved (GPT-J) — wrong"
    # lowest frequency is the last index of each half (largest j)
    assert an.freq_values_sorted[0] <= an.freq_values_sorted[-1]


# --- GQA k_proj indexing ---------------------------------------------------

class _Attn(nn.Module):
    def __init__(self, n_heads, n_kv, head_dim, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, n_kv * head_dim, bias=False)


class _Layer(nn.Module):
    def __init__(self, *a):
        super().__init__()
        self.self_attn = _Attn(*a)


class _GQAModel:
    def __init__(self, n_heads, n_kv, head_dim, hidden):
        self.config = SimpleNamespace(
            num_attention_heads=n_heads, num_key_value_heads=n_kv,
            head_dim=head_dim, hidden_size=hidden,
        )
        self.model = SimpleNamespace(layers=[_Layer(n_heads, n_kv, head_dim, hidden)])

    def eval(self):
        return self


def test_gqa_kproj_maps_query_head_to_kv_head():
    # 4 query heads, 2 KV heads, head_dim 4 → group_size 2.
    n_heads, n_kv, head_dim, hidden = 4, 2, 4, 8
    model = _GQAModel(n_heads, n_kv, head_dim, hidden)
    patcher = ActivationPatcher(model, tokenizer=None, config={})
    assert patcher._get_n_heads_kv() == (4, 2)

    layer = model.model.layers[0]
    x = torch.randn(1, 3, hidden)
    base_k = layer.self_attn.k_proj(x).detach().clone()
    base_q = layer.self_attn.q_proj(x).detach().clone()

    # Patch query head 3 → KV head 3//2 = 1 → k cols [1*4+0, 1*4+1] = [4, 5].
    with patcher.patch_head(0, head_idx=3, dims_to_zero=[0, 1], mode="qk"):
        pk = layer.self_attn.k_proj(x)
        pq = layer.self_attn.q_proj(x)

    assert torch.allclose(pk[..., 4], torch.zeros_like(pk[..., 4]))
    assert torch.allclose(pk[..., 5], torch.zeros_like(pk[..., 5]))
    for c in [0, 1, 2, 3, 6, 7]:                      # other KV cols untouched
        assert torch.allclose(pk[..., c], base_k[..., c])
    # q head 3 → cols 12,13 zeroed (q_proj has full 16 cols)
    assert torch.allclose(pq[..., 12], torch.zeros_like(pq[..., 12]))
    assert torch.allclose(pq[..., 13], torch.zeros_like(pq[..., 13]))


def test_gqa_kproj_never_indexes_out_of_bounds():
    # The OLD bug used head_idx*head_dim into k_proj → col 12 >= width 8 → crash.
    # The fix maps to KV head AND guards col < width, so the highest query head
    # must not raise and must stay in bounds.
    model = _GQAModel(4, 2, 4, 8)
    patcher = ActivationPatcher(model, tokenizer=None, config={})
    layer = model.model.layers[0]
    x = torch.randn(1, 2, 8)
    with patcher.patch_head(0, head_idx=3, dims_to_zero=[0, 1, 2, 3], mode="k"):
        out = layer.self_attn.k_proj(x)        # must not raise
    assert out.shape[-1] == 8
