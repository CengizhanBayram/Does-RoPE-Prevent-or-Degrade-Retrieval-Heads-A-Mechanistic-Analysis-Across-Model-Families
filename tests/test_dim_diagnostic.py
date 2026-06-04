"""Unit tests for the Figure-2 spike diagnostic (diagnose_dimension_norms)."""

import numpy as np

from src.dimension_utility import diagnose_dimension_norms

HEAD_DIM = 8  # half = 4


def _norms_with_dip(dip_dim, dip_value=0.02, base=1.0):
    # (n_layers, n_heads, head_dim) all ~base, one dimension driven near zero.
    n = np.full((4, 6, HEAD_DIM), base, dtype=np.float64)
    n[:, :, dip_dim] = dip_value
    return n


def test_genuine_low_norm_off_boundary():
    res = diagnose_dimension_norms(_norms_with_dip(2), HEAD_DIM, n_heads=6, n_kv_heads=6)
    assert res["min_dim"] == 2
    assert res["min_ratio_to_median"] < 0.2
    assert res["min_at_rotate_half_boundary"] is False
    assert res["verdict"].startswith("GENUINE")


def test_boundary_dip_flagged_as_artifact():
    # dip exactly at head_dim/2 (the rotate_half split) → suspect artifact.
    res = diagnose_dimension_norms(_norms_with_dip(HEAD_DIM // 2), HEAD_DIM)
    assert res["min_at_rotate_half_boundary"] is True
    assert "ARTIFACT" in res["verdict"]


def test_uniform_norms_no_dip():
    n = np.full((4, 6, HEAD_DIM), 1.0)
    res = diagnose_dimension_norms(n, HEAD_DIM)
    assert res["min_ratio_to_median"] > 0.9
    assert "no strong dip" in res["verdict"]
