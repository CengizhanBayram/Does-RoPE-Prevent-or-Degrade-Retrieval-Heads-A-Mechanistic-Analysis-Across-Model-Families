"""
Diagnostic for the Figure-2 spikes / GQA question — VERIFY before concluding.

Prints, per model, on the REAL weights:
  - num_attention_heads vs num_key_value_heads (MHA if equal, GQA if not),
  - q_proj / k_proj / v_proj weight shapes vs the expected
    (n_heads*head_dim) / (n_kv_heads*head_dim),
  - whether DimensionUtilityAnalyzer's q_proj reshape is correct (it uses q_proj
    ONLY, which is always n_heads*head_dim → GQA-safe),
  - whether the RoPE frequency pairing is rotate_half (dim j and j+head_dim/2
    share a frequency) — the half-boundary spike in Fig 2 is the signature of
    the OLD interleaved convention, now fixed.

Usage (Colab):
    !python scripts/diagnose_gqa.py --model llama3
    !python scripts/diagnose_gqa.py --model all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dimension_utility import DimensionUtilityAnalyzer
from src.model_loader import load_model, purge_hf_cache


def diagnose(model_name: str, model_cfg: dict, config: dict) -> None:
    print("=" * 70)
    print(f"MODEL: {model_name}  ({model_cfg['hf_id']})")
    model, _ = load_model(model_cfg, model_name)
    try:
        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // n_heads)
        kind = "MHA" if n_kv == n_heads else f"GQA (group_size={n_heads // n_kv})"
        print(f"  num_attention_heads = {n_heads}")
        print(f"  num_key_value_heads = {n_kv}   → {kind}")
        print(f"  head_dim            = {head_dim}")

        attn = model.model.layers[0].self_attn
        for name in ("q_proj", "k_proj", "v_proj"):
            proj = getattr(attn, name, None)
            if proj is None:
                print(f"  {name}: (absent)")
                continue
            out_dim = tuple(proj.weight.shape)[0]
            exp = n_heads * head_dim if name == "q_proj" else n_kv * head_dim
            ok = "OK" if out_dim == exp else "!! MISMATCH"
            print(f"  {name}.weight out_dim = {out_dim:5d}  expected {exp:5d}  [{ok}]")

        # DimensionUtilityAnalyzer uses q_proj ONLY → confirm n_q_heads == n_heads.
        n_q_heads = tuple(attn.q_proj.weight.shape)[0] // head_dim
        print(f"  analyzer q_proj reshape: n_q_heads={n_q_heads} "
              f"(== n_heads={n_heads}? {n_q_heads == n_heads}) → utility is GQA-safe")

        # RoPE pairing check (rotate_half ⇒ dim j and j+half share frequency).
        an = DimensionUtilityAnalyzer(model, config)
        fpd = an.freq_per_dim
        half = head_dim // 2
        rotate_half_ok = all(np.isclose(fpd[j], fpd[j + half]) for j in range(half))
        interleaved = np.isclose(fpd[0], fpd[1])
        print(f"  RoPE pairing: rotate_half={rotate_half_ok}  "
              f"interleaved={interleaved}  (want rotate_half=True, interleaved=False)")
    finally:
        del model
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        purge_hf_cache(model_cfg["hf_id"])


def main() -> None:
    p = argparse.ArgumentParser(description="GQA / RoPE-pairing diagnostic")
    p.add_argument("--model", default="all")
    p.add_argument("--config", default="configs/config.yaml")
    args = p.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    models = list(config["models"]) if args.model == "all" else [args.model]
    for m in models:
        try:
            diagnose(m, config["models"][m], config)
        except Exception as exc:
            print(f"  [{m}] diagnosis failed: {exc}")
    print("=" * 70)


if __name__ == "__main__":
    main()
