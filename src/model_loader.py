"""
Centralised model loading with pinned revisions and quantization control.

Replaces the four near-identical ``_load_model`` helpers that were copied across
the run scripts. Adds:
    - ``revision`` pinning so a run is tied to an exact weight commit (item C1).
    - An explicit 8-bit / fp16 switch for the quantization ablation (item A2).
    - A warning when the revision is unpinned ("main"/null).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def load_model(
    model_cfg: dict,
    model_name: str = "model",
    *,
    load_in_8bit: bool | None = None,
) -> tuple[Any, Any]:
    """
    Load a HuggingFace causal LM + tokenizer at a pinned revision.

    Args:
        model_cfg: Per-model config dict (keys: hf_id, revision, load_in_8bit).
        model_name: Friendly name for logging.
        load_in_8bit: Overrides ``model_cfg["load_in_8bit"]`` when not None.
            Pass False to run the fp16 quantization ablation (item A2).

    Returns:
        (model, tokenizer) tuple. ``model`` is in eval mode.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    hf_id = model_cfg["hf_id"]
    revision = model_cfg.get("revision") or "main"
    if revision == "main":
        logger.warning(
            "Model '%s' revision is unpinned ('main'). For a reproducible, "
            "reportable run pin a commit SHA in config.yaml (item C1).",
            model_name,
        )

    want_8bit = model_cfg.get("load_in_8bit", True) if load_in_8bit is None else load_in_8bit

    try:
        import bitsandbytes  # noqa: F401
        has_bnb = True
    except ImportError:
        has_bnb = False
        if want_8bit:
            logger.warning("bitsandbytes not found; loading %s in fp16.", model_name)

    tokenizer = AutoTokenizer.from_pretrained(
        hf_id, revision=revision, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: dict = {
        "revision": revision,
        "device_map": "auto",
        "trust_remote_code": True,
    }
    use_8bit = want_8bit and has_bnb
    if use_8bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16

    logger.info(
        "Loading %s (%s @ %s, 8bit=%s) …", model_name, hf_id, revision, use_8bit
    )
    model = AutoModelForCausalLM.from_pretrained(hf_id, **kwargs)
    model.eval()

    if torch.cuda.is_available():
        logger.info("VRAM after load: %.1f GB", torch.cuda.memory_allocated() / 1e9)
    return model, tokenizer
