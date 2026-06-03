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
import os
import shutil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _hf_hub_cache_dir() -> Path:
    """Locate the HuggingFace hub cache directory across versions/platforms."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE  # type: ignore
        return Path(HF_HUB_CACHE)
    except Exception:
        env = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME")
        if env:
            p = Path(env)
            return p / "hub" if p.name != "hub" else p
        return Path.home() / ".cache" / "huggingface" / "hub"


def purge_hf_cache(hf_id: str) -> float:
    """
    Delete a model's downloaded weights from the HF hub cache to free disk.

    On Colab the local disk fills after a few 7B downloads, crashing later cells
    with ``OSError: [Errno 28] No space left on device``. Call this after a model
    is done (it will be re-downloaded automatically if needed again).

    Args:
        hf_id: Repo id, e.g. "meta-llama/Meta-Llama-3.1-8B".

    Returns:
        Approximate GB freed (0.0 if nothing was cached).
    """
    repo_folder = "models--" + hf_id.replace("/", "--")
    path = _hf_hub_cache_dir() / repo_folder
    if not path.exists():
        logger.debug("No HF cache to purge at %s", path)
        return 0.0
    freed = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    shutil.rmtree(path, ignore_errors=True)
    gb = freed / 1e9
    logger.info("Purged HF cache for %s (~%.1f GB freed).", hf_id, gb)
    return gb


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
