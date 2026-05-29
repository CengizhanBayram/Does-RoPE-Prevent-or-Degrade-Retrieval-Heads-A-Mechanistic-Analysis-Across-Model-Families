"""
OLMo-2 checkpoint management for training-dynamics analysis (Layer B).

OLMo-2 checkpoints are available on HuggingFace under the revision format
"step{N}-tokens{M}B". This module lists, loads, and clears those checkpoints
with resume-safe progress tracking.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_OLMO_REPO = "allenai/OLMo-2-1124-7B"


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except ImportError:
        return False


class OLMoCheckpointLoader:
    """
    Manages loading and unloading of OLMo-2 training checkpoints from HuggingFace.

    Checkpoints are loaded one at a time with 8-bit quantization to fit on an
    L4 GPU (24 GB). After each analysis step the model is explicitly deleted and
    CUDA memory is cleared before the next checkpoint is loaded.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Global config dict (olmo_checkpoints section used).
        """
        self.config = config
        ckpt_cfg = config.get("olmo_checkpoints", {})
        self.step_stride: int = ckpt_cfg.get("step_stride", 10000)
        self.cache_dir: str = ckpt_cfg.get("cache_dir", "./cache/olmo_checkpoints")
        self.results_dir: str = config.get("output", {}).get("results_dir", "./results")

        self._available_steps: list[int] | None = None

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_available_checkpoints(self) -> list[str]:
        """
        List all checkpoint revisions available for OLMo-2 on HuggingFace.

        Returns:
            List of revision strings matching the "step{N}-tokens{M}B" pattern.
        """
        try:
            from huggingface_hub import list_repo_refs  # type: ignore

            logger.info("Fetching available revisions for %s …", _OLMO_REPO)
            refs = list_repo_refs(_OLMO_REPO)
            pattern = re.compile(r"^step\d+-tokens\d+B$")
            branches = [
                b.name
                for b in refs.branches
                if pattern.match(b.name)
            ]
            branches.sort(key=lambda x: int(re.search(r"step(\d+)", x).group(1)))
            logger.info("Found %d checkpoint revisions.", len(branches))
            return branches
        except Exception as exc:
            logger.error("Could not fetch repo refs: %s", exc)
            return []

    def get_step_from_revision(self, revision: str) -> int | None:
        """Extract step number from a revision string like 'step12345-tokens50B'."""
        m = re.search(r"step(\d+)", revision)
        return int(m.group(1)) if m else None

    def get_filtered_checkpoints(self, step_stride: int | None = None) -> list[str]:
        """
        Return only checkpoints whose step is divisible by step_stride.

        Args:
            step_stride: Overrides self.step_stride if provided.

        Returns:
            Filtered list of revision strings.
        """
        stride = step_stride if step_stride is not None else self.step_stride
        all_revisions = self.list_available_checkpoints()
        filtered = [
            rev
            for rev in all_revisions
            if (step := self.get_step_from_revision(rev)) is not None
            and step % stride == 0
        ]
        logger.info(
            "Filtered to %d checkpoints (stride=%d).", len(filtered), stride
        )
        return filtered

    # ------------------------------------------------------------------
    # Loading / unloading
    # ------------------------------------------------------------------

    def load_checkpoint(
        self, revision: str, load_in_8bit: bool = True
    ) -> tuple[Any, Any]:
        """
        Load an OLMo-2 checkpoint by revision string.

        Falls back to fp16 if bitsandbytes is unavailable.

        Args:
            revision: HuggingFace revision string (e.g. "step10000-tokens50B").
            load_in_8bit: Whether to use 8-bit quantization.

        Returns:
            (model, tokenizer) tuple.
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        use_8bit = load_in_8bit and _has_bitsandbytes()
        if load_in_8bit and not use_8bit:
            logger.warning(
                "bitsandbytes not available; falling back to fp16 for revision %s.",
                revision,
            )

        logger.info("Loading OLMo-2 revision '%s' (8bit=%s) …", revision, use_8bit)

        quantization_config = None
        if use_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        kwargs: dict = {
            "revision": revision,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if use_8bit:
            kwargs["quantization_config"] = quantization_config
        else:
            kwargs["torch_dtype"] = torch.float16

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            kwargs["cache_dir"] = self.cache_dir

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                _OLMO_REPO,
                revision=revision,
                trust_remote_code=True,
                cache_dir=self.cache_dir if self.cache_dir else None,
            )
            model = AutoModelForCausalLM.from_pretrained(_OLMO_REPO, **kwargs)
            model.eval()
            logger.info(
                "Loaded revision '%s'. VRAM: %.1f GB",
                revision,
                torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            )
            return model, tokenizer
        except Exception as exc:
            logger.error("Failed to load revision '%s': %s", revision, exc)
            raise

    def clear_checkpoint(self, model: Any) -> None:
        """
        Delete a model from memory and release CUDA cache.

        Args:
            model: The model to delete.
        """
        import torch

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Checkpoint cleared from memory.")

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_checkpoint_metadata(self) -> pd.DataFrame:
        """
        Return a DataFrame of available checkpoints with step and token counts.

        Returns:
            DataFrame with columns: revision, step, tokens_billions.
        """
        revisions = self.list_available_checkpoints()
        records = []
        for rev in revisions:
            step_match = re.search(r"step(\d+)", rev)
            tokens_match = re.search(r"tokens(\d+)B", rev)
            records.append(
                {
                    "revision": rev,
                    "step": int(step_match.group(1)) if step_match else None,
                    "tokens_billions": int(tokens_match.group(1)) if tokens_match else None,
                }
            )
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def is_computed(self, revision: str, results_dir: str | None = None) -> bool:
        """
        Check whether results for a checkpoint revision already exist on disk.

        Args:
            revision: Revision string (e.g. "step10000-tokens50B").
            results_dir: Base results directory; defaults to self.results_dir.

        Returns:
            True if the results file exists and is non-empty.
        """
        base = results_dir or self.results_dir
        step = self.get_step_from_revision(revision)
        if step is None:
            return False
        result_path = Path(base) / "layer_b" / f"checkpoint_{step}" / "results.json"
        if not result_path.exists():
            return False
        try:
            with open(result_path) as f:
                data = json.load(f)
            return bool(data)
        except (json.JSONDecodeError, OSError):
            return False
