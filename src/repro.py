"""
Reproducibility utilities: determinism control and environment capture.

Two responsibilities:
    1. ``set_determinism`` — seed every RNG and (optionally) force deterministic
       CUDA kernels so a run can be repeated bit-for-bit where the hardware
       allows it.
    2. ``capture_environment`` — record the exact software/hardware/git state
       that produced a result, so it can be reported in the paper and audited
       later. This is written into every results JSON.

Reference: items C2 (environment capture) and C4 (determinism) of the
publication-readiness checklist.
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def set_determinism(seed: int = 42, *, strict: bool = False) -> None:
    """
    Seed all RNGs and optionally enforce deterministic algorithms.

    Args:
        seed: Seed applied to ``random``, ``numpy``, and ``torch``.
        strict: If True, also set cuDNN to deterministic mode and call
            ``torch.use_deterministic_algorithms(True)``. This can slow
            kernels and may raise if an op has no deterministic implementation;
            keep it off for throughput, on for the final reportable run.
    """
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if strict:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # CUBLAS workspace config is required for deterministic matmul.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception as exc:  # pragma: no cover
                logger.warning("Could not enable deterministic algorithms: %s", exc)
    except ImportError:
        logger.debug("torch not available; seeded random + numpy only.")

    logger.info("Determinism set (seed=%d, strict=%s).", seed, strict)


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def _git_sha(short: bool = False) -> str | None:
    """Return the current git commit SHA of the repo, or None if unavailable."""
    cmd = ["git", "-C", str(_REPO_ROOT), "rev-parse"]
    cmd += ["--short", "HEAD"] if short else ["HEAD"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:  # pragma: no cover - git missing / not a repo
        pass
    return None


def _git_dirty() -> bool | None:
    """Return True if the working tree has uncommitted changes, None if unknown."""
    try:
        out = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:  # pragma: no cover
        pass
    return None


def _package_versions() -> dict[str, str]:
    """Collect versions of the packages that materially affect results."""
    versions: dict[str, str] = {}
    for pkg in ("torch", "transformers", "numpy", "scipy", "bitsandbytes",
                "accelerate", "datasets", "huggingface_hub"):
        try:
            module = __import__(pkg)
            versions[pkg] = getattr(module, "__version__", "unknown")
        except Exception:
            versions[pkg] = "not-installed"
    return versions


def _hardware_info() -> dict[str, Any]:
    """Return GPU / CUDA info (CPU fallback)."""
    info: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["vram_total_gb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1e9, 1
            )
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["n_gpus"] = torch.cuda.device_count()
        else:
            info.update({"gpu": "CPU", "vram_total_gb": 0, "cuda_version": "N/A"})
    except ImportError:
        info.update({"gpu": "unknown", "vram_total_gb": 0, "cuda_version": "N/A"})
    return info


def capture_environment(extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Capture a complete reproducibility fingerprint of the current run.

    Returns a dict suitable for embedding under a ``"environment"`` key in any
    results JSON. Includes Python/OS, package versions, hardware, and the git
    commit (with a dirty-tree flag).

    Args:
        extra: Optional additional fields to merge in (e.g. CLI args).

    Returns:
        Dict with keys: python, platform, packages, hardware, git_commit,
        git_dirty, plus anything from ``extra``.
    """
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": _package_versions(),
        "hardware": _hardware_info(),
        "git_commit": _git_sha(),
        "git_dirty": _git_dirty(),
    }
    if env["git_dirty"]:
        logger.warning(
            "Working tree is dirty — results are not tied to a clean commit. "
            "Commit before the reportable run for full reproducibility."
        )
    if extra:
        env.update(extra)
    return env
