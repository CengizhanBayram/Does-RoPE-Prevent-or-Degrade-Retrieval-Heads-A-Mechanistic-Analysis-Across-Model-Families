"""Pytest configuration: make the repo root importable as ``src.*``."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
