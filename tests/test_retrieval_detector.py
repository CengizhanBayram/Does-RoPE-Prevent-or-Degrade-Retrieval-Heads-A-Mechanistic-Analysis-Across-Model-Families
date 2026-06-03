"""Unit tests for needle-localisation logic in RetrievalHeadDetector (item D1)."""

import pytest

pytest.importorskip("torch")

from src.retrieval_head_detector import RetrievalHeadDetector  # noqa: E402


class _DummyModel:
    """Minimal stand-in so the detector can be constructed without weights."""

    def eval(self):
        return self


def _make_detector():
    return RetrievalHeadDetector(_DummyModel(), tokenizer=None, config={}, seed=7)


def test_find_needle_token_range_found():
    det = _make_detector()
    prompt = [10, 11, 12, 13, 14, 15]
    needle = [12, 13]
    assert det._find_needle_token_range(prompt, needle) == (2, 4)


def test_find_needle_token_range_at_start_and_end():
    det = _make_detector()
    assert det._find_needle_token_range([1, 2, 3], [1]) == (0, 1)
    assert det._find_needle_token_range([1, 2, 3], [3]) == (2, 3)


def test_find_needle_token_range_not_found():
    det = _make_detector()
    assert det._find_needle_token_range([1, 2, 3], [9, 9]) is None


def test_generate_code_is_seeded_and_well_formed():
    det = _make_detector()
    det._rng.seed(7)
    code = det._generate_code()
    assert len(code) == 5 and code.isalnum() and code.isupper() or any(c.isdigit() for c in code)
    # Reseed → identical sequence (local RNG determinism, item C5).
    det._rng.seed(7)
    assert det._generate_code() == code
