"""
Regression tests for paired-seed control (§2.3 / item C5).

Guards the property the whole cross-model comparison rests on: the SAME seed
must yield byte-identical NIAH data regardless of which tokenizers are used,
and the intersection-drop must score every model on an equal number of samples.
"""

import zlib

import pytest

pytest.importorskip("torch")  # retrieval_head_detector imports torch at module load

from src.retrieval_head_detector import (  # noqa: E402
    build_samples_from_specs,
    generate_niah_specs,
    paired_spec_subset,
    valid_spec_ids,
)

CL = [128, 256]
POS = [0.1, 0.5, 0.9]
N = 12


# --- fake tokenizers (no torch / no network) -------------------------------

class _Row(list):
    def tolist(self):
        return list(self)


class FakeTokenizer:
    """Whitespace tokenizer with a salted id map; optional word truncation."""

    bos_token_id = None

    def __init__(self, salt: int = 0, max_words: int | None = None):
        self.salt = salt
        self.max_words = max_words

    def _encode(self, text: str) -> list[int]:
        words = text.split()
        if self.max_words is not None:
            words = words[: self.max_words]
        return [zlib.crc32(w.encode()) ^ self.salt for w in words]

    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, add_special_tokens=True):
        ids = self._encode(text)
        if return_tensors == "pt":
            return {"input_ids": [_Row(ids)]}
        return {"input_ids": ids}


# --- tests -----------------------------------------------------------------

def test_specs_deterministic_per_seed():
    a = generate_niah_specs(1, N, CL, POS)
    b = generate_niah_specs(1, N, CL, POS)
    assert a == b                                   # byte-identical
    assert [s["prompt"] for s in a] == [s["prompt"] for s in b]


def test_specs_differ_across_seeds():
    a = generate_niah_specs(1, N, CL, POS)
    b = generate_niah_specs(2, N, CL, POS)
    assert [s["prompt"] for s in a] != [s["prompt"] for s in b]


def test_specs_are_tokenizer_independent():
    """The same specs feed every model — prompts never depend on a tokenizer."""
    specs = generate_niah_specs(7, N, CL, POS)
    tok_a = FakeTokenizer(salt=0)
    tok_b = FakeTokenizer(salt=12345)
    samples_a, _ = build_samples_from_specs(specs, tok_a)
    samples_b, _ = build_samples_from_specs(specs, tok_b)
    # Different tokenizers → possibly different token ids, but identical prompts
    # in identical order (the data is the same).
    assert [s["prompt"] for s in samples_a] == [s["prompt"] for s in samples_b]
    assert [s["spec_id"] for s in samples_a] == [s["spec_id"] for s in samples_b]


def test_intersection_drop_equalises_n():
    """A tokenizer that truncates drops high-position needles; the paired subset
    must then score BOTH tokenizers on the identical spec set (equal n)."""
    specs = generate_niah_specs(3, N, CL, POS)
    full = FakeTokenizer(salt=0)
    truncating = FakeTokenizer(salt=1, max_words=5)  # cuts most needles out

    subset, dropped = paired_spec_subset(specs, {"full": full, "trunc": truncating})

    # Something was dropped by the truncating tokenizer.
    assert len(dropped) > 0
    assert len(subset) + len(dropped) == len(specs)

    # On the paired subset, BOTH tokenizers keep exactly the same spec_ids.
    ids_full = valid_spec_ids(subset, full)
    ids_trunc = valid_spec_ids(subset, truncating)
    assert ids_full == ids_trunc
    assert len(ids_full) == len(subset)  # every surviving spec valid in both
