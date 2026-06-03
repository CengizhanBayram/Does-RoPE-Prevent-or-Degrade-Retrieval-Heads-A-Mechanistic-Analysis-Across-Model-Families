"""Unit tests for src/corpus.py (item D1)."""

import random

from src.corpus import FALLBACK_SENTENCES, build_haystack


def test_fallback_corpus_nonempty_and_unique_enough():
    assert len(FALLBACK_SENTENCES) >= 40
    # The distinctive needle template must not collide with the corpus.
    assert all("secret passphrase" not in s for s in FALLBACK_SENTENCES)


def test_build_haystack_is_deterministic_per_seed():
    sents = FALLBACK_SENTENCES * 10
    h1 = build_haystack(512, sents, random.Random(42))
    h2 = build_haystack(512, sents, random.Random(42))
    assert h1 == h2


def test_build_haystack_different_seeds_differ():
    sents = FALLBACK_SENTENCES * 10
    h1 = build_haystack(512, sents, random.Random(1))
    h2 = build_haystack(512, sents, random.Random(2))
    assert h1 != h2


def test_build_haystack_respects_char_budget():
    sents = FALLBACK_SENTENCES * 50
    target_tokens = 256
    h = build_haystack(target_tokens, sents, random.Random(0), chars_per_token=4)
    # Stops once the budget is reached; should be in the right ballpark.
    assert len(h) >= target_tokens * 4 - 200
