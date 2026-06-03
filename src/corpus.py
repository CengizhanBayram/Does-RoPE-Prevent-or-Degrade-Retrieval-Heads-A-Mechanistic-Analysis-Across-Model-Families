"""
Shared haystack corpus for NIAH experiments.

Previously the corpus (PG-19 loader + hard-coded fallback) was duplicated in
``retrieval_head_detector.py`` and ``niah_evaluator.py`` with *different*
fallback sentence lists, so a PG-19 load failure silently produced different
data in the two code paths. This module is the single source of truth.

Reference: items C3 (single corpus source) and C5 (local RNG) of the
publication-readiness checklist.
"""

from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

# Hard-coded fallback corpus, used only when PG-19 cannot be downloaded.
# Kept deliberately neutral/declarative so the distinctive needle sentence
# ("The secret passphrase is XXXXX.") remains easy to localise.
FALLBACK_SENTENCES: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "Scientists have discovered a new species of deep-sea fish near the Pacific Ridge.",
    "The annual rainfall in the Amazon basin exceeds three thousand millimetres.",
    "Astronomers have detected a rogue planet drifting through interstellar space.",
    "The central bank raised interest rates for the third consecutive quarter.",
    "A new study suggests that sleep deprivation impairs decision-making more than alcohol.",
    "The ancient library of Alexandria contained hundreds of thousands of scrolls.",
    "Engineers have developed a new alloy that is twice as strong as titanium.",
    "The migratory route of monarch butterflies spans over four thousand kilometres.",
    "Archaeologists unearthed a Roman mosaic beneath a construction site in London.",
    "The human genome contains approximately three billion base pairs.",
    "A long drought in the Horn of Africa displaced millions of subsistence farmers.",
    "Quantum computers can solve certain optimization problems exponentially faster.",
    "The mountain range stretches across three continents and twelve countries.",
    "Philosophers have debated the nature of consciousness for more than two millennia.",
    "The ocean contains an estimated one point three five billion cubic kilometres of water.",
    "Neural networks learn by adjusting the weights of millions of parameters.",
    "The treaty was signed by forty-seven nations after three years of negotiation.",
    "The virus mutated rapidly, producing dozens of new variants each month.",
    "Ancient trade routes connected China to the Mediterranean via the Silk Road.",
    "The spacecraft will reach the asteroid belt in approximately two years.",
    "Children who learn a second language early show improved cognitive flexibility.",
    "The rainfall deficit has reduced crop yields by nearly forty per cent.",
    "A single teaspoon of soil contains billions of microscopic organisms.",
    "The new bridge design can withstand earthquakes of magnitude eight or greater.",
    "Linguists estimate that roughly half of the world's languages will disappear by 2100.",
    "The deep-ocean trench reaches a depth of eleven kilometres below sea level.",
    "Researchers found that regular exercise slows cognitive decline in older adults.",
    "The city was founded in the third century by a Greek trading colony.",
    "Satellite imagery shows that the glacier has retreated by twelve kilometres since 1980.",
    "The algorithm processes ten million transactions per second with near-zero latency.",
    "The new vaccine showed ninety-four per cent efficacy in phase three trials.",
    "Migration patterns suggest the two populations diverged roughly fifty thousand years ago.",
    "The telescope can resolve objects separated by less than a milliarcsecond.",
    "The treaty established a demilitarized zone along the entire border.",
    "Chemists synthesized the compound in seventeen steps from common precursors.",
    "The eruption deposited a layer of volcanic ash over an area of ten thousand square kilometres.",
    "Historians believe the plague killed between a third and half of Europe's population.",
    "The semiconductor factory operates under strict clean-room protocols.",
    "A new analysis of tree rings reveals a prolonged drought eight centuries ago.",
    "The submarine can dive to depths exceeding six hundred metres.",
    "Urban heat islands can raise local temperatures by up to five degrees Celsius.",
    "The manuscript was carbon-dated to the eleventh century.",
    "Conservationists reintroduced wolves to the national park in the nineteen nineties.",
    "The molecule adopts a double-helix structure stabilized by hydrogen bonds.",
    "The empire at its height stretched from the Atlantic coast to central Asia.",
    "Radio telescopes detected a repeating fast-radio burst from a distant galaxy.",
    "The dam stores enough water to supply the entire region for three years.",
    "Psychologists found that framing effects influence economic decisions significantly.",
    "The undersea cable transmits data at a rate of over a hundred terabits per second.",
]

# Module-level cache so PG-19 is downloaded at most once per process.
_CORPUS_CACHE: list[str] | None = None


def load_haystack_corpus(max_sentences: int = 5000) -> list[str]:
    """
    Load the haystack corpus: PG-19 sentences, with a hard-coded fallback.

    The result is cached process-wide. Identical across all callers, fixing the
    previous detector/evaluator divergence.

    Args:
        max_sentences: Maximum number of PG-19 sentences to collect.

    Returns:
        List of sentence strings.
    """
    global _CORPUS_CACHE
    if _CORPUS_CACHE is not None:
        return _CORPUS_CACHE

    try:
        from datasets import load_dataset  # type: ignore

        logger.info("Loading PG-19 haystack corpus (max %d sentences) …", max_sentences)
        ds = load_dataset("pg19", split="train", streaming=True)
        sentences: list[str] = []
        for example in ds:
            for sent in example["text"].split(". "):
                sent = sent.strip()
                if 20 < len(sent) < 200:
                    sentences.append(sent + ".")
                if len(sentences) >= max_sentences:
                    break
            if len(sentences) >= max_sentences:
                break
        if sentences:
            logger.info("Loaded %d sentences from PG-19.", len(sentences))
            _CORPUS_CACHE = sentences
            return sentences
    except Exception as exc:
        logger.warning("Could not load PG-19 (%s) — using fallback corpus.", exc)

    # Repeat the fallback so there are always enough sentences for long contexts.
    _CORPUS_CACHE = FALLBACK_SENTENCES * 40
    return _CORPUS_CACHE


def build_haystack(
    context_length: int,
    sentences: list[str],
    rng: random.Random,
    chars_per_token: int = 4,
) -> str:
    """
    Assemble a haystack of roughly ``context_length`` tokens.

    Args:
        context_length: Target length in tokens (approximate; see chars_per_token).
        sentences: Sentence pool.
        rng: A ``random.Random`` instance for deterministic shuffling. Passing an
            explicit RNG (rather than the global one) keeps generation
            independent of unrelated calls — item C5.
        chars_per_token: Rough characters-per-token estimate used for budgeting.
            NOTE: the resulting token count is approximate; callers that report
            exact context length must re-tokenize (item A5).

    Returns:
        The haystack string (needle not yet inserted).
    """
    pool = sentences.copy()
    rng.shuffle(pool)

    parts: list[str] = []
    total_chars = 0
    char_budget = context_length * chars_per_token
    for sent in pool:
        parts.append(sent)
        total_chars += len(sent) + 1
        if total_chars >= char_budget:
            break
    return " ".join(parts)
