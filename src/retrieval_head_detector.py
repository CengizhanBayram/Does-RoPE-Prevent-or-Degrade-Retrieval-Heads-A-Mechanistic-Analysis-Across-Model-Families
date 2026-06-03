"""
Retrieval head detection via NIAH (Needle-in-a-Haystack) scoring.
Implements Wu et al. (ICLR 2025) methodology.
"""

from __future__ import annotations

import gc
import logging
import random
import string
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

_FALLBACK_SENTENCES: list[str] = [
    "The quick brown fox jumps over the lazy dog.",
    "In the beginning God created the heavens and the earth.",
    "It was the best of times, it was the worst of times.",
    "Call me Ishmael.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "It is a truth universally acknowledged, that a single man in possession of a good fortune, must be in want of a wife.",
    "The sky above the port was the color of television, tuned to a dead channel.",
    "It was a bright cold day in April, and the clocks were striking thirteen.",
    "Lolita, light of my life, fire of my loins.",
    "It was the epoch of belief, it was the epoch of incredulity.",
    "We were somewhere around Barstow on the edge of the desert when the drugs began to take hold.",
    "Many years later, as he faced the firing squad, Colonel Aureliano Buendía was to remember that distant afternoon.",
    "The man in black fled across the desert, and the gunman followed.",
    "It was a dark and stormy night; the rain fell in torrents.",
    "You don't know about me without you have read a book by the name of The Adventures of Tom Sawyer.",
    "Whether I shall turn out to be the hero of my own life, or whether that station will be held by anybody else, these pages must show.",
    "Last night I dreamt I went to Manderley again.",
    "It was the day my grandmother exploded.",
    "The sun shone, having no alternative, on the nothing new.",
    "Far out in the uncharted backwaters of the unfashionable end of the western spiral arm of the Galaxy.",
    "Once upon a time and a very good time it was there was a moocow coming down along the road.",
    "As Gregor Samsa awoke one morning from uneasy dreams he found himself transformed in his bed into a gigantic insect.",
    "Someone must have slandered Josef K., for one morning, without having done anything wrong, he was arrested.",
    "It was a pleasure to burn.",
    "In my younger and more vulnerable years my father gave me some advice.",
    "They shoot the white girl first.",
    "If you really want to hear about it, the first thing you'll probably want to know is where I was born.",
    "There was a boy called Eustace Clarence Scrubb, and he almost deserved it.",
    "The drought had lasted now for ten million years, and the reign of the terrible lizards had long since ended.",
    "All children, except one, grow up.",
    "One morning, when Gregor Samsa woke from troubled dreams, he found himself transformed in his bed into a horrible vermin.",
    "In the late summer of that year we lived in a house in a village that looked across the river and the plain to the mountains.",
    "Robert Cohn was once middleweight boxing champion of Princeton.",
    "The studio was filled with the rich odour of roses.",
    "124 was spiteful. Full of a baby's venom.",
    "It was love at first sight.",
    "I am an invisible man.",
    "The most merciful thing in the world, I think, is the inability of the human mind to correlate all its contents.",
    "Somewhere in la Mancha, in a place whose name I do not care to remember.",
    "A screaming comes across the sky.",
    "It was a wrong number that started it, the telephone ringing three times in the dead of night.",
    "Of man's first disobedience, and the fruit of that forbidden tree whose mortal taste brought death into the world.",
    "The world is what it is; men who are nothing, who allow themselves to become nothing, have no place in it.",
    "Time is not a line but a dimension, like the dimensions of space.",
    "Science has now proven that raindrops keep falling on my head.",
    "The empire at its height stretched from the Atlantic coast to central Asia.",
    "Radio telescopes detected a repeating fast-radio burst from a distant galaxy.",
    "Quantum computers can solve certain optimization problems exponentially faster.",
    "The human genome contains approximately three billion base pairs.",
    "Neural networks learn by adjusting the weights of millions of parameters.",
]


class RetrievalHeadDetector:
    """
    Detects retrieval heads in transformer models using NIAH scoring.

    Following Wu et al. (ICLR 2025): a head is considered a retrieval head
    if, during a Needle-in-a-Haystack task, its attention argmax at the final
    token position lands on needle tokens.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: dict,
        score_threshold: float = 0.1,
        seed: int = 42,
    ) -> None:
        """
        Args:
            model: HuggingFace causal LM model.
            tokenizer: Corresponding tokenizer.
            config: Global config dict (niah section used).
            score_threshold: Minimum hit-rate to classify a head as retrieval.
            seed: Random seed for reproducibility.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.score_threshold = score_threshold
        self.seed = seed

        self.model.eval()
        self._haystack_corpus: list[str] | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_device(self) -> torch.device:
        """Return the first non-meta parameter device (safe for quantized models)."""
        for p in self.model.parameters():
            if p.device.type != "meta":
                return p.device
        # All params on meta (rare) — fall back to cuda/cpu
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _get_layers(self):
        """Return model layer list, handling LLaMA / OLMo / Qwen variants."""
        model_inner = getattr(self.model, "model", self.model)
        for attr in ("layers", "transformer.blocks", "transformer.h", "decoder.layers"):
            obj = model_inner
            for part in attr.split("."):
                obj = getattr(obj, part, None)
                if obj is None:
                    break
            if obj is not None:
                return obj
        raise AttributeError(
            "Cannot locate layer list. Tried: layers, transformer.blocks, "
            "transformer.h, decoder.layers"
        )

    def _get_n_layers_heads(self) -> tuple[int, int]:
        """Return (n_layers, n_heads) from model config."""
        try:
            return (
                self.model.config.num_hidden_layers,
                self.model.config.num_attention_heads,
            )
        except AttributeError:
            layers = self._get_layers()
            n_layers = len(layers)
            n_heads = getattr(layers[0].self_attn, "num_heads", None) or \
                      getattr(layers[0].self_attn, "num_attention_heads", 32)
            return n_layers, n_heads

    # ------------------------------------------------------------------
    # Haystack corpus
    # ------------------------------------------------------------------

    def _load_haystack_corpus(self) -> list[str]:
        """Load PG-19 sentences; fall back to hardcoded sentences."""
        if self._haystack_corpus is not None:
            return self._haystack_corpus

        try:
            from datasets import load_dataset  # type: ignore

            logger.info("Loading PG-19 haystack corpus …")
            ds = load_dataset("pg19", split="train", streaming=True)
            sentences: list[str] = []
            for example in ds:
                text: str = example["text"]
                for sent in text.split(". "):
                    sent = sent.strip()
                    if 20 < len(sent) < 200:
                        sentences.append(sent + ".")
                    if len(sentences) >= 5000:
                        break
                if len(sentences) >= 5000:
                    break
            if sentences:
                logger.info("Loaded %d sentences from PG-19.", len(sentences))
                self._haystack_corpus = sentences
                return sentences
        except Exception as exc:
            logger.warning("Could not load PG-19: %s — using fallback corpus.", exc)

        self._haystack_corpus = _FALLBACK_SENTENCES * 20
        return self._haystack_corpus

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate_code(self) -> str:
        """Generate a random 5-character alphanumeric passphrase code."""
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

    def _build_haystack(self, context_length: int, sentences: list[str]) -> str:
        """Build a haystack string of approximately context_length tokens."""
        rng_sentences = sentences.copy()
        random.shuffle(rng_sentences)

        parts: list[str] = []
        total_chars = 0
        char_budget = context_length * 4  # rough estimate: 1 token ≈ 4 chars

        for sent in rng_sentences:
            parts.append(sent)
            total_chars += len(sent) + 1
            if total_chars >= char_budget:
                break
        return " ".join(parts)

    def _insert_needle(
        self, haystack: str, needle: str, position: float
    ) -> tuple[str, int]:
        """
        Insert needle at the given fractional word position in haystack.

        Returns:
            (full_text, char_offset_of_needle_start)
        """
        words = haystack.split()
        insert_idx = max(0, min(int(len(words) * position), len(words) - 1))
        words.insert(insert_idx, needle)
        full_text = " ".join(words)
        # Use find() + assertion to ensure we locate the correct occurrence
        char_offset = full_text.find(needle)
        assert char_offset >= 0, f"Needle not found after insertion: {needle[:30]}"
        return full_text, char_offset

    def _find_needle_token_range(
        self,
        prompt_ids: list[int],
        needle_ids: list[int],
    ) -> tuple[int, int] | None:
        """
        Find the start/end token indices of needle_ids inside prompt_ids.

        Returns:
            (start, end_exclusive) or None if not found.
        """
        n, m = len(prompt_ids), len(needle_ids)
        for i in range(n - m + 1):
            if prompt_ids[i : i + m] == needle_ids:
                return i, i + m
        return None

    def generate_niah_samples(
        self,
        n_samples: int,
        context_lengths: list[int],
        needle_positions: list[float],
    ) -> list[dict]:
        """
        Generate NIAH evaluation samples.

        Args:
            n_samples: Total number of samples to generate (distributed evenly
                over all context_length × needle_position combinations).
            context_lengths: List of context lengths in tokens.
            needle_positions: List of fractional needle positions [0, 1].

        Returns:
            List of sample dicts with keys: prompt, prompt_ids, code,
            needle_token_ids, needle_start_idx, needle_end_idx,
            context_length, needle_position.
        """
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        sentences = self._load_haystack_corpus()
        samples: list[dict] = []
        skipped = 0

        combos = [
            (cl, pos) for cl in context_lengths for pos in needle_positions
        ]
        samples_per_combo = max(1, n_samples // len(combos))

        for context_length, needle_position in combos:
            generated_this_combo = 0
            attempts = 0
            # Allow extra attempts to compensate for skipped samples
            max_attempts = samples_per_combo * 3

            while generated_this_combo < samples_per_combo and attempts < max_attempts:
                attempts += 1
                code = self._generate_code()
                needle = f"The secret passphrase is {code}."
                query = "What is the secret passphrase?"

                try:
                    haystack = self._build_haystack(context_length, sentences)
                    full_text, _ = self._insert_needle(haystack, needle, needle_position)
                except AssertionError:
                    skipped += 1
                    continue

                prompt = f"{full_text}\n\n{query}\n\nAnswer:"

                encoding = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=context_length + 64,
                )
                prompt_ids: list[int] = encoding["input_ids"][0].tolist()

                # Tokenize needle WITHOUT special tokens, then try WITH leading space
                # to handle tokenizers that merge the first token differently
                needle_ids: list[int] | None = None
                for prefix in ("", " "):
                    candidate = self.tokenizer(
                        prefix + needle, add_special_tokens=False
                    )["input_ids"]
                    # Strip BOS if accidentally included
                    bos = self.tokenizer.bos_token_id
                    if bos is not None and candidate and candidate[0] == bos:
                        candidate = candidate[1:]
                    token_range = self._find_needle_token_range(prompt_ids, candidate)
                    if token_range is not None:
                        needle_ids = candidate
                        break

                if needle_ids is None or token_range is None:
                    skipped += 1
                    logger.debug(
                        "Needle not found in tokenized prompt "
                        "(context_length=%d, pos=%.2f). Skipping.",
                        context_length,
                        needle_position,
                    )
                    continue

                start_idx, end_idx = token_range
                samples.append(
                    {
                        "prompt": prompt,
                        "prompt_ids": prompt_ids,
                        "code": code,
                        "needle_token_ids": needle_ids,
                        "needle_start_idx": start_idx,
                        "needle_end_idx": end_idx,
                        "context_length": context_length,
                        "needle_position": needle_position,
                    }
                )
                generated_this_combo += 1

        if skipped:
            logger.warning(
                "%d sample attempts skipped (needle not found in tokens). "
                "Generated %d/%d requested samples.",
                skipped,
                len(samples),
                n_samples,
            )
        logger.info("Generated %d NIAH samples.", len(samples))
        return samples

    # ------------------------------------------------------------------
    # Head scoring — hook-based, layer-by-layer (O(seq_len²) peak, not O(L·seq_len²))
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score_heads(self, samples: list[dict]) -> np.ndarray:
        """
        Score every attention head on NIAH hit rate.

        Uses forward hooks on each self_attn module so that each layer's
        attention matrix is processed and freed before the next layer runs.
        Peak VRAM is O(seq_len²) per layer rather than O(n_layers · seq_len²).

        Args:
            samples: Output of generate_niah_samples().

        Returns:
            np.ndarray of shape (n_layers, n_heads) with hit rates in [0, 1].
        """
        device = self._get_device()
        n_layers, n_heads = self._get_n_layers_heads()
        layers = self._get_layers()

        hit_counts = np.zeros((n_layers, n_heads), dtype=np.int32)
        total_counts = np.zeros((n_layers, n_heads), dtype=np.int32)

        # Track whether any layer ever returned attention weights
        attn_weights_seen: list[bool] = [False]

        for i, sample in enumerate(tqdm(samples, desc="Scoring retrieval heads")):
            if i % 10 == 0:
                logger.info("Progress: %d/%d samples processed.", i, len(samples))

            needle_start = sample["needle_start_idx"]
            needle_end = sample["needle_end_idx"]

            # Per-sample attention capture dict: layer_idx → (n_heads, seq_len) on CPU
            captured: dict[int, torch.Tensor] = {}
            hooks = []

            def _make_hook(layer_idx: int):
                def hook(module, input_, output):
                    # self_attn returns (attn_output, attn_weights, past_kv) or just attn_output
                    if isinstance(output, tuple) and len(output) >= 2:
                        attn_w = output[1]
                        if attn_w is not None:
                            # attn_w: (batch, n_heads_attn, seq_len, seq_len)
                            # Move to CPU immediately and keep only last-token row
                            last_row = attn_w[0, :, -1, :].float().cpu()
                            captured[layer_idx] = last_row
                            attn_weights_seen[0] = True
                            # Replace with None in the returned tuple to free GPU memory
                            return (output[0], None) + output[2:]
                    return output
                return hook

            try:
                for layer_idx, layer in enumerate(layers):
                    h = layer.self_attn.register_forward_hook(_make_hook(layer_idx))
                    hooks.append(h)

                input_ids = torch.tensor(
                    [sample["prompt_ids"]], dtype=torch.long, device=device
                )
                # output_attentions=True asks each self_attn to return its weights
                self.model(
                    input_ids=input_ids,
                    output_attentions=True,
                    use_cache=False,
                )

            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    logger.warning(
                        "OOM on sample %d (seq_len=%d). Skipping.",
                        i,
                        len(sample["prompt_ids"]),
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    captured.clear()
                else:
                    raise
            finally:
                for h in hooks:
                    h.remove()
                if "input_ids" in dir():
                    del input_ids

            # Update hit/total counts from captured attention rows
            for layer_idx, attn_last in captured.items():
                # attn_last: (n_heads_captured, seq_len)
                n_heads_captured = attn_last.shape[0]
                argmax_positions = attn_last.argmax(dim=-1).numpy()

                # Handle MQA: single KV head shared across all Q heads
                if n_heads_captured == 1 and n_heads > 1:
                    argmax_positions = np.repeat(argmax_positions, n_heads)
                    n_heads_captured = n_heads

                for head_idx in range(min(n_heads_captured, n_heads)):
                    pos = int(argmax_positions[head_idx])
                    total_counts[layer_idx, head_idx] += 1
                    if needle_start <= pos < needle_end:
                        hit_counts[layer_idx, head_idx] += 1

            captured.clear()
            if i % 10 == 9:
                gc.collect()
                torch.cuda.empty_cache()

        if not attn_weights_seen[0]:
            logger.error(
                "No attention weights were captured (output_attentions=True may not "
                "be supported for this model / quantization config). "
                "All scores will be 0. Try loading without 8-bit quantization."
            )

        with np.errstate(invalid="ignore"):
            scores = np.where(
                total_counts > 0,
                hit_counts / total_counts,
                0.0,
            )

        logger.info(
            "Scoring complete. Max score: %.3f, heads above threshold (%.2f): %d",
            scores.max(),
            self.score_threshold,
            int((scores >= self.score_threshold).sum()),
        )
        return scores.astype(np.float32)

    def get_retrieval_heads(self, scores: np.ndarray) -> list[tuple[int, int]]:
        """
        Return (layer, head) pairs whose score exceeds the threshold.

        Args:
            scores: (n_layers, n_heads) array from score_heads().

        Returns:
            Sorted list of (layer_idx, head_idx) tuples.
        """
        layer_idxs, head_idxs = np.where(scores >= self.score_threshold)
        heads = sorted(zip(layer_idxs.tolist(), head_idxs.tolist()))
        logger.info(
            "Found %d retrieval heads (threshold=%.2f).", len(heads), self.score_threshold
        )
        return heads

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_scores(self, scores: np.ndarray, path: str | Path) -> None:
        """Save score matrix to a .npy file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, scores)
        logger.info("Scores saved to %s", path)

    def load_scores(self, path: str | Path) -> np.ndarray:
        """Load score matrix from a .npy file."""
        scores = np.load(Path(path))
        logger.info("Scores loaded from %s, shape=%s", path, scores.shape)
        return scores
