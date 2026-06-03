"""
Retrieval head detection via NIAH (Needle-in-a-Haystack) scoring.

METHOD NOTE (item A1 — read before citing):
This is a *simplified proxy* adapted from Wu et al. (ICLR 2025), not a
re-implementation of their full retrieval score. Wu et al. accumulate a
copy-paste score over the tokens the model generates while answering. Here we
take a single forward pass and check whether each head's attention argmax at
the final input position falls inside the needle span. This is cheaper and
sufficient for *relative* comparisons across heads/models, but it is a
different (coarser) operationalisation. Describe it accurately in the paper as
"an adapted, single-pass attention-argmax variant of Wu et al. (2025)".
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

from src.corpus import build_haystack, load_haystack_corpus

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paired-seed NIAH sample generation (items C5 / §2.3 paired-seed control)
#
# Generation is split into two stages so that the *data* is identical across
# models:
#   1. generate_niah_specs(seed)        — tokenizer-INDEPENDENT text specs.
#      RNG consumption depends only on the seed, never on a tokenizer, so the
#      same seed yields byte-identical specs for every model.
#   2. build_samples_from_specs(specs, tokenizer) — per-model tokenization.
# For cross-model comparison, paired_spec_subset() keeps only the specs that
# tokenize successfully in EVERY model (intersection), so all models are scored
# on the exact same n samples.
# ---------------------------------------------------------------------------

def find_subsequence(seq: list[int], sub: list[int]) -> tuple[int, int] | None:
    """Return (start, end_exclusive) of the first occurrence of sub in seq."""
    n, m = len(seq), len(sub)
    if m == 0:
        return None
    for i in range(n - m + 1):
        if seq[i : i + m] == sub:
            return i, i + m
    return None


def generate_niah_specs(
    seed: int,
    n_samples: int,
    context_lengths: list[int],
    needle_positions: list[float],
    max_corpus_sentences: int = 5000,
) -> list[dict]:
    """
    Generate tokenizer-independent NIAH specifications.

    Deterministic given ``seed``: identical for every model because no tokenizer
    is involved. Each spec is a dict with: spec_id, prompt, code, needle,
    context_length, needle_position.
    """
    rng = random.Random(seed)
    sentences = load_haystack_corpus(max_sentences=max_corpus_sentences)
    combos = [(cl, pos) for cl in context_lengths for pos in needle_positions]
    per_combo = max(1, n_samples // len(combos))

    specs: list[dict] = []
    spec_id = 0
    for context_length, needle_position in combos:
        for _ in range(per_combo):
            code = "".join(rng.choices(string.ascii_uppercase + string.digits, k=5))
            needle = f"The secret passphrase is {code}."
            query = "What is the secret passphrase?"
            haystack = build_haystack(context_length, sentences, rng)
            words = haystack.split()
            insert_idx = max(0, min(int(len(words) * needle_position), len(words) - 1))
            words.insert(insert_idx, needle)
            full_text = " ".join(words)
            prompt = f"{full_text}\n\n{query}\n\nAnswer:"
            specs.append({
                "spec_id": spec_id,
                "prompt": prompt,
                "code": code,
                "needle": needle,
                "context_length": context_length,
                "needle_position": needle_position,
            })
            spec_id += 1
    return specs


def locate_needle(prompt: str, needle: str, context_length: int, tokenizer) -> dict | None:
    """
    Tokenize ``prompt`` and locate the needle token span for a given tokenizer.

    Returns a dict (prompt_ids, needle_token_ids, needle_start_idx,
    needle_end_idx) or None if the needle cannot be located after tokenization.
    """
    encoding = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=context_length + 64
    )
    prompt_ids: list[int] = encoding["input_ids"][0].tolist()

    # Try without and with a leading space (tokenizers merge the first token
    # differently); strip an accidental BOS.
    for prefix in ("", " "):
        candidate = tokenizer(prefix + needle, add_special_tokens=False)["input_ids"]
        bos = tokenizer.bos_token_id
        if bos is not None and candidate and candidate[0] == bos:
            candidate = candidate[1:]
        token_range = find_subsequence(prompt_ids, candidate)
        if token_range is not None:
            return {
                "prompt_ids": prompt_ids,
                "needle_token_ids": candidate,
                "needle_start_idx": token_range[0],
                "needle_end_idx": token_range[1],
            }
    return None


def build_samples_from_specs(specs: list[dict], tokenizer) -> tuple[list[dict], list[int]]:
    """
    Tokenize specs for one model. Returns (samples, valid_spec_ids).

    Specs whose needle cannot be located for this tokenizer are skipped (and
    excluded from valid_spec_ids).
    """
    samples: list[dict] = []
    valid: list[int] = []
    for spec in specs:
        loc = locate_needle(spec["prompt"], spec["needle"], spec["context_length"], tokenizer)
        if loc is None:
            continue
        samples.append({
            "prompt": spec["prompt"],
            "prompt_ids": loc["prompt_ids"],
            "code": spec["code"],
            "needle_token_ids": loc["needle_token_ids"],
            "needle_start_idx": loc["needle_start_idx"],
            "needle_end_idx": loc["needle_end_idx"],
            "context_length": spec["context_length"],
            "actual_token_length": len(loc["prompt_ids"]),
            "needle_position": spec["needle_position"],
            "spec_id": spec["spec_id"],
        })
        valid.append(spec["spec_id"])
    return samples, valid


def valid_spec_ids(specs: list[dict], tokenizer) -> set[int]:
    """Return the set of spec_ids whose needle tokenizes/locates for a tokenizer."""
    return set(build_samples_from_specs(specs, tokenizer)[1])


def paired_spec_subset(
    specs: list[dict], tokenizers: dict[str, object]
) -> tuple[list[dict], list[int]]:
    """
    Keep only specs valid in EVERY tokenizer (intersection-drop, §2.3).

    Guarantees all models are scored on the same n samples. Returns
    (subset_specs, dropped_spec_ids).
    """
    keep: set[int] | None = None
    for tok in tokenizers.values():
        ids = valid_spec_ids(specs, tok)
        keep = ids if keep is None else (keep & ids)
    keep = keep or set()
    subset = [s for s in specs if s["spec_id"] in keep]
    dropped = [s["spec_id"] for s in specs if s["spec_id"] not in keep]
    if dropped:
        logger.warning(
            "Paired-seed intersection dropped %d/%d specs (failed tokenization "
            "in >=1 model): indices %s",
            len(dropped), len(specs), dropped,
        )
    return subset, dropped


class RetrievalHeadDetector:
    """
    Detects retrieval heads in transformer models using NIAH scoring.

    Adapted (simplified) from Wu et al. (ICLR 2025): a head is flagged as a
    retrieval head if, on a Needle-in-a-Haystack task, its attention argmax at
    the final input position lands inside the needle span, aggregated as a hit
    rate over samples. See the module docstring for how this differs from the
    original copy-paste retrieval score.
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
        # Local RNG so sample generation is independent of unrelated global
        # random() calls elsewhere in the pipeline (item C5).
        self._rng = random.Random(seed)

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
        """Load the shared haystack corpus (PG-19 with fallback) — item C3."""
        if self._haystack_corpus is None:
            self._haystack_corpus = load_haystack_corpus(max_sentences=5000)
        return self._haystack_corpus

    # ------------------------------------------------------------------
    # Sample generation
    # ------------------------------------------------------------------

    def _generate_code(self) -> str:
        """Generate a random 5-character alphanumeric passphrase code."""
        return "".join(self._rng.choices(string.ascii_uppercase + string.digits, k=5))

    def _build_haystack(self, context_length: int, sentences: list[str]) -> str:
        """Build a haystack of ~context_length tokens (shared impl, local RNG)."""
        return build_haystack(context_length, sentences, self._rng)

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
        Generate NIAH samples for THIS model (single-model convenience path).

        Equivalent to ``generate_niah_specs(seed, ...)`` followed by
        ``prepare_samples(specs)``. For cross-model PAIRED comparison, generate
        the specs once and use ``paired_spec_subset`` + ``prepare_samples`` so
        all models share the identical sample set (§2.3).

        Returns:
            List of sample dicts (see prepare_samples for keys).
        """
        specs = generate_niah_specs(
            self.seed, n_samples, context_lengths, needle_positions
        )
        return self.prepare_samples(specs)

    def prepare_samples(self, specs: list[dict]) -> list[dict]:
        """
        Tokenize pre-generated (tokenizer-independent) specs for this model.

        Args:
            specs: Output of ``generate_niah_specs`` (optionally already passed
                through ``paired_spec_subset`` for paired comparison).

        Returns:
            List of sample dicts with keys: prompt, prompt_ids, code,
            needle_token_ids, needle_start_idx, needle_end_idx, context_length,
            actual_token_length, needle_position, spec_id.
        """
        samples, valid = build_samples_from_specs(specs, self.tokenizer)
        dropped = len(specs) - len(valid)
        if dropped:
            logger.warning(
                "prepare_samples: %d/%d specs failed tokenization for this model. "
                "For paired comparison pre-filter with paired_spec_subset.",
                dropped, len(specs),
            )
        if samples:
            lengths = np.array([s["actual_token_length"] for s in samples])
            logger.info(
                "Prepared %d NIAH samples. Actual token length: "
                "mean=%.0f min=%d max=%d.",
                len(samples), lengths.mean(), lengths.min(), lengths.max(),
            )
        else:
            logger.info("Prepared 0 NIAH samples.")
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
