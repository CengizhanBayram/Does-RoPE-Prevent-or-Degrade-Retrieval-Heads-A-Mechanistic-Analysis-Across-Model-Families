"""
NIAH (Needle-in-a-Haystack) evaluation harness.

Provides end-to-end generation-based evaluation: the model must reproduce the
hidden passphrase code when queried after reading a long context.
"""

from __future__ import annotations

import gc
import logging
import random
import string
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.corpus import load_haystack_corpus

logger = logging.getLogger(__name__)



class NIAHEvaluator:
    """
    End-to-end NIAH evaluator using generation-based accuracy.

    Accuracy is defined as the fraction of samples where the model's generated
    output contains the correct secret passphrase code.
    """

    def __init__(self, model: Any, tokenizer: Any, config: dict, seed: int = 42) -> None:
        """
        Args:
            model: HuggingFace causal LM.
            tokenizer: Corresponding tokenizer.
            config: Global config dict.
            seed: Random seed.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.seed = seed

        self.model.eval()
        self._corpus: list[str] | None = None
        # Local RNG for reproducibility independent of global state (item C5).
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def load_haystack_corpus(self) -> list[str]:
        """Load the shared haystack corpus (PG-19 with fallback) — item C3."""
        if self._corpus is None:
            self._corpus = load_haystack_corpus(max_sentences=3000)
        return self._corpus

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _generate_code(self) -> str:
        return "".join(self._rng.choices(string.ascii_uppercase + string.digits, k=5))

    def build_prompt(
        self,
        context_length: int,
        needle_position: float,
        code: str,
        sentences: list[str],
    ) -> str:
        """
        Build a single NIAH prompt string.

        Args:
            context_length: Approximate token budget for haystack.
            needle_position: Fractional position [0, 1] to insert needle.
            code: 5-character passphrase code.
            sentences: Pool of haystack sentences.

        Returns:
            Full prompt string ending with "Answer:".
        """
        parts: list[str] = []
        total_chars = 0
        char_budget = context_length * 4
        rng_sents = sentences.copy()
        self._rng.shuffle(rng_sents)
        for sent in rng_sents:
            parts.append(sent)
            total_chars += len(sent) + 1
            if total_chars >= char_budget:
                break
        haystack = " ".join(parts)

        needle = f"The secret passphrase is {code}."
        words = haystack.split()
        insert_idx = max(0, min(int(len(words) * needle_position), len(words) - 1))
        words.insert(insert_idx, needle)
        full_text = " ".join(words)

        query = "What is the secret passphrase?"
        return f"{full_text}\n\n{query}\n\nAnswer:"

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(
        self,
        context_lengths: list[int],
        needle_positions: list[float],
        n_samples: int,
    ) -> np.ndarray:
        """
        Evaluate the model across all (context_length, needle_position) combinations.

        Args:
            context_lengths: List of context lengths in tokens.
            needle_positions: List of fractional needle positions.
            n_samples: Total number of samples (divided equally over combinations).

        Returns:
            np.ndarray of shape (len(context_lengths), len(needle_positions))
            with accuracy values in [0, 1].
        """
        self._rng.seed(self.seed)

        sentences = self.load_haystack_corpus()
        device = next(
            (p for p in self.model.parameters() if p.device.type != "meta"),
            next(self.model.parameters()),
        ).device
        n_combo = len(context_lengths) * len(needle_positions)
        samples_per_combo = max(1, n_samples // n_combo)

        acc_matrix = np.zeros(
            (len(context_lengths), len(needle_positions)), dtype=np.float32
        )

        for i, ctx_len in enumerate(tqdm(context_lengths, desc="Context lengths")):
            for j, pos in enumerate(needle_positions):
                correct = 0
                total = 0
                for _ in range(samples_per_combo):
                    code = self._generate_code()
                    prompt = self.build_prompt(ctx_len, pos, code, sentences)

                    input_ids = None  # initialise so finally block is safe
                    out = None
                    try:
                        encoding = self.tokenizer(
                            prompt,
                            return_tensors="pt",
                            truncation=True,
                            max_length=ctx_len + 64,
                        )
                        input_ids = encoding["input_ids"].to(device)

                        out = self.model.generate(
                            input_ids,
                            max_new_tokens=20,
                            do_sample=False,
                        )
                        generated_ids = out[0, input_ids.shape[1]:]
                        generated_text = self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        )
                        if code in generated_text:
                            correct += 1
                        total += 1

                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            logger.warning("OOM at ctx_len=%d; skipping sample.", ctx_len)
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        # FIX #17: Guard del so NameError cannot occur if
                        # the assignment above raised before completing.
                        if input_ids is not None:
                            del input_ids
                        if out is not None:
                            del out
                        torch.cuda.empty_cache()

                acc_matrix[i, j] = correct / total if total > 0 else 0.0

        return acc_matrix

    @torch.no_grad()
    def evaluate_with_head_masking(
        self,
        heads_to_mask: list[tuple[int, int]],
        context_lengths: list[int],
        needle_positions: list[float],
        n_samples: int,
    ) -> np.ndarray:
        """
        Evaluate NIAH accuracy with specified attention heads masked.

        Masking zeroes each listed head's output contribution just before
        the o_proj linear layer (i.e. the concatenated head outputs).

        Args:
            heads_to_mask: List of (layer_idx, head_idx) tuples.
            context_lengths: Same as evaluate().
            needle_positions: Same as evaluate().
            n_samples: Same as evaluate().

        Returns:
            np.ndarray of shape (len(context_lengths), len(needle_positions)).
        """
        head_dim = self._get_head_dim()
        hooks = []

        # FIX #4: Use register_forward_pre_hook (signature: module, input_tuple)
        # to zero the concatenated head outputs BEFORE o_proj processes them.
        # The previous code used register_forward_pre_hook but with a 3-arg
        # signature (module, input, output) which is wrong — pre_hooks only
        # receive (module, input).
        def _make_pre_hook(h_idx: int):
            def hook(module, input_: tuple):
                if not input_:
                    return input_
                inp = input_[0]  # (batch, seq_len, n_heads * head_dim)
                inp = inp.clone()
                start = h_idx * head_dim
                end = start + head_dim
                inp[:, :, start:end] = 0.0
                return (inp,) + input_[1:]
            return hook

        try:
            layers = self._get_layers()
            for (layer_idx, head_idx) in heads_to_mask:
                layer = layers[layer_idx]
                h = layer.self_attn.o_proj.register_forward_pre_hook(
                    _make_pre_hook(head_idx)
                )
                hooks.append(h)
            result = self.evaluate(context_lengths, needle_positions, n_samples)
        finally:
            for h in hooks:
                h.remove()

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_head_dim(self) -> int:
        """Return per-head dimension."""
        cfg = getattr(self.model, "config", None)
        if cfg is not None:
            if hasattr(cfg, "head_dim"):
                return cfg.head_dim
            try:
                return cfg.hidden_size // cfg.num_attention_heads
            except AttributeError:
                pass
        logger.warning(
            "Could not derive head_dim from model config; falling back to 128 "
            "(item D4). Head-masking indices may be wrong for this model."
        )
        return 128

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
        raise AttributeError("Cannot locate layer list in model architecture.")
