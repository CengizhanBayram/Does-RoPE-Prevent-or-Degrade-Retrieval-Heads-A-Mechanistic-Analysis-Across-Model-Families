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

logger = logging.getLogger(__name__)

_FALLBACK_SENTENCES: list[str] = [
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

    # ------------------------------------------------------------------
    # Corpus
    # ------------------------------------------------------------------

    def load_haystack_corpus(self) -> list[str]:
        """Load PG-19 sentences; return fallback corpus on failure."""
        if self._corpus is not None:
            return self._corpus

        try:
            from datasets import load_dataset  # type: ignore

            ds = load_dataset("pg19", split="train", streaming=True)
            sentences: list[str] = []
            for example in ds:
                for sent in example["text"].split(". "):
                    sent = sent.strip()
                    if 20 < len(sent) < 200:
                        sentences.append(sent + ".")
                    if len(sentences) >= 3000:
                        break
                if len(sentences) >= 3000:
                    break
            if sentences:
                self._corpus = sentences
                return sentences
        except Exception as exc:
            logger.warning("PG-19 load failed (%s); using fallback.", exc)

        self._corpus = _FALLBACK_SENTENCES * 40
        return self._corpus

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _generate_code() -> str:
        return "".join(random.choices(string.ascii_uppercase + string.digits, k=5))

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
        # Build haystack
        parts: list[str] = []
        total_chars = 0
        char_budget = context_length * 4
        rng_sents = sentences.copy()
        random.shuffle(rng_sents)
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
        random.seed(self.seed)
        np.random.seed(self.seed)

        sentences = self.load_haystack_corpus()
        device = next(self.model.parameters()).device
        n_combo = len(context_lengths) * len(needle_positions)
        samples_per_combo = max(1, n_samples // n_combo)

        acc_matrix = np.zeros((len(context_lengths), len(needle_positions)), dtype=np.float32)

        for i, ctx_len in enumerate(tqdm(context_lengths, desc="Context lengths")):
            for j, pos in enumerate(needle_positions):
                correct = 0
                for _ in range(samples_per_combo):
                    code = self._generate_code()
                    prompt = self.build_prompt(ctx_len, pos, code, sentences)

                    encoding = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=ctx_len + 64,
                    )
                    input_ids = encoding["input_ids"].to(device)

                    try:
                        out = self.model.generate(
                            input_ids,
                            max_new_tokens=20,
                            do_sample=False,
                            temperature=1.0,
                        )
                        generated_ids = out[0, input_ids.shape[1]:]
                        generated_text = self.tokenizer.decode(
                            generated_ids, skip_special_tokens=True
                        )
                        if code in generated_text:
                            correct += 1
                    except RuntimeError as exc:
                        if "out of memory" in str(exc).lower():
                            logger.warning("OOM at ctx_len=%d; skipping.", ctx_len)
                            gc.collect()
                            torch.cuda.empty_cache()
                        else:
                            raise
                    finally:
                        del input_ids
                        torch.cuda.empty_cache()

                acc_matrix[i, j] = correct / samples_per_combo

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

        Masking is implemented by zeroing the attention output for each
        listed (layer, head) pair via forward hooks.

        Args:
            heads_to_mask: List of (layer_idx, head_idx) tuples.
            context_lengths: Same as evaluate().
            needle_positions: Same as evaluate().
            n_samples: Same as evaluate().

        Returns:
            np.ndarray of shape (len(context_lengths), len(needle_positions)).
        """
        hooks = []
        head_dim = self.model.config.hidden_size // self.model.config.num_attention_heads

        def _make_output_zero_hook(h_idx: int):
            def hook(module, input_, output):
                out = output[0] if isinstance(output, tuple) else output
                start = h_idx * head_dim
                end = start + head_dim
                out = out.clone()
                out[:, :, start:end] = 0.0
                return (out,) + output[1:] if isinstance(output, tuple) else out
            return hook

        try:
            for (layer_idx, head_idx) in heads_to_mask:
                layer = self.model.model.layers[layer_idx]
                h = layer.self_attn.o_proj.register_forward_pre_hook(
                    _make_output_zero_hook(head_idx)
                )
                hooks.append(h)
            result = self.evaluate(context_lengths, needle_positions, n_samples)
        finally:
            for h in hooks:
                h.remove()

        return result
