"""This file checks that the generated answer is supported by the retrieved sources."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
from ..text import extract_numbers, split_sentences, tokenize


class GroundingChecker:
    """Verifies that generated text is actually supported by retrieved sources."""

    def __init__(self, idf: dict[str, float], default_idf: float, threshold: float) -> None:
        self.idf = idf
        self.default_idf = default_idf
        self.threshold = threshold

    @staticmethod
    def _ngrams(tokens: Sequence[str], n: int) -> set:
        return {" ".join(tokens[i : i + n]) for i in range(max(0, len(tokens) - n + 1))}

    def _support(
        self, sentence: str, source_tokens: Sequence[set], source_grams: Sequence[set]
    ) -> float:
        toks = tokenize(sentence)
        if not toks:
            return 1.0  # punctuation-only fragment, harmless
        weights = {t: self.idf.get(t, self.default_idf) for t in set(toks)}
        total = sum(weights.values()) or 1.0
        best = 0.0
        for st in source_tokens:
            hit = sum(w for t, w in weights.items() if t in st)
            best = max(best, hit / total)
        grams = self._ngrams(toks, 4)
        if grams and any(grams & sg for sg in source_grams):
            best = max(best, 0.85)
        return best

    def check(self, answer: str, sources: Sequence[str]) -> dict[str, Any]:
        if not answer.strip():
            return {"grounded": False, "score": 0.0, "kept": [], "dropped": [], "citations": []}
        if not sources:
            return {
                "grounded": False,
                "score": 0.0,
                "kept": [],
                "dropped": split_sentences(answer),
                "citations": [],
            }

        source_tokens = [set(tokenize(s)) for s in sources]
        source_grams = [self._ngrams(tokenize(s), 4) for s in sources]
        source_numbers = set()
        for s in sources:
            source_numbers.update(extract_numbers(s))

        kept: list[str] = []
        dropped: list[str] = []
        citations: list[dict[str, Any]] = []
        scores: list[float] = []

        for sent in split_sentences(answer):
            support = self._support(sent, source_tokens, source_grams)
            numbers = extract_numbers(sent)
            bad_numbers = [n for n in numbers if n not in source_numbers]
            if bad_numbers:
                support = min(support, 0.2)  # fabricated figure - not recoverable
            scores.append(support)
            if support >= self.threshold:
                kept.append(sent)
                best_i = max(
                    range(len(source_tokens)),
                    key=lambda i: self._support(sent, [source_tokens[i]], [source_grams[i]]),
                )
                citations.append(
                    {"sentence": sent, "source_index": best_i, "support": round(support, 3)}
                )
            else:
                dropped.append(sent)

        overall = (sum(scores) / len(scores)) if scores else 0.0
        grounded = bool(kept) and (len(kept) / max(1, len(kept) + len(dropped))) >= 0.5
        return {
            "grounded": grounded,
            "score": round(overall, 3),
            "kept": kept,
            "dropped": dropped,
            "citations": citations,
        }
