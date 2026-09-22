"""This file reranks candidates, scoring each with an absolute relevance from 0 to 1."""

from __future__ import annotations
import difflib
from collections.abc import Sequence
from .models import Candidate
from .text import (
    DOMAIN_ALIASES,
    INTERROGATIVES,
    LOW_SIGNAL_TERMS,
    LOW_SIGNAL_WEIGHT,
    expand_query_terms,
    salient_terms,
    sigmoid,
    tokenize,
)


class Reranker:
    name = "base"

    def score(self, query: str, candidates: Sequence[Candidate]) -> list[float]:
        raise NotImplementedError


class LexicalSemanticReranker(Reranker):
    """Zero-dependency reranker producing an absolute relevance in [0, 1]."""

    name = "lexical-semantic"
    # Applied when a candidate shares no word with the user's question
    NO_QUESTION_OVERLAP_FACTOR = 0.8

    def __init__(self, idf: dict[str, float], default_idf: float) -> None:
        self.idf = idf
        self.default_idf = default_idf

    def _weight(self, term: str) -> float:
        """IDF, with low-signal terms discounted."""
        weight = self.idf.get(term, self.default_idf)
        if term in LOW_SIGNAL_TERMS:
            weight *= LOW_SIGNAL_WEIGHT
        return weight

    def _is_known(self, term: str) -> bool:
        """Does the corpus cover this term, directly or through a synonym?"""
        if term in self.idf:
            return True
        for alias in DOMAIN_ALIASES.get(term, ()):
            if any(part in self.idf for part in tokenize(alias)):
                return True
        return False

    def _weighted_coverage(self, query_terms: Sequence[str], chunk_terms: Sequence[str]) -> float:
        if not query_terms:
            return 0.0
        chunk_set = set(chunk_terms)
        total = 0.0
        hit = 0.0
        for t in set(query_terms):
            w = self._weight(t)
            total += w
            if t in chunk_set:
                hit += w
        return hit / total if total else 0.0

    def _weighted_f1(self, query_terms: Sequence[str], cand_terms: Sequence[str]) -> float:
        """F1 over token sets, weighted by informativeness."""
        sa, sb = set(query_terms), set(cand_terms)
        inter = sa & sb
        if not inter or not sa or not sb:
            return 0.0
        w_inter = sum(self._weight(t) for t in inter)
        w_a = sum(self._weight(t) for t in sa)
        w_b = sum(self._weight(t) for t in sb)
        if w_a <= 0 or w_b <= 0:
            return 0.0
        recall, precision = w_inter / w_a, w_inter / w_b
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _intent_agreement(query_terms: Sequence[str], cand_terms: Sequence[str]) -> float:
        """Do the two questions ask the same kind of thing?"""
        q = {t for t in query_terms if t in INTERROGATIVES}
        c = {t for t in cand_terms if t in INTERROGATIVES}
        if not q or not c:
            return 0.5
        return 1.0 if q & c else 0.0

    def score(self, query: str, candidates: Sequence[Candidate]) -> list[float]:
        q_terms = tokenize(query)
        q_expanded = expand_query_terms(query)
        q_salient = salient_terms(q_terms)
        q_salient_text = " ".join(q_salient)

        # Penalise query terms the corpus has never seen, by IDF over salient terms
        # An unknown proper noun means out of domain, an unknown verb means nothing
        # Scales every candidate equally so it moves confidence, never ranking
        if q_salient:
            known_mass = sum(self._weight(t) for t in q_salient if self._is_known(t))
            total_mass = sum(self._weight(t) for t in q_salient)
            vocab_ratio = known_mass / total_mass if total_mass else 1.0
        else:
            vocab_ratio = 1.0
        oov_factor = 0.55 + 0.45 * vocab_ratio

        out: list[float] = []
        for c in candidates:
            chunk_terms = tokenize(c.text)
            cand_q_terms = tokenize(c.question)
            coverage = max(
                self._weighted_coverage(q_terms, chunk_terms),
                0.9 * self._weighted_coverage(q_expanded, chunk_terms),
            )

            # Compare salient terms only so two questions cannot match on "What is"
            # Aliases bridge the user's wording to the catalogue's, damped so they cannot win
            tok_f1 = max(
                self._weighted_f1(q_terms, cand_q_terms),
                0.9 * self._weighted_f1(q_expanded, cand_q_terms),
            )
            cand_salient_text = " ".join(salient_terms(cand_q_terms))
            char_sim = max(
                difflib.SequenceMatcher(None, q_salient_text, cand_salient_text).ratio(),
                0.9
                * difflib.SequenceMatcher(
                    None, " ".join(salient_terms(q_expanded)), cand_salient_text
                ).ratio(),
            )
            # Character similarity refines lexical agreement, it never replaces it
            q_sim = max(tok_f1, char_sim * min(1.0, tok_f1 / 0.25))
            # Question kind is a small signal separate from question subject
            q_sim *= 0.90 + 0.10 * self._intent_agreement(q_terms, cand_q_terms)

            a_sim = self._weighted_f1(q_terms, tokenize(c.answer_slice))

            # Weights grid searched over the golden set, the optimum is a broad plateau
            rel = (0.35 * coverage + 0.45 * q_sim + 0.20 * a_sim) * oov_factor

            # No word shared with the FAQ question means the match rests on the answer text alone
            # Damped so a brand only mentioned inside an answer cannot clear the abstain threshold
            if q_sim == 0.0:
                rel *= self.NO_QUESTION_OVERLAP_FACTOR

            # A near verbatim question match is decisive on its own
            if q_sim >= 0.88:
                rel = max(rel, 0.80 + 0.2 * q_sim)
            out.append(max(0.0, min(1.0, rel)))
        return out


class CrossEncoderReranker(Reranker):
    """CrossEncoder reranking, calibrated to a probability via sigmoid."""

    name = "cross-encoder"

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder  # pyright: ignore[reportMissingImports]

        self.model = CrossEncoder(model_name, device="cpu")
        self.model_name = model_name

    def score(self, query: str, candidates: Sequence[Candidate]) -> list[float]:
        if not candidates:
            return []
        pairs = [[query, c.text] for c in candidates]
        logits = self.model.predict(pairs, show_progress_bar=False, batch_size=32)
        return [float(sigmoid(float(x))) for x in logits]
