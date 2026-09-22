"""This file contains the default generator, which answers using verified source sentences."""

from __future__ import annotations
import difflib
import re
from collections.abc import Sequence
from ..config import Config
from ..models import Candidate, Generation
from ..store import MetadataStore
from ..text import clean_text, expand_query_terms, normalize_question, split_sentences, tokenize
from .base import Generator

# A sentence opening like this continues the one before it
_CONTINUES_RE = re.compile(r"^(It|Its|This|These|That|Those|They|Their|Them)\b")


class ExtractiveComposer(Generator):
    """Default generator: composes the answer out of verified source sentences."""

    name = "extractive"

    def __init__(self, cfg: Config, idf: dict[str, float], default_idf: float) -> None:
        self.cfg = cfg
        self.idf = idf
        self.default_idf = default_idf

    def _sentence_score(self, q_terms: Sequence[str], sentence: str) -> float:
        s_terms = set(tokenize(sentence))
        if not s_terms or not q_terms:
            return 0.0
        weights = {t: self.idf.get(t, self.default_idf) for t in set(q_terms)}
        total = sum(weights.values()) or 1.0
        hit = sum(w for t, w in weights.items() if t in s_terms)
        return hit / total

    def generate(
        self, question: str, candidates: Sequence[Candidate], store: MetadataStore
    ) -> Generation:
        if not candidates:
            return Generation(text="", backend=self.name, abstained=True)

        top = candidates[0]
        parent = store.doc(top.parent_id)
        parent_answer = parent["answer"] if parent else top.answer_slice
        q_norm = normalize_question(question)
        title_ratio = difflib.SequenceMatcher(
            None, q_norm, normalize_question(top.question)
        ).ratio()

        # Serve the answer whole when the user asked the FAQ question, or when it fits the budget
        # Trimming a short answer saves little and can drop the one sentence that answers
        parent_sentences = split_sentences(parent_answer)
        fits = (
            len(parent_sentences) <= self.cfg.max_answer_sentences
            and len(clean_text(parent_answer)) <= self.cfg.max_answer_chars
        )
        if "exact" in top.sources or title_ratio >= self.cfg.exact_match_ratio or fits:
            return Generation(
                text=self._truncate(parent_answer),
                backend=self.name,
                used_source_indices=[0],
                raw=parent_answer,
            )

        # Otherwise select the supporting sentences, preserving source order
        q_terms = expand_query_terms(question)

        def sentences_of(cand: Candidate) -> list[tuple[float, int, str]]:
            doc = store.doc(cand.parent_id)
            body = doc["answer"] if doc else cand.answer_slice
            return [
                (self._sentence_score(q_terms, sent), si, sent)
                for si, sent in enumerate(split_sentences(body))
            ]

        # Anchor on the top ranked article
        # Ranking globally lets a generic definition outscore the real answer
        primary = sentences_of(top)
        if not primary:
            return Generation(text="", backend=self.name, abstained=True)
        best_primary = max(sc for sc, _, _ in primary)
        keep: list[tuple[int, int, str]] = [
            (0, si, sent) for sc, si, sent in primary if sc >= max(0.18, best_primary * 0.45)
        ][: self.cfg.max_answer_sentences]
        if not keep:
            top_sent = max(primary, key=lambda t: t[0])
            keep = [(0, top_sent[1], top_sent[2])]

        # Keep an It or This sentence together with the sentence it refers back to
        by_index = {si: sent for _, si, sent in primary}
        kept = {si for _, si, _ in keep}
        for si in sorted(kept):
            for partner in (si - 1, si + 1):
                later = max(si, partner)
                if (
                    partner in by_index
                    and partner not in kept
                    and _CONTINUES_RE.match(by_index[later])
                    and len(kept) < self.cfg.max_answer_sentences
                ):
                    kept.add(partner)
        keep = [(0, si, by_index[si]) for si in sorted(kept)]

        # A second article contributes only if it beats everything the first offered
        if len(candidates) > 1 and candidates[1].relevance >= 0.9 * top.relevance:
            extra = [
                (1, si, sent) for sc, si, sent in sentences_of(candidates[1]) if sc > best_primary
            ]
            keep.extend(extra[:1])

        keep.sort(key=lambda t: (t[0], t[1]))
        text = " ".join(sent for _, _, sent in keep)
        used = sorted({ci for ci, _, _ in keep})
        return Generation(
            text=self._truncate(text), backend=self.name, used_source_indices=used, raw=text
        )

    def _truncate(self, text: str) -> str:
        text = clean_text(text)
        if len(text) <= self.cfg.max_answer_chars:
            return text
        cut = text[: self.cfg.max_answer_chars]
        stop = max(cut.rfind(". "), cut.rfind("! "), cut.rfind("? "))
        return (cut[: stop + 1] if stop > 200 else cut).strip()
