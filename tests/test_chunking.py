"""This file tests chunking, including two bugs that only real articles trigger."""

from __future__ import annotations
import pytest
from tngd_faq_rag.chunking import _split_long_sentence, chunk_documents
from tngd_faq_rag.models import FaqDoc


def doc(answer: str) -> FaqDoc:
    return FaqDoc("id", "Question?", answer, "https://example.test/a", "Category")


class TestAtomicity:
    def test_short_answers_stay_whole(self, cfg):
        chunks = chunk_documents([doc("A brief answer.")], cfg)
        assert len(chunks) == 1

    def test_long_answers_are_split(self, cfg):
        body = " ".join(f"Sentence number {i} explains something." for i in range(120))
        assert len(chunk_documents([doc(body)], cfg)) > 1

    def test_every_chunk_carries_its_question(self, cfg):
        body = " ".join(f"Sentence number {i} explains something." for i in range(120))
        assert all(c.text.startswith("Q: Question?") for c in chunk_documents([doc(body)], cfg))

    def test_parent_linking(self, cfg):
        body = " ".join(f"Sentence number {i} explains something." for i in range(120))
        chunks = chunk_documents([doc(body)], cfg)
        assert {c.parent_id for c in chunks} == {"id"}
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


PATHOLOGICAL = {
    "one huge sentence": "word " * 2000,
    "huge bullet line": "- " + ("item detail " * 900),
    "comma run-on": ", ".join(f"clause {i} here" for i in range(400)) + ".",
    "semicolon run-on": "; ".join(f"clause {i} here" for i in range(400)) + ".",
    "colon run-on": ": ".join(f"clause {i}" for i in range(500)) + ".",
    "bullet list": "\n".join(f"- Item {i} with detail" for i in range(400)),
    "single word": "Yes.",
}


class TestPathologicalArticles:
    """Real help-centre articles contain flattened tables and run-on bullet lines."""

    @pytest.mark.parametrize("name", sorted(PATHOLOGICAL))
    def test_terminates_and_respects_budget(self, cfg, name):
        chunks = chunk_documents([doc(PATHOLOGICAL[name])], cfg)
        assert chunks
        assert max(c.token_count for c in chunks) <= cfg.max_chunk_tokens * 1.35

    def test_oversized_sentence_is_split_at_clause_boundaries(self, cfg):
        sentence = "; ".join(f"clause {i} with words" for i in range(200))
        pieces = _split_long_sentence(sentence, cfg.max_chunk_tokens)
        assert len(pieces) > 1
        assert all(p.strip() for p in pieces)
