"""This file tests retrieval and scoring."""

from __future__ import annotations
import pytest
from tngd_faq_rag.indexes import Bm25Index, VectorIndex, load_vectors, save_vectors
from tngd_faq_rag.reranking import LexicalSemanticReranker


class TestVectorIndex:
    def test_search_returns_cosine_similarity(self):
        idx = VectorIndex(3, use_faiss=False)
        idx.add(["a", "b"], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        results = dict(idx.search([1.0, 0.0, 0.0], top_k=2))
        assert results["a"] == pytest.approx(1.0, abs=1e-5)
        assert results["b"] == pytest.approx(0.0, abs=1e-5)

    def test_normalises_on_insert(self):
        idx = VectorIndex(2, use_faiss=False)
        idx.add(["a"], [[3.0, 4.0]])  # magnitude 5
        assert idx.search([3.0, 4.0], top_k=1)[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_empty_index_returns_nothing(self):
        assert VectorIndex(4, use_faiss=False).search([1, 0, 0, 0], top_k=3) == []

    def test_persistence_round_trip(self, tmp_path):
        idx = VectorIndex(3, use_faiss=False)
        idx.add(["x", "y"], [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        save_vectors(tmp_path, idx)
        restored = load_vectors(tmp_path, use_faiss=False)
        assert restored is not None
        assert restored.search([1.0, 0.0, 0.0], top_k=1)[0][0] == "x"


class TestBm25:
    def test_ranks_the_matching_document_first(self):
        idx = Bm25Index()
        idx.build(["a", "b"], ["road tax renewal for vehicles", "gold investment via e-Mas"])
        assert idx.search("how do I renew road tax", top_k=2)[0][0] == "a"

    def test_finds_rare_terms_a_dense_model_would_smear(self):
        idx = Bm25Index()
        idx.build(["a", "b"], ["DuitNow transfer instructions", "generic payment information"])
        assert idx.search("DuitNow", top_k=1)[0][0] == "a"

    def test_unknown_query_returns_nothing(self):
        idx = Bm25Index()
        idx.build(["a"], ["road tax renewal"])
        assert idx.search("zzzz", top_k=3) == []


class TestCalibratedScores:
    """Relevance must keep an absolute meaning."""

    def test_good_match_scores_far_above_junk(self, ask):
        good = ask("What is TNG eWallet SOS Balance?")
        junk = ask("How do I bake a chocolate cake?")
        assert good["confidence"] > junk["confidence"] + 0.3

    def test_out_of_vocabulary_terms_lower_confidence(self, system):
        """Query terms the corpus has never seen are evidence of being out of domain."""
        in_domain = system.ask("What is CardMatch?")["confidence"]
        out_domain = system.ask("What is quantum chromodynamics?")["confidence"]
        assert out_domain < in_domain

    def test_all_candidates_scored_within_unit_interval(self, system):
        response = system.ask("How do I use CardMatch?")
        for chunk in response["retrieved_chunks"]:
            assert 0.0 <= chunk["scores"]["relevance"] <= 1.0


class TestFusion:
    def test_records_which_retrievers_matched(self, system):
        response = system.ask("What is TNG eWallet SOS Balance?")
        matched = set()
        for chunk in response["retrieved_chunks"]:
            matched.update(chunk["matched_by"])
        assert matched & {"dense", "lexical", "fuzzy", "exact"}

    def test_mmr_avoids_returning_one_article_repeatedly(self, system):
        response = system.ask("verify my saved card")
        parents = [c["parent_id"] for c in response["retrieved_chunks"]]
        assert len(set(parents)) == len(parents)


class TestOutOfDomainRejection:
    """Queries about things the corpus does not cover must be refused."""

    @pytest.mark.parametrize(
        "question",
        [
            "What is CIMB bank",
            "What is Maybank",
            "How do I open a Grab account?",
            "What is bitcoin?",
            "What is a durian",
            "Who is the prime minister?",
        ],
    )
    def test_unknown_named_entities_abstain(self, ask, question):
        response = ask(question)
        assert response["decision"].startswith("abstain"), (
            f"{question!r} answered at confidence {response.get('confidence')}"
        )

    def test_shared_question_prefix_is_not_a_match(self, system):
        """A shared 'What is X' prefix must not imply a shared subject."""
        assert system.ask("What is CIMB bank")["confidence"] < 0.25
        assert system.ask("What is CardMatch?")["confidence"] > 0.9

    def test_interrogative_still_discriminates(self, system):
        """The fix above must not undo the reason interrogatives were kept."""
        why = system.ask("Why must I verify my saved cards?")
        soon = system.ask("How soon must I verify my saved cards?")
        assert "why" in why["sources"][0]["question"].lower()
        assert "how soon" in soon["sources"][0]["question"].lower()

    def test_synonyms_still_bridge_to_the_catalogue(self, system):
        """Out-of-domain rejection must not break legitimate paraphrase."""
        response = system.ask("am I allowed to buy gold on e-Mas")
        assert any("e-Mas" in s["question"] for s in response["sources"][:2])


class TestNoQuestionOverlapDamping:
    """A word found only inside an answer must not carry a match on its own."""

    @pytest.fixture
    def reranker(self, system):
        reranker = system.retriever.reranker
        if not isinstance(reranker, LexicalSemanticReranker):
            pytest.skip("only the built-in reranker applies this damping")
        return reranker

    def test_answer_only_match_is_refused(self, system, reranker, monkeypatch):
        # "countdown" appears in one seed answer and in no seed question
        query = "What is countdown?"
        candidates = system.retriever.retrieve(query)
        damped = reranker.score(query, candidates)
        assert system.ask(query)["decision"] == "abstain_low_confidence"
        monkeypatch.setattr(LexicalSemanticReranker, "NO_QUESTION_OVERLAP_FACTOR", 1.0)
        undamped = reranker.score(query, candidates)
        assert all(d <= u for d, u in zip(damped, undamped))
        # Without the damping it would clear the abstain threshold and be answered
        assert max(undamped) >= system.cfg.abstain_threshold > max(damped)

    def test_question_overlap_is_not_damped(self, system, reranker, monkeypatch):
        query = "What is CardMatch?"
        candidates = system.retriever.retrieve(query)
        damped = reranker.score(query, candidates)
        monkeypatch.setattr(LexicalSemanticReranker, "NO_QUESTION_OVERLAP_FACTOR", 1.0)
        assert reranker.score(query, candidates)[0] == damped[0]
