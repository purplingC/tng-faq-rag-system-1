"""This file tests the answerability grader."""

from __future__ import annotations
import pytest
from tngd_faq_rag.guardrails.answerability import (
    AnswerabilityGrader,
    LLMAnswerabilityGrader,
)
from tngd_faq_rag.llm import ChatResult

PASSAGES = [
    "SOS Balance is an exclusive feature for toll payments.",
    "CardMatch lets you browse and apply for credit cards.",
]


class StubClient:
    """Returns scripted replies and counts calls."""

    def __init__(self, *replies: str, ok: bool = True, error: str = "") -> None:
        self.replies = list(replies) or [""]
        self.ok = ok
        self.error = error
        self.calls: list[list[dict]] = []

    def complete(self, messages, **kwargs) -> ChatResult:
        self.calls.append(messages)
        reply = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return ChatResult(text=reply, ok=self.ok, error=self.error, latency_ms=1.0)


class TestParsing:
    """One parser, two reply shapes."""

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ("NONE", []),
            ("none", []),
            ("1", [0]),
            ("1, 2", [0, 1]),
            ("Passage 2", [1]),
            ("2,1", [0, 1]),
        ],
    )
    def test_free_text_replies(self, reply, expected):
        assert LLMAnswerabilityGrader._parse(reply, 2) == expected

    @pytest.mark.parametrize(
        "reply,expected",
        [
            ('{"supporting": [1]}', [0]),
            ('{"supporting": [1, 2]}', [0, 1]),
            ('{"supporting": [2, 1]}', [0, 1]),
        ],
    )
    def test_structured_replies(self, reply, expected):
        assert LLMAnswerabilityGrader._parse(reply, 2) == expected

    def test_structured_empty_list_is_a_real_none(self):
        """The schema turns "unanswerable" into data instead of into prose."""
        assert LLMAnswerabilityGrader._parse('{"supporting": []}', 2) == []

    @pytest.mark.parametrize("reply", ["", "   ", "I am not sure", "9"])
    def test_unusable_replies_are_no_judgement(self, reply):
        """None means "no judgement", which is not the same as "unanswerable"."""
        assert LLMAnswerabilityGrader._parse(reply, 2) is None

    def test_hallucinated_index_fails_open(self):
        """A non-empty list of out-of-range indices is a botched reply."""
        assert LLMAnswerabilityGrader._parse('{"supporting": [9]}', 2) is None


class TestGrading:
    def test_none_marks_the_question_unanswerable(self):
        grader = LLMAnswerabilityGrader(StubClient("NONE"))
        result = grader.grade("How do we spell SOS balance?", PASSAGES)
        assert not result.answerable
        assert result.graded
        assert result.supporting == []

    def test_endorsed_passages_are_reported(self):
        grader = LLMAnswerabilityGrader(StubClient("1"))
        result = grader.grade("What is SOS Balance?", PASSAGES)
        assert result.answerable
        assert result.supporting == [0]
        assert result.graded

    def test_no_passages_is_unanswerable(self):
        assert not LLMAnswerabilityGrader(StubClient("1")).grade("q", []).answerable


class TestFailureModes:
    """The gate is an addition to a pipeline that is already safe without it."""

    def test_unreachable_endpoint_fails_open(self):
        client = StubClient("", ok=False, error="Connection refused")
        result = LLMAnswerabilityGrader(client).grade("q", PASSAGES)
        assert result.answerable
        assert not result.graded
        assert "unavailable" in result.reason

    def test_unparsable_reply_fails_open(self):
        result = LLMAnswerabilityGrader(StubClient("perhaps?")).grade("q", PASSAGES)
        assert result.answerable
        assert not result.graded
        assert "unparsable" in result.reason

    def test_disabled_grader_allows_everything(self):
        result = AnswerabilityGrader().grade("anything at all", PASSAGES)
        assert result.answerable
        assert not result.graded


class TestCaching:
    def test_repeated_question_is_not_re_graded(self):
        client = StubClient("NONE")
        grader = LLMAnswerabilityGrader(client)
        for _ in range(4):
            grader.grade("How do we spell SOS balance?", PASSAGES)
        assert len(client.calls) == 1

    def test_different_questions_are_graded_separately(self):
        client = StubClient("NONE", "1")
        grader = LLMAnswerabilityGrader(client)
        grader.grade("first question", PASSAGES)
        grader.grade("second question", PASSAGES)
        assert len(client.calls) == 2

    def test_cache_is_bounded(self):
        grader = LLMAnswerabilityGrader(StubClient("1"), cache_size=3)
        for i in range(10):
            grader.grade(f"question {i}", PASSAGES)
        assert len(grader._cache) <= 3


class TestPipelineIntegration:
    def test_unanswerable_question_abstains(self, system, monkeypatch):
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(StubClient("NONE")))
        response = system.ask("How do we spell SOS balance?")
        assert response["decision"] == "abstain_unanswerable"
        assert not response["blocked"]
        assert response["safety"]["answerability"]["graded"] is True

    def test_endorsed_question_is_answered(self, system, monkeypatch):
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(StubClient("1")))
        response = system.ask("What is TNG eWallet SOS Balance?")
        assert not response["decision"].startswith("abstain")
        assert "SOS Balance" in response["final_answer"]

    def test_grading_is_recorded_in_the_trace(self, system, monkeypatch):
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(StubClient("NONE")))
        trace = system.ask("How do we spell SOS balance?")["trace"]
        assert any(t.startswith("answerability:fail:graded") for t in trace)

    def test_grader_narrows_the_passages_given_to_the_generator(self, system, monkeypatch):
        """Corrective-RAG behaviour: endorsed passages only reach generation."""
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(StubClient("2")))
        question = "How come I have to confirm the cards I saved in TNG eWallet?"
        assert system.ask(question)["confidence"] < system.cfg.high_confidence_threshold
        response = system.ask(question)
        assert len(response["retrieved_chunks"]) == 1

    def test_exact_match_skips_the_model_entirely(self, system, monkeypatch):
        """A verbatim FAQ question does not need a second opinion."""
        client = StubClient("NONE")  # would abstain if it were ever consulted
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(client))
        response = system.ask("What is TNG eWallet SOS Balance?")
        assert client.calls == []
        assert response["decision"] == "exact_faq"
        assert "answerability:pass:skipped_exact_match" in response["trace"]

    def test_high_confidence_near_miss_is_still_graded(self, system, monkeypatch):
        """A high score is not an exact match, and a near miss can answer the wrong question."""
        client = StubClient("NONE")
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(client))
        response = system.ask("What is the SOS Balance feature?")
        assert response["confidence"] >= system.cfg.high_confidence_threshold
        assert len(client.calls) == 1
        assert response["decision"] == "abstain_unanswerable"

    def test_borderline_confidence_is_still_graded(self, system, monkeypatch):
        """The skip must not swallow the cases the gate was built for."""
        client = StubClient("NONE")
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(client))
        response = system.ask("How do we spell SOS balance?")
        assert len(client.calls) == 1
        assert response["decision"] == "abstain_unanswerable"

    def test_unreachable_grader_leaves_behaviour_unchanged(self, system, monkeypatch):
        client = StubClient("", ok=False, error="Connection refused")
        monkeypatch.setattr(system, "answerability", LLMAnswerabilityGrader(client))
        response = system.ask("What is TNG eWallet SOS Balance?")
        assert response["decision"] == "exact_faq"
