"""This file tests the optional event log and its redaction."""

from __future__ import annotations
import json
import pytest
from tngd_faq_rag.events import record, redact, summarise


class TestRedaction:
    def test_card_numbers_are_removed(self):
        assert redact("my card is 4539578763621486 please") == "my card is [card] please"

    def test_a_reference_number_is_kept(self):
        # Not a valid card number, so it is an ordinary reference the support team needs
        assert "1234567890123" in redact("reference 1234567890123")

    def test_emails_and_phones_are_removed(self):
        out = redact("write to sam@example.com or call 012-345 6789")
        assert "[email]" in out and "[phone]" in out
        assert "sam@example.com" not in out and "6789" not in out

    def test_ordinary_text_is_untouched(self):
        assert redact("How do I reload my eWallet?") == "How do I reload my eWallet?"


class TestRecording:
    def test_nothing_is_written_when_logging_is_off(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TNGD_EVENT_LOG", raising=False)
        record("a question", {"decision": "exact_faq"})
        assert not list(tmp_path.iterdir())

    def test_one_line_per_question(self, tmp_path, monkeypatch):
        log = tmp_path / "events.jsonl"
        monkeypatch.setenv("TNGD_EVENT_LOG", str(log))
        record(
            "What is SOS Balance?",
            {
                "decision": "exact_faq",
                "language": "en",
                "confidence": 1.0,
                "latency_ms": 12.5,
                "final_answer": "SOS Balance is ...",
                "sources": [{"url": "https://e.test/1"}],
                "safety": {"answerability": {"graded": True, "latency_ms": 800}},
            },
        )
        event = json.loads(log.read_text(encoding="utf-8").strip())
        assert event["decision"] == "exact_faq"
        assert event["llm_called"] is True
        assert event["source_url"] == "https://e.test/1"
        assert "ts" in event

    def test_the_key_is_never_written(self, tmp_path, monkeypatch):
        log = tmp_path / "events.jsonl"
        monkeypatch.setenv("TNGD_EVENT_LOG", str(log))
        monkeypatch.setenv("LLM_API_KEY", "AQ.super-secret-key-value")
        record("hello", {"decision": "exact_faq", "safety": {"answerability": {"graded": True}}})
        assert "super-secret" not in log.read_text(encoding="utf-8")

    def test_a_broken_log_path_does_not_break_answering(self, tmp_path, monkeypatch):
        # A directory cannot be appended to, and the request must still succeed
        monkeypatch.setenv("TNGD_EVENT_LOG", str(tmp_path))
        record("hello", {"decision": "exact_faq"})

    def test_pipeline_writes_one_event_per_question(self, system, tmp_path, monkeypatch):
        log = tmp_path / "pipeline.jsonl"
        monkeypatch.setenv("TNGD_EVENT_LOG", str(log))
        system.ask("What is TNG eWallet SOS Balance?")
        system.ask("How do I bake a chocolate cake?")
        lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
        assert len(lines) == 2
        assert lines[1]["decision"].startswith("abstain")


class TestSummary:
    @pytest.fixture
    def log(self, tmp_path):
        path = tmp_path / "events.jsonl"
        rows = [
            {"decision": "exact_faq", "language": "en", "llm_called": False, "latency_ms": 10},
            {
                "decision": "abstain_low_confidence",
                "language": "ms",
                "llm_called": True,
                "latency_ms": 30,
                "question": "kek coklat?",
            },
            {
                "decision": "abstain_low_confidence",
                "language": "en",
                "llm_called": True,
                "latency_ms": 20,
                "question": "cake?",
            },
        ]
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
        return path

    def test_counts_and_latency(self, log):
        out = summarise(log)
        assert out["events"] == 3
        assert out["llm_calls"] == 2
        assert out["languages"] == {"en": 2, "ms": 1}
        assert out["median_latency_ms"] == 20

    def test_unanswered_questions_are_listed(self, log):
        assert set(dict(summarise(log)["top_unanswered"])) == {"kek coklat?", "cake?"}

    def test_a_damaged_line_is_skipped(self, log):
        with log.open("a", encoding="utf-8") as fh:
            fh.write("not json\n")
        assert summarise(log)["events"] == 3
