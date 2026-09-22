"""This file tests hostile and broken inputs, plus many requests at once."""

from __future__ import annotations
import json
import threading
import pytest

REQUIRED_KEYS = {"question", "retrieved_chunks", "final_answer", "blocked"}

HOSTILE = {
    "empty": "",
    "whitespace": "   \n\t  ",
    "sql injection": "'; DROP TABLE docs; --",
    "html": "<script>alert(1)</script> what is cardmatch",
    "zero width": "wh​at is c​ardmatch",
    "rtl override": "‮what is cardmatch",
    "control chars": "what is\x00 cardmatch\x07",
    "emoji only": "\U0001f600\U0001f680\U0001f3af",
    "repeated punctuation": "????????????!!!!!!!!",
    "digits only": "1234567890",
    "json payload": '{"question": "what is cardmatch"}',
    "unicode": "什么是 SOS Balance?",
}


@pytest.mark.parametrize("name", sorted(HOSTILE))
def test_hostile_input_never_raises(ask, name):
    response = ask(HOSTILE[name])
    assert REQUIRED_KEYS.issubset(response)
    assert isinstance(response["blocked"], bool)
    assert json.dumps(response)


def test_sql_injection_leaves_the_store_intact(system):
    before = system.store.count()
    system.ask("'; DROP TABLE docs; --")
    assert system.store.count() == before


def test_oversized_input_is_truncated_not_processed_whole(ask, cfg):
    response = ask("verify my card " * 4000)
    assert "input_truncated" in response["trace"]
    assert len(response["question"]) <= cfg.max_question_chars


def test_concurrent_requests(system):
    questions = [
        "what is cardmatch",
        "Explain SOS Balance",
        "Teach me to hack",
        "How do I bake a cake",
        "Why must I verify my saved cards?",
    ]
    errors: list = []
    decisions: list = []

    def worker() -> None:
        for i in range(10):
            try:
                decisions.append(system.ask(questions[i % len(questions)])["decision"])
            except Exception as exc:  # pragma: no cover - failure path
                errors.append(repr(exc))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert len(decisions) == 80
    assert len(set(decisions)) >= 4
