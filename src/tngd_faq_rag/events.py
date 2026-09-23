"""This file records one JSON line per answered question, when a log path is set."""

from __future__ import annotations
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any
from .logging_utils import LOG
from .text import luhn_valid

# A card number, a long account or reference number, an email, a phone number
_CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_PHONE_RE = re.compile(r"\b(?:\+?60|0)1\d[- ]?\d{3,4}[- ]?\d{4}\b")

_LOCK = threading.Lock()


def redact(text: str) -> str:
    """Replace card numbers, phone numbers and emails before anything is written."""
    out = _EMAIL_RE.sub("[email]", text or "")
    out = _PHONE_RE.sub("[phone]", out)

    def _card(match: re.Match) -> str:
        digits = re.sub(r"\D", "", match.group(0))
        return "[card]" if luhn_valid(digits) else match.group(0)

    return _CARD_RE.sub(_card, out)


def log_path() -> Path | None:
    """Where events are written, or None when logging is off (the default)."""
    configured = os.environ.get("TNGD_EVENT_LOG", "").strip()
    return Path(configured) if configured else None


def record(question: str, response: dict[str, Any]) -> None:
    """Append one event. Never raises: logging must not break answering."""
    path = log_path()
    if path is None:
        return
    try:
        answerability = (response.get("safety") or {}).get("answerability") or {}
        sources = response.get("sources") or []
        event = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "question": redact(question)[:500],
            "language": response.get("language", ""),
            "decision": response.get("decision", ""),
            "blocked": bool(response.get("blocked")),
            "confidence": response.get("confidence"),
            "latency_ms": response.get("latency_ms"),
            "answer_chars": len(response.get("final_answer") or ""),
            "source_url": (sources[0].get("url") if sources else ""),
            # Which model was asked, never the key that authorised it
            "llm_called": bool(answerability.get("graded")),
            "llm_model": os.environ.get("LLM_MODEL", "") if answerability.get("graded") else "",
            "llm_ms": answerability.get("latency_ms") if answerability.get("graded") else None,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - a log must never break a request
        LOG.debug("event log write failed: %s", exc)


def summarise(path: Path, top: int = 10) -> dict[str, Any]:
    """Count decisions, languages and LLM calls in an event log."""
    from collections import Counter

    decisions: Counter[str] = Counter()
    languages: Counter[str] = Counter()
    unanswered: Counter[str] = Counter()
    latencies: list[float] = []
    llm_calls = total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        total += 1
        decisions[event.get("decision", "")] += 1
        languages[event.get("language", "")] += 1
        llm_calls += bool(event.get("llm_called"))
        if isinstance(event.get("latency_ms"), (int, float)):
            latencies.append(float(event["latency_ms"]))
        if str(event.get("decision", "")).startswith("abstain"):
            unanswered[event.get("question", "")] += 1
    latencies.sort()
    return {
        "events": total,
        "decisions": dict(decisions.most_common()),
        "languages": dict(languages.most_common()),
        "llm_calls": llm_calls,
        "median_latency_ms": round(latencies[len(latencies) // 2], 2) if latencies else None,
        "slowest_latency_ms": round(latencies[-1], 2) if latencies else None,
        "top_unanswered": unanswered.most_common(top),
    }
