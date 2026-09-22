"""This file defines the shapes of data passed around the system."""

from __future__ import annotations
import dataclasses
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FaqDoc:
    """One verified FAQ entry. The unit of provenance."""

    doc_id: str
    question: str
    answer: str
    url: str
    category: str

    def as_dict(self) -> dict[str, str]:
        return dataclasses.asdict(self)


class IngestionReport(dict[str, Any]):
    """A plain dict, so the report serialises to JSON as it is."""


@dataclass
class Chunk:
    """A slice of one FAQ entry, sized to embed and index."""

    chunk_id: str
    parent_id: str
    text: str  # The text that gets embedded and indexed
    answer_slice: str  # Answer portion only, for extraction and citation
    question: str
    url: str
    category: str
    chunk_index: int
    n_chunks: int
    token_count: int
    char_span: tuple[int, int]

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["char_span"] = list(self.char_span)
        return d


@dataclass
class Candidate:
    """A chunk moving through retrieval, carrying the scores it earned."""

    chunk_id: str
    parent_id: str
    text: str
    answer_slice: str
    question: str
    url: str
    category: str
    chunk_index: int = 0
    n_chunks: int = 1
    dense_score: float = 0.0
    lexical_score: float = 0.0
    fuzzy_score: float = 0.0
    rrf_score: float = 0.0
    relevance: float = 0.0  # Absolute and calibrated, in [0, 1], never normalised across candidates
    sources: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "parent_id": self.parent_id,
            "question": self.question,
            "url": self.url,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "n_chunks": self.n_chunks,
            "chunk_text": self.text,
            "answer_slice": self.answer_slice,
            "scores": {
                "dense": round(self.dense_score, 6),
                "lexical": round(self.lexical_score, 6),
                "fuzzy": round(self.fuzzy_score, 6),
                "rrf": round(self.rrf_score, 6),
                "relevance": round(self.relevance, 6),
            },
            "matched_by": self.sources,
        }


class Action:
    """What a guardrail decided to do about the text it saw."""

    ALLOW = "allow"
    BLOCK = "block"
    SAFE_COMPLETE = "safe_complete"  # Refuse, but answer with a helpful message


@dataclass
class Verdict:
    """One guardrail's ruling on a question or an answer."""

    action: str = Action.ALLOW
    category: str = ""
    severity: float = 0.0  # How serious the match is, from 0 to 1
    reason: str = ""
    matched: list[str] = field(default_factory=list)
    message: str = ""

    @property
    def blocked(self) -> bool:
        return self.action != Action.ALLOW

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "category": self.category,
            "severity": round(self.severity, 3),
            "reason": self.reason,
            "matched": self.matched[:5],
        }


@dataclass
class Rule:
    """A pattern a guardrail matches text against, and what to do on a hit."""

    name: str
    category: str
    severity: float
    pattern: re.Pattern
    exempt: re.Pattern | None = None
    action: str = Action.BLOCK
    on_normalized: bool = False  # Match the deobfuscated text instead of the raw text
    reason: str = ""


@dataclass
class Generation:
    """An answer produced by one generator backend."""

    text: str
    backend: str
    used_source_indices: list[int] = field(default_factory=list)
    raw: str = ""
    abstained: bool = False
