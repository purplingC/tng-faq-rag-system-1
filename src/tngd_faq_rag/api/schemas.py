"""This file contains the request and response shapes the API accepts and returns."""

from __future__ import annotations
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    """A question to answer."""

    model_config = ConfigDict(
        extra="forbid",  # a typo'd field is a client bug; say so rather than ignoring it
        json_schema_extra={
            "examples": [{"question": "What is TNG eWallet SOS Balance?"}],
        },
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The user's question about the Touch 'n Go eWallet.",
    )


class Source(BaseModel):
    """One retrieved FAQ article, with its score and whether it was cited."""

    model_config = ConfigDict(extra="allow")

    rank: int
    question: str
    url: str
    category: str = ""
    relevance: float = 0.0
    cited: bool = False
    snippet: str = ""


class AskResponse(BaseModel):
    """The pipeline's answer."""

    model_config = ConfigDict(
        extra="allow",
        json_schema_extra={
            "examples": [
                {
                    "question": "What is TNG eWallet SOS Balance?",
                    "final_answer": "SOS Balance is an exclusive feature ...",
                    "blocked": False,
                    "decision": "exact_faq",
                    "confidence": 1.0,
                    "url": "https://support.tngdigital.com.my/hc/en-my/articles/...",
                    "retrieved_chunks": [],
                    "sources": [],
                }
            ]
        },
    )

    question: str
    retrieved_chunks: list[dict[str, Any]]
    final_answer: str
    blocked: bool

    url: str = ""
    sources: list[Source] = Field(default_factory=list)
    decision: str = ""
    confidence: float = 0.0
    safety: dict[str, Any] = Field(default_factory=dict)
    backends: dict[str, str] = Field(default_factory=dict)
    latency_ms: float = 0.0
    trace: list[str] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Liveness and readiness are different questions."""

    status: str
    version: str
    detail: dict[str, Any] = Field(default_factory=dict)


class InfoResponse(BaseModel):
    """Which backends are actually in use, for debugging a deployment."""

    version: str
    backends: dict[str, str]
    thresholds: dict[str, float]
    documents: int
    chunks: int


class ErrorResponse(BaseModel):
    """A machine-readable error, so clients branch on `error` not on prose."""

    error: str
    detail: str = ""
    request_id: str = ""
