"""This file holds every setting, each one overridable by an environment variable."""

from __future__ import annotations
import dataclasses
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from .text import DOMAIN_ALIASES, DOMAIN_STOPWORDS, STOPWORDS


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


# Chat endpoints, both speaking the same widely used request format
GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"
OPENAI_BASE = "https://api.openai.com/v1"

# Where an API key may come from, in precedence order
# GOOGLE_API_KEY is last because the name may hold a key for any Google service
_API_KEY_VARS = ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")


def _resolve_api_key() -> tuple[str, str]:
    """Return (key, which variable it came from). Empty strings when unset."""
    for name in _API_KEY_VARS:
        value = os.environ.get(name, "").strip()
        if value:
            return value, name
    return "", ""


def _resolve_api_base() -> str:
    """Pick the endpoint, inferring the provider from the key's variable name."""
    explicit = os.environ.get("LLM_API_BASE", "").strip()
    if explicit:
        return explicit
    _, source = _resolve_api_key()
    if source in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        return GEMINI_OPENAI_BASE
    return OPENAI_BASE


DEFAULT_USER_AGENT = (
    "tngd-faq-rag/3.0 (+https://github.com/purplingC/tng-faq-rag-system-1; educational take-home)"
)


@dataclass
class Config:
    """All tunables in one place."""

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("TNGD_DATA_DIR", "data")))
    index_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("TNGD_INDEX_DIR", ".tngd_index"))
    )
    kb_file: str = field(default_factory=lambda: os.environ.get("TNGD_KB_FILE", "tngd_faq.json"))

    # Chunking
    max_chunk_tokens: int = field(default_factory=lambda: _env_int("TNGD_MAX_CHUNK_TOKENS", 320))
    sentence_overlap: int = field(default_factory=lambda: _env_int("TNGD_SENTENCE_OVERLAP", 1))
    min_chunk_tokens: int = field(default_factory=lambda: _env_int("TNGD_MIN_CHUNK_TOKENS", 24))

    # Embedding
    embedding_dim: int = field(default_factory=lambda: _env_int("TNGD_EMBEDDING_DIM", 768))
    st_model: str = field(
        default_factory=lambda: os.environ.get(
            "TNGD_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        )
    )
    prefer_sentence_transformers: bool = field(
        default_factory=lambda: os.environ.get("TNGD_NO_ST", "") == ""
    )

    # Retrieval
    dense_top_k: int = field(default_factory=lambda: _env_int("TNGD_DENSE_TOP_K", 30))
    lexical_top_k: int = field(default_factory=lambda: _env_int("TNGD_LEXICAL_TOP_K", 30))
    rrf_k: int = field(default_factory=lambda: _env_int("TNGD_RRF_K", 60))
    rerank_candidates: int = field(default_factory=lambda: _env_int("TNGD_RERANK_CANDIDATES", 20))
    final_top_k: int = field(default_factory=lambda: _env_int("TNGD_FINAL_TOP_K", 4))
    mmr_lambda: float = field(default_factory=lambda: _env_float("TNGD_MMR_LAMBDA", 0.7))

    # Decision thresholds, absolute and calibrated
    abstain_threshold: float = field(
        default_factory=lambda: _env_float("TNGD_ABSTAIN_THRESHOLD", 0.34)
    )
    high_confidence_threshold: float = field(
        default_factory=lambda: _env_float("TNGD_HIGH_CONF", 0.72)
    )
    exact_match_ratio: float = field(default_factory=lambda: _env_float("TNGD_EXACT_RATIO", 0.92))
    grounding_threshold: float = field(
        default_factory=lambda: _env_float("TNGD_GROUNDING_THRESHOLD", 0.42)
    )

    # Generation, one of auto, extractive, local or api
    generator: str = field(default_factory=lambda: os.environ.get("TNGD_GENERATOR", "auto"))
    local_model: str = field(
        default_factory=lambda: os.environ.get("TNGD_LOCAL_MODEL", "google/flan-t5-base")
    )
    api_base: str = field(default_factory=_resolve_api_base)
    api_key: str = field(default_factory=lambda: _resolve_api_key()[0])
    api_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "gpt-4o-mini"))
    max_question_chars: int = field(
        default_factory=lambda: _env_int("TNGD_MAX_QUESTION_CHARS", 2000)
    )
    max_answer_sentences: int = field(
        default_factory=lambda: _env_int("TNGD_MAX_ANSWER_SENTENCES", 6)
    )
    max_answer_chars: int = field(default_factory=lambda: _env_int("TNGD_MAX_ANSWER_CHARS", 1200))

    # Answerability grading, auto turns it on when an LLM endpoint is configured
    answerability: str = field(default_factory=lambda: os.environ.get("TNGD_ANSWERABILITY", "auto"))
    answerability_model: str = field(
        default_factory=lambda: os.environ.get("TNGD_ANSWERABILITY_MODEL", "")
    )
    answerability_max_passages: int = field(
        default_factory=lambda: _env_int("TNGD_ANSWERABILITY_MAX_PASSAGES", 3)
    )
    llm_timeout: float = field(default_factory=lambda: _env_float("TNGD_LLM_TIMEOUT", 30.0))

    # UI and scraper
    ui_host: str = field(default_factory=lambda: os.environ.get("TNGD_UI_HOST", "127.0.0.1"))
    ui_port: int = field(default_factory=lambda: _env_int("TNGD_UI_PORT", 8000))
    scrape_delay: float = field(default_factory=lambda: _env_float("TNGD_SCRAPE_DELAY", 0.4))
    scrape_page_size: int = field(default_factory=lambda: _env_int("TNGD_SCRAPE_PAGE_SIZE", 100))
    # Zero means no limit
    scrape_max_articles: int = field(default_factory=lambda: _env_int("TNGD_SCRAPE_MAX", 0))
    user_agent: str = field(
        default_factory=lambda: os.environ.get("TNGD_USER_AGENT", DEFAULT_USER_AGENT)
    )

    def fingerprint(self) -> str:
        """Hash of everything that changes what gets indexed."""
        relevant = {
            k: str(v)
            for k, v in dataclasses.asdict(self).items()
            if k
            in {
                "max_chunk_tokens",
                "sentence_overlap",
                "min_chunk_tokens",
                "embedding_dim",
                "st_model",
            }
        }
        relevant["analysis"] = hashlib.sha256(
            repr(
                (sorted(STOPWORDS), sorted(DOMAIN_STOPWORDS), sorted(DOMAIN_ALIASES.items()))
            ).encode()
        ).hexdigest()[:16]
        return hashlib.sha256(json.dumps(relevant, sort_keys=True).encode()).hexdigest()[:16]
