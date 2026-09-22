"""This file picks which generator writes the final answer."""

from __future__ import annotations
import os
from ..config import Config
from ..deps import OPT
from ..logging_utils import LOG
from .base import Generator, PromptBuilder
from .extractive import ExtractiveComposer
from .local_seq2seq import LocalSeq2SeqGenerator
from .openai_compatible import OpenAICompatibleGenerator

__all__ = [
    "ExtractiveComposer",
    "Generator",
    "LocalSeq2SeqGenerator",
    "OpenAICompatibleGenerator",
    "PromptBuilder",
    "build_generator",
]


def build_generator(
    cfg: Config, idf: dict[str, float], default_idf: float, prompt_builder: PromptBuilder
) -> Generator:
    """Choose a generator from configuration and what is actually installed."""
    choice = (cfg.generator or "auto").lower()

    if choice in ("api", "auto") and cfg.api_key:
        LOG.info("Generator: OpenAI-compatible API (%s)", cfg.api_model)
        return OpenAICompatibleGenerator(cfg, prompt_builder)

    opted_into_local = choice == "local" or bool(os.environ.get("TNGD_ALLOW_LOCAL_LLM"))
    local_available = OPT.has("transformers") and OPT.has("torch")
    if choice in ("local", "auto") and opted_into_local and local_available:
        try:
            generator = LocalSeq2SeqGenerator(cfg, prompt_builder)
            LOG.info("Generator: local seq2seq (%s)", cfg.local_model)
            return generator
        except Exception as exc:
            LOG.warning("Local model unavailable (%s); falling back to extractive", exc)

    if choice not in ("extractive", "auto"):
        LOG.warning("Unknown generator %r; falling back to extractive", choice)
    LOG.info("Generator: extractive composer (grounded by construction)")
    return ExtractiveComposer(cfg, idf, default_idf)
