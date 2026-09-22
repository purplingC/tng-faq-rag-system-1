"""This file exposes the public entry points of the TNG eWallet FAQ assistant."""

from __future__ import annotations
from .config import Config
from .models import Candidate, Chunk, FaqDoc, Verdict
from .pipeline import RagSystem, ask_tngd_bot, build_system, get_system
from .version import __version__

__all__ = [
    "Candidate",
    "Chunk",
    "Config",
    "FaqDoc",
    "RagSystem",
    "Verdict",
    "__version__",
    "ask_tngd_bot",
    "build_system",
    "get_system",
]
