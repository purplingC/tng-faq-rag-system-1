"""This file contains fixtures shared by every test."""

from __future__ import annotations
import sys
from pathlib import Path
import pytest

# Allow pytest to run straight from a clone without installing first
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
from tngd_faq_rag.config import Config  # noqa: E402
from tngd_faq_rag.pipeline import build_system  # noqa: E402


@pytest.fixture(scope="session")
def cfg(tmp_path_factory) -> Config:
    config = Config()
    config.index_dir = tmp_path_factory.mktemp("index")
    config.data_dir = tmp_path_factory.mktemp("data")
    return config


@pytest.fixture(scope="session")
def system(cfg: Config):
    return build_system(cfg, use_seed=True, force_rebuild=True)


@pytest.fixture(scope="session")
def ask(system):
    return system.ask
