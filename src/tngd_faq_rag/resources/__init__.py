"""This file loads the data files shipped with the package, like the seed FAQ.

The 30-entry seed knowledge base ships inside the package so that a fresh clone
answers questions with no setup, no network and no scraping. It is package data
rather than a repo-level file so it survives `pip install` and is importable
from anywhere, and it is JSON rather than a Python literal so it can be diffed,
validated and replaced by the scraper output without touching code.
"""

from __future__ import annotations
import json
from typing import Any

SEED_FILENAME = "tngd_faq_seed.json"


def load_seed_records() -> list[dict[str, Any]]:
    """Return the embedded seed FAQ records."""
    try:  # Python 3.9+
        from importlib.resources import files

        raw = (files(__package__) / SEED_FILENAME).read_text(encoding="utf-8")
    except Exception:  # pragma: no cover - very old interpreters / zipapp quirks
        from pathlib import Path

        raw = (Path(__file__).parent / SEED_FILENAME).read_text(encoding="utf-8")
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("seed knowledge base is empty or malformed")
    return records
