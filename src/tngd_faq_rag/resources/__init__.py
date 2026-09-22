"""This file loads the data files shipped with the package, like the seed FAQ."""

from __future__ import annotations
import json
from typing import Any

SEED_FILENAME = "tngd_faq_seed.json"


def load_seed_records() -> list[dict[str, Any]]:
    """Return the embedded seed FAQ records."""
    try:
        from importlib.resources import files

        raw = (files(__package__) / SEED_FILENAME).read_text(encoding="utf-8")
    except Exception:
        from pathlib import Path

        raw = (Path(__file__).parent / SEED_FILENAME).read_text(encoding="utf-8")
    records = json.loads(raw)
    if not isinstance(records, list) or not records:
        raise ValueError("seed knowledge base is empty or malformed")
    return records
