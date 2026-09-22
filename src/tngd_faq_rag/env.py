"""This file loads settings from a .env file into the environment."""

from __future__ import annotations
import os
from pathlib import Path
from .logging_utils import LOG

__all__ = ["load_dotenv"]


def _strip_value(raw: str) -> str:
    value = raw.strip()
    # Quoted values keep a trailing hash as text, unquoted ones treat it as a comment
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value.split(" #", 1)[0].strip()


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from `path` into os.environ."""
    file = Path(path)
    if not file.is_file():
        return {}

    applied: dict[str, str] = {}
    try:
        lines = file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        LOG.debug("could not read %s: %s", file, exc)
        return {}

    for number, line in enumerate(lines, start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if text.startswith("export "):
            text = text[len("export ") :].lstrip()
        key, sep, raw = text.partition("=")
        key = key.strip()
        if not sep or not key.isidentifier():
            LOG.debug("%s:%d ignored, not a KEY=VALUE line", file, number)
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = _strip_value(raw)
        applied[key] = os.environ[key]

    if applied:
        # Names only, since logging a value would put an API key in the logs
        LOG.debug("loaded %d variable(s) from %s: %s", len(applied), file, ", ".join(applied))
    return applied
