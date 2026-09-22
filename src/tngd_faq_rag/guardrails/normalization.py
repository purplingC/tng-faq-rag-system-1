"""This file undoes tricks like leetspeak and hidden characters used to sneak past filters."""

from __future__ import annotations
import re
import unicodedata
from ..text import _ZERO_WIDTH_RE


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE | re.VERBOSE)


_LEET = str.maketrans(
    {
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "@": "a",
        "$": "s",
        "!": "i",
        "|": "l",
    }
)


_B64_RE = re.compile(r"\b[A-Za-z0-9+/]{24,}={0,2}\b")


def deobfuscate(text: str) -> str:
    """Collapse the tricks used to sneak blocked intent past a naive matcher."""
    s = unicodedata.normalize("NFKD", text or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = _ZERO_WIDTH_RE.sub("", s).lower()
    s = s.translate(_LEET)

    extra = []
    for m in _B64_RE.finditer(text or ""):
        try:
            import base64

            decoded = base64.b64decode(m.group(0) + "==", validate=False).decode("utf-8", "ignore")
            if decoded and sum(c.isprintable() for c in decoded) / len(decoded) > 0.8:
                extra.append(decoded.lower())
        except Exception:
            pass

    # De-pad "i.g.n.o.r.e" and "i g n o r e" back into words
    depadded = re.sub(r"(?<=\b\w)[\s._\-*](?=\w\b)", "", s)
    depadded = re.sub(r"(?<=\w)[\s._\-*](?=\w\b)", "", depadded)
    collapsed = re.sub(r"[^a-z0-9]+", "", s)
    return " ".join([s, depadded, collapsed, *extra])
