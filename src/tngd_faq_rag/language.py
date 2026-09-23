"""This file guesses whether a question is written in Malay or English."""

from __future__ import annotations
import re

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = ("en", "ms")

# Function words, not topic words, so the guess does not depend on the subject
_MALAY_WORDS = (
    "adakah apa apakah bagaimana bila berapa boleh bolehkah dan dalam dari daripada "
    "dengan di guna ialah ini itu kenapa ke kepada macam mana mengapa menggunakan "
    "nak pada perlu saya semak sudah tak tidak untuk yang anda akaun atau belum "
    "juga kalau lagi saja sahaja seperti tetapi ada akan jika kerana siapa cara "
    "bagi oleh lebih tentang melalui semua"
)
_ENGLISH_WORDS = (
    "a an and are be can do does for from how i if in is it me my of on or should "
    "that the there this to was what when where which why will with you your am at "
    "by get have need"
)
MALAY_MARKERS = frozenset(_MALAY_WORDS.split())
ENGLISH_MARKERS = frozenset(_ENGLISH_WORDS.split())

_WORD_RE = re.compile(r"[a-z']+")


def _is_malay_word(word: str) -> bool:
    # The -kah suffix turns a Malay word into a question: bagaimanakah, berapakah
    return word in MALAY_MARKERS or (len(word) > 5 and word.endswith("kah"))


def detect_language(text: str) -> str:
    """Return 'ms' when the wording looks Malay, otherwise 'en'."""
    words = _WORD_RE.findall((text or "").lower())
    if not words:
        return DEFAULT_LANGUAGE
    malay = sum(_is_malay_word(word) for word in words)
    english = sum(word in ENGLISH_MARKERS for word in words)
    # A tie goes to Malay: on the real corpus that gains Malay titles and costs no English ones
    return "ms" if malay and malay >= english else DEFAULT_LANGUAGE
