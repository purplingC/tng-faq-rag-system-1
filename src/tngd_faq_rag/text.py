"""This file cleans, tokenises and splits text, and holds small numeric helpers."""

from __future__ import annotations
import hashlib
import html
import math
import re
import unicodedata
from collections.abc import Sequence
from html.parser import HTMLParser
from typing import Any, ClassVar
from .deps import np

# Any run of spaces, tabs or newlines, collapsed to one space
_WS_RE = re.compile(r"\s+")


# One word of letters or digits, keeping contractions like "don't" whole
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


# Sentence break after . ! ? when a capital follows, or on a blank line
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\"'])|\n{2,}|(?<=[.!?])\s*\n")


# A number, with optional RM prefix, comma groups, decimals and percent sign
_NUMERIC_RE = re.compile(r"(?:rm\s?)?\d[\d,]*(?:\.\d+)?\s?%?", re.IGNORECASE)


# Invisible characters, which are used to smuggle text past a filter
_ZERO_WIDTH_RE = re.compile("[​-‏‪-‮⁠-⁤﻿­]")


# Stands in for the dots in "e.g." so they cannot end a sentence
_DOT_SENTINEL = "<!D!>"


# Domain aliases, since the FAQ says reload where users say top up
DOMAIN_ALIASES: dict[str, tuple[str, ...]] = {
    "topup": ("reload",),
    "reload": ("topup", "top up"),
    "ewallet": ("wallet", "tng", "tngd"),
    "wallet": ("ewallet",),
    "tng": ("ewallet", "touch n go", "tngd"),
    "tngd": ("tng", "ewallet"),
    "verify": ("verification", "confirm", "authenticate"),
    "verification": ("verify", "confirm"),
    "confirm": ("verify",),
    "card": ("cards",),
    "refund": ("reimburse", "money back", "reversal"),
    "cancel": ("terminate", "stop"),
    "fee": ("charge", "cost", "fees"),
    "charge": ("fee", "cost"),
    "limit": ("maximum", "cap", "quota"),
    "transfer": ("send money", "duitnow"),
    "toll": ("rfid", "paydirect", "highway"),
    "gold": ("emas", "e-mas"),
    "invest": ("investment", "investing", "buy"),
    "buy": ("purchase", "invest"),
    "allow": ("eligible", "permitted", "qualify"),
    "eligible": ("allowed", "qualify", "entitled"),
    "zakat": ("tithe",),
    "scam": ("fraud", "unauthorised", "unauthorized"),
    "fraud": ("scam", "unauthorised"),
    "unauthorised": ("unauthorized", "fraud", "scam"),
    "unauthorized": ("unauthorised", "fraud", "scam"),
    "login": ("log in", "sign in", "signin"),
    "ev": ("electric vehicle", "charging"),
    "invoice": ("einvoice", "e-invoice", "receipt"),
    "points": ("rewards", "tngd points"),
    "cashback": ("rebate", "money back"),
    "sos": ("emergency",),
    "delivery": ("shipping", "arrive", "arrival"),
}


STOPWORDS = frozenset(
    [
        "a",
        "about",
        "above",
        "after",
        "again",
        "against",
        "all",
        "am",
        "an",
        "and",
        "any",
        "are",
        "aren't",
        "as",
        "at",
        "be",
        "because",
        "been",
        "before",
        "being",
        "below",
        "between",
        "both",
        "but",
        "by",
        "can",
        "cannot",
        "could",
        "couldn't",
        "did",
        "didn't",
        "do",
        "does",
        "doesn't",
        "doing",
        "don't",
        "down",
        "during",
        "each",
        "few",
        "for",
        "from",
        "further",
        "had",
        "hadn't",
        "has",
        "hasn't",
        "have",
        "haven't",
        "having",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "i",
        "if",
        "in",
        "into",
        "is",
        "isn't",
        "it",
        "it's",
        "its",
        "itself",
        "just",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "off",
        "on",
        "once",
        "only",
        "or",
        "other",
        "ought",
        "our",
        "ours",
        "ourselves",
        "out",
        "over",
        "own",
        "same",
        "shan't",
        "she",
        "should",
        "shouldn't",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "until",
        "up",
        "very",
        "was",
        "wasn't",
        "we",
        "were",
        "weren't",
        "while",
        "with",
        "won't",
        "would",
        "wouldn't",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
        "please",
        "kindly",
    ]
)


# Words in nearly every article, so they carry no discriminative signal
DOMAIN_STOPWORDS = frozenset({"tng", "ewallet", "touch", "go", "digital", "tngd"})

# Interrogatives break retrieval in two opposite ways if handled wrongly
# Deleting them makes "Why must I verify" identical to "How soon must I verify"
# Keeping them at full weight lets "What is" alone match an unrelated article
# So they stay as tokens but are down weighted wherever similarity is scored
INTERROGATIVES = frozenset(
    {
        "what",
        "when",
        "where",
        "which",
        "who",
        "whom",
        "whose",
        "why",
        "how",
        "can",
        "could",
        "do",
        "doe",
        "did",
        "is",
        "are",
        "was",
        "were",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "must",
        "need",
        "there",
    }
)

# Terms that must never, on their own, make two questions look alike
LOW_SIGNAL_TERMS = DOMAIN_STOPWORDS | INTERROGATIVES

# Weight a low signal term keeps, weak evidence rather than none at all
LOW_SIGNAL_WEIGHT = 0.15


def salient_terms(terms: Sequence[str]) -> list[str]:
    """The terms that say what a question is about."""
    return [t for t in terms if t not in LOW_SIGNAL_TERMS]


class _HTMLTextExtractor(HTMLParser):
    """Stdlib HTML to text, with each table row as its own sentence."""

    _SKIP: ClassVar[frozenset] = frozenset({"script", "style", "noscript", "head", "meta", "link"})
    _BLOCK: ClassVar[frozenset] = frozenset(
        {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "ul", "ol"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0
        # One entry per open table, so a table nested in a cell renders into that cell
        self._tables: list[dict[str, Any]] = []

    def _cell(self) -> list[str] | None:
        return self._tables[-1]["cell"] if self._tables else None

    def handle_starttag(self, tag: str, attrs) -> None:
        cell = self._cell()
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag == "table":
            self._tables.append({"rows": [], "row": None, "cell": None, "th": False})
        elif self._tables and tag == "tr":
            self._tables[-1]["row"] = []
        elif self._tables and tag in ("td", "th"):
            table = self._tables[-1]
            if table["row"] is None:
                table["row"] = []
            table["cell"], table["th"] = [], tag == "th"
        elif cell is not None and (tag == "li" or tag in self._BLOCK):
            cell.append(" ")
        elif tag == "li":
            self._parts.append("\n- ")
        elif tag in self._BLOCK:
            self._parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif self._tables and tag in ("td", "th"):
            table = self._tables[-1]
            if table["cell"] is not None and table["row"] is not None:
                text = " ".join("".join(table["cell"]).split())
                table["row"].append((text, table["th"]))
            table["cell"] = None
        elif self._tables and tag == "tr":
            table = self._tables[-1]
            if table["row"]:
                table["rows"].append(table["row"])
            table["row"] = None
        elif self._tables and tag == "table":
            rendered = self._render_table(self._tables.pop()["rows"])
            outer = self._cell()
            if outer is not None:
                outer.append(" " + rendered + " ")
            else:
                self._parts.append("\n" + rendered + "\n")
        elif self._cell() is None and tag in self._BLOCK:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        cell = self._cell()
        if cell is not None:
            cell.append(data)
        elif not self._tables or data.strip():
            # Text between cells is only layout whitespace, but a caption is kept
            self._parts.append(data)

    @staticmethod
    def _render_table(rows: list[list[tuple[str, bool]]]) -> str:
        """Turn table rows into one labelled sentence each."""
        rows = [row for row in rows if any(text for text, _ in row)]
        if not rows:
            return ""
        width = max(len(row) for row in rows)
        # A first row of th cells, or any first row of a table 3 or more columns wide, is the header
        has_header = len(rows) > 1 and (all(th for _, th in rows[0]) or width >= 3)
        header = [text for text, _ in rows[0]] if has_header else []
        lines = []
        for row in rows[1:] if has_header else rows:
            cells = [text for text, _ in row]
            if header and len(cells) >= 3:
                label = f"{header[0]} {cells[0]}".strip()
                pairs = [f"{h} {v}".strip() if h else v for h, v in zip(header[1:], cells[1:]) if v]
                pairs += [v for v in cells[len(header) :] if v]
                line = f"{label}: " + "; ".join(pairs)
            elif len(cells) >= 2 and cells[0]:
                line = f"{cells[0]}: " + "; ".join(v for v in cells[1:] if v)
            else:
                line = "; ".join(v for v in cells if v)
            line = line.rstrip(" :;")
            if line and line[-1] not in ".!?":
                line += "."
            if line:
                lines.append(line)
        return "\n".join(lines)

    def text(self) -> str:
        return "".join(self._parts)


def html_to_text(raw: str) -> str:
    """Convert help-centre HTML into clean, readable plain text."""
    if not raw:
        return ""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(raw)
        parser.close()
        out = parser.text()
    except Exception:  # malformed markup -> fall back to tag stripping
        out = re.sub(r"<[^>]+>", " ", raw)
    out = html.unescape(out)
    out = out.replace(" ", " ")
    out = _ZERO_WIDTH_RE.sub("", out)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n\s*\n\s*\n+", "\n\n", out)
    return "\n".join(line.strip() for line in out.split("\n")).strip()


_PUNCT_MAP = {
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "–": "-",
    "—": "-",
    " ": " ",
    "…": "...",
}


def clean_text(s: Any) -> str:
    """Unicode-normalise, unify smart punctuation, collapse whitespace."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _ZERO_WIDTH_RE.sub("", s)
    for src, dst in _PUNCT_MAP.items():
        s = s.replace(src, dst)
    return _WS_RE.sub(" ", s).strip()


def normalize_question(s: str) -> str:
    """Aggressive normalisation used only for exact/fuzzy question matching."""
    s = clean_text(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return _WS_RE.sub(" ", s).strip()


def _light_stem(word: str) -> str:
    """Cheap suffix stripper. Not linguistically perfect, but stable and fast."""
    for suf in (
        "ization",
        "isation",
        "ments",
        "ement",
        "ings",
        "ing",
        "ies",
        "ied",
        "es",
        "ed",
        "s",
    ):
        if len(word) > len(suf) + 2 and word.endswith(suf):
            if suf == "ies":
                return word[: -len(suf)] + "y"
            if suf in ("es", "s") and word.endswith(("ss", "us", "is")):
                return word
            return word[: -len(suf)]
    return word


def tokenize(text: str, *, stem: bool = True, keep_stopwords: bool = False) -> list[str]:
    """Lowercase word tokens, optionally stemmed and stopword-filtered."""
    words = _WORD_RE.findall(clean_text(text).lower())
    out: list[str] = []
    for w in words:
        if not keep_stopwords and (w in STOPWORDS or len(w) == 1):
            continue
        out.append(_light_stem(w) if stem else w)
    return out


def expand_query_terms(text: str) -> list[str]:
    """Query-side alias expansion: adds domain synonyms as extra soft terms."""
    base = tokenize(text)
    extra: list[str] = []
    low = clean_text(text).lower()
    base_set = set(base)
    for key, aliases in DOMAIN_ALIASES.items():
        if _light_stem(key) in base_set or key in low:
            for alias in aliases:
                extra.extend(tokenize(alias))
    return base + extra


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that respects bullet lists and common abbreviations."""
    if not text:
        return []
    protected = text
    for abbr in ("e.g.", "i.e.", "etc.", "vs.", "Mr.", "Ms.", "Dr.", "No.", "approx."):
        protected = protected.replace(abbr, abbr.replace(".", _DOT_SENTINEL))
    pieces: list[str] = []
    for block in protected.split("\n"):
        block = block.strip()
        if not block:
            continue
        if block.startswith("- "):
            pieces.append(block)
            continue
        pieces.extend(p for p in _SENT_SPLIT_RE.split(block) if p and p.strip())
    return [
        p.replace(_DOT_SENTINEL, ".").strip()
        for p in pieces
        if p.replace(_DOT_SENTINEL, ".").strip()
    ]


def _token_estimate(text: str) -> float:
    """Fractional token estimate."""
    if not text:
        return 0.0
    words = text.split()
    punct = sum(text.count(c) for c in ".,;:!?()[]{}\"'/-")
    return len(words) * 1.3 + punct / 2.0


def approx_token_count(text: str) -> int:
    """Model-agnostic token estimate (~1.3 tokens per word plus punctuation)."""
    return int(_token_estimate(text))


def extract_numbers(text: str) -> list[str]:
    """Numbers/percentages/amounts, normalised for the numeric grounding check."""
    out = []
    for m in _NUMERIC_RE.finditer(text or ""):
        tok = m.group(0).lower().replace(",", "").replace(" ", "")
        if tok.startswith("rm"):
            tok = tok[2:]
        tok = tok.rstrip(".")
        if tok and any(ch.isdigit() for ch in tok):
            out.append(tok)
    return out


def sha1(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode("utf-8", "ignore"))
        h.update(b"\x00")
    return h.hexdigest()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity that works with or without NumPy."""
    if np is not None:
        va, vb = np.asarray(a, dtype="float32"), np.asarray(b, dtype="float32")
        na, nb = float(np.linalg.norm(va)), float(np.linalg.norm(vb))
        return 0.0 if na == 0 or nb == 0 else float(va @ vb / (na * nb))
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 60.0)))
    e = math.exp(max(x, -60.0))
    return e / (1.0 + e)


def luhn_valid(digits: str) -> bool:
    """Luhn checksum - the difference between 'a card number' and 'any 16 digits'."""
    d = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(d) <= 19:
        return False
    total, parity = 0, len(d) % 2
    for i, n in enumerate(d):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0
