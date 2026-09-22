"""This file splits FAQ entries into searchable chunks, see docs/chunking.md for why."""

from __future__ import annotations
import re
from collections.abc import Sequence
from .config import Config
from .logging_utils import LOG
from .models import Chunk, FaqDoc
from .text import _token_estimate, approx_token_count, split_sentences


def _split_long_sentence(sentence: str, max_tokens: float) -> list[str]:
    """Break a sentence that alone exceeds the chunk budget."""
    if _token_estimate(sentence) <= max_tokens:
        return [sentence]
    pieces: list[str] = []
    for clause in re.split(r"(?<=[;:])\s+|\s+-\s+", sentence):
        if _token_estimate(clause) <= max_tokens:
            pieces.append(clause)
            continue
        buf: list[str] = []
        for part in re.split(r"(?<=,)\s+", clause):
            if _token_estimate(" ".join([*buf, part])) > max_tokens and buf:
                pieces.append(" ".join(buf))
                buf = []
            buf.append(part)
        if buf:
            pieces.append(" ".join(buf))
    out: list[str] = []
    step = max(20, int(max_tokens / 1.35))
    for piece in pieces:
        if _token_estimate(piece) <= max_tokens:
            out.append(piece)
            continue
        words = piece.split()
        out.extend(" ".join(words[i : i + step]) for i in range(0, len(words), step))
    return [p.strip() for p in out if p.strip()] or [sentence]


def _pack_sentences(sentences: Sequence[str], max_tokens: float, overlap: int) -> list[list[int]]:
    """Greedily pack whole sentences into token-budgeted windows with overlap."""
    windows: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0.0
    i = 0
    guard = 0
    limit = 4 * len(sentences) + 8  # loop cannot outlive the input
    while i < len(sentences):
        guard += 1
        if guard > limit:  # defensive: never spin forever
            LOG.warning("chunk packing guard tripped at sentence %d/%d", i, len(sentences))
            if current:
                windows.append(current)
            windows.extend([j] for j in range(i, len(sentences)))
            return windows
        tok = _token_estimate(sentences[i])
        if current and current_tokens + tok > max_tokens:
            windows.append(current)
            carry = current[-overlap:] if overlap > 0 else []
            carry_tokens = sum(_token_estimate(sentences[j]) for j in carry)
            # Dropping the overlap is the only escape from an unchanged index
            if carry and carry_tokens + tok > max_tokens:
                carry, carry_tokens = [], 0
            current, current_tokens = list(carry), carry_tokens
            if not current and tok > max_tokens:
                # A sentence larger than the budget becomes its own window
                windows.append([i])
                i += 1
            continue
        current.append(i)
        current_tokens += tok
        i += 1
    if current:
        windows.append(current)
    return windows or [[]]


def chunk_documents(docs: Sequence[FaqDoc], cfg: Config) -> list[Chunk]:
    """Turn verified FAQ docs into retrievable chunks."""
    chunks: list[Chunk] = []
    header_budget = 0
    for doc in docs:
        q_tokens = _token_estimate(doc.question)
        body_budget = max(cfg.max_chunk_tokens - q_tokens - 6, 64)
        full_tokens = _token_estimate(doc.answer)

        if full_tokens <= body_budget:
            # Atomic, since the entry is already one semantic unit
            windows_text = [(doc.answer, (0, len(doc.answer)))]
        else:
            # Paragraph first, then sentence packing inside the budget
            sentences: list[str] = []
            spans: list[tuple[int, int]] = []
            cursor = 0
            for para in doc.answer.split("\n"):
                for sent in split_sentences(para):
                    for piece in _split_long_sentence(sent, body_budget):
                        pos = doc.answer.find(piece, cursor)
                        if pos < 0:
                            pos = cursor
                        sentences.append(piece)
                        spans.append((pos, pos + len(piece)))
                        cursor = pos + len(piece)
            if not sentences:
                sentences, spans = [doc.answer], [(0, len(doc.answer))]
            windows = _pack_sentences(sentences, body_budget, cfg.sentence_overlap)
            windows_text = []
            for win in windows:
                if not win:
                    continue
                text = " ".join(sentences[j] for j in win)
                windows_text.append((text, (spans[win[0]][0], spans[win[-1]][1])))

        n = len(windows_text)
        for idx, (body, span) in enumerate(windows_text):
            # Contextual header so chunk 3 of 5 still knows its topic
            text = f"Q: {doc.question}\nA: {body}"
            if n > 1:
                text += f"\n[part {idx + 1} of {n} | {doc.category}]"
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}:{idx}",
                    parent_id=doc.doc_id,
                    text=text,
                    answer_slice=body,
                    question=doc.question,
                    url=doc.url,
                    category=doc.category,
                    chunk_index=idx,
                    n_chunks=n,
                    token_count=approx_token_count(text),
                    char_span=span,
                )
            )
        header_budget += n
    LOG.info(
        "Chunked %d docs into %d chunks (avg %.2f chunks/doc)",
        len(docs),
        len(chunks),
        len(chunks) / max(1, len(docs)),
    )
    return chunks
