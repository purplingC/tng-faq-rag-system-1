"""This file saves documents and chunks in a SQLite database."""

from __future__ import annotations
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path
from .models import Chunk, FaqDoc
from .text import normalize_question


class MetadataStore:
    """SQLite store for docs and chunks."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS docs (
        doc_id TEXT PRIMARY KEY, question TEXT NOT NULL, question_norm TEXT NOT NULL,
        answer TEXT NOT NULL, url TEXT, category TEXT
    );
    CREATE TABLE IF NOT EXISTS chunks (
        chunk_id TEXT PRIMARY KEY, parent_id TEXT NOT NULL, text TEXT NOT NULL,
        answer_slice TEXT NOT NULL, question TEXT, url TEXT, category TEXT,
        chunk_index INTEGER, n_chunks INTEGER, token_count INTEGER,
        char_start INTEGER, char_end INTEGER,
        FOREIGN KEY(parent_id) REFERENCES docs(doc_id)
    );
    CREATE INDEX IF NOT EXISTS idx_docs_qnorm ON docs(question_norm);
    CREATE INDEX IF NOT EXISTS idx_chunks_parent ON chunks(parent_id);
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self.conn.executescript(self.SCHEMA)

    @property
    def conn(self) -> sqlite3.Connection:
        """A connection owned by the calling thread."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    def replace_all(self, docs: Sequence[FaqDoc], chunks: Sequence[Chunk]) -> None:
        with self._write_lock, self.conn:
            self.conn.execute("DELETE FROM chunks")
            self.conn.execute("DELETE FROM docs")
            self.conn.executemany(
                "INSERT INTO docs(doc_id, question, question_norm, answer, url, category)"
                " VALUES (?,?,?,?,?,?)",
                [
                    (
                        d.doc_id,
                        d.question,
                        normalize_question(d.question),
                        d.answer,
                        d.url,
                        d.category,
                    )
                    for d in docs
                ],
            )
            self.conn.executemany(
                "INSERT INTO chunks(chunk_id, parent_id, text, answer_slice, question, url,"
                " category, chunk_index, n_chunks, token_count, char_start, char_end)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.parent_id,
                        c.text,
                        c.answer_slice,
                        c.question,
                        c.url,
                        c.category,
                        c.chunk_index,
                        c.n_chunks,
                        c.token_count,
                        c.char_span[0],
                        c.char_span[1],
                    )
                    for c in chunks
                ],
            )

    def chunk(self, chunk_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM chunks WHERE chunk_id=?", (chunk_id,))
        return cur.fetchone()

    def chunks(self, chunk_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        qs = ",".join("?" * len(chunk_ids))
        cur = self.conn.execute(f"SELECT * FROM chunks WHERE chunk_id IN ({qs})", tuple(chunk_ids))
        return {r["chunk_id"]: r for r in cur.fetchall()}

    def doc(self, doc_id: str) -> sqlite3.Row | None:
        cur = self.conn.execute("SELECT * FROM docs WHERE doc_id=?", (doc_id,))
        return cur.fetchone()

    def all_docs(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM docs").fetchall()

    def all_chunks(self) -> list[sqlite3.Row]:
        return self.conn.execute("SELECT * FROM chunks ORDER BY chunk_id").fetchall()

    def exact_question(self, question: str) -> sqlite3.Row | None:
        """Exact match on the normalised question on both sides."""
        cur = self.conn.execute(
            "SELECT * FROM docs WHERE question_norm=?", (normalize_question(question),)
        )
        return cur.fetchone()

    def count(self) -> tuple[int, int]:
        d = self.conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
        c = self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        return int(d), int(c)

    def close(self) -> None:
        """Close this thread's connection. Other threads keep their own."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None
