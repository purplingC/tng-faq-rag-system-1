"""This file contains the vector and BM25 indexes, and how they are saved to disk."""

from __future__ import annotations
import array
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from .constants import VECTORS_BIN, VECTORS_META
from .deps import OPT, np
from .logging_utils import LOG
from .text import DOMAIN_STOPWORDS, expand_query_terms, tokenize


class VectorIndex:
    """Cosine-similarity index."""

    def __init__(self, dim: int, use_faiss: bool = True) -> None:
        self.dim = dim
        self._ids: list[str] = []
        self._backend = "python"
        self._faiss_index = None
        self._matrix = None
        self._rows: list[list[float]] = []
        if use_faiss and OPT.has("faiss"):
            try:
                self._faiss_index = OPT.faiss.IndexFlatIP(dim)
                self._backend = "faiss"
            except Exception as exc:  # pragma: no cover
                LOG.warning("FAISS init failed (%s); falling back", exc)
        if self._backend == "python" and np is not None:
            self._backend = "numpy"

    @property
    def backend(self) -> str:
        return self._backend

    @staticmethod
    def _l2(vec: Sequence[float]) -> list[float]:
        n = math.sqrt(sum(v * v for v in vec))
        return [v / n for v in vec] if n else list(vec)

    def add(self, ids: Sequence[str], vectors: Sequence[Sequence[float]]) -> None:
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors length mismatch")
        normed = [self._l2(v) for v in vectors]
        self._ids.extend(ids)
        if self._backend == "faiss" and self._faiss_index is not None:
            self._faiss_index.add(np.asarray(normed, dtype="float32"))
        elif self._backend == "numpy":
            arr = np.asarray(normed, dtype="float32")
            self._matrix = arr if self._matrix is None else np.vstack([self._matrix, arr])
        else:
            self._rows.extend(normed)

    def search(self, query: Sequence[float], top_k: int) -> list[tuple[str, float]]:
        if not self._ids:
            return []
        q = self._l2(query)
        k = min(top_k, len(self._ids))
        if self._backend == "faiss" and self._faiss_index is not None:
            scores, idxs = self._faiss_index.search(np.asarray([q], dtype="float32"), k)
            return [
                (self._ids[i], float(s))
                for s, i in zip(scores[0], idxs[0])
                if 0 <= int(i) < len(self._ids)
            ]
        if self._backend == "numpy":
            sims = self._matrix @ np.asarray(q, dtype="float32")
            order = np.argsort(-sims)[:k]
            return [(self._ids[int(i)], float(sims[int(i)])) for i in order]
        sims = [
            (self._ids[i], sum(a * b for a, b in zip(row, q))) for i, row in enumerate(self._rows)
        ]
        sims.sort(key=lambda t: -t[1])
        return sims[:k]

    def vector_for(self, chunk_id: str) -> list[float] | None:
        try:
            pos = self._ids.index(chunk_id)
        except ValueError:
            return None
        if self._backend == "numpy" and self._matrix is not None:
            return [float(x) for x in self._matrix[pos]]
        if self._backend == "faiss" and self._faiss_index is not None:
            return [float(x) for x in self._faiss_index.reconstruct(pos)]
        return self._rows[pos]

    def to_payload(self) -> dict[str, Any]:
        """Serialisable view of the index, whichever backend is in use."""
        rows: list[list[float]]
        if self._backend == "numpy" and self._matrix is not None:
            rows = [[float(x) for x in row] for row in self._matrix]
        elif self._backend == "faiss" and self._faiss_index is not None:
            rows = [
                [float(x) for x in self._faiss_index.reconstruct(i)] for i in range(len(self._ids))
            ]
        else:
            rows = self._rows
        return {"dim": self.dim, "ids": self._ids, "vectors": rows}

    @classmethod
    def from_payload(cls, payload: dict[str, Any], use_faiss: bool = True) -> VectorIndex:
        idx = cls(int(payload["dim"]), use_faiss=use_faiss)
        if payload["ids"]:
            idx.add(payload["ids"], payload["vectors"])
        return idx


class Bm25Index:
    """Okapi BM25, implemented directly - it is 40 lines and avoids a dependency."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_ids: list[str] = []
        self.doc_len: list[int] = []
        self.avgdl = 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

    def build(self, ids: Sequence[str], texts: Sequence[str]) -> None:
        self.doc_ids = list(ids)
        self.postings = defaultdict(list)
        self.doc_len = []
        df: Counter = Counter()
        for i, text in enumerate(texts):
            toks = [t for t in tokenize(text) if t not in DOMAIN_STOPWORDS]
            self.doc_len.append(len(toks))
            tf = Counter(toks)
            for term, count in tf.items():
                self.postings[term].append((i, count))
            df.update(tf.keys())
        n = max(1, len(texts))
        self.avgdl = sum(self.doc_len) / n
        self.idf = {t: math.log(1.0 + (n - c + 0.5) / (c + 0.5)) for t, c in df.items()}

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        terms = [t for t in expand_query_terms(query) if t not in DOMAIN_STOPWORDS]
        if not terms or not self.doc_ids:
            return []
        scores: dict[int, float] = defaultdict(float)
        seen_terms: Counter = Counter(terms)
        for term, qtf in seen_terms.items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            # Damp alias duplicates so a synonym cannot outweigh the user's wording
            qweight = 1.0 if qtf == 1 else 1.0 + 0.25 * math.log(qtf)
            for doc_i, tf in postings:
                dl = self.doc_len[doc_i] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_i] += qweight * idf * (tf * (self.k1 + 1)) / denom
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]
        return [(self.doc_ids[i], float(s)) for i, s in ranked]

    def to_payload(self) -> dict[str, Any]:
        return {
            "k1": self.k1,
            "b": self.b,
            "doc_ids": self.doc_ids,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "idf": self.idf,
            "postings": dict(self.postings.items()),
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Bm25Index:
        idx = cls(float(payload["k1"]), float(payload["b"]))
        idx.doc_ids = payload["doc_ids"]
        idx.doc_len = payload["doc_len"]
        idx.avgdl = float(payload["avgdl"])
        idx.idf = payload["idf"]
        idx.postings = defaultdict(
            list, {t: [tuple(x) for x in p] for t, p in payload["postings"].items()}
        )
        return idx


def save_vectors(path: Path, index: VectorIndex) -> None:

    payload = index.to_payload()
    flat = array.array("f")
    for row in payload["vectors"]:
        flat.extend(row)
    with open(path / VECTORS_BIN, "wb") as fh:
        flat.tofile(fh)
    with open(path / VECTORS_META, "w", encoding="utf-8") as fh:
        json.dump({"dim": payload["dim"], "ids": payload["ids"]}, fh)


def load_vectors(path: Path, use_faiss: bool) -> VectorIndex | None:

    meta_file, bin_file = path / VECTORS_META, path / VECTORS_BIN
    if not (meta_file.exists() and bin_file.exists()):
        return None
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    dim, ids = int(meta["dim"]), meta["ids"]
    flat = array.array("f")
    with open(bin_file, "rb") as fh:
        flat.fromfile(fh, dim * len(ids))
    rows = [list(flat[i * dim : (i + 1) * dim]) for i in range(len(ids))]
    idx = VectorIndex(dim, use_faiss=use_faiss)
    if ids:
        idx.add(ids, rows)
    return idx
