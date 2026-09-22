"""This file turns text into vectors with a built-in and an optional neural backend."""

from __future__ import annotations
import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Any
from .config import Config
from .deps import OPT
from .logging_utils import LOG
from .text import clean_text, expand_query_terms, tokenize


class EmbeddingBackend:
    name = "base"
    dim = 0

    def fit(self, corpus: Sequence[str]) -> None:
        """Optional corpus-statistics pass (IDF). No-op for neural backends."""

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        raise NotImplementedError

    def encode_one(self, text: str, *, is_query: bool = False) -> list[float]:
        return self.encode([text], is_query=is_query)[0]

    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:
        pass


class HashedTfidfEmbedder(EmbeddingBackend):
    """Deterministic, dependency-free embedder over hashed word and character n-grams."""

    name = "hashed-tfidf"

    def __init__(self, dim: int = 768, char_n: tuple[int, ...] = (4, 5)) -> None:
        self.dim = dim
        self.char_n = char_n
        self.idf: dict[str, float] = {}
        self._default_idf = 1.0

    def _features(self, text: str, is_query: bool) -> Counter:
        feats: Counter = Counter()
        toks = expand_query_terms(text) if is_query else tokenize(text)
        for t in toks:
            feats[f"w:{t}"] += 1
        for a, b in zip(toks, toks[1:]):
            feats[f"b:{a}_{b}"] += 1
        flat = re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()
        for n in self.char_n:
            for i in range(0, max(0, len(flat) - n + 1)):
                gram = flat[i : i + n]
                if gram.strip():
                    feats[f"c{n}:{gram}"] += 1
        return feats

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        """Signed hashing: the sign halves collision bias (Weinberger et al.)."""
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(h, "big")
        return v % dim, 1.0 if (v >> 63) & 1 else -1.0

    def fit(self, corpus: Sequence[str]) -> None:
        df: Counter = Counter()
        for text in corpus:
            df.update(set(self._features(text, is_query=False)))
        n = max(1, len(corpus))
        self.idf = {f: math.log((n + 1) / (c + 0.5)) + 1.0 for f, c in df.items()}
        self._default_idf = math.log((n + 1) / 0.5) + 1.0
        LOG.debug("HashedTfidfEmbedder fitted: %d features, n=%d", len(self.idf), n)

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            feats = self._features(text, is_query)
            for f, tf in feats.items():
                weight = (1.0 + math.log(tf)) * self.idf.get(f, self._default_idf)
                if f.startswith("w:"):
                    weight *= 1.6  # whole words are the strongest signal
                elif f.startswith("b:"):
                    weight *= 1.3  # bigrams capture multi-word concepts
                else:
                    weight *= 0.55  # char-grams are a robustness backstop
                idx, sign = self._bucket(f, self.dim)
                vec[idx] += sign * weight
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out

    def state(self) -> dict[str, Any]:
        return {"idf": self.idf, "default_idf": self._default_idf, "dim": self.dim}

    def load_state(self, state: dict[str, Any]) -> None:
        self.idf = state.get("idf", {})
        self._default_idf = state.get("default_idf", 1.0)
        self.dim = state.get("dim", self.dim)


class SentenceTransformerEmbedder(EmbeddingBackend):
    """Real dense embeddings when sentence-transformers is installed."""

    name = "sentence-transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer  # pyright: ignore[reportMissingImports]

        self.model = SentenceTransformer(model_name)
        self.dim = int(self.model.get_sentence_embedding_dimension())
        self.model_name = model_name

    def encode(self, texts: Sequence[str], *, is_query: bool = False) -> list[list[float]]:
        vecs = self.model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        return [[float(x) for x in row] for row in vecs]

    def state(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "dim": self.dim}


def build_embedder(cfg: Config) -> EmbeddingBackend:
    if cfg.prefer_sentence_transformers and OPT.has("sentence_transformers"):
        try:
            emb = SentenceTransformerEmbedder(cfg.st_model)
            LOG.info("Embedder: sentence-transformers (%s, dim=%d)", cfg.st_model, emb.dim)
            return emb
        except Exception as exc:  # offline, no cache, etc.
            LOG.warning("sentence-transformers unavailable (%s); using hashed TF-IDF", exc)
    LOG.info("Embedder: hashed TF-IDF (dim=%d, zero-dependency)", cfg.embedding_dim)
    return HashedTfidfEmbedder(dim=cfg.embedding_dim)
