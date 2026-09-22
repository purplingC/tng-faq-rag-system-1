"""This file finds candidate chunks with dense, lexical and fuzzy search combined."""

from __future__ import annotations
import difflib
from collections import defaultdict
from collections.abc import Sequence
from .config import Config
from .embeddings import EmbeddingBackend
from .indexes import Bm25Index, VectorIndex
from .models import Candidate
from .reranking import Reranker
from .store import MetadataStore
from .text import normalize_question, tokenize


class HybridRetriever:
    """Dense + BM25 + exact/fuzzy question match, fused with Reciprocal Rank Fusion."""

    def __init__(
        self,
        cfg: Config,
        store: MetadataStore,
        vectors: VectorIndex,
        lexical: Bm25Index,
        embedder: EmbeddingBackend,
        reranker: Reranker,
        question_index: Sequence[tuple[str, str, str]],
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.vectors = vectors
        self.lexical = lexical
        self.embedder = embedder
        self.reranker = reranker
        # Normalised question, doc id and first chunk id for fuzzy matching
        self.question_index = list(question_index)

    # Individual retrievers
    def _dense(self, query: str) -> list[tuple[str, float]]:
        qv = self.embedder.encode_one(query, is_query=True)
        return self.vectors.search(qv, self.cfg.dense_top_k)

    def _lexical(self, query: str) -> list[tuple[str, float]]:
        return self.lexical.search(query, self.cfg.lexical_top_k)

    def _fuzzy_questions(self, query: str, limit: int = 8) -> list[tuple[str, float]]:
        qn = normalize_question(query)
        if not qn:
            return []
        scored: list[tuple[str, float]] = []
        matcher = difflib.SequenceMatcher()
        matcher.set_seq2(qn)
        for question_norm, _doc_id, chunk_id in self.question_index:
            matcher.set_seq1(question_norm)
            # Cheap length gate before the quadratic ratio
            if abs(len(question_norm) - len(qn)) > max(40, len(qn)):
                continue
            if matcher.real_quick_ratio() < 0.5 or matcher.quick_ratio() < 0.5:
                continue
            scored.append((chunk_id, matcher.ratio()))
        scored.sort(key=lambda t: -t[1])
        return scored[:limit]

    # Fusion
    def retrieve(self, query: str) -> list[Candidate]:
        dense = self._dense(query)
        lexical = self._lexical(query)
        fuzzy = self._fuzzy_questions(query)

        k = self.cfg.rrf_k
        rrf: dict[str, float] = defaultdict(float)
        provenance: dict[str, list[str]] = defaultdict(list)
        raw: dict[str, dict[str, float]] = defaultdict(dict)

        for name, ranked, weight in (
            ("dense", dense, 1.0),
            ("lexical", lexical, 1.0),
            ("fuzzy", fuzzy, 1.2),  # a verbatim FAQ question deserves a nudge
        ):
            for rank, (cid, score) in enumerate(ranked, start=1):
                rrf[cid] += weight / (k + rank)
                provenance[cid].append(name)
                raw[cid][name] = float(score)

        if not rrf:
            return []

        top_ids = [cid for cid, _ in sorted(rrf.items(), key=lambda kv: -kv[1])][
            : self.cfg.rerank_candidates
        ]
        rows = self.store.chunks(top_ids)
        candidates: list[Candidate] = []
        for cid in top_ids:
            row = rows.get(cid)
            if row is None:
                continue
            candidates.append(
                Candidate(
                    chunk_id=row["chunk_id"],
                    parent_id=row["parent_id"],
                    text=row["text"],
                    answer_slice=row["answer_slice"],
                    question=row["question"],
                    url=row["url"],
                    category=row["category"],
                    chunk_index=row["chunk_index"],
                    n_chunks=row["n_chunks"],
                    dense_score=raw[cid].get("dense", 0.0),
                    lexical_score=raw[cid].get("lexical", 0.0),
                    fuzzy_score=raw[cid].get("fuzzy", 0.0),
                    rrf_score=rrf[cid],
                    sources=sorted(set(provenance[cid])),
                )
            )

        for c, rel in zip(candidates, self.reranker.score(query, candidates)):
            c.relevance = rel
        candidates.sort(key=lambda c: -c.relevance)
        return self._mmr(query, candidates)

    def _mmr(self, query: str, candidates: Sequence[Candidate]) -> list[Candidate]:
        """Maximal Marginal Relevance, so the sources shown complement each other."""
        if len(candidates) <= 1:
            return list(candidates)
        lam = self.cfg.mmr_lambda
        pool = list(candidates)
        selected: list[Candidate] = [pool.pop(0)]
        token_cache = {c.chunk_id: set(tokenize(c.text)) for c in candidates}
        while pool and len(selected) < self.cfg.final_top_k:
            best: Candidate = pool[0]
            best_score = -1e9
            for cand in pool:
                sim_to_selected = 0.0
                for sel in selected:
                    a, b = token_cache[cand.chunk_id], token_cache[sel.chunk_id]
                    if a and b:
                        sim_to_selected = max(sim_to_selected, len(a & b) / len(a | b))
                    # Same parent doc means the same article, strongly redundant
                    if cand.parent_id == sel.parent_id:
                        sim_to_selected = max(sim_to_selected, 0.85)
                score = lam * cand.relevance - (1 - lam) * sim_to_selected
                if score > best_score:
                    best, best_score = cand, score
            selected.append(best)
            pool.remove(best)
        return selected
