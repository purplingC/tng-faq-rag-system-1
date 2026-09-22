"""This file runs the full RAG pipeline, from question to checked answer."""

from __future__ import annotations
import difflib
import hashlib
import json
import math
import os
import threading
import time
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from .chunking import chunk_documents
from .config import Config
from .constants import (
    FALLBACK_ANSWER,
    LEXICAL_JSON,
    MANIFEST_NAME,
    SQLITE_NAME,
    TNGD_FAQ_URL,
)
from .deps import OPT
from .embeddings import build_embedder
from .generation import ExtractiveComposer, PromptBuilder, build_generator
from .generation.base import Generator
from .guardrails import (
    AnswerabilityGrader,
    AnswerabilityResult,
    GroundingChecker,
    InputPolicy,
    OutputPolicy,
    build_answerability_grader,
    refusal_message,
)
from .indexes import Bm25Index, VectorIndex, load_vectors, save_vectors
from .ingestion import load_documents
from .logging_utils import LOG
from .models import Candidate, FaqDoc, Verdict
from .reranking import CrossEncoderReranker, LexicalSemanticReranker, Reranker
from .retrieval import HybridRetriever
from .store import MetadataStore
from .text import clean_text, normalize_question, tokenize
from .version import __version__

__all__ = ["RagSystem", "ask_tngd_bot", "build_system", "get_system"]


class RagSystem:
    """Everything wired together. Construct via `build_system()`, not directly."""

    def __init__(
        self,
        cfg: Config,
        store: MetadataStore,
        retriever: HybridRetriever,
        generator: Generator,
        input_policy: InputPolicy,
        output_policy: OutputPolicy,
        grounding: GroundingChecker,
        answerability: AnswerabilityGrader,
        ingestion_report: dict[str, Any],
    ) -> None:
        self.cfg = cfg
        self.store = store
        self.retriever = retriever
        self.generator = generator
        self.input_policy = input_policy
        self.output_policy = output_policy
        self.grounding = grounding
        self.answerability = answerability
        self.ingestion_report = ingestion_report
        self._lock = threading.Lock()

    # Introspection
    def backends(self) -> dict[str, str]:
        n_docs, n_chunks = self.store.count()
        return {
            "embedder": self.retriever.embedder.name,
            "vector_index": self.retriever.vectors.backend,
            "lexical": "bm25-okapi",
            "reranker": self.retriever.reranker.name,
            "generator": self.generator.name,
            "answerability": self.answerability.name,
            "docs": str(n_docs),
            "chunks": str(n_chunks),
            "kb_source": str(self.ingestion_report.get("source", "")),
        }

    # Helpers
    def _fallback(self) -> str:
        return FALLBACK_ANSWER.format(url=TNGD_FAQ_URL)

    def _source_texts(self, candidates: Sequence[Candidate]) -> list[str]:
        """Full parent answers, so grounding judges the same text the answer came from."""
        out = []
        for c in candidates:
            doc = self.store.doc(c.parent_id)
            out.append(doc["answer"] if doc else c.answer_slice)
        return out

    def _is_exact_match(self, question: str, candidate: Candidate) -> bool:
        """True when the user asked a catalogued FAQ question, word for word or nearly."""
        if "exact" in candidate.sources:
            return True
        ratio = difflib.SequenceMatcher(
            None, normalize_question(question), normalize_question(candidate.question)
        ).ratio()
        return ratio >= self.cfg.exact_match_ratio

    def _promote_exact(self, question: str, candidates: list[Candidate]) -> list[Candidate]:
        """If the KB holds this question verbatim, make sure it ranks first."""
        row = self.store.exact_question(question)
        if row is None:
            return candidates
        doc_id = row["doc_id"]
        for i, c in enumerate(candidates):
            if c.parent_id == doc_id:
                if i:
                    candidates.insert(0, candidates.pop(i))
                candidates[0].relevance = max(candidates[0].relevance, 0.99)
                candidates[0].sources = sorted({*candidates[0].sources, "exact"})
                return candidates
        chunk = self.store.conn.execute(
            "SELECT * FROM chunks WHERE parent_id=? ORDER BY chunk_index LIMIT 1", (doc_id,)
        ).fetchone()
        if chunk is None:
            return candidates
        candidates.insert(
            0,
            Candidate(
                chunk_id=chunk["chunk_id"],
                parent_id=chunk["parent_id"],
                text=chunk["text"],
                answer_slice=chunk["answer_slice"],
                question=chunk["question"],
                url=chunk["url"],
                category=chunk["category"],
                chunk_index=chunk["chunk_index"],
                n_chunks=chunk["n_chunks"],
                relevance=0.99,
                sources=["exact"],
            ),
        )
        return candidates[: self.cfg.final_top_k]

    @staticmethod
    def _response(**kw: Any) -> dict[str, Any]:
        """Guarantees the four keys the assignment specifies, in a stable order."""
        base = {
            "question": kw.pop("question", ""),
            "retrieved_chunks": kw.pop("retrieved_chunks", []),
            "final_answer": kw.pop("final_answer", ""),
            "blocked": bool(kw.pop("blocked", False)),
        }
        base.update(kw)
        return base

    # The pipeline
    def ask(self, question: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        trace: list[str] = []
        q = clean_text(question)
        # Cap the input so one request cannot burn unbounded CPU
        if len(q) > self.cfg.max_question_chars:
            q = q[: self.cfg.max_question_chars].rsplit(" ", 1)[0]
            trace.append("input_truncated")

        def finish(**kw: Any) -> dict[str, Any]:
            kw.setdefault("question", q)
            kw.setdefault("backends", self.backends())
            kw["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
            kw["trace"] = trace
            return self._response(**kw)

        # Stage 0 empty input
        if not q:
            trace.append("empty_question")
            return finish(
                final_answer="Please enter a question about the Touch 'n Go eWallet.",
                decision="empty_question",
                url="",
                sources=[],
                confidence=0.0,
                safety={"input": Verdict(reason="empty").as_dict()},
            )

        # Stage 1 input guardrails, before anything expensive
        verdict = self.input_policy.evaluate(q)
        trace.append(f"input_policy:{verdict.action}:{verdict.category or 'clean'}")
        if verdict.blocked:
            LOG.info("BLOCKED [%s] %r (%s)", verdict.category, q[:70], verdict.reason)
            return finish(
                retrieved_chunks=[],
                final_answer=refusal_message(verdict.category),
                blocked=True,
                decision="blocked_input",
                url="",
                sources=[],
                confidence=0.0,
                blocked_reason=verdict.reason,
                blocked_category=verdict.category,
                safety={"input": verdict.as_dict(), "output": None, "grounded": None},
            )

        # Stage 2 hybrid retrieval
        with self._lock:
            candidates = self.retriever.retrieve(q)
            candidates = self._promote_exact(q, candidates)
        candidates = candidates[: self.cfg.final_top_k]
        trace.append(f"retrieved:{len(candidates)}")
        chunk_payload = [c.as_dict() for c in candidates]
        confidence = candidates[0].relevance if candidates else 0.0

        # Stage 3 abstention
        if not candidates or confidence < self.cfg.abstain_threshold:
            trace.append(f"abstain:confidence={confidence:.3f}<{self.cfg.abstain_threshold}")
            LOG.info("NO ANSWER %r (confidence %.3f)", q[:70], confidence)
            return finish(
                retrieved_chunks=chunk_payload,
                final_answer=self._fallback(),
                decision="abstain_low_confidence",
                url=TNGD_FAQ_URL,
                sources=self._sources(candidates),
                confidence=round(confidence, 4),
                safety={"input": verdict.as_dict(), "output": None, "grounded": False},
            )

        # Stage 4 answerability, the only expensive filter so it runs last
        # Term overlap sees topic, not whether the passage answers the question
        source_texts = self._source_texts(candidates)
        if self._is_exact_match(q, candidates[0]):
            # Skip the model only when the user asked a catalogued question word for word
            # A high score alone is not enough, since a near miss can answer the wrong question
            answerability = AnswerabilityResult(
                answerable=True,
                supporting=list(range(len(source_texts))),
                reason="exact FAQ question match, nothing to grade",
                graded=False,
            )
            trace.append("answerability:pass:skipped_exact_match")
        else:
            answerability = self.answerability.grade(q, source_texts)
            trace.append(
                f"answerability:{'pass' if answerability.answerable else 'fail'}"
                f":{'graded' if answerability.graded else 'skipped'}"
            )
        if not answerability.answerable:
            LOG.info("UNANSWERABLE %r (%s)", q[:70], answerability.reason)
            return finish(
                retrieved_chunks=chunk_payload,
                final_answer=self._fallback(),
                decision="abstain_unanswerable",
                url=TNGD_FAQ_URL,
                sources=self._sources(candidates),
                confidence=round(confidence, 4),
                safety={
                    "input": verdict.as_dict(),
                    "output": None,
                    "grounded": False,
                    "answerability": answerability.as_dict(),
                },
            )
        if answerability.graded and answerability.supporting:
            # Keep only the passages the grader endorsed
            kept = [candidates[i] for i in answerability.supporting if i < len(candidates)]
            if kept:
                candidates = kept
                source_texts = self._source_texts(candidates)
                chunk_payload = [c.as_dict() for c in candidates]

        # Stage 5 generation
        generation = self.generator.generate(q, candidates, self.store)
        trace.append(
            f"generate:{generation.backend}:{'abstained' if generation.abstained else 'ok'}"
        )
        if generation.abstained or not generation.text.strip():
            return finish(
                retrieved_chunks=chunk_payload,
                final_answer=self._fallback(),
                decision="abstain_ungrounded",
                url=TNGD_FAQ_URL,
                sources=self._sources(candidates),
                confidence=round(confidence, 4),
                safety={"input": verdict.as_dict(), "output": None, "grounded": False},
            )

        # Stage 6 grounding and citations
        ground = self.grounding.check(generation.text, source_texts)
        trace.append(f"grounding:{ground['score']}:{'pass' if ground['grounded'] else 'fail'}")
        if not ground["grounded"]:
            LOG.info("UNGROUNDED %r (score %.3f)", q[:70], ground["score"])
            return finish(
                retrieved_chunks=chunk_payload,
                final_answer=self._fallback(),
                decision="abstain_ungrounded",
                url=TNGD_FAQ_URL,
                sources=self._sources(candidates),
                confidence=round(confidence, 4),
                safety={
                    "input": verdict.as_dict(),
                    "output": None,
                    "grounded": False,
                    "grounding_score": ground["score"],
                    "dropped_sentences": ground["dropped"],
                },
            )
        answer = " ".join(ground["kept"]) if ground["dropped"] else generation.text

        # Stage 7 output guardrails
        out_verdict = self.output_policy.evaluate(answer, source_texts)
        trace.append(f"output_policy:{out_verdict.action}:{out_verdict.category or 'clean'}")
        if out_verdict.blocked:
            LOG.warning("OUTPUT BLOCKED [%s] for %r", out_verdict.category, q[:70])
            return finish(
                retrieved_chunks=chunk_payload,
                final_answer=refusal_message(out_verdict.category),
                blocked=True,
                decision="blocked_output",
                url="",
                sources=self._sources(candidates),
                confidence=round(confidence, 4),
                blocked_reason=out_verdict.reason,
                blocked_category=out_verdict.category,
                safety={
                    "input": verdict.as_dict(),
                    "output": out_verdict.as_dict(),
                    "grounded": True,
                },
            )

        # Stage 8 attribution, which follows the citations rather than the ranking
        # The top candidate is not always the one the answer was drawn from
        cited_counts = Counter(c["source_index"] for c in ground["citations"])
        if cited_counts:
            primary_index = cited_counts.most_common(1)[0][0]
        elif generation.used_source_indices:
            primary_index = generation.used_source_indices[0]
        else:
            primary_index = 0
        primary_index = max(0, min(primary_index, len(candidates) - 1))
        top = candidates[primary_index]
        cited_indices = sorted(cited_counts) or [primary_index]
        trace.append(f"attribution:source[{primary_index}]")

        if self._is_exact_match(q, top):
            decision = "exact_faq"
        elif confidence >= self.cfg.high_confidence_threshold:
            decision = "high_confidence_faq"
        elif generation.backend == ExtractiveComposer.name:
            decision = "extractive"
        else:
            decision = "generated"
        trace.append(f"decision:{decision}")

        # Say so when a long answer was cut, since the rest is at the source link
        shortened = len(clean_text(generation.raw)) > len(clean_text(generation.text))
        if generation.backend == ExtractiveComposer.name and shortened:
            answer += "\n\n(Shortened. The full answer is at the source link.)"
            trace.append("answer:shortened")

        return finish(
            retrieved_chunks=chunk_payload,
            final_answer=answer,
            blocked=False,
            decision=decision,
            url=top.url,
            sources=self._sources(candidates, cited=cited_indices),
            confidence=round(confidence, 4),
            safety={
                "input": verdict.as_dict(),
                "output": out_verdict.as_dict(),
                "grounded": True,
                "grounding_score": ground["score"],
                "citations": ground["citations"],
                "answerability": answerability.as_dict(),
            },
        )

    def _sources(
        self, candidates: Sequence[Candidate], cited: Sequence[int] | None = None
    ) -> list[dict[str, Any]]:
        cited_set = set(cited or [])
        out = []
        for i, c in enumerate(candidates):
            out.append(
                {
                    "rank": i + 1,
                    "question": c.question,
                    "url": c.url,
                    "category": c.category,
                    "relevance": round(c.relevance, 4),
                    "matched_by": c.sources,
                    "cited": i in cited_set,
                    "snippet": (c.answer_slice[:220] + "...")
                    if len(c.answer_slice) > 220
                    else c.answer_slice,
                }
            )
        # Cited sources first since the list is read top down
        out.sort(key=lambda s: (not s["cited"], s["rank"]))
        return out


def _kb_hash(docs: Sequence[FaqDoc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda x: x.doc_id):
        h.update(d.doc_id.encode())
        h.update(d.answer.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _term_statistics(chunk_texts: Sequence[str]) -> tuple[dict[str, float], float]:
    """Word-level IDF shared by the reranker, grounding checker and composer."""
    df: Counter = Counter()
    for text in chunk_texts:
        df.update(set(tokenize(text)))
    n = max(1, len(chunk_texts))
    idf = {t: math.log((n + 1) / (c + 0.5)) + 1.0 for t, c in df.items()}
    return idf, math.log((n + 1) / 0.5) + 1.0


def build_system(
    cfg: Config | None = None,
    *,
    kb_path: Path | None = None,
    force_rebuild: bool = False,
    use_seed: bool = False,
) -> RagSystem:
    """Build (or load) the full system. Safe to call repeatedly."""
    cfg = cfg or Config()
    cfg.index_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.index_dir / MANIFEST_NAME

    docs, report = load_documents(cfg, kb_path, use_seed=use_seed)
    kb_hash = _kb_hash(docs)
    embedder = build_embedder(cfg)

    manifest = {}
    if manifest_path.exists() and not force_rebuild:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            manifest = {}

    fresh = (
        manifest.get("kb_hash") == kb_hash
        and manifest.get("config") == cfg.fingerprint()
        and manifest.get("embedder") == embedder.name
        and manifest.get("version") == __version__
    )

    store = MetadataStore(cfg.index_dir / SQLITE_NAME)
    vectors: VectorIndex | None = None
    lexical: Bm25Index | None = None

    if fresh:
        try:
            vectors = load_vectors(cfg.index_dir, use_faiss=True)
            lexical_payload = json.loads((cfg.index_dir / LEXICAL_JSON).read_text(encoding="utf-8"))
            lexical = Bm25Index.from_payload(lexical_payload)
            if store.count()[1] == 0:
                vectors, lexical = None, None
        except Exception as exc:
            LOG.warning("Could not load persisted index (%s); rebuilding", exc)
            vectors, lexical = None, None

    chunk_rows = store.all_chunks() if (vectors and lexical) else []
    if vectors is None or lexical is None or not chunk_rows:
        LOG.info("Building index from scratch ...")
        chunks = chunk_documents(docs, cfg)
        store.replace_all(docs, chunks)
        texts = [c.text for c in chunks]
        embedder.fit(texts)
        t0 = time.perf_counter()
        vecs = embedder.encode(texts)
        LOG.info("Embedded %d chunks in %.2fs", len(texts), time.perf_counter() - t0)
        vectors = VectorIndex(embedder.dim, use_faiss=True)
        vectors.add([c.chunk_id for c in chunks], vecs)
        lexical = Bm25Index()
        lexical.build([c.chunk_id for c in chunks], texts)
        save_vectors(cfg.index_dir, vectors)
        (cfg.index_dir / LEXICAL_JSON).write_text(
            json.dumps(lexical.to_payload()), encoding="utf-8"
        )
        manifest_path.write_text(
            json.dumps(
                {
                    "kb_hash": kb_hash,
                    "config": cfg.fingerprint(),
                    "embedder": embedder.name,
                    "version": __version__,
                    "docs": len(docs),
                    "chunks": len(chunks),
                    "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        chunk_texts = texts
    else:
        LOG.info("Loaded persisted index (%d chunks)", len(chunk_rows))
        chunk_texts = [r["text"] for r in chunk_rows]
        # Refit IDF from SQLite so queries encode exactly as the documents did
        embedder.fit(chunk_texts)

    idf, default_idf = _term_statistics(chunk_texts)

    reranker: Reranker
    if OPT.has("sentence_transformers") and os.environ.get("TNGD_NO_CROSS_ENCODER", "") == "":
        try:
            reranker = CrossEncoderReranker()
            LOG.info("Reranker: cross-encoder (sigmoid-calibrated)")
        except Exception as exc:
            LOG.warning("CrossEncoder unavailable (%s); using lexical-semantic reranker", exc)
            reranker = LexicalSemanticReranker(idf, default_idf)
    else:
        reranker = LexicalSemanticReranker(idf, default_idf)
        LOG.info("Reranker: lexical-semantic (zero-dependency, absolute scores)")

    question_index = [
        (normalize_question(r["question"]), r["parent_id"], r["chunk_id"])
        for r in store.conn.execute(
            "SELECT chunk_id, parent_id, question FROM chunks WHERE chunk_index=0"
        ).fetchall()
    ]

    retriever = HybridRetriever(cfg, store, vectors, lexical, embedder, reranker, question_index)
    canary = "CANARY-" + hashlib.sha1(os.urandom(16)).hexdigest()[:12].upper()
    prompt_builder = PromptBuilder(canary)
    generator = build_generator(cfg, idf, default_idf, prompt_builder)
    grounding = GroundingChecker(idf, default_idf, cfg.grounding_threshold)
    answerability = build_answerability_grader(cfg)

    return RagSystem(
        cfg=cfg,
        store=store,
        retriever=retriever,
        generator=generator,
        input_policy=InputPolicy(),
        output_policy=OutputPolicy(canary),
        grounding=grounding,
        answerability=answerability,
        ingestion_report=report,
    )


_SYSTEM: RagSystem | None = None

_SYSTEM_LOCK = threading.Lock()


def get_system(cfg: Config | None = None, force_rebuild: bool = False) -> RagSystem:
    """Lazily build the singleton."""
    global _SYSTEM
    with _SYSTEM_LOCK:
        if _SYSTEM is None or force_rebuild:
            _SYSTEM = build_system(cfg, force_rebuild=force_rebuild)
        return _SYSTEM


def ask_tngd_bot(question: str) -> dict:
    """Answer a question about the Touch 'n Go eWallet from verified FAQ content."""
    return get_system().ask(question)
