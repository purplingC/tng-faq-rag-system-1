# Issues found in the original implementation

This project began as a refactor of an earlier RAG implementation. Each issue
below was reproduced by running that code, not by reading it — several are only
visible numerically. Every one has a regression test in `tests/`.

## 1. Score normalisation made abstention unreachable

The reranker min-max normalised scores across candidates, which maps the best
candidate to `1.0` unconditionally. Feeding it CrossEncoder logits for three
irrelevant chunks:

```
logits [-10.9, -11.0, -11.1]  ->  fused [1.0, 0.5, 0.0]
```

The top score is `1.0` whether the match is perfect or hopeless, so the
`MIN_FUSED_SCORE = 0.12` gate could never fire and the system could not refuse
anything.

**Fix:** absolute, calibrated relevance; no cross-candidate normalisation.
`tests/test_retrieval.py::TestCalibratedScores`.

## 2. The echo detector discarded every correct answer

Output was rejected as an "echo" when more than 50% of its tokens appeared in
the prompt — but the prompt *contains the sources*. A correctly grounded answer
measured **0.86 overlap**, so it was discarded, retried three times with
sampling, and finally replaced by the fallback message.

**Fix:** compare generated text against the *instruction* text only, never
against the sources.

## 3. The system censored its own knowledge base

`moderation_score` — an input-moderation function — was run over retrieved FAQ
text. **6 of 30** verified answers scored ≥ 0.5 and were replaced with "Response
blocked by safety guardrails", including the card-verification article that the
original README screenshots as its showcase.

**Fix:** separate input and output policies.
`tests/test_guardrails.py::TestOutputPolicy::test_never_suppresses_verified_kb_text`.

## 4. Guardrails were wrong in both directions

| | Original | Now |
| --- | --- | --- |
| Injection attacks blocked | 1 / 8 | 23 / 23 |
| Ordinary support questions wrongly blocked | 5 / 13 | 0 / 18 |

Blocked by the original: *"Who is eligible for TNG eWallet SOS Balance?"*,
*"My card number changed…"*, *"how do I kill the app process"*, *"Is it illegal
to…"*, *"What are my account details?"*.

**Fix:** intent-scoped rules with explicit exemptions. [guardrails.md](guardrails.md).

## 5. The exact-match shortcut fired on fragments

Substring matching meant the single-word query `"limit"` or `"user"` bypassed
generation entirely and returned a full FAQ answer verbatim.

**Fix:** normalised comparison with a similarity ratio threshold.

## 6. A relevance model used as a safety classifier

`ms-marco-MiniLM` emits an unbounded relevance logit (≈ −11…+11). It was
compared against `threshold = 0.55` as though it were a probability, so the
guardrail accepted essentially everything.

**Fix:** `sigmoid()` calibration, and the cross-encoder is used for reranking
only — safety is a separate policy layer.

## 7. Prompt truncation removed the question

Sources were emitted before the question. When the prompt exceeded the encoder
limit, truncation removed the question *and* the safety rules, leaving the model
answering a prompt it had never seen.

**Fix:** `PromptBuilder` emits instructions and the question first, then packs
sources into the remaining budget.

## 8. Import-time construction

The pipeline module built an embedder, a FAISS index, a cross-encoder and an LLM
at import time, so merely importing it crashed when no index existed, and
importing it for a unit test paid for four model loads.

**Fix:** lazy singleton (`get_system`), and dependency probing that does not
import.

## 9. Smaller defects

* `requirements.txt` was UTF-16 encoded — `pip install -r` chokes on it.
* `generator.py` computed `answer_part` and then returned `candidate`; the
  "Sources:" stripping was dead code.
* `generate()` was annotated `-> str` but returned a `dict` on the fallback path.
* The README documented `index/build_index.py`, which did not exist.
* `exact_match` compared a whitespace-collapsed query against a merely trimmed
  column, so any internal spacing difference missed.

## Defects introduced during the rewrite, and fixed

Found by testing against real scraped articles and by the concurrency test —
worth recording because they are the ones reading the code would not reveal.

* **Infinite loop in the sentence packer** when a carried overlap sentence plus
  the next sentence exceeded the budget. Synthetic tests passed; real articles
  hung.
* **Non-additive token estimates** — per-sentence rounding drifted ~0.9 tokens
  each, overshooting the chunk budget by hundreds of tokens on a 400-clause
  article.
* **`torch` imported eagerly**, costing ~10 s of startup for a dependency the
  default configuration never touches.
* **Interrogatives treated as stopwords**, making *"Why must I verify my saved
  cards"* and *"How soon must I verify my saved cards"* identical to the matcher;
  retrieval returned the wrong one of the two.
* **One SQLite connection shared across threads**, which raises "bad parameter
  or other API misuse" under the threaded web UI. Now thread-local connections
  with WAL.
* **Attribution followed retrieval rank rather than the grounding citations**, so
  the cited URL could be an article that did not contain the sentence served.
* **Interrogatives at full weight let a shared question prefix carry a match.**
  Two earlier fixes collided: promoting `what`/`how` from stopwords to content
  tokens (needed to tell *"Why must I verify"* from *"How soon must I verify"*)
  also let `what` satisfy the token-agreement gate that was supposed to stop
  character similarity answering out-of-domain questions. *"What is CIMB bank"*
  scored **0.43** against *"What is CardMatch?"* and was answered.
  Fixed by keeping interrogatives as tokens but discounting them to 15%
  wherever similarity is scored, weighting F1 by IDF, comparing salient text
  only, and scoring question KIND separately from question SUBJECT.
  Result: 0.43 -> 0.17, and the why/how distinction is preserved.
