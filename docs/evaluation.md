# Evaluation

## What is measured

`tngd-faq-rag eval` scores a 72-case golden set (`src/tngd_faq_rag/golden_set.py`):

| Group | n | Asks |
| --- | --- | --- |
| Retrieval | 26 | Does a paraphrase find the right article? (recall@1, recall@k, MRR) |
| Abstention | 13 | Are out-of-scope questions refused, including named entities that sit next to the corpus (*CIMB*, *Maybank*, *bitcoin*)? |
| Adversarial | 19 | Are attacks blocked, with the right category? |
| False positives | 14 | Are ordinary support questions left alone? |
| KB self-censorship | all docs | Does the system ever suppress its own verified answers? |

The last two groups exist because the two failure modes are symmetric. A system
that blocks everything scores perfectly on "adversarial" and is useless.

## Which corpus

By default the golden set is scored against the **packaged 30-entry seed
corpus**, not whatever knowledge base happens to be active.

This is deliberate. Scored against a freshly scraped KB, the numbers would move
whenever someone edits the live help centre — articles get retired, near
duplicates appear — and a genuine regression in this code would be
indistinguishable from an edit on someone else's website. A fixed evaluation
corpus is what makes the number a signal.

`tngd-faq-rag eval --live` scores the active knowledge base when that is what you
want.

## Results — seed corpus

Zero-dependency backends, no models downloaded:

```
Retrieval        n=26   recall@1=0.962   recall@4=1.000   MRR=0.981
Abstention       n=13   correct=13/13    rate=1.000
Adversarial      n=19   blocked=19/19    rate=1.000   category_acc=1.000
False positives  n=14   wrongly blocked=0             rate=0.000
KB self-censor   n=30   suppressed answers=0
Latency                 2–5 ms per question, single-threaded, no GPU
```

## Results — full scraped FAQ (2,477 articles, scraped 2026-09-17)

Run with the answerability gate off, so these are retrieval and rules alone:

```
Retrieval        n=26   recall@1=0.769   recall@4=0.885   MRR=0.814
Abstention       n=13   correct=9/13     rate=0.692
Adversarial      n=19   blocked=19/19    rate=1.000   category_acc=1.000
False positives  n=14   wrongly blocked=0             rate=0.000
KB self-censor   n=2477 suppressed answers=0
```

The differences from the seed results are worth stating plainly rather than hiding:

* **2 are the golden set being stale, not the system.** The seed-era article
  *"Can I edit my Near Me review"* no longer exists upstream, and *"How long does
  it take for the Cash Out to be paid"* **does** exist upstream — so answering it
  is correct and the abstention expectation is what is wrong.
* **1 is eval brittleness.** *"am I allowed to buy gold on e-Mas"* returns *"What
  is e-Mas?"*, which answers the question; the golden set names a different valid
  article.
* **The rest are genuine misses without the gate**: the road-tax paraphrase, and
  *CIMB bank*, *Maybank housing loan* and *Grab account* being answered. With the
  gate on, both CIMB questions tested are refused. See
  [retrieval.md](retrieval.md#known-failure-modes).

## Test suite

266 tests in `tests/`, none of which touch the network:

| File | Covers |
| --- | --- |
| `test_text.py` | Normalisation, tokenisation, Luhn, token-estimate additivity |
| `test_chunking.py` | Atomicity, budgets, pathological articles |
| `test_ingestion.py` | Schema coercion, deduplication, validation, the full scraped FAQ file |
| `test_config.py` | Env overrides, index fingerprint invalidation |
| `test_retrieval.py` | Index behaviour, absolute scores, OOV penalty, answer-only damping, MMR, out-of-domain rejection |
| `test_guardrails.py` | Every block case and every allow case, obfuscation |
| `test_grounding.py` | Support, numeric consistency, citations |
| `test_pipeline.py` | Response contract, decisions, attribution, observability |
| `test_robustness.py` | Hostile input, SQL injection, concurrency |
| `test_scraper.py` | Payload mapping against recorded fixtures |
| `test_evaluation.py` | Golden-set thresholds as a regression gate |

Live-network tests are marked and deselected by default: `pytest -m network`.

## Running it

```bash
make eval          # seed corpus (the reproducible number)
make eval-live     # active knowledge base
make test          # pytest
make smoke         # fast end-to-end check, no test framework needed
make check         # lint + types + tests + eval, i.e. what CI runs
```
