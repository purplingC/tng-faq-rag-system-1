# Evaluation

## What is measured

`tngd-faq-rag eval` scores a 95-case golden set (`src/tngd_faq_rag/golden_set.py`):

| Group | n | Asks |
| --- | --- | --- |
| Retrieval | 26 | Does a paraphrase find the right article? (recall@1, recall@k, MRR) |
| Abstention | 13 | Are out-of-scope questions refused, including named entities that sit next to the corpus (*CIMB*, *Maybank*, *bitcoin*)? |
| Malay | — | Attacks written in Malay are in the adversarial set, and real Malay questions are in the allow set |
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
Adversarial      n=23   blocked=23/23    rate=1.000   category_acc=1.000
False positives  n=18   wrongly blocked=0             rate=0.000
KB self-censor   n=30   suppressed answers=0
Malay            n=8    recall@1=1.000   refused=6/7   rate=0.857
Latency                 median 2 ms, p95 4 ms per question, single-threaded, no GPU
```

## Results — full scraped FAQ (2,477 articles, scraped 2026-09-17)

Latency on this corpus is median 12 ms and p95 108 ms per question, measured over
the 62 golden-set queries on one thread with no GPU.

Run with the answerability gate off, so these are retrieval and rules alone:

```
Retrieval        n=26   recall@1=0.769   recall@4=0.885   MRR=0.814
Abstention       n=13   correct=9/13     rate=0.692
Adversarial      n=23   blocked=23/23    rate=1.000   category_acc=1.000
False positives  n=18   wrongly blocked=0             rate=0.000
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

## Results — Malay

The seed corpus ships in Malay too (29 of the 30 English seed articles have a
translation), so `tngd-faq-rag eval` scores Malay with the same reproducibility as
English:

```
Malay            n=8    recall@1=1.000  refused=6/7  rate=0.857
```

The one case that is answered instead of refused is *"Apakah itu bank CIMB?"*, the
same limitation as English: CIMB appears in the corpus, so word overlap cannot tell a
definition question from an article that merely mentions the bank. **With the
answerability gate on it is refused**, in Malay. The evaluation names the failing case
rather than hiding it, and the pass threshold covers Malay whenever its corpus is loaded.

The wider corpus is measured separately, against the parallel corpus (1,955 of the
1,975 Malay articles are translations of an English one).

| | |
| --- | --- |
| Language guessed correctly | English 2,476/2,477 (**99.96%**), Malay 1,955/1,975 (**98.99%**) |
| Malay retrieval, asking an article its own question | **recall@1 0.987** on a 150-article sample |
| Questions routed to the Malay corpus | 148/150 |
| Real FAQ questions wrongly blocked by the input policy | 0/2,477 English, 0/1,975 Malay |

Of the 13 apparent retrieval misses, 12 were the same question text published under
more than one article id, so the answer returned was identical. The remaining one,
*"Lupa PIN 6 Digit"*, is a label rather than a question.

Malay attacks sit in the adversarial set and real Malay questions in the allow set, so
`tngd-faq-rag eval` covers both languages. Abstention and grounding have **no Malay
golden set**: they are assumed to behave as they do in English, not proven to.

## Test suite

322 tests in `tests/`, none of which touch the network:

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
| `test_scraper.py` | Payload mapping against recorded fixtures, language selection |
| `test_language.py` | The Malay and English language guess |
| `test_malay.py` | Malay routing, refusal language, Malay guardrails and the Malay seed |
| `test_events.py` | Event log redaction, writing and the stats summary |
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
