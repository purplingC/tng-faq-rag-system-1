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

---

## Known limitations

* **The zero-dependency embedder matches words, not meanings.** It handles FAQ
  paraphrase well, but *"renew road tax for more than one car"* loses to an
  insurance article because the incidental word "car" gives more token overlap.
  Installing `sentence-transformers` replaces it transparently.
* **The extractive generator cannot synthesise across two articles.** That is a
  deliberate accuracy/safety trade: it cannot hallucinate, but it also cannot
  combine two FAQs into one answer. Set `LLM_API_KEY` for abstractive answers —
  the grounding gate still applies.
* **Guardrails are rule-based.** English and Malay patterns cover the highest-risk
  categories — injection, illicit requests and other people's data — and both were
  checked against every real FAQ question for false positives. A fine-tuned safety
  classifier would still raise recall on novel paraphrased attacks.
* **Malay safety copy has not been reviewed by a native speaker.** Refusals and
  abstentions follow the question's language, and the crisis helpline numbers are
  unchanged, but the Malay wording should be checked before this is used publicly.
* **Mixed Malay and English questions are fragile.** Malaysians write both at once.
  *"teach me how to use sos balance"* is answered, while *"**ajar** me how to use sos
  balance"* is refused: the one unknown word changes which articles rank first. And
  *"boleh tak I check my sos balance"* is answered **in Malay**, because the language
  vote picks one language for a question written in two.
* **Chinese is not supported.** The site has 10 Chinese articles, and the tokenizer
  cannot read Chinese characters. It would need character n-gram tokenising.
* **Malay grounding is not separately measured.** Retrieval and abstention now have
  a Malay golden set (recall@1 1.000, refusals 6/7 without an LLM key), but grounding
  is assumed to behave as it does in English rather than proven to.
* **No conversational memory.** Each question is answered independently;
  multi-turn would need query rewriting plus re-screening of the rewritten query.
* **The golden set is 95 cases** — enough for regression, not for statistically
  strong claims. Reranker weights were tuned on 26 of them; the optimum is a
  broad plateau rather than a knife-edge, but it is still a dev set.
* **Topically-relevant but unanswerable questions need an LLM to catch.**
  Without one configured, *"How do we spell SOS balance?"* returns the SOS
  Balance article. See [the answerability gate](#optional-the-answerability-gate)
  — it is off by default because it requires a network call.
* **Without an LLM, some off-topic brand questions are answered on the full FAQ.**
  When a brand appears in an FAQ question, word overlap cannot tell *"What is CIMB
  bank?"* from *"Who should I contact if I need help with my BizCash application
  from CIMB?"*. The answerability gate refuses both CIMB questions tested. Brands
  mentioned only inside answers, such as *Maybank* and *Shopee*, are refused
  either way.
* **Some questions retrieve a related article instead of the right one.** *"How do
  I top up my eWallet?"* retrieves articles about topping up *with* the eWallet,
  and the gate refuses rather than answering the wrong one. *"What is the loan
  amount for BizCash?"* retrieves the how-to-apply article rather than the one
  whose table lists the amount. Meaning-based embeddings would fix both.
* **Tables with merged cells still read awkwardly.** Rows become sentences, but a
  table using merged or multi-level headers, such as the Tokio Marine premium
  grid, loses which column a number belongs to.
* **One legitimate paraphrase now abstains.** *"am I allowed to buy gold on
  e-Mas"* scores 0.277 against a 0.34 threshold. That is the deliberate cost of
  the conservative threshold described in
  [docs/retrieval.md](docs/retrieval.md#measured-separation) - it fails safely,
  by pointing at the help centre rather than answering wrongly.

Full detail, including what was wrong with the earlier implementation and how it
was proven, is in [docs/original-issues.md](docs/original-issues.md).

---

## What I would do next

In the order I would actually do them, with the reason rather than the buzzword.

1. **Meaning-based retrieval.** Install `sentence-transformers` and make it the default
   when present. It is the single change that fixes the most open problems: the
   *"top up my eWallet"* miss, *"loan amount for BizCash"* picking the wrong article, and
   most of the mixed-language fragility above. Cost: a 2.5 GB download and slower cold start.
2. **Handle code-switching properly.** Score a question against both corpora instead of
   voting for one language, and answer from whichever scores higher. That removes the
   "one Malay word flips the answer" failure without needing a language guess at all.
3. **A Malay golden set for grounding.** Retrieval and abstention are measured; grounding
   is assumed to behave as it does in English. Same method as the retrieval set: the
   parallel corpus makes the expected article unambiguous.
4. **A native speaker reviews the Malay safety copy.** The refusals are mine, and the
   crisis helplines matter too much to ship on my translation alone.
5. **Conversational memory.** Every question is answered alone today. Multi-turn needs the
   follow-up rewritten into a standalone question, then re-screened by the guardrails —
   rewriting is exactly where an injected instruction could sneak back in.
6. **Feed the event log back into the FAQ.** The log already lists the questions that got
   no answer. That list is the highest-value input to both the knowledge base and the
   golden set, and it costs nothing to collect.
7. **Cache the answerability judgements across processes.** The cache is per process
   today, so a restarted API pays for the same judgement again. A small shared cache would
   cut calls well under the free tier's daily limit.
8. **Tables with merged cells.** Rows become sentences now, but a multi-level header such
   as the Tokio Marine premium grid loses which column a number belongs to.
9. **Chinese, if the corpus grows.** It needs character n-gram tokenising, and today the
   site has only 10 Chinese articles — not worth it yet.
10. **A fine-tuned safety classifier** such as Llama Guard, alongside the rules, for novel
    paraphrased attacks. `InputPolicy` already takes a rule list so it can be added as an
    extra layer rather than a rewrite.
