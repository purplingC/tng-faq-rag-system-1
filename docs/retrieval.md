# Retrieval

## Three retrievers, fused

| Retriever | What it is for |
| --- | --- |
| **Dense** | Cosine over normalised embeddings. Catches paraphrase: *"Explain SOS balance"* → *"What is TNG eWallet SOS Balance?"* |
| **BM25** (Okapi, k1=1.5, b=0.75) | Catches rare, high-signal tokens that embedding models smear into their neighbourhoods: *DuitNow*, *RFID*, *e-Mas*, *PayDirect*, *NSRC* |
| **Exact / fuzzy question match** | An FAQ bot's most common query shape is the FAQ question itself. Normalised comparison plus `difflib` ratio |

Fusion is **Reciprocal Rank Fusion** (k=60), not a weighted sum of raw scores.
The three retrievers live on incompatible scales — cosine in `[-1, 1]`, BM25
unbounded, `difflib` ratio in `[0, 1]` — and RRF consumes only ranks, so nothing
needs calibrating and adding a fourth retriever cannot blow up the scale.

Then a reranker scores `(question, chunk)` pairs, and **MMR** (λ=0.7) removes
near-duplicates so the generator sees complementary evidence rather than the
same paragraph three times.

## Calibrated confidence — the important part

Relevance is an **absolute** value in `[0, 1]`:

* **CrossEncoder path** — `sigmoid(logit)`, a genuine probability. The raw
  ms-marco logit is unbounded (roughly −11…+11); comparing it directly against a
  0–1 threshold is a category error that accepts everything.
* **Fallback path** — `0.35·coverage + 0.45·q_sim + 0.20·a_sim`, scaled by an
  out-of-vocabulary factor.

Scores are deliberately **not** min-max normalised across candidates. Min-max
always maps the best candidate to 1.0, which erases the difference between *a
great match* and *the least bad of five terrible matches*, and makes every
abstention threshold unreachable.

### The three signals

**`coverage`** — IDF-weighted recall of the query's content terms by the chunk.
Independent of the other candidates, which is precisely why it can support a
threshold.

**`q_sim`** — question-to-question similarity, weighted highest because an FAQ
bot is fundamentally matching the user's question to a catalogued one. Three
things keep it honest:

*Salient terms only.* Interrogatives and modals (`what`, `how`, `is`, `can`) are
kept as tokens — deleting them made *"**Why** must I verify my saved cards"*
identical to *"**How soon** must I verify my saved cards"* — but discounted to
15% wherever similarity is scored. At full weight they carry a match on their
own: *"What is CIMB bank"* scored **0.43** against *"What is CardMatch?"* purely
because both begin "what is".

*IDF-weighted F1.* Unweighted F1 counts every shared token equally, so "what"
was worth as much as "cimb". Weighting by IDF makes agreement on a rare term
worth far more than agreement on a common one.

*Character similarity gated behind token agreement, and computed on salient
text*:

```python
q_sim = max(tok_f1, char_sim * min(1.0, tok_f1 / 0.25))
q_sim *= 0.90 + 0.10 * intent_agreement   # same KIND of question?
```

`difflib` scores *"How do I bake a chocolate cake?"* against *"How do I make a
claim?"* at **0.71** on the shared prefix alone, while the two share no content
word whatsoever. Question KIND is scored separately from question SUBJECT, so a
shared "what is" can never imply a shared topic.

**`a_sim`** — similarity between the question and the answer text.

### Out-of-vocabulary penalty

```python
oov_factor = 0.55 + 0.45 * (known_idf_mass / total_idf_mass)   # salient terms only
```

A query term the corpus has never seen — *"cimb"*, *"durian"* — contributes
nothing to coverage, so coverage silently rescales to the terms that *did* match
and the system reports high confidence about a question it cannot answer.
Unmatched terms are evidence of being out of domain, so they cost something.

Two details matter:

* It is weighted by **IDF mass over salient terms**, not a count of all terms.
  An unknown proper noun means the question is about something absent from the
  corpus; an unknown verb (*"explain"*, *"spell"*) means nothing at all. Counting
  terms equally cannot tell those apart and penalises legitimate paraphrase.
* A term counts as known if **it or any of its domain aliases** is in the
  vocabulary. *"allowed"* and *"buy"* appear nowhere in the corpus, but map to
  *"eligible"* and *"invest"*, which do — so *"am I allowed to buy gold on
  e-Mas"* is not mistaken for an out-of-domain question. *"CIMB"* has no such
  bridge.

The factor scales every candidate equally: it changes confidence, never ranking.

### Measured separation

On the seed corpus, including named entities that sit in the same semantic
neighbourhood as the corpus (banks, Malaysian brands, money):

| | confidence |
| --- | --- |
| Out of domain (*CIMB bank*, *Maybank*, *bitcoin*, *durian*, *bake a cake*, greetings) | 0.00 – 0.22 |
| In domain (paraphrase through verbatim) | 0.28 – 1.00 |

`abstain_threshold` sits at 0.34. It is deliberately on the conservative side of
the gap rather than at its midpoint: for a payments FAQ, refusing a question you
could have answered costs a user one click to the help centre, while answering a
question you should have refused is actively misleading. Raise recall by setting
`TNGD_ABSTAIN_THRESHOLD=0.25`, at the cost of that margin.

### Answer-only matches

An FAQ bot is mostly matching the user's question to a catalogued question. When
a candidate shares **no** word with its FAQ question, even through aliases, the
match rests on the answer text alone, so its relevance is multiplied by 0.8
(`LexicalSemanticReranker.NO_QUESTION_OVERLAP_FACTOR`).

This targets brands mentioned only inside an answer. Measured on the full FAQ:

| | before | after |
| --- | --- | --- |
| *What is Maybank?* | 0.34, answered | 0.27, refused |
| *What is Shopee?* | 0.36, answered | 0.29, refused |
| Seed golden set | recall@1 0.962, 13/13 refused | unchanged |
| Full-FAQ golden set | retrieval 0.769 / 0.885, 8/13 refused | retrieval unchanged, 9/13 refused |

It cannot help when the brand is in a question title. *"What is CIMB bank?"*
scores 0.52 against *"Who should I contact if I need help with my BizCash
application from CIMB?"*, inside the range of genuine paraphrases such as
*"I forgot my PIN"* (0.66). The answerability gate is what refuses it.

## Two languages, two indexes

A Malay question is answered from the Malay corpus, an English one from the English
corpus. Each language has its own index, because merging the corpora would change the
IDF statistics that every threshold here was measured against.

`language.py` guesses the language from function words (*apakah, bagaimana, boleh,
saya*) plus the `-kah` question suffix, never from topic words. On the real corpora it
reads 2,476/2,477 English titles and 1,955/1,975 Malay titles correctly. A tie goes to
Malay: measured, that gains 4 Malay titles and costs no English ones.

Asking a Malay article its own question returns that article for 0.987 of a
150-article sample. Of the 13 apparent misses, 12 were the same question text
published under more than one article id, so the answer returned was identical.

When a Malay question scores below the abstain threshold, the English corpus is tried
as a fallback, since many answers exist only in English.

## Reranker weights

`0.35 / 0.45 / 0.20` was grid-searched over the 26 retrieval cases in the golden
set. Several neighbouring settings score identically, so the optimum is a broad
plateau rather than a knife-edge — which is what makes it a defensible choice
rather than a fit to 26 examples. Re-run the search if the knowledge base
changes substantially.

## Known failure modes

Both stem from the same root cause — the zero-dependency embedder matches words,
not meanings — and both are resolved by installing `sentence-transformers`:

* *"can I renew road tax for more than one car"* returns *"Can I renew my car
  insurance without Road Tax?"*, because the incidental word "car" gives it more
  token overlap than the correct article.
* *"What is the interest rate on a Maybank housing loan?"* answers from TNG's own
  BizCash article at confidence 0.47 on the full FAQ instead of refusing; every term in the query
  exists somewhere in the corpus, so the OOV signal never fires.
