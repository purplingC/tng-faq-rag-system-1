# Chunking

## Strategy

An FAQ is not prose. Each entry is already a self-contained semantic unit, so
the chunker is structure-aware rather than a blind sliding window.

**1. Atomic by default.** One FAQ entry becomes one chunk as long as it fits the
token budget (320 by default). Splitting a 60-word answer in half halves recall
and guarantees truncated answers.

**2. Paragraph, then sentence packing.** Long answers split on paragraph
boundaries first, then pack whole sentences into token-budgeted windows, so a
chunk boundary never lands mid-sentence.

**3. Oversized sentences are the exception.** A sentence that alone exceeds the
budget is broken at clause boundaries (`;` `:` ` - `), then at commas, and only
as a last resort at a word boundary. Real help-centre articles contain 300-token
run-on bullet lines and flattened tables; left whole, every embedding model
silently truncates them and the tail becomes unretrievable.

**4. Contextual chunk headers.** Every chunk is stored as:

```
Q: <the FAQ question>
A: <this slice of the answer>
[part 2 of 4 | Reload Card Verification]
```

This is the highest-leverage detail in the whole chunker. Without the header,
chunk 3 of a long answer loses its subject and becomes unretrievable by any
query that names the topic.

**5. Sentence-level overlap** (one sentence by default) rather than token
overlap, so the seam between chunks never destroys a fact.

**6. Small-to-big / parent linking.** Chunks are what we *retrieve*; the parent
answer is what we *answer from*. Every chunk stores `parent_id`, so once a chunk
wins retrieval we expand back to the complete verified answer. A user never
receives half a verified answer because a boundary fell in the wrong place.

**7. Metadata per chunk:** `parent_id`, `question`, `url`, `category`,
`chunk_index`, `n_chunks`, `token_count`, `char_span`.

## Why 320 tokens

It keeps each chunk inside the comfortable range of small embedding models
(MiniLM truncates at 256 word-pieces, so oversized chunks are silently
decapitated), and it lets four to six sources fit into even a 512-token
generator prompt.

## Two bugs worth knowing about

Both were found by running the chunker over real scraped articles rather than
synthetic text, and both are covered by regression tests in
`tests/test_chunking.py`.

**Infinite loop.** When the carried overlap sentence plus the next sentence
exceeded the budget, the packer emitted a window, carried the same sentence
again, and re-entered with the index unchanged — forever. Dropping the overlap
is always the correct escape, and there is now an iteration guard as a backstop.

**Non-additive token estimates.** `approx_token_count` rounded to an integer per
sentence. Summing 400 of those drifted by roughly 0.9 tokens each, overshooting
the budget by several hundred tokens on a run-on article. The packer now
accumulates a fractional estimate (`_token_estimate`) and only rounds for
display.

## Results on the real corpus

2,477 scraped articles produce 2,668 chunks — an average of 1.08 chunks per
document. Most FAQ answers are genuinely atomic, which is exactly what the
strategy predicts.
