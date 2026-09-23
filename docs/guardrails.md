# Guardrails

Defence in depth, six layers. Two principles drive every decision.

**Intent, not keywords.** A blocklist of bare words (*card number*, *kill*,
*illegal*, *attack*) blocks the support questions the bot exists to answer.
Rules are verb+object shaped and carry explicit exemptions.

**Input policy ≠ output policy.** User input is untrusted and screened hard.
Retrieved knowledge-base text is trusted and must never be screened with input
rules — doing so makes the system delete its own verified answers.

---

## L1 — Input policy

A rule engine over `(pattern, category, severity, exemption)`. Categories:
`prompt_injection`, `pii_request`, `credential_request`, `payment_data`,
`illicit_activity`, `self_harm`.

A **benign allowlist** protects domain vocabulary, checked before the soft rules:

| Question | Must be | Why a keyword list gets it wrong |
| --- | --- | --- |
| "My **card number** changed, how do I update it?" | allowed | `card number` is a support topic, not a request to disclose one |
| "The app crashed, how do I **kill** the app process?" | allowed | no violent intent |
| "Is it **illegal** to let someone use my eWallet?" | allowed | asking about legality is not illicit |
| "My account was **hacked**, how do I report it?" | allowed | the user is the victim |
| "I forgot my **PIN**, what should I do?" | allowed | recovery, not disclosure |
| "What is **my PIN**?" | blocked | a request to disclose a secret |

## L2 — Prompt-injection detection

Instruction override, system-prompt exfiltration, jailbreak personas, fake
authority claims, delimiter and control-token injection, "repeat everything
above", translation-of-instructions attacks, and encoded payloads.

Obfuscation is normalised away first (`deobfuscate`): unicode confusables and
accents, zero-width characters, leetspeak, `l e t t e r   s p a c i n g`,
punctuation padding (`i.g.n.o.r.e`), and base64 blobs, which are decoded and
appended so their contents get screened too.

## L3 — PII, credentials and payment data

Card numbers are matched by regex **and** validated with the Luhn checksum, at a
13-digit floor. Twelve digits is too weak: the digit run `6011 5166 2025`
appears in a real help-centre article and is Luhn-valid by coincidence — at a
12-digit floor it suppressed a legitimate answer.

Credentials, NRIC, phone and email are detected in *request* context ("tell me
X") rather than *mention* context.

## L4 — Answerability gate (optional, needs an LLM)

Term overlap answers *"is this passage about the same thing as the question?"*.
It cannot answer *"does this passage actually answer the question?"*. Those come
apart whenever a query is on topic but asks for something the corpus does not
contain:

```
"How do we spell SOS balance?"   0.4189   <- unanswerable from the article
"Explain SOS Balance"            0.4196   <- perfectly answerable
"Tell me about SOS Balance"      0.4475   <- perfectly answerable
```

All three retrieve the same correct article, because to a bag of words they
**are** the same query: same subject, same salient terms, same best match. No
threshold separates 0.4189 from 0.4196, so this is an architecture problem, not
a tuning problem.

A small LLM settles it in one call - judging whether a text answers a question
is exactly what language models are good at and term statistics are not. This is
the **CRAG / Self-RAG** pattern: grade the retrieved passages before generating,
and abstain if none of them answer. Endorsed passages are also the only ones
passed on to the generator.

The gate runs **last**, after the cheap filters have discarded the obvious
rubbish, so the model is only ever asked about plausible candidates. Results are
cached per (question, passages).

**It fails open, deliberately.** If the endpoint is unreachable or returns
nonsense, the system keeps its previous behaviour rather than refusing
everything: this is an *additional* gate on a pipeline that is already safe
without it, and a misconfigured endpoint should degrade quality, never take the
bot offline. Every result carries a `graded` flag, so a silent fallback appears
in the response trace instead of being invisible.

Enable it with any OpenAI-compatible endpoint - see the README for free options.

## L5 — Grounding gate

Every generated sentence must be supported by retrieved text: IDF-weighted
content-word overlap or a shared 4-gram, **plus a numeric consistency check** —
every number, percentage and currency amount in the answer must appear in a
source. A model that writes "within 7 days" where the FAQ says "24 hours" fails
here even though the sentence is otherwise well grounded, and that is exactly
the class of hallucination that matters for a payments FAQ.

Unsupported sentences are dropped; if nothing survives, the system abstains.
Citations are attached per sentence and drive the answer's attribution.

## L6 — Output policy and canary

The system prompt carries a random canary token; if it ever appears in output,
the response is blocked as a leak. The policy also catches instruction echo,
profanity, PANs, and email addresses absent from the sources.

It is applied **only to generated text**. `tests/test_guardrails.py` asserts
that no verified knowledge-base answer is ever suppressed — a regression test
for a bug in the original implementation that silently replaced 6 of 30 verified
answers, including the flagship card-verification article, with "Response
blocked by safety guardrails."

## Abstention is a guardrail

Below the confidence threshold the system says it does not know and links the
official FAQ, rather than improvising. Measured: 13/13 out-of-scope questions
refused on the seed corpus.

## Self-harm

Routed to `SAFE_COMPLETE` rather than a bare block, and the response carries
real Malaysian crisis lines (Talian Kasih 15999, Befrienders KL 03-7627 2929).
A generic "request blocked" is the wrong answer to that message.

## Measured

| | Result |
| --- | --- |
| Adversarial prompts blocked | 23/23, English and Malay |
| Correct category assigned | 23/23 |
| Ordinary support questions wrongly blocked | 0/18 |
| Verified KB answers suppressed | 0/30 seed, 0/2,477 full FAQ |
| Real FAQ questions wrongly blocked by the input policy | 0/2,477 English, 0/1,975 Malay |

That last row found two real defects. *"What is OTP?"* and *"What is a TNG Reload
Pin?"* were blocked as credential requests, and questions about a **jailbroken
device** — a genuine support topic — were blocked as jailbreak attempts. Both now
have benign-context exemptions, and both are in the golden set.

## Limitations

Rule-based, in English and Malay. Malay covers the highest-risk categories only:
instruction override, prompt extraction, illicit requests and other people's data.
Refusals follow the language of the question, in English or Malay, with the crisis
helpline numbers unchanged. The Malay wording has not been reviewed by a native
speaker and should be before public use. A fine-tuned classifier such as Llama Guard would raise recall on novel paraphrased
attacks; `InputPolicy` takes a rule list and was designed to accept one as an
additional layer without restructuring.
