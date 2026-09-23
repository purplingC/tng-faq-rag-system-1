# TNG FAQ RAG System

> A **Retrieval-Augmented Generation** system that answers Touch 'n Go eWallet questions
> **only** from verified TNG Digital FAQ content, in **English and Malay**, and refuses
> everything else — out-of-scope questions, prompt injection, PII requests and illicit instructions.

> **Rebuild of an earlier project of mine**, [`purplingC/faq-rag-system`](https://github.com/purplingC/faq-rag-system).
> That version worked but had defects that only showed up under measurement. This one fixes
> them, proves each fix with a regression test, and records the evidence in
> [docs/original-issues.md](docs/original-issues.md).

---

## 1. Project Overview and Features

The eWallet help centre holds about **2,500 articles**. A general chatbot will happily invent
an answer that sounds right and is wrong — for a payments app, that is worse than saying nothing.

This system **quotes rather than invents**. Answers are assembled from sentences in the official
article, every reply names the article it came from, and when nothing matches well enough the
system says so and links the help centre.

```python
from tngd_faq_rag import ask_tngd_bot

ask_tngd_bot("What is TNG eWallet SOS Balance?")
```

### Features

| Feature | What it does |
| --- | --- |
| **Grounded answers** | Assembled from the article's own sentences, so a fee or deadline cannot be invented |
| **Abstention** | Below a calibrated confidence threshold, it refuses and links the FAQ |
| **Six guardrail layers** | Injection defence, illicit requests, credentials, payment data, grounding, output policy |
| **Bilingual** | English and Malay, each with its own index; a Malay question gets a Malay answer and a Malay refusal |
| **Zero runtime dependencies** | The default pipeline is standard library only; every heavy backend is optional |
| **Answerability gate** | Optional LLM check for questions that are on topic but unanswerable |
| **Three interfaces** | CLI, a chat web page, and a FastAPI REST service |
| **Measured, not asserted** | A 95-case golden set, 322 tests, and CI on 3 operating systems |

### Measured results

| | |
| --- | --- |
| Knowledge base | **2,477** English + **1,975** Malay articles, shipped in the repo |
| Retrieval (seed corpus) | recall@1 **0.962** · recall@4 **1.000** · MRR **0.981** |
| Malay (golden set) | recall@1 **1.000** · refusals **6 / 7** without an LLM key |
| Out-of-scope questions refused | **13 / 13** |
| Adversarial prompts blocked | **23 / 23**, correct category 23 / 23, English and Malay |
| Ordinary questions wrongly blocked | **0 / 18** golden set · **0 / 4,452** real FAQ questions |
| Verified answers ever suppressed | **0** |
| Latency | median **12 ms** on the full FAQ (p95 108 ms), **2 ms** on the seed corpus |
| Tests | **322** passing, none touching the network |

Full methodology and per-corpus results: [docs/evaluation.md](docs/evaluation.md).

---

## 2. Tech Stack, APIs, and Other Resources

### Technical highlights

| Component | Technology / Approach |
| --- | --- |
| **Language** | Python 3.9 – 3.13 |
| **Embeddings** | Hashed TF-IDF (built in) · sentence-transformers (optional) |
| **Vector index** | Pure Python · NumPy · FAISS, whichever is installed |
| **Keyword search** | Okapi BM25, implemented directly |
| **Fusion & ranking** | Reciprocal Rank Fusion → absolute-score reranking → MMR diversification |
| **Storage** | SQLite, thread-local connections |
| **Generation** | Extractive composer (default) · hosted chat API · local seq2seq |
| **Guardrails** | Rule-based, intent-scoped, English + Malay, with Luhn card detection |
| **Answerability** | Gemini `gemini-3.5-flash-lite` via an OpenAI-compatible endpoint (optional) |
| **API layer** | FastAPI + Uvicorn, Swagger UI at `/docs` |
| **Web UI** | Standard-library HTTP server, single HTML page |
| **Packaging** | PEP 621 `pyproject.toml`, src layout, optional extras |
| **Quality** | pytest · ruff · mypy · GitHub Actions (8 jobs) · Docker |

### APIs used

| API | Purpose | Required? |
| --- | --- | --- |
| **TNG Digital help centre** (Zendesk public JSON) | Source of every FAQ article, in 3 languages | Only to re-scrape |
| **Google AI Studio / Gemini** | The optional answerability check | No — the system works without it |

### This project's own API

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/ask` | Answer a question; returns the answer, sources, confidence and safety record |
| `GET /v1/health/live` | Is the process up? |
| `GET /v1/health/ready` | Is the index built and ready to serve? |
| `GET /v1/info` | Active backends and thresholds (API key required when keys are set) |

Details, auth and rate limits: [docs/api.md](docs/api.md).

### Resources in this repository

| Path | Contents |
| --- | --- |
| `data/tngd_faq.json` | 2,477 English articles (1.7 MB) |
| `data/tngd_faq_ms.json` | 1,975 Malay articles (1.5 MB) |
| `data/tngd_faq_zh.json` | 10 Chinese articles, stored but not answered from |
| `src/tngd_faq_rag/resources/` | 30 English + 29 Malay seed articles, used by tests and the evaluation |
| `docs/` | Architecture, retrieval, chunking, guardrails, evaluation, API, original issues |

---

## 3. Getting Started: Setup and Running Instructions

### Option A — Docker (recommended, nothing else to install)

1. Get the code:

   ```bash
   git clone https://github.com/purplingC/tng-faq-rag-system-1.git
   cd tng-faq-rag-system-1
   ```

2. Start **Docker Desktop** and wait until it is running.

3. Build and run:

   ```bash
   docker compose up --build
   ```

4. Wait for `API ready: ... docs=2477, chunks=2668`, then open
   [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) and try `POST /v1/ask`.
   `Ctrl+C` stops it.

The FAQ and a pre-built index are inside the image, so there is no download, no scrape and no key.

### Option B — Python

1. Get the code:

   ```bash
   git clone https://github.com/purplingC/tng-faq-rag-system-1.git
   cd tng-faq-rag-system-1
   ```

2. Create and activate a virtual environment:

   ```bash
   python3 -m venv .venv
   # macOS / Linux
   source .venv/bin/activate
   # Windows
   .venv\Scripts\activate
   ```

3. Install the package:

   ```bash
   pip install -e ".[api]"      # drop [api] for the CLI only
   ```

4. Run it — this builds the index in about three seconds, answers a scripted set of
   questions, then opens a prompt:

   ```bash
   tngd-faq-rag
   ```

### Everyday commands

| Command | What it does |
| --- | --- |
| `tngd-faq-rag ask "..."` | Answer one question; add `--json` for the full record |
| `tngd-faq-rag demo` | One example of each of the six reply types |
| `tngd-faq-rag chat` | Interactive prompt |
| `tngd-faq-rag ui` | Chat page on [http://127.0.0.1:8000](http://127.0.0.1:8000) |
| `tngd-faq-rag-api` | REST service on [http://127.0.0.1:8080/docs](http://127.0.0.1:8080/docs) |
| `tngd-faq-rag eval` | The scored golden set |
| `tngd-faq-rag smoke` | Fast end-to-end check, no test framework needed |
| `tngd-faq-rag scrape` | Rebuild the knowledge base; `--locale ms-my` for Malay |
| `tngd-faq-rag stats` | Summarise the event log, when one is enabled |
| `make help` | Every `make` shortcut |

### Optional: a free Gemini key

Without a key the system answers from the FAQ as normal. With one, it adds the answerability
check, which catches questions that are on topic but unanswerable.

```bash
cp .env.example .env            # then put your key on the LLM_API_KEY= line
tngd-faq-rag info               # look for "reachable": true
```

Get a key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey). `.env` is
gitignored and the key is never logged.

### Verifying it works

```bash
make check
```

Expected: `All checks passed!`, `322 passed`, and `RESULT: PASS`.

```
Retrieval        n=26   recall@1=0.962  recall@4=1.000  MRR=0.981
Abstention       n=13   correct=13/13  rate=1.000
Adversarial      n=23   blocked=23/23  rate=1.000  category_acc=1.000
False positives  n=18   wrongly blocked=0  rate=0.000
KB self-censor   n=30   suppressed answers=0
Malay            n=8    recall@1=1.000  refused=6/7  rate=0.857
RESULT: PASS
```

A full walkthrough, with the output of every command, is in
[the introduction PDF](TNG-FAQ-RAG-System-Introduction.pdf).

### If something goes wrong

| Message | Fix |
| --- | --- |
| `Cannot connect to the Docker daemon` | Docker Desktop is not running |
| `error: externally-managed-environment` | Create the virtual environment first |
| `command not found: tngd-faq-rag` | Activate it: `source .venv/bin/activate` |
| `Address already in use` | `TNGD_API_PORT=8090 tngd-faq-rag-api` |
| Answers look stale after editing the FAQ | `tngd-faq-rag index --rebuild` |

---

## 4. How to Contribute and Report Issues

### Before opening a pull request

```bash
make check      # lint + types + 322 tests + the scored evaluation
```

Everything must pass. CI runs the same checks on Python 3.9–3.13 across Linux, macOS and
Windows, plus a Docker build and a bare-interpreter install.

| Expectation | Why |
| --- | --- |
| **A test with every behaviour change** | The golden set and the suite are the only proof a change works |
| **Numbers come from a run, not a guess** | Every figure in this README was measured |
| **`ruff` and `mypy` clean** | `make lint` and `make typecheck` |
| **No secrets, ever** | `.env`, keys and the event log are gitignored |
| **New rules need a false-positive check** | Guardrail changes are tested against all 4,452 real FAQ questions |

### Reporting an issue

Open an issue at
[github.com/purplingC/tng-faq-rag-system-1/issues](https://github.com/purplingC/tng-faq-rag-system-1/issues)
with:

1. The **question** you asked, exactly as typed.
2. The output of `tngd-faq-rag ask "your question" --json` — it contains the decision,
   the confidence, the sources and the stage-by-stage `trace`.
3. What you expected instead.

**Never paste an API key** into an issue. The `--json` output does not contain one.

---

## 5. Conclusion and License

This project set out to answer eWallet questions **accurately or not at all**. What makes it
trustworthy is not the model — there is barely one — but the discipline around it: absolute
scores so refusing is reachable, grounding so numbers cannot drift, guardrails tested against
every real question for false positives, and an evaluation that prints its failures by name.

### Known limitations

* **Word matching, not meaning.** Installing `sentence-transformers` replaces it and fixes most open cases.
* **Mixed Malay and English is fragile.** One unknown word can change which article ranks first.
* **Brand questions need the LLM gate.** Without a key, *"What is CIMB bank?"* is answered from an article that merely mentions CIMB.
* **Chinese is not answered.** 10 articles, and the tokenizer cannot read the characters.
* **Malay safety copy is unreviewed** by a native speaker, though the crisis helpline numbers are correct.
* **No memory between questions.**

The full list, and what I would do next in priority order, is at the end of
[docs/evaluation.md](docs/evaluation.md) and in the
[introduction PDF](TNG-FAQ-RAG-System-Introduction.pdf).

### Documentation

| Document | Contents |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | Diagrams, module map, design rules |
| [docs/retrieval.md](docs/retrieval.md) | Fusion, calibration, language routing, failure modes |
| [docs/chunking.md](docs/chunking.md) | Chunking strategy and rationale |
| [docs/guardrails.md](docs/guardrails.md) | Threat model and the six layers |
| [docs/evaluation.md](docs/evaluation.md) | Methodology, results, test suite |
| [docs/api.md](docs/api.md) | Endpoints, authentication, rate limits |
| [docs/original-issues.md](docs/original-issues.md) | What was wrong in the earlier version, and the proof |

### License

**MIT — see [LICENSE](LICENSE).** The licence covers the code in this repository only.

The FAQ content under `data/` and `src/tngd_faq_rag/resources/` belongs to TNG Digital,
retrieved from their public help centre. Every record keeps its source URL, and every answer is
either verbatim source text or grounded and cited against it. This project is unaffiliated with
TNG Digital and claims no rights over that content.
