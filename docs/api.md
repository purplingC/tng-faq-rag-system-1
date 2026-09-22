# HTTP API

`src/tngd_faq_rag/api/` is the machine-facing service. It is separate from
`src/tngd_faq_rag/web/`, which serves the human-facing demo chat page on the
standard library. Both share one `RagSystem`, so there is a single pipeline and
no duplicated logic between the CLI, the UI and the API.

## Running it

```bash
pip install -e ".[api]"
tngd-faq-rag-api                   # or: make api
```

Then open **http://127.0.0.1:8080/docs** for interactive, generated
documentation.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/v1/ask` | Answer a question |
| `GET` | `/v1/health/live` | Liveness — is the process up? |
| `GET` | `/v1/health/ready` | Readiness — can it serve traffic? |
| `GET` | `/v1/info` | Active backends and thresholds |
| `GET` | `/docs`, `/redoc`, `/openapi.json` | Generated documentation |

```bash
curl -X POST http://127.0.0.1:8080/v1/ask \
  -H 'Content-Type: application/json' \
  -H 'X-API-Key: your-key' \
  -d '{"question": "What is TNG eWallet SOS Balance?"}'
```

Versioned under `/v1` from the start: adding a version once clients exist is a
migration, having one from the beginning is free.

## Design decisions

**Liveness and readiness are different questions.** Liveness asks whether the
process is up and deliberately does no work — a liveness probe that touches a
dependency turns that dependency's outage into a restart loop. Readiness asks
whether the index is built and the corpus loaded. An orchestrator *restarts* on
failed liveness but only *withdraws traffic* on failed readiness, so conflating
them turns a slow start into a crash loop.

**The pipeline runs in a worker thread.** `RagSystem.ask` is synchronous: CPU-bound
in retrieval, I/O-bound in the answerability call. Awaiting it directly on the
event loop would serialise every other caller behind it, which is the most common
way a FastAPI service loses the concurrency it was chosen for.
`run_in_threadpool` keeps the loop free.

**An abstention is a `200`, not an error.** An out-of-scope question returns a
normal response with `decision: "abstain_low_confidence"`. The system worked
correctly and declined to guess; that is not a failure, and a client should not
have to catch an exception to handle it. `4xx` means the *request* was wrong;
`5xx` means *we* were.

**Unknown request fields are rejected.** `extra="forbid"` on `AskRequest` means a
typo'd field returns a 422 naming it, instead of being silently ignored and
leaving the caller to wonder why their parameter did nothing.

**Errors share one shape.** Every failure — validation, auth, rate limit,
unhandled — returns `{"error", "detail", "request_id"}`, so clients branch on
`error` rather than parsing prose. Internal errors return a generic message: a
traceback in an HTTP body is an information leak, and the request ID ties it to
the full trace in the logs.

**Request IDs travel.** An inbound `X-Request-ID` is honoured rather than
replaced, so a trace survives a proxy instead of restarting at our door. It comes
back in the response headers and appears in every log line for that request.

## Security

| | |
| --- | --- |
| **Auth** | `X-API-Key` header, compared with `secrets.compare_digest` — a plain `==` on a secret leaks it one byte at a time through timing. |
| **Open by default** | With no `TNGD_API_KEYS` set the API is open, which is right for `docker run` on a laptop and wrong for anything reachable. It is logged loudly at startup rather than left to be discovered. |
| **Rate limiting** | Sliding-window, per key, or per source address for anonymous callers. |
| **CORS** | Off unless `TNGD_CORS_ORIGINS` is set. |

**The rate limiter is in-process and therefore per-worker.** Four uvicorn workers
with a limit of 60 admit 240 requests a minute, not 60. That is fine for a single
instance and wrong for a fleet; a fleet needs Redis. The limitation is stated in
the code rather than implied away.

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `TNGD_API_HOST` | `0.0.0.0` | Bind address |
| `TNGD_API_PORT` | `8080` | Port |
| `TNGD_API_WORKERS` | `1` | uvicorn workers |
| `TNGD_API_KEYS` | *(unset)* | Comma-separated keys; unset means open |
| `TNGD_RATE_LIMIT` | `60` | Requests per minute; `0` disables |
| `TNGD_CORS_ORIGINS` | *(unset)* | Comma-separated allowed origins |

Everything in [.env.example](../.env.example) applies too — the API and the CLI
read the same `Config`.

## Container

```bash
docker build -t tngd-faq-rag .
docker run --rm -p 8080:8080 tngd-faq-rag
# or: docker compose up --build
```

The image is multi-stage: the builder carries pip and wheels, the runtime carries
only the virtualenv and the source. It runs as an unprivileged user, and the
index is built at **image build time**, so a cold container is ready in
milliseconds rather than seconds. A different knowledge base mounted at runtime
rebuilds automatically, because the manifest stores a content hash of the corpus.

CI builds the image, runs it, and asserts that it answers a real question — so
the Dockerfile cannot rot silently, and you do not need Docker installed locally
to know it works.

## Tests

`tests/test_api.py` — 25 tests through FastAPI's `TestClient`, in-process. No
server, no port, no container, no network. They inject the shared `system`
fixture, so the API tests cost no extra index build.

## Choosing a grading model

The answerability gate sends ~900 tokens in and caps output at 24. It is a
one-word classification, not reasoning or generation, and it sits **in the
request path** - every graded question waits for it.

So the selection criterion is latency, not capability. A **flash-lite** class
model is the right choice; a reasoning model answering "does this passage answer
this question?" is slower for no measurable gain.

```bash
export LLM_MODEL=gemini-3.5-flash-lite     # current generation, free tier
```

Google's own description of that model is almost a statement of this use case:
*"optimized for high-throughput, low-cost execution for subagent tasks... where
latency and API cost are the primary constraints."* It is a **stable** ID, not a
preview, which matters twice over: previews get withdrawn, and Google documents
that "rate limits are more restricted for experimental and preview models".

Where the provider supports it, the grading call uses a **constrained JSON
schema** (`response_format`), so the model cannot return something unparsable
and the "unusable reply" fallback becomes unreachable rather than merely
handled. Providers without that support fall back to free text, which the same
parser still reads. A 400 that names `response_format` is retried once without
it, so a stricter Gemini and a looser Ollama both work from one code path.

Avoid `-preview` model IDs in anything you submit or deploy: they get withdrawn.

Model IDs change and free-tier rate limits are per-project and no longer
published in the docs, so verify both against your own account rather than
trusting a README:

```bash
curl -s -H "Authorization: Bearer $LLM_API_KEY" \
  https://generativelanguage.googleapis.com/v1beta/openai/models \
  | python3 -c "import json,sys; [print(' ', m['id']) for m in json.load(sys.stdin)['data']]"
```

Rate limits: <https://aistudio.google.com/rate-limit>

Three facts about free-tier limits that are easy to get wrong:

* They are **per project, not per API key** - a second key buys no headroom.
* Three dimensions are enforced independently (RPM, input TPM, RPD) and
  **exceeding any one** returns 429.
* **RPD resets at midnight Pacific**, not on a rolling window.

Google does not publish the per-model numbers anywhere; the AI Studio dashboard
is the only place they appear. Observed for `gemini-3.5-flash-lite` on the free
tier, as a rough guide rather than a guarantee:

| Dimension | Free-tier limit |
| --- | --- |
| Requests per minute | 15 |
| Input tokens per minute | ~250,000 |
| Requests per day | 500 |

For interactive use that is a lot of room: one graded call per answered question,
and out-of-scope or blocked questions never reach the grader at all. **RPM is the
binding constraint, not RPD** - a burst of more than 15 questions inside a minute
will 429 long before you approach 500 in a day.

That is exactly what `tngd-faq-rag eval` does, which is the practical reason to run
it with grading off:

```bash
TNGD_ANSWERABILITY=off tngd-faq-rag eval
```

The evaluation fires roughly 30-40 graded calls as fast as it can. Beyond the
quota, the golden-set numbers are meant to be reproducible offline, so a network
service has no business influencing them.

A 429 is not an outage here: the grader fails open, the answer is served from
retrieval alone, and the response records `graded: false` so the degradation is
visible rather than silent.

If you hit 429s, either drop to a smaller model or turn the gate off without
unconfiguring anything else:

```bash
export TNGD_ANSWERABILITY=off
```
