# syntax=docker/dockerfile:1
#
# Multi stage build so the runtime carries only the virtualenv and the source

# Builder stage
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy dependency metadata first so editing a source file does not reinstall it
COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install ".[api]"

# Runtime stage
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="tngd-faq-rag" \
      org.opencontainers.image.description="Grounded, guardrailed RAG over the Touch 'n Go eWallet FAQ" \
      org.opencontainers.image.source="https://github.com/purplingC/tng-faq-rag-system-1"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    TNGD_INDEX_DIR=/app/.tngd_index \
    TNGD_API_HOST=0.0.0.0 \
    TNGD_API_PORT=8080

# An unprivileged user, since containers run as root by default
RUN useradd --create-home --uid 10001 appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser pyproject.toml README.md LICENSE ./
COPY --chown=appuser:appuser data/tngd_faq.json ./data/

# The index is written here at build time, so the directory must belong to appuser
RUN chown appuser:appuser /app

USER appuser

# Build the index at image build time so a cold container is ready immediately
RUN python -m tngd_faq_rag index --rebuild --quiet

EXPOSE 8080

# Readiness rather than liveness, since the question is whether it can serve traffic
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/v1/health/ready', timeout=4).status==200 else 1)"

CMD ["tngd-faq-rag-api"]
