# Every target uses the virtualenv in .venv, created by make venv
VENV    ?= .venv
PY      ?= $(VENV)/bin/python
PIP     ?= $(VENV)/bin/pip
SYSPY   ?= python3

.DEFAULT_GOAL := help
.PHONY: help venv install install-dev install-ml install-api run demo chat ui ask eval \
        eval-live smoke index scrape api docker-build docker-run test test-cov lint format \
        typecheck check clean distclean

# Show this help
help:
	@awk -F: '/^# / { c = substr($$0, 3); next } \
	/^[a-zA-Z_-]+:/ { if (c) printf "  \033[36m%-14s\033[0m %s\n", $$1, c } \
	{ c = "" }' $(MAKEFILE_LIST)

# Create the virtualenv
venv: $(VENV)/bin/python

# A real file, so the virtualenv is built once instead of on every install
$(VENV)/bin/python:
	$(SYSPY) -m venv $(VENV)
	$(PIP) install --upgrade pip
	@echo "Created $(VENV). Activate with: source $(VENV)/bin/activate"

# Install the package
install: venv
	$(PIP) install -e .

# Install with the test and lint tools
install-dev: venv
	$(PIP) install -e ".[dev]"

# Install the REST API extras
install-api: venv
	$(PIP) install -e ".[api]"

# Install the ML backends, roughly 2.5 GB
install-ml: venv
	$(PIP) install -e ".[ml]"

# Build the index, run the demo, then chat
run:
	$(PY) -m tngd_faq_rag

# Answer a scripted set of normal, out-of-scope and adversarial queries
demo:
	$(PY) -m tngd_faq_rag demo

# Interactive prompt
chat:
	$(PY) -m tngd_faq_rag chat

# Serve the web chat UI on http://127.0.0.1:8000
ui:
	$(PY) -m tngd_faq_rag ui

# Serve the REST API on port 8080, docs at /docs
api:
	TNGD_API_PORT=8080 $(VENV)/bin/tngd-faq-rag-api

# Build the container image
docker-build:
	docker build -t tngd-faq-rag:local .

# Run the container on port 8080
docker-run: docker-build
	docker run --rm -p 8080:8080 tngd-faq-rag:local

# Answer one question, make ask Q="what is SOS Balance?"
ask:
	$(PY) -m tngd_faq_rag ask "$(Q)"

# Evaluate against the frozen seed corpus
eval:
	$(PY) -m tngd_faq_rag eval

# Evaluate against the active knowledge base
eval-live:
	$(PY) -m tngd_faq_rag eval --live

# Fast end-to-end check
smoke:
	$(PY) -m tngd_faq_rag smoke

# Rebuild the index
index:
	$(PY) -m tngd_faq_rag index --rebuild

# Rebuild the knowledge base from the live help centre
scrape:
	$(PY) -m tngd_faq_rag scrape

# Run the tests
test:
	$(VENV)/bin/pytest

# Run the tests with coverage
test-cov:
	$(VENV)/bin/pytest --cov=tngd_faq_rag --cov-report=term-missing

# Lint
lint:
	$(VENV)/bin/ruff check src tests

# Auto-fix and format
format:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

# Static type check
typecheck:
	$(VENV)/bin/mypy

# Everything CI runs
check: lint typecheck test eval

# Remove caches and the index
clean:
	rm -rf .tngd_index .tngd_index_eval .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +

# Remove the virtualenv, scraped data and build artefacts
distclean: clean
	rm -rf $(VENV) build dist src/*.egg-info data/tngd_faq.csv
