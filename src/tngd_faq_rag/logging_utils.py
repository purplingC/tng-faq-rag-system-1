"""This file sets up logging for the whole package."""

from __future__ import annotations
import logging
import sys

LOG = logging.getLogger("tngd_faq_rag")

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S %z"


def setup_logging(verbose: bool = False) -> None:
    """Send logs to stderr, so stdout stays clean for JSON output."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))

    LOG.handlers.clear()
    LOG.addHandler(handler)
    LOG.setLevel(logging.DEBUG if verbose else logging.INFO)

    LOG.propagate = False
