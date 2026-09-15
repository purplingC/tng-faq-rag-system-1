"""Logging setup."""

from __future__ import annotations
import logging
import sys

LOG = logging.getLogger("tngd_rag")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    LOG.handlers[:] = [handler]
    LOG.setLevel(level)
    LOG.propagate = False
