"""This file exposes the REST API, installed with `pip install -e ".[api]"`."""

from __future__ import annotations
from .app import create_app, run

__all__ = ["create_app", "run"]
