"""This file checks which optional packages are installed, without importing them."""

from __future__ import annotations
from typing import Any
from .logging_utils import LOG


class _Optional:
    """Probes with find_spec, so nothing heavy loads at startup."""

    _NAMES = ("numpy", "faiss", "sentence_transformers", "transformers", "torch", "requests")

    def __init__(self) -> None:
        import importlib.util

        self._available = {name: importlib.util.find_spec(name) is not None for name in self._NAMES}
        self._cache: dict[str, Any] = {}
        self.numpy = self._load("numpy")  # Loaded now, the vector code binds it at module scope

    def _load(self, name: str):
        """Imports on first real use, not at startup."""
        if name in self._cache:
            return self._cache[name]
        module = None
        if self._available.get(name):
            try:
                import importlib

                module = importlib.import_module(name)
            except Exception as exc:  # pragma: no cover - environment dependent
                LOG.debug("optional import %s failed: %s", name, exc)
                self._available[name] = False
        self._cache[name] = module
        return module

    def __getattr__(self, name: str):
        if name in self._NAMES:
            return self._load(name)
        raise AttributeError(name)

    def has(self, name: str) -> bool:
        """Is it installed? Answers without importing it."""
        return bool(self._available.get(name))

    def summary(self) -> dict[str, str]:
        """Per package: absent, installed but not loaded, or its version."""
        out = {}
        for name in self._NAMES:
            if not self._available.get(name):
                out[name] = "absent"
            elif name in self._cache and self._cache[name] is not None:
                out[name] = getattr(self._cache[name], "__version__", "present")
            else:
                out[name] = "installed (not loaded)"
        return out


OPT = _Optional()


np = OPT.numpy  # May be None, every numeric path has a pure Python fallback
