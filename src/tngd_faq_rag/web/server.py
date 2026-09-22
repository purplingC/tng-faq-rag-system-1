"""This file serves the web chat page using Python's built-in web server."""

from __future__ import annotations
import json
from typing import Any
from ..logging_utils import LOG
from ..pipeline import RagSystem
from ..version import __version__


def _load_page() -> str:
    """Read the single-page UI from package data."""
    try:
        from importlib.resources import files

        return (files(__package__) / "static" / "index.html").read_text(encoding="utf-8")
    except Exception:  # pragma: no cover
        from pathlib import Path

        return (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")


def serve_ui(system: RagSystem, host: str, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        server_version = "tngd-faq-rag/" + __version__

        def log_message(self, fmt: str, *args: Any) -> None:
            LOG.debug("ui %s", fmt % args)

        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self' 'unsafe-inline'")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                self._send(200, _load_page().encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/healthz":
                self._send(
                    200, json.dumps({"ok": True, **system.backends()}).encode(), "application/json"
                )
            else:
                self._send(404, b'{"error":"not found"}', "application/json")

        def do_POST(self) -> None:
            if self.path != "/api/ask":
                self._send(404, b'{"error":"not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
                if length > 16_384:
                    raise ValueError("request too large")
                payload = json.loads(self.rfile.read(length) or b"{}")
                question = str(payload.get("question", ""))[:2000]
                result = system.ask(question)
                self._send(
                    200,
                    json.dumps(result, ensure_ascii=False).encode("utf-8"),
                    "application/json; charset=utf-8",
                )
            except Exception as exc:
                LOG.exception("UI request failed")
                self._send(400, json.dumps({"error": str(exc)}).encode(), "application/json")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"\n  TNG eWallet FAQ Assistant -> http://{host}:{port}\n  Ctrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()
