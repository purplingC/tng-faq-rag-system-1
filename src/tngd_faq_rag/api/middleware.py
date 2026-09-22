"""This file gives each request an ID and logs how long it took."""

from __future__ import annotations
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
from ..logging_utils import LOG

REQUEST_ID_HEADER = "X-Request-ID"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request ID, time the request, log one structured line."""

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Honour an upstream ID so a trace survives a proxy
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            LOG.exception(
                "request failed %s",
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "duration_ms": round(elapsed, 2),
                    }
                ),
            )
            raise

        elapsed = (time.perf_counter() - started) * 1000
        response.headers[REQUEST_ID_HEADER] = request_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"

        # One JSON line per request, greppable by hand and parseable by a shipper
        if request.url.path not in ("/v1/health/live", "/v1/health/ready"):
            LOG.info(
                "%s",
                json.dumps(
                    {
                        "request_id": request_id,
                        "method": request.method,
                        "path": request.url.path,
                        "status": response.status_code,
                        "duration_ms": round(elapsed, 2),
                    }
                ),
            )
        return response
