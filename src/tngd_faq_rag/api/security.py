"""This file checks API keys and limits how many requests each caller can make."""

from __future__ import annotations
import secrets
import threading
import time
from collections import deque
from fastapi import Header, HTTPException, Request, status
from ..logging_utils import LOG


def _configured_keys(request: Request) -> set[str]:
    return set(request.app.state.api_keys or [])


async def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> str:
    """Authenticate a caller."""
    keys = _configured_keys(request)
    if not keys:
        return "anonymous"
    if x_api_key and any(secrets.compare_digest(x_api_key, k) for k in keys):
        return x_api_key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Missing or invalid X-API-Key header.",
        headers={"WWW-Authenticate": "ApiKey"},
    )


class RateLimiter:
    """Fixed-window-free sliding log limiter, per caller."""

    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, caller: str) -> tuple[bool, int]:
        """Returns (allowed, seconds to wait before retrying)."""
        if self.limit <= 0:
            return True, 0
        now = time.monotonic()
        window_start = now - 60.0
        with self._lock:
            hits = self._hits.setdefault(caller, deque())
            while hits and hits[0] < window_start:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, int(hits[0] + 60.0 - now) + 1)
            hits.append(now)
            # Keep the table bounded on a long lived process
            if len(self._hits) > 10_000:
                for key in [k for k, v in self._hits.items() if not v][:5_000]:
                    self._hits.pop(key, None)
            return True, 0


async def enforce_rate_limit(request: Request, caller: str) -> None:
    limiter: RateLimiter | None = getattr(request.app.state, "rate_limiter", None)
    if limiter is None:
        return
    # Limit by key, or by source address when anonymous
    identity = caller if caller != "anonymous" else (request.client.host if request.client else "?")
    allowed, retry_after = limiter.allow(identity)
    if not allowed:
        LOG.warning("rate limit exceeded for %s", identity)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit of {limiter.limit} requests/minute exceeded.",
            headers={"Retry-After": str(retry_after)},
        )
