"""This file builds the FastAPI app and starts the server."""

from __future__ import annotations
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from ..config import Config
from ..env import load_dotenv
from ..logging_utils import LOG, setup_logging
from ..pipeline import RagSystem, build_system
from ..version import __version__
from .middleware import RequestContextMiddleware
from .routes import router
from .schemas import ErrorResponse
from .security import RateLimiter

DESCRIPTION = """
Answers questions about the Touch 'n Go eWallet using **only verified FAQ
content**, and refuses everything else - out-of-scope questions, prompt
injection, PII requests and illicit instructions.

Every answer is grounded in a cited source. An out-of-scope question returns a
`200` with an abstention, not an error: the system worked correctly and declined
to guess.

Authenticate with an `X-API-Key` header when the deployment configures keys.
"""


def _split_env_list(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def _serve_address() -> tuple[str, int]:
    """Host and port the server binds to, read from the environment."""
    return os.environ.get("TNGD_API_HOST", "0.0.0.0"), int(os.environ.get("TNGD_API_PORT", "8080"))


def create_app(cfg: Config | None = None, *, system: RagSystem | None = None) -> FastAPI:
    """Build the application."""
    load_dotenv()  # before Config reads os.environ
    cfg = cfg or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Build the index once at startup, with readiness false until it finishes
        app.state.system = system
        if app.state.system is None:
            LOG.info("Building the RAG system ...")
            app.state.system = build_system(cfg)
        backends = app.state.system.backends()
        LOG.info("API ready: %s", ", ".join(f"{k}={v}" for k, v in backends.items()))
        if not app.state.api_keys:
            LOG.warning(
                "No API keys configured (TNGD_API_KEYS is unset): this API is OPEN. "
                "Set keys before exposing it beyond localhost."
            )
        host, port = _serve_address()
        # A browser cannot open 0.0.0.0, so show the local address instead
        shown = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        LOG.info("Open http://%s:%d/docs to try the API, Ctrl+C to stop", shown, port)
        yield
        app.state.system = None

    app = FastAPI(
        title="TNG eWallet FAQ Assistant",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.state.api_keys = _split_env_list("TNGD_API_KEYS")
    rate_limit = int(os.environ.get("TNGD_RATE_LIMIT", "60") or 0)
    app.state.rate_limiter = RateLimiter(rate_limit) if rate_limit > 0 else None

    app.add_middleware(RequestContextMiddleware)

    origins = _split_env_list("TNGD_CORS_ORIGINS")
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["Content-Type", "X-API-Key", "X-Request-ID"],
        )

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    def _request_id(request: Request) -> str:
        return getattr(request.state, "request_id", "")

    # One error shape whatever raised it, so clients branch on error not on prose
    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        body = ErrorResponse(
            error="http_error", detail=str(exc.detail), request_id=_request_id(request)
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=body.model_dump(),
            headers=getattr(exc, "headers", None),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                detail="; ".join(
                    f"{'.'.join(str(p) for p in e['loc'][1:])}: {e['msg']}" for e in exc.errors()
                ),
                request_id=_request_id(request),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        # Generic message, since a traceback in an HTTP body is an information leak
        LOG.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred.",
                request_id=_request_id(request),
            ).model_dump(),
        )

    return app


def run() -> None:  # pragma: no cover - process entry point
    """`tngd-faq-rag-api` console script: serve with uvicorn."""
    import uvicorn

    setup_logging(verbose=os.environ.get("TNGD_VERBOSE", "") != "")
    host, port = _serve_address()
    uvicorn.run(
        "tngd_faq_rag.api.app:create_app",
        factory=True,
        host=host,
        port=port,
        workers=int(os.environ.get("TNGD_API_WORKERS", "1")),
        log_config=None,
    )


app_factory: Any = create_app
