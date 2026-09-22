"""This file contains the API endpoints: ask, health checks and info."""

from __future__ import annotations
from typing import Any
from fastapi import APIRouter, Depends, Request
from starlette.concurrency import run_in_threadpool
from ..version import __version__
from .schemas import AskRequest, AskResponse, HealthResponse, InfoResponse
from .security import enforce_rate_limit, require_api_key

router = APIRouter(prefix="/v1")


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question from the verified TNG eWallet FAQ",
    responses={
        401: {"description": "Missing or invalid API key"},
        429: {"description": "Rate limit exceeded"},
        503: {"description": "The index is not ready yet"},
    },
)
async def ask(
    payload: AskRequest,
    request: Request,
    caller: str = Depends(require_api_key),
) -> dict[str, Any]:
    """Answer one question."""
    await enforce_rate_limit(request, caller)
    system = request.app.state.system

    # A worker thread keeps the event loop free, since ask is synchronous
    # Awaiting it directly would serialise every other caller behind it
    return await run_in_threadpool(system.ask, payload.question)


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def health_live() -> HealthResponse:
    """Is the process up?"""
    return HealthResponse(status="alive", version=__version__)


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness probe",
    responses={503: {"description": "Not ready to serve traffic"}},
)
async def health_ready(request: Request) -> Any:
    """Can it actually serve traffic - index built, knowledge base loaded?"""
    from fastapi.responses import JSONResponse

    system = getattr(request.app.state, "system", None)
    if system is None:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="not_ready",
                version=__version__,
                detail={"reason": "index is still building"},
            ).model_dump(),
        )
    documents, chunks = system.store.count()
    if chunks == 0:
        return JSONResponse(
            status_code=503,
            content=HealthResponse(
                status="not_ready",
                version=__version__,
                detail={"reason": "index contains no chunks"},
            ).model_dump(),
        )
    return HealthResponse(
        status="ready",
        version=__version__,
        detail={"documents": documents, "chunks": chunks},
    )


@router.get("/info", response_model=InfoResponse, summary="Active backends and thresholds")
async def info(request: Request, caller: str = Depends(require_api_key)) -> InfoResponse:
    """Which backends are in use. For debugging a deployment, not for clients."""
    system = request.app.state.system
    documents, chunks = system.store.count()
    cfg = system.cfg
    return InfoResponse(
        version=__version__,
        backends=system.backends(),
        thresholds={
            "abstain": cfg.abstain_threshold,
            "high_confidence": cfg.high_confidence_threshold,
            "grounding": cfg.grounding_threshold,
        },
        documents=documents,
        chunks=chunks,
    )
