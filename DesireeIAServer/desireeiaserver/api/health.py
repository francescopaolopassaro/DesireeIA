"""Health and version probes."""

from __future__ import annotations

import os
import time

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from .. import __version__
from .. import engine as engine_mod

router = APIRouter(tags=["health"])


@router.get("/health")
@router.get("/v1/health")
def health(request: Request):
    slots = getattr(request.app.state, "slots", None)
    state = "ok"
    status_code = 200
    if slots is not None and slots.slots:
        states = {slot.state for slot in slots.slots}
        if "loading" in states:
            state, status_code = "loading", 503
        elif "loaded" not in states:
            state, status_code = "unavailable", 503

    uptime_s = round(time.monotonic() - request.app.state.started_at)
    body = {
        "status": state,
        "version": __version__,
        "engine_available": engine_mod.engine_available(),
        "engine_version": engine_mod.engine_version(),
        "engine_import_error": engine_mod.engine_import_error(),
        "pid": os.getpid(),
        "uptime_s": uptime_s,
        "slots": _slot_summaries(request),
    }
    return JSONResponse(body, status_code=status_code)


def _slot_summaries(request: Request) -> list:
    slots = getattr(request.app.state, "slots", None)
    if slots is None:
        return []
    return [{
        "id": slot.model_id,
        "state": slot.state,
        "path": slot.model_path,
    } for slot in slots.slots]


@router.get("/version")
def version():
    return {
        "server": __version__,
        "engine": engine_mod.engine_version(),
        "engine_available": engine_mod.engine_available(),
    }