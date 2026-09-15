# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Models-folder management: GET /models (scan), POST /models/upload
(streamed straight to disk), DELETE /models/{name}. Distinct from
/v1/models (api/v1.py), which lists currently loaded SLOTS in the OpenAI
shape — this is filesystem management of what's AVAILABLE to load, and
isn't part of the OpenAI-compatible surface, so it lives outside /v1.
"""

from __future__ import annotations

import asyncio
import json
import time

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import models_store as store
from ..config import Settings
from .errors import InvalidRequestError, NotFoundError

router = APIRouter(tags=["models"])


def _model_dict(m: store.ModelFile) -> dict:
    return {
        "name": m.name,
        "size_bytes": m.size_bytes,
        "modified_at": m.modified_at,
        "format": m.format,
        "valid": m.valid,
    }


@router.get("/models")
def list_models(request: Request) -> dict:
    settings: Settings = request.app.state.settings
    models = [_model_dict(m) for m in store.list_models(settings.resolved_models_dir)]
    return {"models": models}


@router.post("/models/upload")
async def upload_model(request: Request):
    settings: Settings = request.app.state.settings
    filename = request.query_params.get("filename")
    if not filename:
        raise InvalidRequestError("query parameter 'filename' is required", param="filename")
    try:
        store.validate_filename(filename)
    except store.InvalidFilename as exc:
        raise InvalidRequestError(str(exc), param="filename") from exc

    max_bytes = settings.max_upload_mb * 1024 * 1024

    # An honest Content-Length lets a too-large upload be rejected before a
    # single byte is written; a chunked request (no Content-Length at all)
    # still gets the same cap enforced incrementally by ChunkedUpload.write
    # as bytes actually arrive.
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > max_bytes:
                raise InvalidRequestError(
                    f"upload exceeds the {settings.max_upload_mb} MB limit",
                    status=413, code="file_too_large", param="filename",
                )
        except ValueError:
            pass

    settings.ensure_models_dir()
    upload = store.ChunkedUpload(settings.resolved_models_dir, filename, max_bytes)
    try:
        async for chunk in request.stream():
            if chunk:
                await asyncio.to_thread(upload.write, chunk)
    except store.FileTooLarge as exc:
        upload.abort()
        raise InvalidRequestError(str(exc), status=413, code="file_too_large", param="filename") from exc
    except store.BadMagic as exc:
        upload.abort()
        raise InvalidRequestError(str(exc), param="filename") from exc
    except Exception:
        upload.abort()
        raise

    model = upload.finish()
    # Router mode registers a model from what's on disk, so a freshly
    # uploaded checkpoint is invisible to /v1/models (and to resolve())
    # until the manager re-scans. Single-model mode's refresh_from_folder
    # is a no-op, so this is harmless there too.
    request.app.state.slots.refresh_from_folder()
    request.app.state.models_events.publish("models_reload", reason="upload", name=model.name)
    return {"model": _model_dict(model)}


@router.delete("/models/{name}")
def delete_model(name: str, request: Request):
    settings: Settings = request.app.state.settings
    try:
        store.delete_model(settings.resolved_models_dir, name)
    except store.InvalidFilename as exc:
        raise InvalidRequestError(str(exc), param="name") from exc
    except FileNotFoundError as exc:
        raise NotFoundError(f"model {name!r} not found in the models folder") from exc
    request.app.state.slots.refresh_from_folder()
    request.app.state.models_events.publish("models_reload", reason="delete", name=name)
    return {"deleted": name}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def models_sse(request: Request) -> StreamingResponse:
    bus = request.app.state.models_events

    async def stream():
        yield _sse({"event": "connected", "time": time.time()})
        async for event in bus.subscribe():
            yield _sse(event)

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
