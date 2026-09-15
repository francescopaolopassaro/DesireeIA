"""OpenAI-compatible /v1 routes: models list plus stubs for reserved endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from .errors import NotImplementedEndpointError
from .models_files import models_sse as _models_sse

PENDING = "This endpoint is scaffolded and will be implemented in a later step."


def build_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])

    @router.get("/models")
    def list_models(request: Request) -> dict:
        manager = request.app.state.slots
        model_list = []
        # One row per DISTINCT model, not per replica: with --parallel > 1
        # a model has several replica Slots (manager.slots is that flat,
        # repeats-by-design list), but OpenAI's /v1/models lists MODELS.
        # meta.info comes from whichever replica happens to be loaded, or
        # None if all of them currently are not — the group's aggregate
        # state still reports correctly either way.
        for model_id, group in sorted(manager.groups.items()):
            info = next((r.info() for r in group.replicas if r.info() is not None), None)
            model_list.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "desireeia",
                "meta": info or {"id": model_id, "state": group.aggregate_state()},
                "replicas": len(group.replicas),
            })
        return {"object": "list", "data": model_list}

    @router.get("/models/sse")
    async def models_sse(request: Request):
        return await _models_sse(request)

    @router.post("/responses")
    def responses(body: Any = None):
        raise NotImplementedEndpointError("responses (" + PENDING)

    return router