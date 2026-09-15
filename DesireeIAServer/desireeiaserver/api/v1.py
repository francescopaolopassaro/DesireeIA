"""OpenAI-compatible /v1 routes: models list plus stubs for reserved endpoints."""

from __future__ import annotations

import time
from typing import Any

from fastapi import APIRouter, Request

from .errors import NotImplementedEndpointError

PENDING = "This endpoint is scaffolded and will be implemented in a later step."


def build_router() -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])

    @router.get("/models")
    def list_models(request: Request) -> dict:
        manager = request.app.state.slots
        model_list = []
        for slot in manager.slots:
            info = slot.info()
            model_list.append({
                "id": slot.model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "desireeia",
                "meta": info or {"id": slot.model_id, "state": slot.state},
            })
        return {"object": "list", "data": model_list}

    @router.get("/models/sse")
    def models_sse():
        raise NotImplementedEndpointError("model events (" + PENDING)

    @router.post("/responses")
    def responses(body: Any = None):
        raise NotImplementedEndpointError("responses (" + PENDING)

    return router