"""OpenAI-compatible /v1 routes: models list plus stubs for reserved endpoints."""

from __future__ import annotations

import time
from typing import Any

import anyio
from fastapi import APIRouter, Request

from .errors import NotFoundError, NotImplementedEndpointError, UnavailableError
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
            # `state` is always set explicitly here rather than relying on
            # info() to supply it: Slot.info() only includes "state" in its
            # own fallback shape (when not loaded) - when a model IS loaded
            # it returns real engine fields instead (context_size, etc.)
            # with no "state" key at all, which left every LOADED model
            # reporting an empty state to clients that read meta.state.
            meta = dict(info) if info else {"id": model_id}
            meta["state"] = group.aggregate_state()
            model_list.append({
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "desireeia",
                "meta": meta,
                "replicas": len(group.replicas),
            })
        return {"object": "list", "data": model_list}

    @router.get("/models/sse")
    async def models_sse(request: Request):
        return await _models_sse(request)

    def _group_or_404(request: Request, name: str):
        manager = request.app.state.slots
        group = manager.groups.get(name)
        if group is None:
            raise NotFoundError(f"model {name!r} not found")
        return group

    @router.post("/models/{name}/unload")
    async def unload_model(name: str, request: Request) -> dict:
        # Frees whatever VRAM/RAM/CPU threads the checkpoint was holding
        # without touching the folder listing or the group's registration -
        # the next chat request (or a POST .../load) reloads it on demand.
        group = _group_or_404(request, name)
        await anyio.to_thread.run_sync(group.unload_all)
        request.app.state.models_events.publish("models_reload", reason="unload", name=name)
        return {"model": name, "state": group.aggregate_state()}

    @router.post("/models/{name}/load")
    async def load_model(name: str, request: Request) -> dict:
        group = _group_or_404(request, name)
        try:
            for replica in group.replicas:
                await anyio.to_thread.run_sync(replica.ensure_loaded)
        except Exception as exc:
            raise UnavailableError(f"failed to load model {name!r}: {exc}") from exc
        request.app.state.models_events.publish("models_reload", reason="load", name=name)
        return {"model": name, "state": group.aggregate_state()}

    @router.post("/responses")
    def responses(body: Any = None):
        raise NotImplementedEndpointError("responses (" + PENDING)

    return router