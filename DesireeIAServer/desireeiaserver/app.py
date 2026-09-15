"""FastAPI application factory for the DesireeIA server."""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import anyio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import __version__
from .api import chat, embeddings, errors as api_errors, health, models_files, props, settings_api, tokenize
from .api.v1 import build_router as build_v1_router
from .config import Settings
from .logging_setup import get_logger
from .models_events import ModelsEventBus
from .slots import SlotManager

logger = get_logger(__name__)


def _static_directory(settings: Settings) -> Optional[Path]:
    candidates = [settings.ui_dir, Path(__file__).parent / "static"]
    for candidate in candidates:
        if candidate and candidate.is_dir() and (candidate / "index.html").exists():
            return candidate
    return None


async def _background_load_all(manager: SlotManager) -> None:
    """Loads the first replica of every currently-registered model
    sequentially (not concurrently: an unbounded number of simultaneous
    multi-gigabyte loads is a startup-time memory spike nobody asked for,
    and this only runs once, so serial cost is a one-time wait, not an
    ongoing tax). Single-model mode has exactly one group here, so this
    subsumes what used to be a dedicated single-model code path."""
    for model_id, group in manager.groups.items():
        try:
            await anyio.to_thread.run_sync(group.replicas[0].ensure_loaded)
        except Exception as exc:
            logger.warning("background load of %r failed: %s", model_id, exc)


async def _idle_sweep_loop(manager: SlotManager, settings: Settings) -> None:
    # Polls rather than scheduling a per-replica timer: with N replicas
    # across M models this is one lock-guarded pass instead of tracking
    # NxM independent timers, and the cost of a poll that finds nothing to
    # unload is a handful of monotonic() reads under one lock — negligible
    # next to the interval it runs at.
    interval = max(5.0, min(30.0, settings.sleep_idle_seconds / 2))
    while True:
        await asyncio.sleep(interval)
        try:
            unloaded = await anyio.to_thread.run_sync(manager.sweep_idle)
        except Exception:
            logger.exception("idle sweep failed")
            continue
        for model_id in unloaded:
            logger.info("unloaded %r after %ds idle", model_id, settings.sleep_idle_seconds)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    manager = SlotManager(settings)
    app.state.slots = manager
    background_tasks = []
    if settings.models_autoload:
        background_tasks.append(asyncio.create_task(_background_load_all(manager)))
    if settings.sleep_idle_seconds > 0:
        background_tasks.append(asyncio.create_task(_idle_sweep_loop(manager, settings)))
    try:
        yield
    finally:
        for task in background_tasks:
            if not task.done():
                task.cancel()
        manager.shutdown()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="DesireeIA Server", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.state.started_at = time.monotonic()
    app.state.models_events = ModelsEventBus()

    settings.ensure_models_dir()

    app.add_exception_handler(StarletteHTTPException, api_errors.exception_handler)
    app.add_exception_handler(Exception, api_errors.exception_handler)

    app.include_router(health.router)
    app.include_router(props.router)
    app.include_router(settings_api.router)
    app.include_router(tokenize.router)
    app.include_router(chat.router)
    app.include_router(embeddings.router)
    app.include_router(models_files.router)
    app.include_router(build_v1_router())

    static_dir = _static_directory(settings)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app