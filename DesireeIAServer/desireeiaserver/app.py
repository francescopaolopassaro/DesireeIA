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
from .api import chat, embeddings, errors as api_errors, health, props, settings_api, tokenize
from .api.v1 import build_router as build_v1_router
from .config import Settings
from .logging_setup import get_logger
from .slots import SlotManager

logger = get_logger(__name__)


def _static_directory(settings: Settings) -> Optional[Path]:
    candidates = [settings.ui_dir, Path(__file__).parent / "static"]
    for candidate in candidates:
        if candidate and candidate.is_dir() and (candidate / "index.html").exists():
            return candidate
    return None


@asynccontextmanager
async def _lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    manager = SlotManager(settings)
    app.state.slots = manager
    load_task = None
    if settings.model:
        async def _background_load():
            try:
                await anyio.to_thread.run_sync(manager.resolve(None).ensure_loaded)
            except Exception as exc:
                logger.warning("background model load failed: %s", exc)
        load_task = asyncio.create_task(_background_load())
    try:
        yield
    finally:
        if load_task is not None and not load_task.done():
            load_task.cancel()
        manager.shutdown()


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="DesireeIA Server", version=__version__, lifespan=_lifespan)
    app.state.settings = settings
    app.state.started_at = time.monotonic()

    settings.ensure_models_dir()

    app.add_exception_handler(StarletteHTTPException, api_errors.exception_handler)
    app.add_exception_handler(Exception, api_errors.exception_handler)

    app.include_router(health.router)
    app.include_router(props.router)
    app.include_router(settings_api.router)
    app.include_router(tokenize.router)
    app.include_router(chat.router)
    app.include_router(embeddings.router)
    app.include_router(build_v1_router())

    static_dir = _static_directory(settings)
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="ui")

    return app