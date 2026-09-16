"""Server properties: current settings and the loaded model's info."""

from __future__ import annotations

from fastapi import APIRouter, Request

from .. import __version__
from .. import engine as engine_mod
from ..config import Settings

router = APIRouter(tags=["props"])


@router.get("/props")
def props(request: Request):
    settings: Settings = request.app.state.settings
    slots = getattr(request.app.state, "slots", None)
    model_info = None
    if slots is not None and slots.slots:
        model_info = slots.slots[0].info()
    return {
        "default_generation_settings": engine_mod.default_generation_settings(settings),
        "alias": settings.alias,
        "model": settings.model,
        "model_info": model_info,
        "models_dir": str(settings.resolved_models_dir),
        "config_path": str(settings.resolved_config_path),
        "parallel": settings.parallel,
        "server_version": __version__,
        "tools": {"run_python": settings.enable_python_tool},
        "whisper": settings.enable_whisper,
    }