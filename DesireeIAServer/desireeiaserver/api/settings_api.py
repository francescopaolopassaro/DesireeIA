"""Runtime configuration: GET/POST /config to read, update and persist settings."""

from __future__ import annotations

import json

from fastapi import APIRouter, Body, Request

from ..config import Settings, save_config, _apply_overrides
from .errors import InvalidRequestError

router = APIRouter(tags=["config"])


def _settings_dict(settings: Settings) -> dict:
    data = settings.to_dict()
    data["resolved_data_dir"] = str(settings.resolved_data_dir)
    data["resolved_models_dir"] = str(settings.resolved_models_dir)
    data["resolved_config_path"] = str(settings.resolved_config_path)
    return data


@router.get("/config")
def get_config(request: Request):
    settings: Settings = request.app.state.settings
    return {"settings": _settings_dict(settings), "path": str(settings.resolved_config_path)}


@router.post("/config")
def post_config(request: Request, body: dict = Body(...)):
    settings: Settings = request.app.state.settings
    updates = body.get("settings")
    if not isinstance(updates, dict):
        raise InvalidRequestError("field 'settings' must be a JSON object", param="settings")

    new_settings = _apply_overrides(settings, updates)
    save_config(new_settings, new_settings.resolved_config_path)
    request.app.state.settings = new_settings

    try:
        request.app.state.slots.reconfigure(new_settings)
    except Exception:
        pass

    return {
        "saved": str(new_settings.resolved_config_path),
        "settings": _settings_dict(new_settings),
    }