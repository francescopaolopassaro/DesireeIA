# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Tests for POST /v1/models/{name}/unload and /load - manual model
lifecycle control (freeing VRAM/RAM without restarting the server, and
loading it back on demand).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(model="my-model.gguf", temperature=0.0,
                     models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _aggregate_state(client, model_id: str) -> str:
    return client.app.state.slots.groups[model_id].aggregate_state()


def test_unload_frees_a_loaded_model(client):
    client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert _aggregate_state(client, "my-model") == "loaded"

    r = client.post("/v1/models/my-model/unload")
    assert r.status_code == 200
    assert r.json()["state"] == "unloaded"
    assert _aggregate_state(client, "my-model") == "unloaded"


def test_load_brings_a_model_back(client):
    client.post("/v1/models/my-model/unload")
    r = client.post("/v1/models/my-model/load")
    assert r.status_code == 200
    assert r.json()["state"] == "loaded"
    assert _aggregate_state(client, "my-model") == "loaded"


def test_unload_unknown_model_404s(client):
    r = client.post("/v1/models/nope/unload")
    assert r.status_code == 404


def test_load_unknown_model_404s(client):
    r = client.post("/v1/models/nope/load")
    assert r.status_code == 404


def test_unload_is_idempotent(client):
    r1 = client.post("/v1/models/my-model/unload")
    r2 = client.post("/v1/models/my-model/unload")
    assert r1.status_code == 200 and r2.status_code == 200
    assert r2.json()["state"] == "unloaded"


def test_chat_after_unload_reloads_automatically(client):
    client.post("/v1/models/my-model/unload")
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert _aggregate_state(client, "my-model") == "loaded"
