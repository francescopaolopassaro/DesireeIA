# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""HTTP tests for GET/POST/DELETE /models and /v1/models/sse."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _settings(tmp_path, **overrides) -> Settings:
    kwargs = dict(models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    kwargs.update(overrides)
    return Settings(**kwargs)


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_list_models_empty(client):
    assert client.get("/models").json() == {"models": []}


def test_upload_then_list_then_delete(client):
    body = b"GGUF" + b"\x00" * 40
    r = client.post("/models/upload?filename=demo.gguf", content=body)
    assert r.status_code == 200
    assert r.json()["model"]["valid"] is True

    r = client.get("/models")
    names = [m["name"] for m in r.json()["models"]]
    assert names == ["demo.gguf"]

    r = client.delete("/models/demo.gguf")
    assert r.status_code == 200
    assert client.get("/models").json() == {"models": []}


def test_upload_missing_filename_param(client):
    r = client.post("/models/upload", content=b"GGUF")
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "filename"


def test_upload_rejects_unsupported_extension(client):
    r = client.post("/models/upload?filename=x.exe", content=b"GGUF")
    assert r.status_code == 400


def test_upload_rejects_bad_magic(client):
    r = client.post("/models/upload?filename=x.gguf", content=b"NOTGGUF!!")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request_error"


def test_upload_rejects_oversize(tmp_path, fake_desireeia):
    app = create_app(_settings(tmp_path, max_upload_mb=1))
    client = TestClient(app, raise_server_exceptions=False)
    big = b"GGUF" + b"\x00" * (2 * 1024 * 1024)
    r = client.post("/models/upload?filename=big.gguf", content=big)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "file_too_large"
    # the oversized upload must not leave a partial file behind
    assert list((tmp_path / "models").glob(".upload-*.partial")) == []


def test_delete_missing_returns_404(client):
    r = client.delete("/models/nope.gguf")
    assert r.status_code == 404
    assert r.json()["error"]["type"] == "not_found_error"


def test_delete_traversal_filename_rejected(client):
    # Starlette's own routing normalizes ".." in the path before it ever
    # reaches this handler, so a literal "/models/.." doesn't even match
    # the route (405, no matching path) - the real boundary this exercises
    # is delete_model()'s own containment check, covered directly in
    # test_models_store.py::test_delete_model_rejects_traversal. What's
    # worth confirming at the HTTP layer is that the attempt doesn't
    # somehow succeed.
    r = client.delete("/models/..")
    assert r.status_code != 200


@pytest.mark.anyio
async def test_models_sse_returns_event_stream(tmp_path, fake_desireeia):
    # Deliberately NOT a live client.stream(...) round-trip: this endpoint
    # streams forever by design (a "models changed" push channel), and
    # httpx's ASGI test transport only recognizes a client disconnect once
    # the app's own response finishes — which an infinite stream never
    # does on its own. Driving it through a real HTTP round-trip in a test
    # hangs waiting for a disconnect signal the test transport never sends
    # (confirmed directly against StreamingResponse's own disconnect
    # listener). The event *content* is already covered without HTTP by
    # test_models_events.py; this only checks the route is wired to it
    # and answers with the right shape.
    from starlette.requests import Request

    from desireeiaserver.app import create_app
    from desireeiaserver.api.models_files import models_sse

    app = create_app(_settings(tmp_path))
    scope = {
        "type": "http", "method": "GET", "path": "/v1/models/sse",
        "headers": [], "query_string": b"", "app": app,
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    response = await models_sse(request)
    assert response.status_code == 200
    assert response.media_type == "text/event-stream"

    # Pull exactly one chunk from the underlying generator without ever
    # driving it through the ASGI send loop, so there's nothing here that
    # can block waiting for a disconnect.
    first_chunk = await response.body_iterator.__anext__()
    assert "connected" in first_chunk
    await response.body_iterator.aclose()
