"""HTTP smoke tests for /health, /props, /v1/models, and /config endpoint."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _settings(tmp_path) -> Settings:
    return Settings(models_dir=tmp_path / "models", data_dir=tmp_path / "data")


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def test_health_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine_available"] is True
    assert body["engine_version"] == "9.9.9-fake"


def test_health_v1_alias(client):
    assert client.get("/v1/health").status_code == 200


def test_version_endpoint(client):
    body = client.get("/version").json()
    assert body["server"]
    assert body["engine"] == "9.9.9-fake"


def test_props_expose_sampling_defaults(client, tmp_path):
    body = client.get("/props").json()
    assert body["default_generation_settings"]["temperature"] == 0.0
    assert body["models_dir"] == str(tmp_path / "models")
    assert "config_path" in body


def test_props_reflect_settings(tmp_path, fake_desireeia):
    settings = Settings(
        models_dir=tmp_path / "models",
        temperature=0.6,
        n_predict=256,
        data_dir=tmp_path / "data",
    )
    client = TestClient(create_app(settings), raise_server_exceptions=False)
    body = client.get("/props").json()
    assert body["default_generation_settings"]["temperature"] == 0.6
    assert body["default_generation_settings"]["n_predict"] == 256


def test_list_models_openai_shape(client):
    body = client.get("/v1/models").json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)


def test_pending_responses_returns_501(client):
    response = client.post("/v1/responses", json={})
    assert response.status_code == 501


def test_unknown_route_returns_error_envelope(client):
    response = client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["type"] == "not_found_error"


def test_ui_served_at_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "DesireeIA" in response.text


def test_models_dir_created_on_startup(tmp_path, fake_desireeia):
    create_app(Settings(models_dir=tmp_path / "auto", data_dir=tmp_path / "data"))
    assert (tmp_path / "auto").is_dir()