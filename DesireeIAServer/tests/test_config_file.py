"""Config file load, save, round-trip, and the /config endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings, load_config, save_config


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "cfg.json"
    settings = Settings(temperature=0.42, port=9999, model="x.gguf")
    save_config(settings, path)
    loaded = load_config(path, Settings())
    assert loaded.temperature == 0.42
    assert loaded.port == 9999
    assert loaded.model == "x.gguf"


def test_load_missing_file_returns_defaults():
    loaded = load_config(Path("/no/such/file.json"), Settings(port=1234))
    assert loaded.port == 1234


def test_parse_args_loads_config_file(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"temperature": 0.77, "port": 7777}), encoding="utf-8")

    from desireeiaserver.config import parse_args

    settings = parse_args(["--config", str(cfg)])
    assert settings.temperature == 0.77
    assert settings.port == 7777


def test_cli_overrides_config_file(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"temperature": 0.11}), encoding="utf-8")

    from desireeiaserver.config import parse_args

    settings = parse_args(["--config", str(cfg), "--temperature", "0.88"])
    assert settings.temperature == 0.88


def test_config_endpoint_and_post(tmp_path, fake_desireeia):
    settings = Settings(model="m.gguf", temperature=0.0, data_dir=tmp_path / "data", models_dir=tmp_path / "models")
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as client:
        initial = client.get("/config").json()
        assert initial["settings"]["temperature"] == 0.0

        r = client.post("/config", json={"settings": {"temperature": 0.99}})
        assert r.status_code == 200
        assert r.json()["settings"]["temperature"] == 0.99

        refreshed = client.get("/config").json()
        assert refreshed["settings"]["temperature"] == 0.99

        persisted = json.loads(Path(r.json()["saved"]).read_text(encoding="utf-8"))
        assert persisted["temperature"] == 0.99