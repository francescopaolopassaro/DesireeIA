"""/v1/decide (decision models such as Laya) and their directories in the models folder, driven through a fake model."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from desireeiaserver import models_store
from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(model="laya-typed", temperature=0.0, models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


CHOICE = {"type": "choice", "instruction": "Route it.", "options": {"billing": "payments", "infra": "outages"}}


def test_decide_returns_typed_answers(client):
    body = {"state": "The API is down.", "questions": [
        CHOICE,
        {"type": "score", "instruction": "Urgency?", "options": ["low", "medium", "high"]},
        {"type": "noul", "instruction": "Is it an outage?"},
    ]}
    r = client.post("/v1/decide", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["object"] == "decision"
    assert data["model"] == "laya-typed"
    assert [a["type"] for a in data["answers"]] == ["choice", "score", "noul"]
    first = data["answers"][0]
    assert first["options"] == ["billing", "infra"]
    assert first["choice"] == "billing"
    assert abs(sum(first["probabilities"]) - 1.0) < 1e-9
    assert data["answers"][1]["options"] == ["0", "1", "2"]
    assert data["answers"][2]["options"] == ["false", "true"]


def test_decide_hands_state_and_flags_to_the_engine(client):
    state = {"ticket": "double charge", "tier": "gold"}
    r = client.post("/v1/decide", json={"state": state, "questions": [CHOICE], "calibrate": False,
                                        "truncate_left": True})
    assert r.status_code == 200
    model = client.app.state.slots.resolve(None).ensure_loaded()
    assert model.last_decide["state"] == state
    assert model.last_decide["calibrate"] is False
    assert model.last_decide["truncate_left"] is True


@pytest.mark.parametrize("body, param", [
    ({"questions": [CHOICE]}, "state"),
    ({"state": "x"}, "questions"),
    ({"state": "x", "questions": []}, "questions"),
    ({"state": "x", "questions": ["nope"]}, "questions"),
    ({"state": "x", "questions": [{"type": "rank", "instruction": "?"}]}, "questions"),
    ({"state": "x", "questions": [{"type": "choice", "instruction": "?"}]}, "questions"),
    ({"state": "x", "questions": [{"type": "score", "instruction": "?", "options": {"a": "b"}}]}, "questions"),
    ({"state": "x", "questions": [CHOICE], "calibrate": "yes"}, "calibrate"),
    ({"state": "x", "questions": [CHOICE] * 33}, "questions"),
])
def test_decide_rejects_bad_requests(client, body, param):
    r = client.post("/v1/decide", json=body)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == param


def test_engine_side_rejection_is_a_400(client):
    r = client.post("/v1/decide", json={"state": "x", "questions": [
        {"type": "choice", "instruction": "?", "options": ["only-one"]}]})
    assert r.status_code == 400
    assert "at least two options" in r.json()["error"]["message"]


def test_a_regular_model_is_not_a_decision_model(tmp_path, fake_desireeia):
    app = create_app(_make_settings(tmp_path, model="my-model.gguf"))
    with TestClient(app, raise_server_exceptions=False) as c:
        r = c.post("/v1/decide", json={"state": "x", "questions": [CHOICE]})
        assert r.status_code == 400
        assert "not a decision model" in r.json()["error"]["message"]


def test_an_engine_without_decide_says_so(client, monkeypatch):
    model = client.app.state.slots.resolve(None).ensure_loaded()
    monkeypatch.setattr(type(model), "decide", None)
    monkeypatch.delattr(type(model), "decide")
    r = client.post("/v1/decide", json={"state": "x", "questions": [CHOICE]})
    assert r.status_code == 400
    assert "interface 6" in r.json()["error"]["message"]


# -- decision model directories in the models folder ------------------------------------------

def _make_decision_dir(root, name="laya", magic_ok=True):
    d = root / name
    (d / "encoder").mkdir(parents=True)
    (d / "tokenizer").mkdir()
    (d / "rl_agent_config.json").write_text("{}")
    (d / "encoder" / "config.json").write_text("{}")
    (d / "tokenizer" / "tokenizer.json").write_text("{}")
    header = json.dumps({}).encode()
    weights = len(header).to_bytes(8, "little") + header if magic_ok else b"not a safetensors file at all"
    (d / "model.safetensors").write_bytes(weights)
    return d


def test_models_folder_lists_a_decision_model_directory(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    _make_decision_dir(root)
    (root / "plain-folder").mkdir()
    found = list(models_store.list_models(root))
    assert [(m.name, m.format, m.valid) for m in found] == [("laya", "decision", True)]


def test_an_incomplete_decision_directory_is_ignored(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    d = _make_decision_dir(root)
    (d / "tokenizer" / "tokenizer.json").unlink()
    assert list(models_store.list_models(root)) == []


def test_a_decision_directory_with_bad_weights_is_listed_invalid(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    _make_decision_dir(root, magic_ok=False)
    assert [(m.name, m.valid) for m in models_store.list_models(root)] == [("laya", False)]


def test_router_mode_registers_the_decision_directory(tmp_path, fake_desireeia):
    models = tmp_path / "models"
    models.mkdir()
    _make_decision_dir(models)
    app = create_app(_make_settings(tmp_path, model="", models_dir=models))
    with TestClient(app, raise_server_exceptions=False) as c:
        ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
        assert ids == ["laya"]
        r = c.post("/v1/decide", json={"state": "x", "questions": [CHOICE]})
        assert r.status_code == 200
        assert r.json()["model"] == "laya"


def test_chat_on_a_model_that_cannot_generate_is_refused_cleanly(client, monkeypatch):
    """A decision model has no tokenizer for prompts: chat must say so (400), not fail as an opaque 500."""
    model = client.app.state.slots.resolve(None).ensure_loaded()
    monkeypatch.setattr(type(model), "has_tokenizer", property(lambda self: False))
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 400
    assert "/v1/decide" in r.json()["error"]["message"]
