"""Live HTTP tests for chat, completions, tokenizer and embeddings."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(
        model="my-model.gguf",
        temperature=0.0,
        models_dir=tmp_path / "models",
        data_dir=tmp_path / "data",
    )
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _ok(body=None):
    return body or {"messages": [{"role": "user", "content": "hi"}]}


def test_chat_sync(client):
    r = client.post("/v1/chat/completions", json=_ok())
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "TokenA TokenB "
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"]["completion_tokens"] == 2
    assert body["timings"]["predicted_n"] == 2


def test_chat_sync_with_full_sampling_override(client):
    # Regression test: slots.py::_apply_sampling used to build its
    # dataclasses.replace() kwargs with OpenAI-shaped field names
    # (repeat_penalty, frequency_penalty, presence_penalty, repeat_last_n)
    # instead of the real SamplingOptions names (penalty_repeat,
    # penalty_frequency, penalty_presence, penalty_last_n) — replace()
    # raised TypeError for an unrecognized kwarg, but only when at least
    # one of these fields was actually set to a non-falsy value. The only
    # other sampling test here sends temperature=0.0 and nothing else, so
    # that branch was never exercised: this failed on every real chat UI
    # request (its sliders always send non-default values) while every
    # existing test stayed green. Every field that maps to a differently-
    # named SamplingOptions field is set here specifically to catch that
    # class of mismatch again if it comes back.
    r = client.post("/v1/chat/completions", json=_ok({
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.8, "top_p": 0.9, "top_k": 50,
        "repeat_penalty": 1.1, "frequency_penalty": 0.2,
        "presence_penalty": 0.1, "repeat_last_n": 32,
    }))
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "TokenA TokenB "


def test_chat_stream(client):
    with client.stream("POST", "/v1/chat/completions", json={**_ok(), "stream": True}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    lines = [line for line in text.splitlines() if line.startswith("data: ")]
    payloads = [json.loads(line[6:]) for line in lines if line[6:] != "[DONE]"]

    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert content == "TokenA TokenB "

    final = payloads[-1]
    assert final["choices"][0]["finish_reason"] == "stop"
    assert final["usage"]["completion_tokens"] == 2
    assert "timings" in final
    assert lines[-1] == "data: [DONE]"


def test_completions_sync(client):
    r = client.post("/v1/completions", json={"prompt": "P>"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "text_completion"
    assert body["choices"][0]["finish_reason"] == "stop"


def test_completions_echo(client):
    r = client.post("/v1/completions", json={"prompt": "P>", "echo": True})
    text = r.json()["choices"][0]["text"]
    assert text.startswith("P>")


def test_completions_stream(client):
    with client.stream("POST", "/v1/completions", json={"prompt": "hello", "stream": True}) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())
    lines = [line for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    final = json.loads(lines[-1][6:])
    assert final["choices"][0]["finish_reason"] == "stop"


def test_tokenize(client):
    r = client.post("/tokenize", json={"content": "hello", "with_pieces": True})
    assert r.json()["tokens"] == [1, 2, 3, 4]
    assert len(r.json()["pieces"]) == 4


def test_detokenize(client):
    r = client.post("/detokenize", json={"tokens": [10, 11]})
    assert r.json()["text"] == "TokenA TokenB "


def test_apply_template(client):
    r = client.post("/apply-template", json={"messages": [{"role": "user", "content": "hi"}]})
    assert "TEMPLATE|" in r.json()["prompt"]


def test_embeddings_unsupported(client):
    r = client.post("/v1/embeddings", json={"input": "hello"})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "not_supported_error"


def test_models_lists_single(client):
    body = client.get("/v1/models").json()
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "my-model"


def test_chat_invalid_messages(client):
    r = client.post("/v1/chat/completions", json={"messages": "not-a-list"})
    assert r.status_code == 400


def test_chat_empty_messages(client):
    r = client.post("/v1/chat/completions", json={"messages": []})
    assert r.status_code == 400


def test_chat_stream_no_stream_flag(client):
    r = client.post("/v1/chat/completions", json=_ok())
    assert r.json()["object"] == "chat.completion"