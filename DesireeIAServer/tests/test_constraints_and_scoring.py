"""response_format enforcement, logit_bias and /v1/score (engine 0.3.0 features, driven through a fake model)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _make_settings(tmp_path, **overrides) -> Settings:
    defaults = dict(model="my-model.gguf", temperature=0.0, models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    defaults.update(overrides)
    return Settings(**defaults)


@pytest.fixture
def client(tmp_path, fake_desireeia):
    app = create_app(_make_settings(tmp_path))
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


def _model(client):
    return client.app.state.slots.resolve(None).ensure_loaded()


def _chat(client, **extra):
    body = {"messages": [{"role": "user", "content": "hi"}]}
    body.update(extra)
    return client.post("/v1/chat/completions", json=body)


SCHEMA = {"type": "object", "properties": {"a": {"type": "integer"}}, "required": ["a"]}


def test_json_schema_is_handed_to_the_engine(client):
    r = _chat(client, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": SCHEMA}})
    assert r.status_code == 200
    assert json.loads(_model(client).json_schema) == SCHEMA


def test_json_object_means_any_object_and_no_format_turns_it_off(client):
    assert _chat(client, response_format={"type": "json_object"}).status_code == 200
    assert json.loads(_model(client).json_schema) == {"type": "object"}
    assert _chat(client).status_code == 200
    assert _model(client).json_schema is None            # the constraint persists on the model: a plain request clears it


def test_a_schema_the_engine_refuses_is_a_400(client):
    r = _chat(client, response_format={"type": "json_schema", "json_schema": {"name": "x", "schema": "BAD"}})
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "response_format"


def test_logit_bias_is_passed_and_cleared(client):
    assert _chat(client, logit_bias={"10": -100, "11": 5.5}).status_code == 200
    assert _model(client).logit_bias == {10: -100.0, 11: 5.5}
    assert _chat(client).status_code == 200
    assert _model(client).logit_bias is None


@pytest.mark.parametrize("bad", [[1], {"x": 1}, {"10": "high"}, {"10": 101}, {"-1": 1}, {"10": True}])
def test_logit_bias_is_validated(client, bad):
    r = _chat(client, logit_bias=bad)
    assert r.status_code == 400
    assert r.json()["error"]["param"] == "logit_bias"


def test_score_text(client):
    r = client.post("/v1/score", json={"prompt": "Rate this", "text": "good day to you"})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "score"
    assert len(body["tokens"]) == len(body["logprobs"]) == len(body["entropy"]) == 4
    assert body["total_logprob"] == pytest.approx(-2.0)
    assert body["mean_logprob"] == pytest.approx(-0.5)
    assert body["perplexity"] == pytest.approx(2.718281828 ** 0.5, rel=1e-3)


def test_score_candidates_picks_the_likeliest(client):
    r = client.post("/v1/score", json={"prompt": "Answer:", "candidates": ["bad answer", "good answer", "meh"]})
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "score.ranking"
    assert body["best"] == 1
    assert len(body["candidates"]) == 3


@pytest.mark.parametrize("body", [
    {},                                                    # neither text nor candidates
    {"text": "a", "candidates": ["b"], "prompt": "p"},     # both
    {"text": ""},
    {"text": 5},
    {"prompt": "p", "candidates": []},
    {"prompt": "p", "candidates": ["ok", ""]},
    {"candidates": ["a", "b"]},                            # ranking needs a prompt
    {"prompt": 3, "text": "a"},
])
def test_score_validation(client, body):
    assert client.post("/v1/score", json=body).status_code == 400


def test_n_answers_to_one_prompt(client):
    r = _chat(client, n=3)
    assert r.status_code == 200
    body = r.json()
    assert [c["index"] for c in body["choices"]] == [0, 1, 2]
    assert all(c["message"]["content"] for c in body["choices"])
    # the prompt is counted once, every answer adds its own tokens
    one = _chat(client).json()["usage"]
    assert body["usage"]["prompt_tokens"] == one["prompt_tokens"]
    assert body["usage"]["completion_tokens"] >= 3
    assert body["usage"]["total_tokens"] == body["usage"]["prompt_tokens"] + body["usage"]["completion_tokens"]


def test_n_cannot_be_streamed(client):
    assert _chat(client, n=2, stream=True).status_code == 400 or _chat(client, n=2, stream=True).status_code == 501
