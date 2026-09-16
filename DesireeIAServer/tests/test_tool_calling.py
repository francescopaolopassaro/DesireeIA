# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Tests for OpenAI-style tool/function calling: the toolcalling.py parsing
helpers, and end-to-end wiring through /v1/chat/completions (sync + stream).
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from desireeiaserver import toolcalling
from desireeiaserver.app import create_app
from desireeiaserver.config import Settings

WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a location.",
        "parameters": {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location"],
        },
    },
}


# -- toolcalling.py unit tests ----------------------------------------------

def test_build_tools_instruction_lists_name_and_schema():
    instruction = toolcalling.build_tools_instruction([WEATHER_TOOL])
    assert "get_weather" in instruction
    assert "Get the current weather" in instruction
    assert '"location"' in instruction
    assert toolcalling.TOOL_CALL_OPEN in instruction


def test_extract_tool_call_valid():
    text = 'Sure, let me check.\n<tool_call>{"name": "get_weather", "arguments": {"location": "Rome"}}</tool_call>'
    result = toolcalling.extract_tool_call(text)
    assert result == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_extract_tool_call_function_call_tag():
    text = '<function_call>{"name": "get_weather", "arguments": {"location": "Rome"}}</function_call>'
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_extract_tool_call_function_named_tag_with_arguments_key():
    text = '<function=get_weather>{"arguments": {"location": "Rome"}}</function>'
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_extract_tool_call_function_named_tag_bare_args():
    text = '<function name="get_weather">{"location": "Rome"}</function>'
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_extract_tool_call_function_inline_name():
    text = '<function>get_weather{"location": "Rome"}</function>'
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_extract_tool_call_arg_key_value_tags():
    # Observed directly from a real local model asked to call list_files:
    # instead of a JSON arguments object it invented its own per-argument
    # tag pair.
    text = "<tool_call>list_files<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>"
    assert toolcalling.extract_tool_call(text) == {"name": "list_files", "arguments": {"path": "."}}


def test_extract_tool_call_arg_key_value_tags_multiple_args():
    text = (
        "<tool_call>write_file"
        "<arg_key>path</arg_key><arg_value>notes.txt</arg_value>"
        "<arg_key>content</arg_key><arg_value>hello</arg_value>"
        "</tool_call>"
    )
    assert toolcalling.extract_tool_call(text) == {
        "name": "write_file", "arguments": {"path": "notes.txt", "content": "hello"},
    }


def test_extract_tool_call_prefers_first_recognized_dialect():
    # A response that happens to contain both should not error - the first
    # match wins, same behavior as before this could ever happen.
    text = '<tool_call>{"name": "a", "arguments": {}}</tool_call> then <function_call>{"name": "b", "arguments": {}}</function_call>'
    assert toolcalling.extract_tool_call(text)["name"] == "a"


def test_extract_tool_call_no_marker_returns_none():
    assert toolcalling.extract_tool_call("just a normal answer") is None


def test_extract_tool_call_malformed_json_returns_none():
    assert toolcalling.extract_tool_call("<tool_call>{not json</tool_call>") is None


def test_extract_tool_call_missing_name_returns_none():
    assert toolcalling.extract_tool_call('<tool_call>{"arguments": {}}</tool_call>') is None


def test_render_tool_call_text_roundtrips():
    text = toolcalling.render_tool_call_text("get_weather", {"location": "Rome"})
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


def test_render_tool_call_text_accepts_json_string_arguments():
    text = toolcalling.render_tool_call_text("get_weather", '{"location": "Rome"}')
    assert toolcalling.extract_tool_call(text) == {"name": "get_weather", "arguments": {"location": "Rome"}}


# -- end-to-end wiring --------------------------------------------------------

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


def _emit_as_single_token(fake_desireeia, monkeypatch, text: str) -> None:
    """Make the fake model's very first (and only meaningful) token equal to
    the full given text, so a test can control exactly what the model
    "said" without needing a real generation loop."""
    monkeypatch.setattr(fake_desireeia.LocalModel, "PIECES", {10: text, 11: "", 12: ""})


def test_chat_without_tools_is_unaffected(client):
    r = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "TokenA TokenB "
    assert "tool_calls" not in r.json()["choices"][0]["message"]


def test_chat_sync_detects_tool_call(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(
        fake_desireeia, monkeypatch,
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Rome"}}</tool_call>',
    )
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "what's the weather in Rome?"}],
        "tools": [WEATHER_TOOL],
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None
    call = choice["message"]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "Rome"}


def test_chat_sync_with_tools_but_no_call_answers_normally(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(fake_desireeia, monkeypatch, "It's sunny today.")
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEATHER_TOOL],
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] != "tool_calls"
    assert choice["message"]["content"] == "It's sunny today."


def test_chat_with_tool_choice_none_disables_detection(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(
        fake_desireeia, monkeypatch,
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Rome"}}</tool_call>',
    )
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather?"}],
        "tools": [WEATHER_TOOL],
        "tool_choice": "none",
    })
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] != "tool_calls"
    assert choice["message"]["content"].startswith("<tool_call>")


def test_chat_stream_detects_tool_call(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(
        fake_desireeia, monkeypatch,
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Milan"}}</tool_call>',
    )
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Milan?"}],
        "tools": [WEATHER_TOOL],
        "stream": True,
    }) as response:
        assert response.status_code == 200
        text = "".join(response.iter_text())

    lines = [line for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    payloads = [json.loads(line[6:]) for line in lines]

    # Content streams through live and unconditionally now (see chat.py's
    # _chat_stream) - the raw "<tool_call>...>" markup IS visible as
    # content deltas while streaming. What matters is that the FINAL chunk
    # still correctly reports it as a tool call, not plain text.
    tool_call_chunks = [p for p in payloads if p["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_call_chunks) == 1
    call = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert json.loads(call["function"]["arguments"]) == {"location": "Milan"}
    assert tool_call_chunks[0]["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_stream_detects_tool_call_prefaced_with_explanatory_text(client, fake_desireeia, monkeypatch):
    # Regression test: a weak model that ignores "respond with ONLY the
    # tool call" and adds explanatory prose FIRST ("To do that, I need to
    # call...") used to never be detected at all - detection only looked
    # at whether the response's OPENING characters matched a known tool-
    # call marker. extract_tool_call() searches the whole accumulated
    # text, not just the start, specifically so this case is still caught.
    _emit_as_single_token(
        fake_desireeia, monkeypatch,
        'To check that, I need to call a tool:\n\n'
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Milan"}}</tool_call>',
    )
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Milan?"}],
        "tools": [WEATHER_TOOL],
        "stream": True,
    }) as response:
        text = "".join(response.iter_text())
    lines = [line for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    payloads = [json.loads(line[6:]) for line in lines]
    tool_call_chunks = [p for p in payloads if p["choices"][0]["delta"].get("tool_calls")]
    assert len(tool_call_chunks) == 1
    call = tool_call_chunks[0]["choices"][0]["delta"]["tool_calls"][0]
    assert call["function"]["name"] == "get_weather"
    assert tool_call_chunks[0]["choices"][0]["finish_reason"] == "tool_calls"


def test_chat_sync_detects_tool_call_prefaced_with_explanatory_text(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(
        fake_desireeia, monkeypatch,
        'To check that, I need to call a tool:\n\n'
        '<tool_call>{"name": "get_weather", "arguments": {"location": "Milan"}}</tool_call>',
    )
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "weather in Milan?"}],
        "tools": [WEATHER_TOOL],
    })
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["function"]["name"] == "get_weather"


def test_chat_stream_without_tool_call_streams_normally(client, fake_desireeia, monkeypatch):
    _emit_as_single_token(fake_desireeia, monkeypatch, "Hello there, friend!")
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEATHER_TOOL],
        "stream": True,
    }) as response:
        text = "".join(response.iter_text())
    lines = [line for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    payloads = [json.loads(line[6:]) for line in lines]
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert content == "Hello there, friend!"


def test_chat_stream_short_reply_resembling_a_tool_call_prefix_is_not_lost(client, fake_desireeia, monkeypatch):
    # A reply that merely starts with "<" and never becomes a real,
    # parseable tool call (extract_tool_call finds nothing) must still
    # reach the client as ordinary content - content streams through live
    # now, so this mostly guards against a regression back to any kind of
    # mid-stream withholding.
    _emit_as_single_token(fake_desireeia, monkeypatch, "<to")
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [WEATHER_TOOL],
        "stream": True,
    }) as response:
        text = "".join(response.iter_text())
    lines = [line for line in text.splitlines() if line.startswith("data: ") and line[6:] != "[DONE]"]
    payloads = [json.loads(line[6:]) for line in lines]
    content = "".join(p["choices"][0]["delta"].get("content", "") for p in payloads)
    assert content == "<to"


def test_tool_result_message_round_trips_through_a_follow_up_request(client, fake_desireeia, monkeypatch):
    """After a tool call, the client is expected to send the conversation
    back with the assistant's tool_calls message and a role="tool" message
    carrying the result - this must not error, and the reply should be
    generated as plain text again."""
    _emit_as_single_token(fake_desireeia, monkeypatch, "It's 21C and sunny in Rome.")
    r = client.post("/v1/chat/completions", json={
        "messages": [
            {"role": "user", "content": "weather in Rome?"},
            {"role": "assistant", "content": None, "tool_calls": [{
                "id": "call_1", "type": "function",
                "function": {"name": "get_weather", "arguments": '{"location": "Rome"}'},
            }]},
            {"role": "tool", "tool_call_id": "call_1", "name": "get_weather",
             "content": '{"temperature_c": 21, "condition": "sunny"}'},
        ],
        "tools": [WEATHER_TOOL],
    })
    assert r.status_code == 200
    assert r.json()["choices"][0]["message"]["content"] == "It's 21C and sunny in Rome."
