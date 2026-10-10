# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Server wiring of the engine context memory: the /tools/open_file and
/tools/memory_* endpoints and the automatic spill of oversized tool
results in /v1/chat/completions. Runs against the real native engine
(skipped when the engine library can't be loaded)."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from desireeiaserver.api.chat import _spill_tool_results
from desireeiaserver.app import create_app
from desireeiaserver.config import Settings

desireeia = pytest.importorskip("desireeia")


def _memory(tmp_path):
    try:
        return desireeia.ContextMemory(str(tmp_path / "mem"), 600)
    except OSError as exc:  # pragma: no cover - no native library here
        pytest.skip(f"native engine not loadable: {exc}")


@pytest.fixture
def client(tmp_path, fake_desireeia):
    settings = Settings(model="m.gguf", models_dir=tmp_path / "models", data_dir=tmp_path / "data",
                        enable_python_tool=True)
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as c:
        app.state.memory = _memory(tmp_path)
        yield c
        app.state.memory.close()


def _workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    lines = [f"def f{i}():\n    return {i}\n" for i in range(5000)]
    lines[2500] = "def target():\n    return 'old'\n"
    (ws / "big.py").write_text("".join(lines), encoding="utf-8")
    return ws


def test_open_search_read_edit_roundtrip(client, tmp_path):
    ws = _workspace(tmp_path)
    base = {"workspace_path": str(ws), "session_id": "chat1"}

    stub = client.post("/tools/open_file", json={**base, "path": "big.py"}).json()
    assert stub["spilled"] is True and stub["total_lines"] == 10000
    assert len(json.dumps(stub)) < 5000  # a stub, not the 100 KB file

    hits = client.post("/tools/memory_search", json={**base, "handle": stub["handle"], "query": "target"}).json()
    assert "def target" in hits["results"][0]["snippet"]

    window = client.post("/tools/memory_read", json={**base, "handle": stub["handle"], "offset": 5001, "limit": 2}).json()
    assert window["content"] == "5001\tdef target():\n5002\t    return 'old'"

    edit = client.post("/tools/memory_replace_lines", json={
        **base, "handle": stub["handle"], "first": 5002, "last": 5002, "text": "    return 'new'"}).json()
    assert edit["ok"] is True and edit["written_to_source"] is True
    assert "    return 'new'\n" in (ws / "big.py").read_text(encoding="utf-8")


def test_open_file_requires_session_and_stays_in_workspace(client, tmp_path):
    ws = _workspace(tmp_path)
    r = client.post("/tools/open_file", json={"workspace_path": str(ws), "path": "big.py"})
    assert r.status_code == 400 and r.json()["error"]["param"] == "session_id"
    r = client.post("/tools/open_file", json={"workspace_path": str(ws), "session_id": "s", "path": "../x"})
    assert r.status_code == 400


def test_memory_append_streams(client):
    first = client.post("/tools/memory_append", json={"session_id": "s", "name": "log", "text": "a\nb\n"}).json()
    second = client.post("/tools/memory_append", json={"session_id": "s", "name": "log", "text": "c\n"}).json()
    assert first["handle"] == second["handle"] and second["total_lines"] == 3 and second["version"] == 2


def test_big_tool_results_are_spilled_with_stable_stubs(tmp_path):
    mem = _memory(tmp_path)
    try:
        big = "\n".join(f"line {i}" for i in range(5000))
        msgs = [{"role": "user", "content": "hi"},
                {"role": "tool", "content": big, "name": "run_python"},
                {"role": "tool", "content": "small", "name": "x"}]
        _spill_tool_results(msgs, mem, "chat9", 4000)
        stub = json.loads(msgs[1]["content"])
        assert stub["spilled"] is True and stub["total_lines"] == 5000
        assert msgs[2]["content"] == "small"

        again = [{"role": "tool", "content": big, "name": "run_python"}]
        _spill_tool_results(again, mem, "chat9", 4000)
        assert again[0]["content"] == msgs[1]["content"]  # identical prefix -> KV reuse
    finally:
        mem.close()


def test_spill_is_skipped_without_session(tmp_path):
    msgs = [{"role": "tool", "content": "x" * 10000}]
    _spill_tool_results(msgs, object(), None, 4000)
    assert msgs[0]["content"] == "x" * 10000
