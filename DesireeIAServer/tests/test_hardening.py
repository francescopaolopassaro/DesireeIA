# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Tests for the hardening middleware (api key, body size, security
headers), the /metrics endpoint, and the run_python sandbox tool.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from desireeiaserver import sandbox
from desireeiaserver.app import create_app
from desireeiaserver.config import Settings
from desireeiaserver.metrics import metrics


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(model="my-model.gguf", temperature=0.0,
                     models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    defaults.update(overrides)
    return Settings(**defaults)


def _client(tmp_path, **overrides):
    app = create_app(_settings(tmp_path, **overrides))
    return TestClient(app, raise_server_exceptions=False)


# -- API key middleware -------------------------------------------------------

def test_no_api_key_configured_leaves_endpoints_open(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200


def test_protected_endpoint_rejects_missing_key(tmp_path, fake_desireeia):
    with _client(tmp_path, api_keys=("secret",)) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401
        assert r.json()["error"]["type"] == "authentication_error"


def test_protected_endpoint_rejects_wrong_key(tmp_path, fake_desireeia):
    with _client(tmp_path, api_keys=("secret",)) as c:
        r = c.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer nope"},
                    json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 401


def test_protected_endpoint_accepts_correct_key(tmp_path, fake_desireeia):
    with _client(tmp_path, api_keys=("secret",)) as c:
        r = c.post("/v1/chat/completions",
                    headers={"Authorization": "Bearer secret"},
                    json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200


def test_health_stays_open_even_with_api_key_configured(tmp_path, fake_desireeia):
    with _client(tmp_path, api_keys=("secret",)) as c:
        r = c.get("/health")
        assert r.status_code == 200


# -- request body size cap ----------------------------------------------------

def test_oversized_body_rejected(tmp_path, fake_desireeia):
    with _client(tmp_path, max_request_mb=1) as c:
        huge_content = "x" * (2 * 1024 * 1024)
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": huge_content}]})
        assert r.status_code == 413
        assert r.json()["error"]["code"] == "request_too_large"


def test_body_within_limit_is_accepted(tmp_path, fake_desireeia):
    with _client(tmp_path, max_request_mb=1) as c:
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert r.status_code == 200


# -- security headers ----------------------------------------------------------

def test_security_headers_present(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.get("/health")
        assert r.headers["x-content-type-options"] == "nosniff"
        assert r.headers["x-frame-options"] == "DENY"
        assert r.headers["referrer-policy"] == "no-referrer"


# -- /metrics -------------------------------------------------------------------

def test_metrics_endpoint_reports_counters(tmp_path, fake_desireeia):
    # `metrics` is a process-wide singleton (by design - see metrics.py),
    # so other tests in this module that also hit /v1/chat/completions
    # bump the same counters. Assert it went up, not an exact absolute
    # value, so this stays order-independent.
    before = metrics.chat_requests_total
    with _client(tmp_path) as c:
        c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        r = c.get("/metrics")
    assert r.status_code == 200
    body = r.text
    assert metrics.chat_requests_total == before + 1
    assert f"desireeia_chat_requests_total {before + 1}" in body
    assert 'desireeia_http_requests_total{method="POST",route="/v1/chat/completions",status="200"}' in body
    assert "desireeia_model_loaded{" in body


# -- run_python sandbox tool ---------------------------------------------------

def test_run_python_disabled_by_default(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.post("/tools/run_python", json={"code": "print(1)"})
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "not_supported_error"


def test_run_python_enabled_executes_code(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/run_python", json={"code": "print(1 + 1)"})
        assert r.status_code == 200
        body = r.json()
        assert body["exit_code"] == 0
        assert body["stdout"].strip() == "2"
        assert body["timed_out"] is False


def test_run_python_captures_stderr_and_nonzero_exit(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/run_python", json={"code": "raise ValueError('boom')"})
        assert r.status_code == 200
        body = r.json()
        assert body["exit_code"] != 0
        assert "boom" in body["stderr"]


def test_run_python_requires_code_field(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/run_python", json={})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "code"


def test_run_python_workspace_persists_across_calls(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r1 = c.post("/tools/run_python", json={
            "code": "open('note.txt', 'w').write('hello')",
            "workspace_id": "sess_abc123",
        })
        assert r1.status_code == 200
        assert r1.json()["exit_code"] == 0

        r2 = c.post("/tools/run_python", json={
            "code": "print(open('note.txt').read())",
            "workspace_id": "sess_abc123",
        })
        assert r2.status_code == 200
        assert r2.json()["stdout"].strip() == "hello"


def test_run_python_different_workspaces_are_isolated(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        c.post("/tools/run_python", json={
            "code": "open('note.txt', 'w').write('a')", "workspace_id": "sess_a",
        })
        r = c.post("/tools/run_python", json={
            "code": "import os; print(os.path.exists('note.txt'))", "workspace_id": "sess_b",
        })
        assert r.json()["stdout"].strip() == "False"


def test_run_python_rejects_path_traversal_workspace_id(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/run_python", json={"code": "print(1)", "workspace_id": "../../etc"})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "workspace_id"


def test_run_python_without_workspace_id_still_works(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/run_python", json={"code": "print(2 + 2)"})
        assert r.status_code == 200
        assert r.json()["stdout"].strip() == "4"


# -- direct file tools (list/read/write/search) --------------------------------

def test_write_then_read_file_roundtrips(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        w = c.post("/tools/write_file", json={
            "workspace_id": "sess_files", "path": "notes/todo.txt", "content": "buy milk",
        })
        assert w.status_code == 200
        body = w.json()
        assert body["path"] == "notes/todo.txt"
        assert body["bytes_written"] == 8
        assert body["absolute_path"].endswith("notes\\todo.txt") or body["absolute_path"].endswith("notes/todo.txt")

        r = c.post("/tools/read_file", json={"workspace_id": "sess_files", "path": "notes/todo.txt"})
        assert r.status_code == 200
        assert r.json()["content"] == "buy milk"


def test_list_files_shows_written_entries(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        c.post("/tools/write_file", json={"workspace_id": "sess_list", "path": "a.txt", "content": "x"})
        c.post("/tools/write_file", json={"workspace_id": "sess_list", "path": "sub/b.txt", "content": "y"})
        r = c.post("/tools/list_files", json={"workspace_id": "sess_list"})
        assert r.status_code == 200
        names = {e["name"] for e in r.json()["entries"]}
        assert names == {"a.txt", "sub"}


def test_search_files_matches_name_and_content(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        c.post("/tools/write_file", json={"workspace_id": "sess_search", "path": "report.txt", "content": "quarterly numbers"})
        c.post("/tools/write_file", json={"workspace_id": "sess_search", "path": "other.txt", "content": "unrelated"})

        by_name = c.post("/tools/search_files", json={"workspace_id": "sess_search", "query": "report"})
        assert [r["path"] for r in by_name.json()["results"]] == ["report.txt"]

        by_content = c.post("/tools/search_files", json={"workspace_id": "sess_search", "query": "quarterly"})
        assert [r["path"] for r in by_content.json()["results"]] == ["report.txt"]


def test_read_file_missing_returns_400(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/read_file", json={"workspace_id": "sess_missing", "path": "nope.txt"})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "path"


def test_write_file_rejects_path_traversal(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/write_file", json={
            "workspace_id": "sess_escape", "path": "../../outside.txt", "content": "x",
        })
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "path"


def test_file_tools_require_workspace_id(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/list_files", json={})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "workspace_id"


def test_file_tools_disabled_by_default(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.post("/tools/write_file", json={"workspace_id": "s", "path": "a.txt", "content": "x"})
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "not_supported_error"


# -- real-directory workspace (workspace_path) + the browse endpoint ----------

def test_write_file_accepts_a_real_workspace_path(tmp_path, fake_desireeia):
    real_dir = tmp_path / "my-real-folder"
    real_dir.mkdir()
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/write_file", json={
            "workspace_path": str(real_dir), "path": "out.txt", "content": "hi",
        })
        assert r.status_code == 200
        assert (real_dir / "out.txt").read_text() == "hi"


def test_write_file_rejects_a_nonexistent_workspace_path(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post("/tools/write_file", json={
            "workspace_path": str(tmp_path / "does-not-exist"), "path": "out.txt", "content": "hi",
        })
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "workspace_path"


def test_browse_lists_subdirectories(tmp_path, fake_desireeia):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "beta").mkdir()
    (tmp_path / "a_file.txt").write_text("x")  # files must not show up
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.get("/tools/workspaces/browse", params={"path": str(tmp_path)})
        assert r.status_code == 200
        body = r.json()
        names = {e["name"] for e in body["entries"]}
        assert {"alpha", "beta"} <= names  # tmp_path also has a "models" dir from ensure_models_dir()
        assert "a_file.txt" not in names
        assert body["parent"] is not None


def test_browse_disabled_by_default(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.get("/tools/workspaces/browse")
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "not_supported_error"


def test_browse_unknown_path_404s(tmp_path, fake_desireeia):
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.get("/tools/workspaces/browse", params={"path": str(tmp_path / "nope")})
        assert r.status_code == 404


def test_malformed_json_body_returns_400_not_500(tmp_path, fake_desireeia):
    # Regression test: request.json() raising on invalid JSON used to
    # bubble out of every /tools/* endpoint as an unhandled exception (a
    # generic 500), instead of the same clean 400 every other malformed
    # body case already gets.
    with _client(tmp_path, enable_python_tool=True) as c:
        r = c.post(
            "/tools/list_files",
            content=b'{"workspace_path": "C:\\bad\\escape"}',  # \b and \e are not valid JSON escapes
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "invalid_request_error"


def test_run_python_rejects_oversized_code_directly():
    with pytest.raises(sandbox.CodeTooLarge):
        sandbox.run_python("x = 1\n" * 10_000)


def test_run_python_kills_on_timeout(monkeypatch):
    monkeypatch.setattr(sandbox, "TIMEOUT_SECONDS", 0.2)
    result = sandbox.run_python("import time; time.sleep(5)")
    assert result["timed_out"] is True
    assert result["exit_code"] is None
