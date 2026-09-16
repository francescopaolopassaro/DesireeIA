# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Local-machine tools: POST /tools/run_python (sandboxed code execution)
and the direct workspace file tools (/tools/list_files, /read_file,
/write_file, /search_files) - see toolcalling.py for how the model is
asked to emit a tool call, and static/index.html for how the browser
executes one. All gated behind the same Settings.enable_python_tool /
--enable-python-tool flag: file tools don't execute code, but they're the
same "let the assistant touch this machine" trust boundary as run_python,
just a lighter-weight path to file I/O it could already reach by writing
a Python snippet.
"""

from __future__ import annotations

from pathlib import Path

import anyio
from fastapi import APIRouter, Request

from .. import sandbox
from ..config import Settings
from ..metrics import metrics
from .errors import InvalidRequestError, NotFoundError, NotSupportedError

router = APIRouter(prefix="/tools", tags=["tools"])


def _require_enabled(settings: Settings) -> None:
    if not settings.enable_python_tool:
        raise NotSupportedError(
            "local tools are disabled on this server "
            "(start it with --enable-python-tool to turn them on)"
        )


async def _read_json_body(request: Request) -> dict:
    # A malformed body (invalid JSON, wrong Content-Type, truncated
    # request) used to bubble straight out of request.json() as an
    # unhandled exception -> a generic 500, instead of the same clean 400
    # every other malformed-body case here already gets.
    try:
        payload = await request.json()
    except Exception as exc:
        raise InvalidRequestError("request body must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidRequestError("request body must be a JSON object")
    return payload


def _resolve_workspace(settings: Settings, payload: dict, *, required: bool):
    """A workspace is EITHER a real absolute directory the person picked
    via the browser's folder-browse modal (workspace_path - see
    sandbox.list_directories/browse_directories below) OR a name-only
    sandboxed folder under this server's own data dir (workspace_id, the
    original scheme, still supported for anything that just wants a
    lightweight per-conversation scratch space without picking a real
    location). workspace_path is NOT confined to any root - the point of
    it is letting the model work in a folder the person actually chose."""
    workspace_path = payload.get("workspace_path")
    if workspace_path is not None:
        if not isinstance(workspace_path, str):
            raise InvalidRequestError("field 'workspace_path' must be a string", param="workspace_path")
        target = Path(workspace_path)
        if not target.is_dir():
            raise InvalidRequestError(f"not a directory: {workspace_path}", param="workspace_path")
        return target.resolve()

    workspace_id = payload.get("workspace_id")
    if workspace_id is None:
        if required:
            raise InvalidRequestError(
                "field 'workspace_id' or 'workspace_path' is required", param="workspace_id")
        return None
    if not isinstance(workspace_id, str):
        raise InvalidRequestError("field 'workspace_id' must be a string", param="workspace_id")
    try:
        workspaces_root = settings.resolved_data_dir / "workspaces"
        return sandbox.workspace_path(workspaces_root, workspace_id)
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="workspace_id") from exc


@router.get("/workspaces/browse")
async def browse_directories(request: Request, path: str = ""):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    try:
        result = await anyio.to_thread.run_sync(sandbox.list_directories, path or None)
    except NotADirectoryError as exc:
        raise NotFoundError(str(exc)) from exc
    return result


@router.post("/run_python")
async def run_python(request: Request):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    payload = await _read_json_body(request)
    if not isinstance(payload.get("code"), str):
        raise InvalidRequestError("field 'code' (string) is required", param="code")

    # workspace_id (the client sends its chat session id) gives run_python
    # a persistent directory across calls in the same conversation instead
    # of a fresh throwaway temp dir every time - without one, the file
    # writes/reads of a multi-step script never see each other.
    workspace_dir = _resolve_workspace(settings, payload, required=False)

    try:
        result = await anyio.to_thread.run_sync(sandbox.run_python, payload["code"], workspace_dir)
    except sandbox.CodeTooLarge as exc:
        raise InvalidRequestError(str(exc), param="code") from exc

    metrics.record_tool_call()
    return result


@router.post("/list_files")
async def list_files(request: Request):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    payload = await _read_json_body(request)
    workspace_dir = _resolve_workspace(settings, payload, required=True)
    rel_path = payload.get("path") or "."
    try:
        entries = await anyio.to_thread.run_sync(sandbox.list_files, workspace_dir, rel_path)
    except (ValueError, NotADirectoryError) as exc:
        raise InvalidRequestError(str(exc), param="path") from exc
    metrics.record_tool_call()
    return {"entries": entries}


@router.post("/read_file")
async def read_file(request: Request):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    payload = await _read_json_body(request)
    if not isinstance(payload.get("path"), str):
        raise InvalidRequestError("field 'path' (string) is required", param="path")
    workspace_dir = _resolve_workspace(settings, payload, required=True)
    try:
        content = await anyio.to_thread.run_sync(sandbox.read_file, workspace_dir, payload["path"])
    except (ValueError, FileNotFoundError) as exc:
        raise InvalidRequestError(str(exc), param="path") from exc
    metrics.record_tool_call()
    return {"path": payload["path"], "content": content}


@router.post("/write_file")
async def write_file(request: Request):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    payload = await _read_json_body(request)
    if not isinstance(payload.get("path"), str):
        raise InvalidRequestError("field 'path' (string) is required", param="path")
    if not isinstance(payload.get("content"), str):
        raise InvalidRequestError("field 'content' (string) is required", param="content")
    workspace_dir = _resolve_workspace(settings, payload, required=True)
    try:
        result = await anyio.to_thread.run_sync(
            sandbox.write_file, workspace_dir, payload["path"], payload["content"]
        )
    except ValueError as exc:
        raise InvalidRequestError(str(exc), param="path") from exc
    except sandbox.ContentTooLarge as exc:
        raise InvalidRequestError(str(exc), param="content") from exc
    metrics.record_tool_call()
    return result


@router.post("/search_files")
async def search_files(request: Request):
    settings: Settings = request.app.state.settings
    _require_enabled(settings)
    payload = await _read_json_body(request)
    if not isinstance(payload.get("query"), str) or not payload["query"]:
        raise InvalidRequestError("field 'query' (non-empty string) is required", param="query")
    workspace_dir = _resolve_workspace(settings, payload, required=True)
    results = await anyio.to_thread.run_sync(sandbox.search_files, workspace_dir, payload["query"])
    metrics.record_tool_call()
    return {"results": results}
