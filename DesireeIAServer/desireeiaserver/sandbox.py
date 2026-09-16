# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Sandboxed Python execution backing the run_python tool.

"Sandboxed" here means: a fresh subprocess (no shared memory/state with the
server process), launched with `python -I` (isolated mode - ignores
PYTHON*/PATH-driven site customization and does not add the script's
directory or the current directory to sys.path), a scrubbed environment, a
throwaway empty working directory, a wall-clock timeout, and truncated
captured output.

This is NOT an isolation boundary against a determined attacker with local
code execution as the actual goal - there is no seccomp/container/VM
sandbox here, and the subprocess still runs as the same OS user as the
server. That level of isolation needs a container or VM, which is out of
scope for this feature. What this defends against is accidental
runaway/hanging/resource-hungry code coming out of a model's own output on
a single-user local server - the threat this tool actually needs to
handle, not a multi-tenant one.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

MAX_CODE_BYTES = 20_000
MAX_OUTPUT_CHARS = 8_000
TIMEOUT_SECONDS = 5.0


class CodeTooLarge(Exception):
    pass


class ContentTooLarge(Exception):
    pass


MAX_FILE_BYTES = 512 * 1024
MAX_LIST_ENTRIES = 500


WORKSPACE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def workspace_path(base_dir: Path, workspace_id: str) -> Path:
    """Resolve `workspace_id` to a persistent directory under `base_dir`,
    creating it if needed. One workspace per chat session (the client
    sends its session id) so run_python's file writes in one call are
    still there for a later call in the same conversation, without
    letting an id escape the workspace root via path separators/"..".
    """
    if not WORKSPACE_ID_RE.match(workspace_id):
        raise ValueError(f"invalid workspace id: {workspace_id!r}")
    base_resolved = base_dir.resolve()
    path = (base_resolved / workspace_id).resolve()
    if path != base_resolved and base_resolved not in path.parents:
        raise ValueError("workspace id escapes the workspace root")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _minimal_env() -> dict:
    # subprocess with an entirely empty environment is unreliable on
    # Windows (Python's own startup needs SystemRoot for socket/crypto
    # init) - keep only what the interpreter needs to boot, nothing the
    # calling process itself was configured with.
    if sys.platform == "win32":
        keys = ("SYSTEMROOT", "SYSTEMDRIVE", "PATH", "TEMP", "TMP")
    else:
        keys = ("PATH",)
    return {key: os.environ[key] for key in keys if key in os.environ}


def run_python(code: str, workspace_dir: Optional[Path] = None) -> dict:
    if len(code.encode("utf-8")) > MAX_CODE_BYTES:
        raise CodeTooLarge(f"code exceeds {MAX_CODE_BYTES} bytes")

    with tempfile.TemporaryDirectory(prefix="desireeia-sandbox-") as tmp:
        # The driver script itself always lives in a throwaway temp dir
        # (auto-cleaned on return); only the working directory - where a
        # relative open("data.csv", "w") actually lands - is the caller's
        # persistent workspace, when one was given.
        script = Path(tmp) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        cwd = str(workspace_dir) if workspace_dir is not None else tmp
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-B", str(script)],
                cwd=cwd,
                env=_minimal_env(),
                capture_output=True,
                timeout=TIMEOUT_SECONDS,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return {
                "timed_out": True,
                "stdout": "",
                "stderr": f"execution exceeded {TIMEOUT_SECONDS}s and was killed",
                "exit_code": None,
            }
        return {
            "timed_out": False,
            "stdout": proc.stdout[:MAX_OUTPUT_CHARS],
            "stderr": proc.stderr[:MAX_OUTPUT_CHARS],
            "exit_code": proc.returncode,
        }


# ---------------------------------------------------------------------------
# Direct file tools (list/read/write/search) - a lighter, more direct path
# to the same workspace run_python already has full access to via ordinary
# Python file I/O. These exist so a model can list/read/write/search files
# without having to write and run a Python snippet for something this
# simple, and so file access can be granted independently of code
# execution on the client's Permissions screen.
# ---------------------------------------------------------------------------

def resolve_in_workspace(workspace_dir: Path, rel_path: str) -> Path:
    """Resolve `rel_path` against `workspace_dir`, refusing anything that
    escapes it (a leading "/", "..", a symlink pointing outside, etc.)."""
    rel_path = rel_path.strip() if rel_path else "."
    if not rel_path:
        rel_path = "."
    workspace_resolved = workspace_dir.resolve()
    candidate = (workspace_resolved / rel_path).resolve()
    if candidate != workspace_resolved and workspace_resolved not in candidate.parents:
        raise ValueError(f"path escapes the workspace: {rel_path!r}")
    return candidate


def list_files(workspace_dir: Path, rel_path: str = ".") -> list:
    target = resolve_in_workspace(workspace_dir, rel_path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {rel_path}")
    entries = []
    for entry in sorted(target.iterdir())[:MAX_LIST_ENTRIES]:
        entries.append({
            "name": entry.name,
            "is_dir": entry.is_dir(),
            "size_bytes": entry.stat().st_size if entry.is_file() else None,
        })
    return entries


def read_file(workspace_dir: Path, rel_path: str) -> str:
    target = resolve_in_workspace(workspace_dir, rel_path)
    if not target.is_file():
        raise FileNotFoundError(f"not a file: {rel_path}")
    return target.read_text(encoding="utf-8", errors="replace")[:MAX_FILE_BYTES]


def write_file(workspace_dir: Path, rel_path: str, content: str) -> dict:
    if len(content.encode("utf-8")) > MAX_FILE_BYTES:
        raise ContentTooLarge(f"content exceeds {MAX_FILE_BYTES} bytes")
    target = resolve_in_workspace(workspace_dir, rel_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    # The real absolute path, not just the workspace-relative one: asked
    # for directly by name in testing ("where did you actually write this
    # file?") - the model has no way to answer that honestly from the
    # relative path alone, since it has never been told where on disk the
    # workspace itself lives.
    return {"path": rel_path, "absolute_path": str(target), "bytes_written": len(content.encode("utf-8"))}


def search_files(workspace_dir: Path, query: str) -> list:
    """Matches file NAMES always; file CONTENT only for files small enough
    to read cheaply (MAX_FILE_BYTES) - a workspace is a handful of files a
    conversation produced, not something worth indexing."""
    workspace_resolved = workspace_dir.resolve()
    query_lower = query.lower()
    results = []
    for path in sorted(workspace_resolved.rglob("*")):
        if len(results) >= MAX_LIST_ENTRIES:
            break
        if not path.is_file():
            continue
        rel = str(path.relative_to(workspace_resolved))
        matched = query_lower in path.name.lower()
        if not matched and path.stat().st_size <= MAX_FILE_BYTES:
            try:
                matched = query_lower in path.read_text(encoding="utf-8", errors="ignore").lower()
            except OSError:
                pass
        if matched:
            results.append({"path": rel})
    return results


# ---------------------------------------------------------------------------
# Real-directory browser, backing the workspace picker: a web page has no
# way to hand JavaScript a real OS path (the File System Access API only
# ever gives back an opaque, sandboxed handle, by design) - but the server
# runs on the same machine as the person using it, so it can walk the real
# filesystem itself and let the browser pick from that instead. Read-only:
# lists directory names, never touches file contents.
# ---------------------------------------------------------------------------

def list_directories(path: Optional[str]) -> dict:
    """List the subdirectories of `path`. `path` of None/"" lists the
    available drive roots on Windows (there's no single filesystem root to
    start from), or "/" on POSIX."""
    if not path:
        if sys.platform == "win32":
            import string

            drives = [f"{letter}:\\" for letter in string.ascii_uppercase if Path(f"{letter}:\\").exists()]
            return {"path": "", "parent": None, "entries": [{"name": d, "path": d} for d in drives]}
        path = "/"

    target = Path(path)
    if not target.is_dir():
        raise NotADirectoryError(f"not a directory: {path}")
    target = target.resolve()

    entries = []
    try:
        for entry in sorted(target.iterdir(), key=lambda p: p.name.lower()):
            try:
                if entry.is_dir():
                    entries.append({"name": entry.name, "path": str(entry)})
            except OSError:
                continue  # a broken symlink or a permission error mid-listing
    except PermissionError:
        pass

    is_drive_root = sys.platform == "win32" and target.parent == target
    parent = None if is_drive_root else str(target.parent)
    return {"path": str(target), "parent": parent, "entries": entries}
