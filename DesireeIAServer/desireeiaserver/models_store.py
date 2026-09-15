# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Filesystem-only operations on the models folder: list checkpoints,
validate and save an uploaded one, delete one. No FastAPI import here on
purpose — this module is exercised directly by unit tests, without going
through HTTP, and the router (api/models_files.py) is a thin layer on top
of it.
"""

from __future__ import annotations

import os
import re
import struct
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

# Same magic bytes GgufReader (the native engine's own reader,
# src/gguf/gguf_reader.cpp) checks: the literal ASCII string "GGUF" as the
# file's first 4 bytes. Not re-deriving it from the engine's numeric
# constant (0x46554747, little-endian for "GGUF") — the string form is the
# one actually documented by the file format and is what every other GGUF
# reader in the wild checks too.
GGUF_MAGIC = b"GGUF"

SUPPORTED_EXTENSIONS = (".gguf", ".safetensors")

# Rejects '..', a leading '/' or '\', and any embedded path separator —
# the upload/delete filename comes straight from client input, so this is
# the one check standing between it and writing/deleting a file anywhere
# reachable from the process, not just inside the models folder.
_SAFE_NAME_RE = re.compile(r"^[^/\\]+$")


class InvalidFilename(ValueError):
    """The requested filename is unsafe or has an unsupported extension."""


class FileTooLarge(ValueError):
    """The upload exceeded the configured max_upload_mb cap."""

    def __init__(self, limit_bytes: int) -> None:
        super().__init__(f"upload exceeds the {limit_bytes} byte limit")
        self.limit_bytes = limit_bytes


class BadMagic(ValueError):
    """The uploaded bytes don't start with the expected format magic."""


@dataclass(frozen=True)
class ModelFile:
    name: str
    size_bytes: int
    modified_at: float
    format: str          # "gguf" | "safetensors" | "unknown"
    valid: bool          # magic/header check passed


def validate_filename(name: str) -> str:
    """Raises InvalidFilename if `name` isn't a safe, supported checkpoint
    filename. Returns it unchanged otherwise (kept as a function, not a
    bool, so callers can't forget to check the result)."""
    if not name or not _SAFE_NAME_RE.match(name):
        raise InvalidFilename(f"{name!r} is not a valid filename (no path separators allowed)")
    ext = Path(name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise InvalidFilename(
            f"unsupported extension {ext!r} (expected one of {SUPPORTED_EXTENSIONS})"
        )
    return name


def _format_of(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".gguf":
        return "gguf"
    if ext == ".safetensors":
        return "safetensors"
    return "unknown"


def _check_gguf_magic(handle) -> bool:
    return handle.read(4) == GGUF_MAGIC


def _check_safetensors_header(handle) -> bool:
    # safetensors has no fixed magic string; its own defined structure is
    # an 8-byte little-endian header length followed by that many bytes of
    # JSON. Checked here only as a sanity bound (a plausible length, and
    # that a JSON object actually starts where it should) — a full parse
    # belongs to the engine's own reader, not this upload gate.
    raw_len = handle.read(8)
    if len(raw_len) != 8:
        return False
    (header_len,) = struct.unpack("<Q", raw_len)
    if header_len == 0 or header_len > 100 * 1024 * 1024:
        return False
    first_byte = handle.read(1)
    return first_byte == b"{"


def check_magic(path: Path) -> bool:
    """Reads just enough of the file to validate its format header. Never
    loads the checkpoint's tensor data."""
    fmt = _format_of(path)
    try:
        with open(path, "rb") as handle:
            if fmt == "gguf":
                return _check_gguf_magic(handle)
            if fmt == "safetensors":
                return _check_safetensors_header(handle)
    except OSError:
        return False
    return False


def list_models(models_dir: Path) -> Iterator[ModelFile]:
    """Scans `models_dir` (non-recursive: checkpoints are single files,
    not project trees) for supported checkpoints. A file with a supported
    extension but a failing magic check is still listed (so a truncated or
    wrong-format upload is visible, not silently hidden) with valid=False."""
    if not models_dir.is_dir():
        return
    for entry in sorted(models_dir.iterdir(), key=lambda p: p.name.lower()):
        if not entry.is_file():
            continue
        if entry.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            stat = entry.stat()
        except OSError:
            continue
        yield ModelFile(
            name=entry.name,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            format=_format_of(entry),
            valid=check_magic(entry),
        )


def delete_model(models_dir: Path, name: str) -> None:
    """Raises InvalidFilename for an unsafe name, FileNotFoundError if it
    doesn't exist under models_dir."""
    validate_filename(name)
    target = models_dir / name
    # Resolve and re-check containment: validate_filename already rejects
    # any embedded separator, but this catches the platform-specific cases
    # a regex alone can miss (e.g. a reserved device name on Windows).
    if target.resolve().parent != models_dir.resolve():
        raise InvalidFilename(f"{name!r} does not resolve inside the models folder")
    target.unlink()  # raises FileNotFoundError if absent, which is correct here


class ChunkedUpload:
    """Writes an incoming byte stream straight to disk, enforcing the size
    cap and the format magic as the data arrives rather than after
    buffering the whole file — the checkpoints this serves are routinely
    multi-gigabyte, so "buffer it all, then check" is not an option.

    Written to a temp file inside `models_dir` first and renamed into place
    only once the stream ends successfully: a failed or aborted upload
    (over the size cap, wrong magic, client disconnect) never leaves a
    broken file visible to list_models().
    """

    # Bytes needed before the magic check can run at all: the safetensors
    # header-length sanity check reads 9 bytes; GGUF's is 4. Buffering up
    # to this many bytes before the first write is negligible next to the
    # file sizes involved.
    _MAGIC_PROBE_BYTES = 9

    def __init__(self, models_dir: Path, filename: str, max_bytes: int) -> None:
        validate_filename(filename)
        self.filename = filename
        self.models_dir = models_dir
        self.max_bytes = max_bytes
        self._fmt = _format_of(Path(filename))
        self._probe = bytearray()
        self._probe_checked = False
        self._written = 0
        self._tmp_path = models_dir / f".upload-{uuid.uuid4().hex}.partial"
        self._handle = open(self._tmp_path, "wb")

    def _check_probe(self) -> None:
        if self._probe_checked:
            return
        if len(self._probe) < self._MAGIC_PROBE_BYTES:
            return
        self._probe_checked = True
        ok = (
            self._probe[:4] == GGUF_MAGIC if self._fmt == "gguf" else
            self._probe[8:9] == b"{" if self._fmt == "safetensors" else
            False
        )
        if not ok:
            raise BadMagic(f"{self.filename!r} does not start with a valid {self._fmt} header")

    def write(self, chunk: bytes) -> None:
        self._written += len(chunk)
        if self._written > self.max_bytes:
            raise FileTooLarge(self.max_bytes)
        if not self._probe_checked:
            self._probe.extend(chunk)
            self._check_probe()
        self._handle.write(chunk)

    def abort(self) -> None:
        self._handle.close()
        self._tmp_path.unlink(missing_ok=True)

    def finish(self) -> ModelFile:
        self._handle.close()
        if not self._probe_checked:
            # The whole file was shorter than the probe needs: definitely
            # not a valid checkpoint of either supported format.
            self._tmp_path.unlink(missing_ok=True)
            raise BadMagic(f"{self.filename!r} is too small to be a valid {self._fmt} file")
        final_path = self.models_dir / self.filename
        os.replace(self._tmp_path, final_path)  # atomic on both POSIX and Windows (NTFS)
        stat = final_path.stat()
        return ModelFile(
            name=self.filename,
            size_bytes=stat.st_size,
            modified_at=stat.st_mtime,
            format=self._fmt,
            valid=True,
        )
