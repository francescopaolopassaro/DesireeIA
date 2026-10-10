# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Thin wrapper over the engine's context memory (desireeia_memory_* in
abi.h). All storage, compression, indexing, search and session expiry run
inside the native engine; this module only marshals strings."""

from __future__ import annotations

import ctypes
import json
from typing import Optional

from ._native import get_lib
from .enums import Error


class ContextMemory:
    def __init__(self, root_dir: str, ttl_seconds: float = 1800.0) -> None:
        self._lib = get_lib()
        handle = ctypes.c_void_p()
        self._check(self._lib.desireeia_memory_open(root_dir.encode("utf-8"), ttl_seconds, ctypes.byref(handle)),
                    "open")
        self._mem = handle

    def close(self) -> None:
        if self._mem:
            self._lib.desireeia_memory_close(self._mem)
            self._mem = None

    def __enter__(self) -> "ContextMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def set_embedder(self, model) -> None:
        """Use a loaded encoder LocalModel for vector search (None = BM25 only)."""
        self._check(self._lib.desireeia_memory_set_embedder(self._mem, model._ctx if model else None), "set_embedder")

    @staticmethod
    def _check(err: int, what: str) -> None:
        if err != Error.OK:
            raise RuntimeError(f"context memory {what} failed: {Error(err).name}")

    def _json(self, fn, *args) -> dict:
        out = ctypes.c_void_p()
        self._check(fn(self._mem, *args, ctypes.byref(out)), fn.__name__)
        try:
            return json.loads(ctypes.string_at(out).decode("utf-8"))
        finally:
            self._lib.desireeia_memory_free_string(out)

    @staticmethod
    def _b(s: Optional[str]) -> Optional[bytes]:
        return None if s is None else s.encode("utf-8")

    def put_file(self, session: str, name: str, path: str) -> dict:
        return self._json(self._lib.desireeia_memory_put_file, self._b(session), self._b(name), self._b(path))

    def put_text(self, session: str, name: str, text: str) -> dict:
        raw = text.encode("utf-8")
        return self._json(self._lib.desireeia_memory_put_text, self._b(session), self._b(name), raw, len(raw))

    def append(self, session: str, name: str, text: str) -> dict:
        raw = text.encode("utf-8")
        return self._json(self._lib.desireeia_memory_append, self._b(session), self._b(name), raw, len(raw))

    def read(self, session: str, handle: str, offset: int = 1, limit: int = 200) -> dict:
        return self._json(self._lib.desireeia_memory_read, self._b(session), self._b(handle), offset, limit)

    def search(self, session: str, query: str, handle: Optional[str] = None, k: int = 5) -> dict:
        return self._json(self._lib.desireeia_memory_search, self._b(session), self._b(query), self._b(handle), k)

    def list(self, session: str) -> dict:
        return self._json(self._lib.desireeia_memory_list, self._b(session))

    def replace_lines(self, session: str, handle: str, first: int, last: int, text: str) -> dict:
        raw = text.encode("utf-8")
        return self._json(self._lib.desireeia_memory_replace_lines, self._b(session), self._b(handle),
                          first, last, raw, len(raw))

    def export(self, session: str, handle: str, path: str) -> dict:
        return self._json(self._lib.desireeia_memory_export, self._b(session), self._b(handle), self._b(path))

    def touch(self, session: str) -> None:
        self._check(self._lib.desireeia_memory_touch(self._mem, self._b(session)), "touch")

    def drop_session(self, session: str) -> None:
        self._check(self._lib.desireeia_memory_drop_session(self._mem, self._b(session)), "drop_session")

    def sweep(self) -> int:
        removed = ctypes.c_int32()
        self._check(self._lib.desireeia_memory_sweep(self._mem, ctypes.byref(removed)), "sweep")
        return removed.value
