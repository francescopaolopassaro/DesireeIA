# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""HTTP hardening and observability middleware:

- ApiKeyMiddleware: enforces `--api-key`/DESIREEIA_API_KEY when configured.
  Before this middleware existed, `Settings.api_keys` was parsed from the
  CLI/env and even round-tripped through argparse, but nothing ever read it
  back - a server started with --api-key looked protected and was not.
- MaxBodySizeMiddleware: caps a regular JSON request body (distinct from
  the checkpoint upload path in models_store.py, which already enforces its
  own, much larger, streamed limit).
- SecurityHeadersMiddleware: a few baseline response headers.
- MetricsMiddleware: records the per-request counters GET /metrics exposes.
"""

from __future__ import annotations

import time
from typing import Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

from .metrics import metrics

# Paths that require the API key when one is configured. The operator
# console itself (static files served at "/") and /health/ /props stay
# reachable without a key - the API surface they call is what actually
# needs protecting, and a health/readiness check must work before any
# credential is known to whatever is probing it.
_PROTECTED_PREFIXES = ("/v1/", "/models", "/config", "/tokenize", "/detokenize",
                        "/apply-template", "/tools/", "/transcribe")


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        {"error": {"message": "invalid API key", "type": "authentication_error",
                   "code": "authentication_error"}},
        status_code=401,
    )


def _too_large(max_bytes: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"message": f"request body exceeds {max_bytes} bytes",
                   "type": "invalid_request_error", "code": "request_too_large"}},
        status_code=413,
    )


class ApiKeyMiddleware:
    """Pure ASGI, not BaseHTTPMiddleware: an unauthorized request is
    rejected before it reaches routing or body parsing at all."""

    def __init__(self, app: ASGIApp, api_keys: Tuple[str, ...]) -> None:
        self.app = app
        self.api_keys = set(api_keys)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not self.api_keys:
            await self.app(scope, receive, send)
            return
        path = scope["path"]
        if not any(path.startswith(prefix) for prefix in _PROTECTED_PREFIXES):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        auth = headers.get(b"authorization", b"").decode("latin-1")
        token = auth[7:] if auth.lower().startswith("bearer ") else None
        if token in self.api_keys:
            await self.app(scope, receive, send)
            return
        await _unauthorized()(scope, receive, send)


class _BodyTooLarge(Exception):
    pass


class MaxBodySizeMiddleware:
    """Rejects an oversized body as early as the Content-Length header
    allows, and enforces the same cap while consuming a chunked body that
    has no Content-Length up front."""

    # Endpoints that stream their own, much larger, dedicated limit and
    # enforce it themselves (models_store.ChunkedUpload's max_upload_mb,
    # transcribe_api's MAX_AUDIO_BYTES) - applying this middleware's much
    # smaller default on top would break them before their own check ever
    # runs, since it would reject the body before the handler sees it.
    exempt_prefixes = ("/models/upload", "/transcribe")

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        if any(scope["path"].startswith(prefix) for prefix in MaxBodySizeMiddleware.exempt_prefixes):
            await self.app(scope, receive, send)
            return
        headers = dict(scope.get("headers") or [])
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await _too_large(self.max_bytes)(scope, receive, send)
                    return
            except ValueError:
                pass

        seen = 0
        max_bytes = self.max_bytes

        async def limited_receive():
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > max_bytes:
                    raise _BodyTooLarge()
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _too_large(self.max_bytes)(scope, receive, send)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        route = request.scope.get("route")
        route_path = route.path if route is not None else request.url.path
        metrics.record_request(request.method, route_path, response.status_code, duration)
        return response
