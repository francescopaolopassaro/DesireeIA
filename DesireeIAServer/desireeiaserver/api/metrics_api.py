# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""GET /metrics: Prometheus text-format exposition. Deliberately left off
the ApiKeyMiddleware-protected prefix list, same as /health - a scraper
typically has no way to attach a bearer token and this data is not
sensitive enough to be worth the operational friction of gating it.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from ..metrics import metrics

router = APIRouter(tags=["metrics"])


@router.get("/metrics")
def get_metrics(request: Request) -> PlainTextResponse:
    body = metrics.render(request.app.state.slots)
    return PlainTextResponse(body, media_type="text/plain; version=0.0.4")
