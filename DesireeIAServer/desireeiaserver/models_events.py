# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""In-process pub/sub for the /v1/models/sse endpoint: every upload or
delete calls publish(), every open SSE connection holds its own
subscription. Deliberately not backed by anything external (no Redis, no
file watcher) — this server is single-process, so an asyncio.Queue per
subscriber is the whole mechanism a UI's "model list changed, refresh"
signal needs.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional


class ModelsEventBus:
    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue] = set()

    def publish(self, event: str, **fields) -> None:
        payload = {"event": event, **fields}
        for queue in list(self._subscribers):
            # A full queue means that subscriber isn't reading (a dead or
            # stalled connection): drop the event for it rather than block
            # every OTHER subscriber, or the publisher, on one straggler.
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    async def subscribe(self, keepalive_seconds: float = 15.0) -> AsyncIterator[dict]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers.add(queue)
        try:
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=keepalive_seconds)
                    yield event
                except asyncio.TimeoutError:
                    # A periodic heartbeat, not a models_reload event: keeps
                    # intermediary proxies from timing out an idle SSE
                    # connection, and lets the client tell "still connected,
                    # nothing changed" apart from "connection died".
                    yield {"event": "ping"}
        finally:
            self._subscribers.discard(queue)
