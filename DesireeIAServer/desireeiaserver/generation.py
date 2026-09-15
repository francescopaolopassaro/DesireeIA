"""Token-by-token prediction loop and the SSE streaming bridge.

The engine wrapper is synchronous and blocking. Generation runs in a worker
thread that pushes steps into an asyncio queue; the SSE endpoint consumes the
queue. Cancellation is cooperative: a `threading.Event` is checked between
tokens, so an in-flight response stops within one token of latency.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional

from .token_stats import TokenTracker


@dataclass
class PredictionStep:
    piece: str = ""
    finish_reason: Optional[str] = None
    final: bool = False


def build_prompt_ids(model, messages, add_assistant: bool = True):
    prompt = model.apply_chat_template(messages, add_assistant)
    ids = model.tokenize(prompt, add_bos=True)
    if ids is None:
        raise RuntimeError("model has no recognized tokenizer")
    return ids


def iter_prediction(model, ids, generation, tracker: TokenTracker, cancelled: threading.Event):
    from desireeia.generation import StopSequenceScanner

    scanner = StopSequenceScanner(generation.stop_sequences)
    t0 = time.perf_counter()
    token = model.predict(ids)
    tracker.prompt_ms = (time.perf_counter() - t0) * 1000.0
    tracker.prompt_n = len(ids)

    finish = "length"
    predicted_start = time.perf_counter()
    for _ in range(generation.max_tokens):
        if cancelled.is_set():
            finish = "cancelled"
            break
        if model.is_end_of_generation(token):
            finish = "stop"
            break
        tracker.predicted_n += 1
        piece = model.token_piece(token) or ""
        emit, stopped = scanner.feed(piece)
        if emit:
            yield PredictionStep(piece=emit)
        if stopped:
            finish = "stop"
            break
        token = model.next_token()
    tail = scanner.flush()
    if tail:
        yield PredictionStep(piece=tail)
    tracker.predicted_ms = (time.perf_counter() - predicted_start) * 1000.0
    tracker.total_ms = (time.perf_counter() - t0) * 1000.0
    yield PredictionStep(finish_reason=finish, final=True)


class GenerationRunner:
    """Run a producer in a background thread and publish PredictionSteps."""

    def __init__(self, producer, tracker: TokenTracker) -> None:
        self._producer = producer
        self._tracker = tracker
        self._queue: Optional[asyncio.Queue] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._cancelled = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def tracker(self) -> TokenTracker:
        return self._tracker

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue = asyncio.Queue()
        self._thread = threading.Thread(target=self._pump, name="desireeia-gen", daemon=True)
        self._thread.start()

    def _put(self, kind: str, payload) -> None:
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (kind, payload))

    def _pump(self) -> None:
        try:
            for step in self._producer(self._cancelled, self._tracker):
                self._put("piece", step)
            self._put("done", None)
        except Exception as exc:
            self._put("error", exc)

    async def events(self):
        while True:
            kind, payload = await self._queue.get()
            if kind == "piece":
                yield payload
            elif kind == "error":
                raise payload
            else:
                return

    def cancel(self) -> None:
        self._cancelled.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=3)


def run_sync(producer) -> tuple:
    """Run a producer to completion on the current thread.

    Returns (text, finish_reason, tracker).
    """
    cancelled = threading.Event()
    tracker = TokenTracker()
    pieces: list[str] = []
    finish = "stop"
    for step in producer(cancelled, tracker):
        if step.final:
            finish = step.finish_reason or "stop"
        elif step.piece:
            pieces.append(step.piece)
    return "".join(pieces), finish, tracker