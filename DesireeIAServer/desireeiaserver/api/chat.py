"""OpenAI-compatible chat and text-completion endpoints, sync and SSE."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, List, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..generation import GenerationRunner, run_sync
from ..sampling import CodegenParams, resolve
from ..slots import Slot
from ..token_stats import TokenTracker
from .errors import InvalidRequestError

router = APIRouter(prefix="/v1", tags=["chat"])

_STREAM_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _request_id(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:24]}"


async def _read_body(request: Request) -> dict:
    try:
        payload = await request.json()
    except Exception:
        raise InvalidRequestError("request body must be a valid JSON object")
    if not isinstance(payload, dict):
        raise InvalidRequestError("request body must be a JSON object")
    return payload


def _normalize_messages(payload: dict) -> List[Tuple[str, str]]:
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        raise InvalidRequestError("field 'messages' must be a non-empty array", param="messages")
    normalized = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise InvalidRequestError(f"messages[{index}] must be an object")
        role = message.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise InvalidRequestError(
                f"messages[{index}].role must be one of system, user, assistant, tool", param="role")
        content = message.get("content")
        if content is None:
            content = ""
        if isinstance(content, list):
            content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in content
            )
        if not isinstance(content, str):
            raise InvalidRequestError(f"messages[{index}].content must be a string or parts array")
        normalized.append((role, content))
    return normalized


def _apply_response_format(messages: List[Tuple[str, str]], params: CodegenParams) -> None:
    if params.response_format is None:
        return
    from desireeia.generation import StructuredOutput

    raw_schema = params.response_format.get("json_schema", {}) if isinstance(
        params.response_format.get("json_schema"), dict) else {}
    schema = raw_schema.get("schema")
    if isinstance(schema, dict):
        schema = json.dumps(schema, ensure_ascii=False)
    instruction = StructuredOutput.build_json_instruction(schema)
    for index in range(len(messages) - 1, -1, -1):
        if messages[index][0] == "system":
            messages[index] = (
                "system",
                messages[index][1] + "\n" + instruction if messages[index][1] else instruction,
            )
            return
    messages.insert(0, ("system", instruction))


def _producer_for(slot: Slot, messages: List[Tuple[str, str]], params: CodegenParams):
    prediction = slot.prediction(messages, params)

    def producer(cancelled, tracker):
        yield from prediction.run(cancelled, tracker)

    return producer


def _chat_chunk(request_id: str, created: int, model_id: str,
                delta: str, finish_reason: str | None) -> dict:
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {} if finish_reason is not None else {"content": delta},
            "finish_reason": finish_reason,
        }],
    }


async def _chat_stream(runner: GenerationRunner, request_id: str, created: int,
                       model_id: str) -> AsyncGenerator[str, None]:
    try:
        async for step in runner.events():
            if step.final:
                final = _chat_chunk(request_id, created, model_id, "", step.finish_reason)
                final["usage"] = runner.tracker.usage(model_id)
                final["timings"] = runner.tracker.timings()
                yield sse(final)
            elif step.piece:
                yield sse(_chat_chunk(request_id, created, model_id, step.piece, None))
        yield "data: [DONE]\n\n"
    finally:
        runner.cancel()


@router.post("/chat/completions")
async def chat_completions(request: Request):
    payload = await _read_body(request)
    params = resolve(payload)
    slot = request.app.state.slots.resolve(payload.get("model"))

    messages = _normalize_messages(payload)
    _apply_response_format(messages, params)

    model_id = slot.model_id
    request_id = _request_id("chatcmpl-")
    created = int(time.time())
    producer = _producer_for(slot, messages, params)

    if params.stream:
        runner = GenerationRunner(producer, TokenTracker())
        runner.start(asyncio.get_running_loop())
        return StreamingResponse(
            _chat_stream(runner, request_id, created, model_id),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    text, finish_reason, tracker = run_sync(producer)
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason,
        }],
        "usage": tracker.usage(model_id),
        "timings": tracker.timings(),
    }


def _normalize_prompt(payload: dict) -> str:
    prompt = payload.get("prompt")
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and all(isinstance(item, str) for item in prompt):
        return "".join(prompt)
    raise InvalidRequestError("field 'prompt' must be a string or a list of strings", param="prompt")


@router.post("/completions")
async def completions(request: Request):
    payload = await _read_body(request)
    params = resolve(payload, allow_n_predict=True)
    slot = request.app.state.slots.resolve(payload.get("model"))

    prompt = _normalize_prompt(payload)
    messages = [("user", prompt)]

    model_id = slot.model_id
    request_id = _request_id("cmpl-")
    created = int(time.time())
    producer = _producer_for(slot, messages, params)

    if params.stream:
        runner = GenerationRunner(producer, TokenTracker())
        runner.start(asyncio.get_running_loop())
        return StreamingResponse(
            _completion_stream(runner, request_id, created, model_id),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    text, finish_reason, tracker = run_sync(producer)
    if params.echo:
        text = prompt.rstrip() + text
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "text": text,
            "finish_reason": finish_reason,
        }],
        "usage": tracker.usage(model_id),
        "timings": tracker.timings(),
    }


def _completion_chunk(request_id: str, created: int, model_id: str,
                      text: str, finish_reason: str | None) -> dict:
    return {
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_id,
        "choices": [{"index": 0, "text": text, "finish_reason": finish_reason}],
    }


async def _completion_stream(runner: GenerationRunner, request_id: str, created: int,
                             model_id: str) -> AsyncGenerator[str, None]:
    try:
        async for step in runner.events():
            if step.final:
                final = _completion_chunk(request_id, created, model_id, "", step.finish_reason)
                final["usage"] = runner.tracker.usage(model_id)
                final["timings"] = runner.tracker.timings()
                yield sse(final)
            elif step.piece:
                yield sse(_completion_chunk(request_id, created, model_id, step.piece, None))
        yield "data: [DONE]\n\n"
    finally:
        runner.cancel()