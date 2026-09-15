"""OpenAI-compatible chat and text-completion endpoints, sync and SSE."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncGenerator, List, Tuple

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from .. import toolcalling
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


def _normalize_messages(payload: dict) -> List[dict]:
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
        normalized.append({
            "role": role,
            "content": content,
            "tool_calls": message.get("tool_calls"),
            "tool_call_id": message.get("tool_call_id"),
            "name": message.get("name"),
        })
    return normalized


def _prepend_system(messages: List[dict], instruction: str) -> None:
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "system":
            messages[index]["content"] = (
                messages[index]["content"] + "\n\n" + instruction if messages[index]["content"] else instruction
            )
            return
    messages.insert(0, {"role": "system", "content": instruction, "tool_calls": None,
                         "tool_call_id": None, "name": None})


def _apply_response_format(messages: List[dict], params: CodegenParams) -> None:
    if params.response_format is None:
        return
    from desireeia.generation import StructuredOutput

    raw_schema = params.response_format.get("json_schema", {}) if isinstance(
        params.response_format.get("json_schema"), dict) else {}
    schema = raw_schema.get("schema")
    if isinstance(schema, dict):
        schema = json.dumps(schema, ensure_ascii=False)
    _prepend_system(messages, StructuredOutput.build_json_instruction(schema))


def _apply_tools(messages: List[dict], payload: dict) -> bool:
    """Merge a tool-use instruction into the system message when the
    request declares `tools`. Returns True if tool-call detection should
    run on this request's output.

    tool_choice is honored only for "auto" (default) and "none"; a forced
    choice of a specific function is accepted but not specially enforced
    beyond the usual instruction - not needed for the single client-side
    tool this project wires up today, and adding real enforcement without
    a model that natively targets this wire format would mean inventing
    behavior no test could validate.
    """
    tools = payload.get("tools")
    tool_choice = payload.get("tool_choice", "auto")
    if not isinstance(tools, list) or not tools or tool_choice == "none":
        return False
    _prepend_system(messages, toolcalling.build_tools_instruction(tools))
    return True


def _to_engine_messages(messages: List[dict]) -> List[Tuple[str, str]]:
    """Translate OpenAI-shaped messages (with a "tool" role and optional
    assistant tool_calls) into the (role, content) pairs the native chat
    template engine understands - it only knows system/user/assistant, for
    every loaded template, so tool turns are folded into a user turn
    rather than extending the native template engine with a new role.
    """
    engine: List[Tuple[str, str]] = []
    for message in messages:
        role = message["role"]
        content = message["content"]
        if role == "tool":
            label = message.get("name") or "tool"
            content = f"[Tool result: {label}]\n{content}"
            role = "user"
        elif role == "assistant" and message.get("tool_calls"):
            call = message["tool_calls"][0]
            fn = call.get("function", {}) if isinstance(call, dict) else {}
            content = toolcalling.render_tool_call_text(fn.get("name", ""), fn.get("arguments", {}))
        engine.append((role, content))
    return engine


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


def _tool_call_message(tool_call: dict) -> dict:
    return {
        "id": _request_id("call_"),
        "type": "function",
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["arguments"], ensure_ascii=False),
        },
    }


def _tool_call_chunk(request_id: str, created: int, model_id: str, tool_call: dict) -> dict:
    call = _tool_call_message(tool_call)
    call["index"] = 0
    return {
        "id": request_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [call]},
            "finish_reason": "tool_calls",
        }],
    }


async def _chat_stream(runner: GenerationRunner, request_id: str, created: int,
                       model_id: str, tools_active: bool) -> AsyncGenerator[str, None]:
    full_text = ""
    tool_filter = toolcalling.ToolCallStreamFilter() if tools_active else None
    try:
        async for step in runner.events():
            if step.final:
                if tool_filter is not None and tool_filter.is_tool_call:
                    tool_call = toolcalling.extract_tool_call(full_text)
                    if tool_call is not None:
                        final = _tool_call_chunk(request_id, created, model_id, tool_call)
                    else:
                        # Looked like a tool call from the opening tag but
                        # failed to parse (malformed JSON) - degrade to
                        # showing the raw text rather than silently
                        # dropping the model's output.
                        final = _chat_chunk(request_id, created, model_id, full_text, step.finish_reason)
                else:
                    final = _chat_chunk(request_id, created, model_id, "", step.finish_reason)
                final["usage"] = runner.tracker.usage(model_id)
                final["timings"] = runner.tracker.timings()
                yield sse(final)
            elif step.piece:
                full_text += step.piece
                forward = step.piece if tool_filter is None else tool_filter.feed(step.piece)
                if forward:
                    yield sse(_chat_chunk(request_id, created, model_id, forward, None))
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
    tools_active = _apply_tools(messages, payload)
    engine_messages = _to_engine_messages(messages)

    model_id = slot.model_id
    request_id = _request_id("chatcmpl-")
    created = int(time.time())
    producer = _producer_for(slot, engine_messages, params)

    if params.stream:
        runner = GenerationRunner(producer, TokenTracker())
        runner.start(asyncio.get_running_loop())
        return StreamingResponse(
            _chat_stream(runner, request_id, created, model_id, tools_active),
            media_type="text/event-stream",
            headers=_STREAM_HEADERS,
        )

    text, finish_reason, tracker = run_sync(producer)
    tool_call = toolcalling.extract_tool_call(text) if tools_active else None
    if tool_call is not None:
        message = {"role": "assistant", "content": None, "tool_calls": [_tool_call_message(tool_call)]}
        finish_reason = "tool_calls"
    else:
        message = {"role": "assistant", "content": text}
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_id,
        "choices": [{
            "index": 0,
            "message": message,
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