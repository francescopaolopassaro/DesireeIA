# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""OpenAI-style tool/function calling: prompt injection + text-based parsing.

The loaded local models are not natively trained on OpenAI's structured
tool_calls wire format, so this module asks the model to emit a single
"<tool_call>{...}</tool_call>" block when it wants to call a tool - the
same convention several open chat-template families (Hermes, Qwen, ...)
already use - and parses that back out of the generated text afterwards.
How reliably the model actually follows this is bounded by the model
itself; this module only gives it a clear instruction and a strict enough
marker to detect when it complies.
"""

from __future__ import annotations

import json
from typing import Any, Optional

TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"


def build_tools_instruction(tools: list[dict]) -> str:
    lines = [
        "You have access to tools. To call one, respond with ONLY the "
        "following and nothing else in the message (no explanation, no "
        "markdown fences):",
        TOOL_CALL_OPEN,
        '{"name": "<tool name>", "arguments": {<arguments as a JSON object>}}',
        TOOL_CALL_CLOSE,
        "Only call a tool when the user's request actually needs it. If no "
        "tool is needed, answer normally in plain text instead.",
        "",
        "Available tools:",
    ]
    for tool in tools:
        fn = tool.get("function", tool) if isinstance(tool, dict) else {}
        name = fn.get("name", "")
        description = fn.get("description", "")
        parameters = fn.get("parameters")
        lines.append(f"- {name}: {description}")
        if parameters:
            lines.append(f"  arguments schema: {json.dumps(parameters, ensure_ascii=False)}")
    return "\n".join(lines)


def render_tool_call_text(name: str, arguments: Any) -> str:
    """Re-serialize a tool call the way the model itself would have emitted
    it, so replaying it back as assistant history on a follow-up request
    stays consistent with the format the system prompt asked for."""
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    payload = json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"


def extract_tool_call(text: str) -> Optional[dict]:
    """Return {"name": ..., "arguments": {...}} if `text` contains a valid
    tool-call block, else None."""
    start = text.find(TOOL_CALL_OPEN)
    if start == -1:
        return None
    from desireeia.generation import StructuredOutput

    after = text[start + len(TOOL_CALL_OPEN):]
    candidate = StructuredOutput.try_extract_json(after)
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(parsed, dict):
        return None
    name = parsed.get("name")
    if not isinstance(name, str) or not name:
        return None
    arguments = parsed.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


class ToolCallStreamFilter:
    """Buffers just enough of the start of a stream to tell whether the
    model is emitting a <tool_call> block instead of normal text, without
    delaying a normal (non-tool) response by more than a few characters.

    feed(piece) returns the text that is now safe to forward to the client
    as a regular content delta ("" while still undecided, or while a tool
    call is confirmed in progress - its raw JSON is never shown to the
    user as chat text).
    """

    def __init__(self) -> None:
        self._pending = ""
        self._decided = False
        self.is_tool_call = False

    def feed(self, piece: str) -> str:
        if self._decided:
            return "" if self.is_tool_call else piece
        self._pending += piece
        stripped = self._pending.lstrip()
        prefix = stripped[: len(TOOL_CALL_OPEN)]
        if TOOL_CALL_OPEN.startswith(prefix):
            if len(prefix) == len(TOOL_CALL_OPEN):
                self._decided = True
                self.is_tool_call = True
                self._pending = ""
                return ""
            return ""  # still an ambiguous partial prefix - keep buffering
        self._decided = True
        self.is_tool_call = False
        flushed, self._pending = self._pending, ""
        return flushed
