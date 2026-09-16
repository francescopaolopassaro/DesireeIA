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

The PARSER deliberately recognizes several dialects beyond the one the
system prompt asks for, not just that one: a weak/local model asked to
call a tool very often reaches for whatever tool-call-shaped text it saw
most during training instead of the exact format requested - <function>,
<function_call>, <function=NAME>, bare <tool_call>{...} without matching
name/arguments keys, etc. are all real, observed outputs, not
hypothetical ones. Refusing everything but one exact tag turns "the model
tried to call a tool and got the shape slightly wrong" into "nothing
happened" from the user's point of view, which is a worse failure than
being a little permissive about what counts as an attempt.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

# The format the system prompt instructs the model to use - authoritative
# for what render_tool_call_text() re-emits when replaying history, even
# though extract_tool_call() accepts more than just this on the way in.
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

_TAG_JSON_PATTERNS = (
    re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.IGNORECASE | re.DOTALL),
    re.compile(r"<function_call>\s*(\{.*?\})\s*</function_call>", re.IGNORECASE | re.DOTALL),
)

# <function=NAME>{...}</function>  or  <function name="NAME">{...}</function>
_FUNCTION_NAMED_RE = re.compile(
    r'<function(?:\s*=\s*|\s+name\s*=\s*["\']?)([A-Za-z0-9_.\-]+)["\']?\s*>(.*?)</function\s*>',
    re.IGNORECASE | re.DOTALL,
)

# <function>NAME{...}</function> - the name precedes the arguments object.
_FUNCTION_INLINE_RE = re.compile(
    r"<function>\s*([A-Za-z0-9_.\-]+)\s*(\{.*?\})\s*</function\s*>",
    re.IGNORECASE | re.DOTALL,
)

# <tool_call>NAME<arg_key>KEY</arg_key><arg_value>VALUE</arg_value>...</tool_call>
# - key/value TAGS instead of a JSON object. Observed directly from a real
# local model (not hypothetical): asked to emit {"name":...,"arguments":
# {...}}, it instead invented its own tag-per-argument shape. The name
# check (no "{") keeps this from ever firing on the ordinary JSON case
# above, which is always tried first and already handles it correctly.
_TOOL_CALL_NAME_KV_RE = re.compile(
    r"<tool_call>\s*([A-Za-z0-9_.\-]+)(.*?)</tool_call>",
    re.IGNORECASE | re.DOTALL,
)
_ARG_KV_RE = re.compile(
    r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>",
    re.IGNORECASE | re.DOTALL,
)


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


def _parse_name_arguments_object(raw: str) -> Optional[dict]:
    """Parse a JSON object shaped {"name": ..., "arguments": {...}} - the
    body of <tool_call>/<function_call>, and also what a <function=NAME>
    tag's body sometimes turns out to be (a model occasionally repeats the
    name inside the JSON too, rather than only in the tag)."""
    from desireeia.generation import StructuredOutput

    candidate = StructuredOutput.try_extract_json(raw)
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


def _parse_json_object(raw: str) -> Optional[dict]:
    from desireeia.generation import StructuredOutput

    candidate = StructuredOutput.try_extract_json(raw.strip())
    if candidate is None:
        return None
    try:
        parsed = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_tool_call(text: str) -> Optional[dict]:
    """Return {"name": ..., "arguments": {...}} for the first recognized
    tool-call block in `text`, trying each known dialect in turn, or None
    if nothing recognizable is there."""
    for pattern in _TAG_JSON_PATTERNS:
        match = pattern.search(text)
        if match:
            result = _parse_name_arguments_object(match.group(1))
            if result is not None:
                return result

    match = _FUNCTION_INLINE_RE.search(text)
    if match:
        name, args_json = match.group(1), match.group(2)
        arguments = _parse_json_object(args_json)
        if arguments is not None:
            return {"name": name, "arguments": arguments}

    match = _FUNCTION_NAMED_RE.search(text)
    if match:
        name, body = match.group(1), match.group(2)
        parsed = _parse_json_object(body)
        if parsed is not None:
            # The body may itself be {"arguments": {...}} (name repeated
            # inside the JSON) or just the arguments object directly.
            arguments = parsed.get("arguments") if isinstance(parsed.get("arguments"), dict) else parsed
            return {"name": name, "arguments": arguments}

    match = _TOOL_CALL_NAME_KV_RE.search(text)
    if match:
        name, body = match.group(1), match.group(2)
        arguments = dict(_ARG_KV_RE.findall(body))
        if arguments:
            return {"name": name, "arguments": arguments}

    return None
