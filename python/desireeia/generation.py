"""Generation utilities — StopSequenceScanner, ToolCalling, StructuredOutput.

Mirrors C# Generation.cs.
"""

from __future__ import annotations

import json
import re
from typing import Iterator, List, Optional, Tuple

from .types import GenerateOptions, ToolCall, ToolDefinition


# ---------------------------------------------------------------------------
# StopSequenceScanner
# ---------------------------------------------------------------------------

class StopSequenceScanner:
    """Incrementally detect stop sequences across token fragments."""

    __slots__ = ("_stops", "_max_stop_len", "_tail")

    def __init__(self, stop_sequences: Optional[List[str]]) -> None:
        self._stops = [s for s in (stop_sequences or []) if s]
        self._max_stop_len = max((len(s) for s in self._stops), default=0)
        self._tail: list[str] = []

    @property
    def has_stops(self) -> bool:
        return len(self._stops) > 0

    def feed(self, piece: str) -> Tuple[str, bool]:
        """Return (safe_text_to_emit, stopped)."""
        if not self._stops:
            return piece, False

        self._tail.append(piece)
        combined = "".join(self._tail)

        for stop in self._stops:
            idx = combined.find(stop)
            if idx >= 0:
                before = combined[:idx]
                self._tail.clear()
                return before, True

        safe_len = max(0, len(combined) - (self._max_stop_len - 1))
        to_emit = combined[:safe_len]
        self._tail.clear()
        if safe_len < len(combined):
            self._tail.append(combined[safe_len:])
        return to_emit, False

    def flush(self) -> str:
        """Return remaining tail text (called when generation ends normally)."""
        s = "".join(self._tail)
        self._tail.clear()
        return s


# ---------------------------------------------------------------------------
# ToolCalling
# ---------------------------------------------------------------------------

_OPEN_TAG = "<tool_call>"
_CLOSE_TAG = "</tool_call>"


class ToolCalling:
    """Prompt-based tool calling (mirrors C# ToolCalling static class)."""

    @staticmethod
    def build_system_prompt(tools: List[ToolDefinition]) -> str:
        lines = [
            "You can call tools to help answer the user. To call a tool, "
            "respond with ONLY this block and nothing else (no other text before or after):",
            _OPEN_TAG,
            '{"name": "<tool name>", "arguments": { ... }}',
            _CLOSE_TAG,
            "If no tool call is needed, answer normally in plain text instead. Available tools:",
        ]
        for t in tools:
            lines.append(f"- {t.name}: {t.description}")
            lines.append(f"  parameters (JSON schema): {t.parameters_json_schema}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def try_parse(response_text: str) -> Optional[ToolCall]:
        if not response_text:
            return None
        start = response_text.find(_OPEN_TAG)
        if start < 0:
            return None
        start += len(_OPEN_TAG)
        end = response_text.find(_CLOSE_TAG, start)
        json_str = response_text[start:end].strip() if end >= 0 else response_text[start:].strip()
        if not json_str:
            return None
        try:
            obj = json.loads(json_str)
            name = obj.get("name")
            if not name:
                return None
            args = json.dumps(obj.get("arguments", {}))
            return ToolCall(name=name, arguments_json=args)
        except (json.JSONDecodeError, AttributeError):
            return None

    @staticmethod
    def build_result_message(tool_name: str, result_json: str) -> Tuple[str, str]:
        return ("user", f'[tool_result name="{tool_name}"] {result_json}')


# ---------------------------------------------------------------------------
# StructuredOutput
# ---------------------------------------------------------------------------

class StructuredOutput:
    """Best-effort JSON extraction from free-form text."""

    @staticmethod
    def build_json_instruction(json_schema: Optional[str] = None) -> str:
        instr = (
            "Respond with ONLY valid JSON and no other text: "
            "no markdown code fences, no explanation before or after."
        )
        if json_schema:
            instr += " The JSON must conform to this schema:\n" + json_schema
        return instr

    @staticmethod
    def try_extract_json(text: str) -> Optional[str]:
        for i, ch in enumerate(text):
            if ch not in ("{", "["):
                continue
            close = "}" if ch == "{" else "]"
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                c = text[j]
                if in_string:
                    if escape:
                        escape = False
                    elif c == "\\":
                        escape = True
                    elif c == '"':
                        in_string = False
                    continue
                if c == '"':
                    in_string = True
                    continue
                if c == ch:
                    depth += 1
                elif c == close:
                    depth -= 1
                    if depth == 0:
                        candidate = text[i : j + 1]
                        try:
                            json.loads(candidate)
                            return candidate
                        except (json.JSONDecodeError, ValueError):
                            break
        return None
