"""Unit tests for the streaming generation utilities."""

import pytest

from desireeia.generation import (
    StopSequenceScanner,
    StructuredOutput,
    ToolCalling,
    ToolCall,
)
from desireeia import ToolDefinition


class TestStopSequenceScanner:
    def test_no_stops_emits_everything_immediately(self):
        sc = StopSequenceScanner(None)
        assert sc.feed("hello ") == ("hello ", False)
        assert sc.feed("world") == ("world", False)
        assert not sc.has_stops

    def test_empty_stops_list_acts_as_no_stops(self):
        sc = StopSequenceScanner([])
        assert sc.feed("x") == ("x", False)

    def test_stop_split_across_pieces_is_detected(self):
        # Pieces concatenate to "hello world goodbye": the whole "goodbye"
        # must be dropped, everything before it emitted.
        sc = StopSequenceScanner(["goodbye"])
        emitted = []
        stopped = False
        for piece in ["hel", "lo wor", "ld go", "odbye"]:
            emit, stopped = sc.feed(piece)
            if emit:
                emitted.append(emit)
            if stopped:
                break
        assert stopped is True
        assert "".join(emitted) == "hello world "

    def test_stop_text_not_included_in_output(self):
        sc = StopSequenceScanner(["STOP"])
        emit, stopped = sc.feed("abcSTOP")
        assert emit == "abc"
        assert stopped is True

    def test_first_stop_in_list_that_matches_wins(self):
        # Order of the list is what matters: "ab" is listed first, so the
        # match at index 2 wins even though "<b>" also appears later.
        sc = StopSequenceScanner(["ab", "</b>"])
        emit, stopped = sc.feed("12ab</b>")
        assert emit == "12"
        assert stopped is True

    def test_partial_stop_held_until_certain(self):
        sc = StopSequenceScanner(["END"])
        # "EN" could be the start of "END", so the tail must not be emitted.
        emit, stopped = sc.feed("xEN")
        assert emit == "x"
        assert stopped is False
        # Completing the held tail ("EN"+"Dmore" = "ENDmore") fires the stop.
        emit, stopped = sc.feed("Dmore")
        assert emit == ""
        assert stopped is True

    def test_flush_emits_held_tail(self):
        sc = StopSequenceScanner(["END"])
        emit, _ = sc.feed("hello EN")
        assert emit == "hello "  # "EN" held back
        assert sc.flush() == "EN"


class TestToolCalling:
    def test_parses_valid_tool_call(self):
        call = ToolCalling.try_parse(
            '<tool_call>{"name": "calc", "arguments": {"a": 1}}</tool_call>'
        )
        assert call is not None
        assert call.name == "calc"
        assert "a" in call.arguments_json

    def test_returns_none_without_tag(self):
        assert ToolCalling.try_parse("just some text") is None

    def test_returns_none_on_invalid_json(self):
        assert ToolCalling.try_parse("<tool_call>{not json}</tool_call>") is None

    def test_returns_none_on_missing_name(self):
        assert ToolCalling.try_parse('<tool_call>{"arguments": {}}</tool_call>') is None

    def test_returns_none_on_empty(self):
        assert ToolCalling.try_parse("") is None

    def test_build_system_prompt(self):
        tools = [ToolDefinition("calc", "does math", "{}")]
        prompt = ToolCalling.build_system_prompt(tools)
        assert "<tool_call>" in prompt
        assert "calc" in prompt

    def test_build_result_message_uses_user_role(self):
        role, content = ToolCalling.build_result_message("calc", '{"ok": true}')
        assert role == "user"
        assert content.startswith('[tool_result name="calc"]')


class TestStructuredOutput:
    def test_extracts_json_from_surrounding_text(self):
        extracted = StructuredOutput.try_extract_json(
            'Here you go: {"a": 1, "b": [1, 2]} thanks!'
        )
        assert extracted is not None
        assert extracted == '{"a": 1, "b": [1, 2]}'

    def test_extracts_json_array(self):
        extracted = StructuredOutput.try_extract_json("List: [1, 2, 3] end")
        assert extracted == "[1, 2, 3]"

    def test_returns_none_for_plain_text(self):
        assert StructuredOutput.try_extract_json("no json here") is None

    def test_skips_json_inside_strings(self):
        # A '{' inside a quoted string must not start an object
        extracted = StructuredOutput.try_extract_json('{"msg": "brace { inside"} tail')
        assert extracted is not None
        assert extracted.startswith('{"msg"')

    def test_build_json_instruction(self):
        assert "ONLY valid JSON" in StructuredOutput.build_json_instruction()
        assert "schema" in StructuredOutput.build_json_instruction("{}")