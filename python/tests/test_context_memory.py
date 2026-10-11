# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""desireeia.ContextMemory: the Python binding over the engine's context
memory (desireeia_memory_*). The behavior itself is covered natively in
tests/agentic_test.cpp; this checks the marshalling end to end. Skipped
when no native library is available."""

import time

import pytest

import desireeia
from desireeia import _native as _nat


def _native_available() -> bool:
    try:
        _nat.get_lib().desireeia_memory_open  # noqa: B018 - an engine build with context memory
        return True
    except (OSError, AttributeError):
        return False


pytestmark = pytest.mark.skipif(not _native_available(), reason="native engine with context memory not available")


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "big.py"
    lines = [f"def f{i}():\n    return {i}\n" for i in range(20000)]
    lines[10000] = "def needle():\n    return 'città'\n"
    path.write_bytes("".join(lines).encode("utf-8"))
    return path


def test_open_read_search_edit(tmp_path, source):
    with desireeia.ContextMemory(str(tmp_path / "mem"), 600) as mem:
        stub = mem.put_file("s", "big.py", str(source))
        assert stub["spilled"] and stub["total_lines"] == 40000
        handle = stub["handle"]

        window = mem.read("s", handle, 20001, 2)
        assert window["content"] == "20001\tdef needle():\n20002\t    return 'città'"

        hits = mem.search("s", "needle", handle, 3)
        assert "def needle" in hits["results"][0]["snippet"]

        edit = mem.replace_lines("s", handle, 20002, 20002, "    return 'nuovo'")
        assert edit["ok"] and edit["written_to_source"]
        assert "    return 'nuovo'\n" in source.read_text(encoding="utf-8")


def test_stream_append_and_list(tmp_path):
    with desireeia.ContextMemory(str(tmp_path / "mem"), 600) as mem:
        mem.append("s", "log", "a\n")
        item = mem.append("s", "log", "b\nc\n")
        assert item["total_lines"] == 3 and item["version"] == 2
        assert [i["name"] for i in mem.list("s")["items"]] == ["log"]


def test_idle_session_is_swept(tmp_path, source):
    with desireeia.ContextMemory(str(tmp_path / "mem"), 1) as mem:
        mem.put_file("s", "big.py", str(source))
        time.sleep(1.5)
        mem.sweep()
        assert not (tmp_path / "mem" / "s").exists()
        assert source.exists()


def test_invalid_session_rejected(tmp_path):
    with desireeia.ContextMemory(str(tmp_path / "mem"), 600) as mem:
        with pytest.raises(RuntimeError):
            mem.put_text("../escape", "x", "data")
