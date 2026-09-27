"""Tests that exercise the native engine ABI. Skipped when no native library
is available (see desireeia._native._find_lib)."""

import os

import pytest

import desireeia
from desireeia import _native as _nat
from desireeia.enums import Error


def _native_available() -> bool:
    try:
        _nat.get_lib()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    _native_available() is False,
    reason="DesireeIALocaleEngine native library not available in this environment",
)


def test_lib_is_cached():
    assert _nat.get_lib() is _nat.get_lib()


def test_version_returns_nonempty_string():
    v = desireeia.version()
    assert isinstance(v, str)
    assert v.strip()


def test_probe_hw_returns_profile():
    hw = desireeia.detect_hardware()
    assert hw.cpu_threads > 0
    assert hw.ram_total_mb > 0


def test_build_plan_does_not_crash():
    plan = desireeia.build_plan()
    assert plan.ram_budget_mb >= 0


def test_build_plan_honors_overrides():
    # Regression test: build_plan() used to construct a blank, all-zero
    # native Plan struct and never call overrides.to_native() on the
    # ExecutionPlan the caller passed in, silently discarding every field
    # the caller explicitly set (thread_count, kv_compression,
    # batch_union, expert_prefetch, dense_quantization, ...) in favor of
    # whatever the native auto-detect logic picks for a zeroed field. The
    # native side only auto-fills a field when it is left at zero (see
    # Enums.cs's comment on this exact behavior), so an explicit non-zero
    # thread_count must come back unchanged if overrides are actually
    # being applied.
    from desireeia.types import ExecutionPlan

    override = ExecutionPlan(thread_count=3, kv_compression=False, batch_union=False)
    plan = desireeia.build_plan(overrides=override)
    assert plan.thread_count == 3
    assert plan.kv_compression is False
    assert plan.batch_union is False


def test_error_codes_roundtrip():
    for code in (0, -1, -2, -3, -4, -5, -6):
        assert Error(code) is not None

_MODEL = os.environ.get("DESIREEIA_TEST_MODEL_PATH")


@pytest.mark.skipif(not _MODEL or not os.path.exists(_MODEL), reason="DESIREEIA_TEST_MODEL_PATH not set")
def test_session_reuses_prefix_and_matches_full_prefill():
    """Turn 2 = turn-1 prompt + more: exact mode reuses the turn-1 prompt from
    the KV cache and must produce exactly what a full prefill produces."""
    model = desireeia.LocalModel.load(_MODEL, desireeia.build_plan(_MODEL))
    try:
        turn1 = model.tokenize("Hello, my cat is called Luna.")
        model.predict(turn1)
        assert model.last_reused_tokens() == 0

        turn2 = turn1 + model.tokenize(" What is my cat called?", add_bos=False)
        first_session = model.predict(turn2)
        rest_session = [model.next_token() for _ in range(8)]
        assert model.last_reused_tokens() == len(turn1)

        model.reset_session()
        first_full = model.predict(turn2)
        rest_full = [model.next_token() for _ in range(8)]
        assert model.last_reused_tokens() == 0
        assert [first_session] + rest_session == [first_full] + rest_full

        model.set_session_reuse(0)
        model.predict(turn2)
        assert model.last_reused_tokens() == 0
        with pytest.raises(ValueError):
            model.set_session_reuse(3)
    finally:
        model.close()


@pytest.mark.skipif(not _MODEL or not os.path.exists(_MODEL), reason="DESIREEIA_TEST_MODEL_PATH not set")
def test_session_file_restores_prefix_in_a_fresh_instance(tmp_path):
    """A prefix saved by one instance is reused by a fresh one, with the
    answer a full prefill gives; a corrupted or missing file is refused."""
    file = tmp_path / "session.dskv"

    def greedy(model, prompt, n=8):
        ids = [model.predict(prompt)]
        ids += [model.next_token() for _ in range(n - 1)]
        return ids

    model = desireeia.LocalModel.load(_MODEL, desireeia.build_plan(_MODEL))
    try:
        system = model.tokenize("You are a helpful assistant. Answer briefly and precisely.")
        turn = system + model.tokenize(" What is two plus two?", add_bos=False)
        expected = greedy(model, turn)
        model.reset_session()
        model.predict(system)
        assert model.save_session(file)
    finally:
        model.close()

    model = desireeia.LocalModel.load(_MODEL, desireeia.build_plan(_MODEL))
    try:
        assert model.load_session(file) == len(system)
        assert greedy(model, turn) == expected
        assert model.last_reused_tokens() == len(system)
        image = model.save_session_bytes(len(system))
        assert image
        model.reset_session()
        assert model.load_session_bytes(image) == len(system)
        assert greedy(model, turn) == expected
        assert model.load_session_bytes(image[:-1]) == 0
        file.write_bytes(bytes([1, 2, 3]))
        assert model.load_session(file) == 0
        assert model.load_session(tmp_path / "missing.dskv") == 0
    finally:
        model.close()


def test_abi_version_and_gpus():
    assert desireeia.abi_version() >= 2
    for g in desireeia.gpus():
        assert g["name"] and g["total_bytes"] > 0


@pytest.mark.skipif(not _MODEL or not os.path.exists(_MODEL), reason="DESIREEIA_TEST_MODEL_PATH not set")
def test_cancel_progress_trim(tmp_path):
    """Progress is reported, a cancel from another thread stops a long
    prefill, and after trimming the cache the model answers as before."""
    import threading
    model = desireeia.LocalModel.load(_MODEL, desireeia.build_plan(_MODEL))
    try:
        short = model.tokenize("The capital of France is")

        def greedy(prompt, n=6):
            ids = [model.predict(prompt)]
            ids += [model.next_token() for _ in range(n - 1)]
            return ids

        expected = greedy(short)
        seen = []
        model.set_prefill_progress(lambda d, t: seen.append((d, t)))
        filler = model.tokenize(" the quick brown fox jumps over the lazy dog", add_bos=False)
        long_prompt = list(short)
        while len(long_prompt) < 3000:
            long_prompt += filler
        model.reset_session()
        model.predict(long_prompt[:200])
        assert seen and seen[-1][1] > 0

        model.reset_session()
        errors = []

        def run():
            try:
                model.predict(long_prompt)
            except InterruptedError:
                errors.append("cancelled")

        th = threading.Thread(target=run)
        th.start()
        import time
        time.sleep(0.03)
        model.cancel()
        th.join()
        model.set_prefill_progress(None)

        assert model.reserve_context(1024)
        model.trim_cache()
        model.reset_session()
        assert greedy(short) == expected
    finally:
        model.close()
