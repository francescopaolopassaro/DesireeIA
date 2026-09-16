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