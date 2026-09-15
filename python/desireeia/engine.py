"""DesireeIAEngine — static entry point. Mirrors C# DesireeIAEngine.cs."""

from __future__ import annotations

from typing import Optional

from . import _native as _nat
from .enums import Error
from .types import ExecutionPlan, HardwareProfile


def version() -> str:
    """Return the native engine version string."""
    return _nat.get_lib().desireeia_version().decode("utf-8")


def detect_hardware() -> HardwareProfile:
    """Probe the machine and return a HardwareProfile."""
    hw = _nat.HwInfo()
    err = _nat.get_lib().desireeia_probe_hw(hw)
    if err != Error.OK:
        raise RuntimeError(f"Hardware probe failed: {Error(err).name}")
    return HardwareProfile.from_native(hw)


def build_plan(
    model_path: Optional[str] = None,
    overrides: Optional[ExecutionPlan] = None,
) -> ExecutionPlan:
    """Let the native engine auto-detect the best plan, optionally with overrides."""
    lib = _nat.get_lib()

    hw = _nat.HwInfo()
    err = lib.desireeia_probe_hw(hw)
    if err != Error.OK:
        raise RuntimeError(f"Hardware probe failed: {Error(err).name}")

    override_native = _nat.Plan()
    override_ptr = _nat.ctypes.pointer(override_native) if overrides is not None else None
    plan_out = _nat.Plan()

    model_bytes = model_path.encode("utf-8") if model_path else None

    err = lib.desireeia_make_plan(
        _nat.ctypes.byref(hw),
        model_bytes,
        override_ptr,
        _nat.ctypes.byref(plan_out),
    )
    if err != Error.OK:
        raise RuntimeError(f"Plan build failed: {Error(err).name}")

    return ExecutionPlan.from_native(plan_out)


def profile_dump() -> str:
    """Human-readable cumulative matmul profiling counters."""
    buf = ( _nat.ctypes.c_char * 2048 )()
    _nat.get_lib().desireeia_profile_dump(buf, 2048)
    raw = buf.value
    return raw.decode("utf-8", errors="replace")


def profile_reset() -> None:
    """Zero the process-global profiling counters."""
    _nat.get_lib().desireeia_profile_reset()
