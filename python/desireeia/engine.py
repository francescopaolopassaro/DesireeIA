"""DesireeIAEngine — static entry point. Mirrors C# DesireeIAEngine.cs."""

from __future__ import annotations

import ctypes
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

    # Bug fixed here: this used to construct a blank, all-zero `_nat.Plan()`
    # and never populate it from `overrides` - `overrides.to_native()` was
    # never called, so every field a caller explicitly set (thread_count,
    # kv_compression, batch_union, expert_prefetch, dense_quantization,
    # ...) was silently discarded and replaced by whatever the native
    # auto-detect logic picks for a zeroed field. The C# binding
    # (DesireeIAEngine.cs BuildPlan, `overrides.ToNative()`) always did
    # this correctly; only this Python wrapper had the gap - which is why
    # a server built on this wrapper (desireeiaserver) measured
    # dramatically lower decode throughput than the CLI on the identical
    # model/hardware, despite requesting the same plan.
    override_native = overrides.to_native() if overrides is not None else _nat.Plan()
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


def abi_version() -> int:
    """Interface version of the loaded native engine (0 when it predates the query)."""
    lib = _nat.get_lib()
    return int(lib.desireeia_abi_version()) if hasattr(lib, "desireeia_abi_version") else 0


def ensure_abi() -> None:
    """Raise a clear error when the native engine is older than this wrapper."""
    v = abi_version()
    if v < _nat.ABI_VERSION:
        raise RuntimeError(
            f"DesireeIA native engine interface {v} is older than this wrapper "
            f"({_nat.ABI_VERSION}): update the native library that ships with the package."
        )


def last_error(ctx=None) -> str:
    """Text of the last native error (the context's, else the calling thread's)."""
    lib = _nat.get_lib()
    if not hasattr(lib, "desireeia_last_error"):
        return ""
    n = int(lib.desireeia_last_error(ctx, None, 0))
    if n == 0:
        return ""
    buf = ctypes.create_string_buffer(n + 1)
    lib.desireeia_last_error(ctx, buf, n + 1)
    return buf.value.decode("utf-8", errors="replace")


def gpus() -> list:
    """GPUs the engine can use, as dicts (empty without CUDA)."""
    lib = _nat.get_lib()
    if not hasattr(lib, "desireeia_gpu_count"):
        return []
    out = []
    for i in range(int(lib.desireeia_gpu_count())):
        info = _nat.GpuInfoNative()
        if lib.desireeia_probe_gpu(i, ctypes.byref(info)) != 0:
            continue
        out.append({
            "index": i,
            "name": info.name.decode("utf-8", errors="replace"),
            "total_bytes": int(info.total_bytes),
            "free_bytes": int(info.free_bytes),
            "compute_capability": (int(info.cc_major), int(info.cc_minor)),
            "multiprocessors": int(info.multiprocessors),
        })
    return out


def profile_reset() -> None:
    """Zero the process-global profiling counters."""
    _nat.get_lib().desireeia_profile_reset()
