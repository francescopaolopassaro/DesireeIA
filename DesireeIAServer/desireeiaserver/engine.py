"""Engine adapter: the inference motor is always DesireeIA.

This module is the only place that imports the `desireeia` wrapper. Layers
above it (routers, slots, UI) talk to this adapter, never to the native
binding directly, so the engine stays swappable in tests via a fake module.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Optional

from .config import Settings

logger = logging.getLogger(__name__)

_desireeia = None
_import_error: Optional[str] = None
try:
    import desireeia as _desireeia
except Exception as exc:  # pragma: no cover - depends on the local environment
    _import_error = f"{type(exc).__name__}: {exc}"

_BACKEND_NAMES = {
    "cpu": "CPU",
    "cuda": "CUDA",
    "metal": "METAL",
    "vulkan": "VULKAN",
    "intel": "INTEL",
    "axelera": "AXELERA",
}


class EngineUnavailable(RuntimeError):
    pass


def engine_available() -> bool:
    return _desireeia is not None


def engine_import_error() -> Optional[str]:
    return _import_error


def _require() -> Any:
    if _desireeia is None:
        raise EngineUnavailable(_import_error or "desireeia is not importable")
    return _desireeia


def engine_version() -> Optional[str]:
    try:
        return _require().version()
    except Exception:
        logger.debug("Engine version probe failed", exc_info=True)
        return None


def hardware_probe() -> Optional[dict]:
    try:
        hw = _require().detect_hardware()
    except Exception:
        logger.debug("Hardware probe failed", exc_info=True)
        return None
    return {
        "cpu_threads": hw.cpu_threads,
        "avx": hw.avx,
        "avx2": hw.avx2,
        "avx512": hw.avx512,
        "neon": hw.neon,
        "cuda_device_count": hw.cuda_device_count,
        "ram_total_mb": hw.ram_total_mb,
        "ram_free_mb": hw.ram_free_mb,
        "intel_gpu_count": hw.intel_gpu_count,
        "axelera_device_count": hw.axelera_device_count,
        "metal": hw.metal,
        "vulkan": hw.vulkan,
    }


def _pick_backend(settings: Settings):
    desireeia = _require()
    if settings.backend:
        name = _BACKEND_NAMES.get(settings.backend.lower())
        if name is None:
            raise ValueError(f"unknown backend {settings.backend!r}")
        return getattr(desireeia.InferenceBackend, name)
    if settings.n_gpu_layers and settings.n_gpu_layers > 0:
        try:
            hw = desireeia.detect_hardware()
        except Exception:
            return desireeia.InferenceBackend.UNCONFIGURED
        if hw.cuda_device_count > 0:
            return desireeia.InferenceBackend.CUDA
        if getattr(hw, "metal", False):
            return desireeia.InferenceBackend.METAL
        if hw.intel_gpu_count > 0:
            return desireeia.InferenceBackend.INTEL
        if hw.vulkan:
            return desireeia.InferenceBackend.VULKAN
    return desireeia.InferenceBackend.UNCONFIGURED


def build_plan(model_path: str, settings: Settings):
    desireeia = _require()
    overrides = desireeia.ExecutionPlan(
        backend=_pick_backend(settings),
        thread_count=settings.threads if settings.threads else (os.cpu_count() or 1),
        ram_budget_mb=settings.ram_budget_mb,
    )
    return desireeia.build_plan(model_path, overrides)


def load_model(
    model_path: str,
    settings: Settings,
    logger_cb: Optional[Callable[[str], None]] = None,
):
    desireeia = _require()
    plan = build_plan(model_path, settings)
    return desireeia.LocalModel.load(model_path, plan, logger=logger_cb)


def sampling_options(settings: Settings):
    desireeia = _require()
    return desireeia.SamplingOptions(
        temperature=settings.temperature,
        top_k=settings.top_k,
        top_p=settings.top_p,
        penalty_repeat=settings.repeat_penalty,
        penalty_frequency=settings.frequency_penalty,
        penalty_presence=settings.presence_penalty,
        penalty_last_n=settings.repeat_last_n,
        seed=settings.seed if settings.seed >= 0 else 0,
    )


def generate_options(
    settings: Settings,
    max_tokens: Optional[int] = None,
    stop_sequences: Optional[list] = None,
):
    desireeia = _require()
    resolved = max_tokens
    if resolved is None or resolved <= 0:
        resolved = settings.n_predict if settings.n_predict > 0 else 512
    return desireeia.GenerateOptions(max_tokens=resolved, stop_sequences=stop_sequences)


def default_generation_settings(settings: Settings) -> dict:
    return {
        "temperature": settings.temperature,
        "top_k": settings.top_k,
        "top_p": settings.top_p,
        "repeat_penalty": settings.repeat_penalty,
        "repeat_last_n": settings.repeat_last_n,
        "frequency_penalty": settings.frequency_penalty,
        "presence_penalty": settings.presence_penalty,
        "seed": settings.seed,
        "n_predict": settings.n_predict,
    }