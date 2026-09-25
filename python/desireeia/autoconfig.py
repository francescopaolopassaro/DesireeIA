"""Automatic configuration of a model for the current machine.

Mirror of the C# ``AutoConfigurator`` (src/DesireeIA/AutoConfig.cs): same
defaults, same rules, so the NuGet and PyPI packages behave identically.

- backend, threads and RAM budget come from the native planner
  (:func:`build_plan`), which probes CPU features, CUDA/Intel/Metal devices
  and free memory;
- sampling is never greedy: greedy decoding falls into repetitive loops on
  long generations;
- the context is as large as the target allows, but never beyond the
  trained length nor beyond what the KV cache can fit in memory;
- the reply length is bounded by the context.
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .engine import build_plan, detect_hardware
from .types import ExecutionPlan, HardwareProfile, SamplingOptions

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048
TARGET_CONTEXT_SIZE = 16384
MIN_CONTEXT_SIZE = 2048
SAFETY_MARGIN_MB = 1024


# ---------------------------------------------------------------------------
# GGUF header (metadata only, no tensors)
# ---------------------------------------------------------------------------

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int) -> Any:
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in (8, 9):
            for _ in range(count):
                _read_value(f, etype)
        else:
            f.seek(struct.calcsize(_SCALARS[etype]) * count, 1)
        return None  # arrays are skipped, like the C# reader's ArrayMarker
    fmt = _SCALARS.get(vtype)
    if fmt is None:
        raise ValueError(f"Unknown GGUF value type {vtype}.")
    (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    return v


def read_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """Key/value metadata of a GGUF file (array values are omitted)."""
    kv: Dict[str, Any] = {}
    with open(model_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file.")
        _version, _tensors, count = struct.unpack("<IQQ", f.read(20))
        for _ in range(count):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            value = _read_value(f, vtype)
            if value is not None:
                kv[key] = value
    return kv


# ---------------------------------------------------------------------------
# Model traits
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelTraits:
    architecture: str
    trained_context_length: int
    layer_count: int
    head_count_kv: int
    key_length: int
    value_length: int
    sliding_window: int
    file_size_bytes: int
    recommended_temperature: Optional[float] = None
    recommended_top_k: Optional[int] = None
    recommended_top_p: Optional[float] = None

    @property
    def kv_bytes_per_token(self) -> int:
        """Upper-bound KV-cache bytes per context token (16-bit, all layers)."""
        return 2 * self.layer_count * self.head_count_kv * (self.key_length + self.value_length)

    @staticmethod
    def read(model_path: str) -> "ModelTraits":
        kv = read_gguf_metadata(model_path)
        arch = kv.get("general.architecture", "unknown")

        def i(key: str) -> Optional[int]:
            v = kv.get(key)
            return int(v) if isinstance(v, (int, bool)) and not isinstance(v, bool) else None

        def fl(key: str) -> Optional[float]:
            v = kv.get(key)
            return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

        head_count = i(f"{arch}.attention.head_count") or 0
        head_count_kv = i(f"{arch}.attention.head_count_kv") or head_count
        embedding = i(f"{arch}.embedding_length") or 0
        head_dim = embedding // head_count if head_count > 0 else 0
        key_length = i(f"{arch}.attention.key_length") or head_dim
        value_length = i(f"{arch}.attention.value_length") or key_length
        top_k = i("general.sampling.top_k")
        return ModelTraits(
            architecture=arch,
            trained_context_length=i(f"{arch}.context_length") or 0,
            layer_count=i(f"{arch}.block_count") or 0,
            head_count_kv=head_count_kv,
            key_length=key_length,
            value_length=value_length,
            sliding_window=i(f"{arch}.attention.sliding_window") or 0,
            file_size_bytes=os.path.getsize(model_path),
            recommended_temperature=fl("general.sampling.temp"),
            recommended_top_k=top_k if top_k is not None and top_k > 0 else None,
            recommended_top_p=fl("general.sampling.top_p"),
        )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AutoConfiguration:
    hardware: HardwareProfile
    plan: ExecutionPlan
    model: ModelTraits
    sampling: SamplingOptions
    context_size: int
    max_tokens: int
    notes: List[str] = field(default_factory=list)


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}".rstrip("0").rstrip(".")


def compute_configuration(hw: HardwareProfile, plan: ExecutionPlan, model: ModelTraits) -> AutoConfiguration:
    """The pure decision, separated from probing so it can be tested."""
    notes = [
        f"backend {plan.backend.name}, {plan.thread_count} threads, RAM budget {plan.ram_budget_mb} MB "
        f"(cpu threads {hw.cpu_threads}, cuda devices {hw.cuda_device_count}, free RAM {hw.ram_free_mb} MB)"
    ]

    context = TARGET_CONTEXT_SIZE
    if 0 < model.trained_context_length < context:
        context = model.trained_context_length
        notes.append(f"context limited to the trained length {model.trained_context_length}")

    budget_mb = plan.ram_budget_mb if plan.ram_budget_mb > 0 else hw.ram_free_mb
    model_mb = max(0, model.file_size_bytes) // (1024 * 1024)
    kv_per_token = model.kv_bytes_per_token
    if budget_mb > 0 and kv_per_token > 0:
        free_for_kv_mb = budget_mb - model_mb - SAFETY_MARGIN_MB if budget_mb > model_mb + SAFETY_MARGIN_MB else 0
        by_memory = free_for_kv_mb * 1024 * 1024 // 2 // kv_per_token
        if by_memory < context:
            context = max(MIN_CONTEXT_SIZE, by_memory)
            notes.append(f"context limited by memory: ~{kv_per_token // 1024} KB of KV cache per token, "
                         f"{free_for_kv_mb} MB available after the model")
    if context > MIN_CONTEXT_SIZE:
        context -= context % 1024
    context = max(MIN_CONTEXT_SIZE, int(context))

    max_tokens = min(DEFAULT_MAX_TOKENS, context // 4)

    sampling = SamplingOptions(temperature=DEFAULT_TEMPERATURE, top_k=DEFAULT_TOP_K, top_p=DEFAULT_TOP_P)
    if (model.recommended_temperature is not None or model.recommended_top_p is not None
            or model.recommended_top_k is not None):
        notes.append(
            f"the model file suggests temp={_fmt(model.recommended_temperature)} "
            f"top-k={model.recommended_top_k if model.recommended_top_k is not None else '-'} "
            f"top-p={_fmt(model.recommended_top_p)}; defaults used: "
            f"temp={DEFAULT_TEMPERATURE} top-k={DEFAULT_TOP_K} top-p={DEFAULT_TOP_P}")

    notes.append(f"context {context} tokens, replies up to {max_tokens} tokens")
    return AutoConfiguration(hw, plan, model, sampling, context, max_tokens, notes)


def auto_configure(model_path: str, overrides: Optional[ExecutionPlan] = None) -> AutoConfiguration:
    """Configure ``model_path`` for the current machine."""
    traits = ModelTraits.read(model_path)
    hw = detect_hardware()
    plan = build_plan(model_path, overrides)
    return compute_configuration(hw, plan, traits)
