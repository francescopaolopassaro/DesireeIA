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
from .enums import InferenceBackend
from .types import ExecutionPlan, HardwareProfile, SamplingOptions

DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 40
DEFAULT_TOP_P = 0.9
DEFAULT_MAX_TOKENS = 2048
TARGET_CONTEXT_SIZE = 16384
# Context to aim for when the KV cache is windowed (Q8_0 cache, a model with local-attention layers): only the global layers
# grow with it, so twice the context costs about what 16384 positions cost with the whole cache.
WINDOWED_TARGET_CONTEXT_SIZE = 32768


def target_context_for(model: "ModelTraits", plan: ExecutionPlan) -> int:
    """The context to aim for with this model and plan: the windowed target when the cache is windowed."""
    return WINDOWED_TARGET_CONTEXT_SIZE if plan.kv_compression and model.local_layer_count > 0 and model.sliding_window > 0         else TARGET_CONTEXT_SIZE
MIN_CONTEXT_SIZE = 2048
SAFETY_MARGIN_MB = 1024
# Video memory kept free after the weights and the KV cache, for the compute buffers of a prefill. Smaller than
# SAFETY_MARGIN_MB on purpose: the free figure of the driver already excludes what the desktop uses. MEASURED with a 4B
# model on a card showing 5080 MB free: weights 3379 MB + KV 1224 MB (16384 tokens) left ~480 MB and ran at 2100 tokens/s.
VIDEO_MARGIN_MB = 384


# ---------------------------------------------------------------------------
# GGUF header (metadata only, no tensors)
# ---------------------------------------------------------------------------

_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i",
            6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _read_str(f) -> str:
    (n,) = struct.unpack("<Q", f.read(8))
    return f.read(n).decode("utf-8", errors="replace")


def _read_value(f, vtype: int, keep: bool = False) -> Any:
    if vtype == 8:
        return _read_str(f)
    if vtype == 9:
        (etype,) = struct.unpack("<I", f.read(4))
        (count,) = struct.unpack("<Q", f.read(8))
        if etype in (8, 9):
            for _ in range(count):
                _read_value(f, etype)
        elif keep and count <= 4096:
            fmt = _SCALARS[etype]
            raw = f.read(struct.calcsize(fmt) * count)
            return [int(x) for x in struct.unpack("<" + fmt[1:] * count, raw)]   # the per-layer local-attention flags
        else:
            f.seek(struct.calcsize(_SCALARS[etype]) * count, 1)
        return None  # other arrays are skipped, like the C# reader's ArrayMarker
    fmt = _SCALARS.get(vtype)
    if fmt is None:
        raise ValueError(f"Unknown GGUF value type {vtype}.")
    (v,) = struct.unpack(fmt, f.read(struct.calcsize(fmt)))
    return v


def read_gguf_metadata(model_path: str) -> Dict[str, Any]:
    """Key/value metadata of a GGUF file (array values are omitted, except the local-attention flags)."""
    kv: Dict[str, Any] = {}
    with open(model_path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("Not a GGUF file.")
        _version, _tensors, count = struct.unpack("<IQQ", f.read(20))
        for _ in range(count):
            key = _read_str(f)
            (vtype,) = struct.unpack("<I", f.read(4))
            value = _read_value(f, vtype, key.endswith("attention.sliding_window_pattern"))
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
    local_layer_count: int = 0          # layers with local attention (they attend to sliding_window positions only)

    # Positions beyond the window a local layer keeps in the engine's windowed cache: the rows behind the current one
    # a rolled-back draft may need (64) plus the block a long prompt goes through at a time (1024).
    WINDOWED_KV_SLACK = 64 + 1024

    @property
    def local_kv_positions(self) -> int:
        """Positions one local layer holds in the windowed cache (a fixed size, whatever the context)."""
        return self.sliding_window - 1 + self.WINDOWED_KV_SLACK if self.sliding_window > 0 else 0

    def _per_layer_bytes(self, quantized: bool) -> int:
        per_layer = self.head_count_kv * (self.key_length + self.value_length)
        return (per_layer * 34 + 31) // 32 if quantized else per_layer * 4

    def _locals(self, quantized: bool) -> int:
        return max(0, min(self.local_layer_count, self.layer_count)) if quantized and self.sliding_window > 0 else 0

    def kv_bytes_for(self, quantized: bool, positions: int) -> int:
        """KV-cache bytes for ``positions`` of context as the engine stores it: with the Q8_0 cache the local layers keep
        only ``local_kv_positions`` positions, so only the global layers grow with the context."""
        if positions <= 0 or self.layer_count <= 0:
            return 0
        per = self._per_layer_bytes(quantized)
        locals_ = self._locals(quantized)
        return (self.layer_count - locals_) * positions * per + locals_ * min(positions, self.local_kv_positions) * per

    def max_positions_for(self, quantized: bool, budget_bytes: int) -> int:
        """The context (positions) a KV cache of at most ``budget_bytes`` holds."""
        whole = self.kv_bytes_for(quantized, 1)
        if whole <= 0:
            return 0
        locals_ = self._locals(quantized)
        if locals_ == 0:
            return budget_bytes // whole
        per = self._per_layer_bytes(quantized)
        globals_ = self.layer_count - locals_
        if globals_ == 0:
            return 2 ** 31 - 1
        fixed = locals_ * self.local_kv_positions * per
        if budget_bytes > fixed:
            by_global = (budget_bytes - fixed) // (globals_ * per)
            if by_global >= self.local_kv_positions:
                return by_global
        return min(budget_bytes // whole, self.local_kv_positions)   # small: every layer holds every position

    @property
    def kv_bytes_per_token(self) -> int:
        """KV-cache bytes per context token with the float32 cache (all layers)."""
        return self.kv_bytes_per_token_for(False)

    def kv_bytes_per_token_for(self, quantized: bool) -> int:
        """KV-cache bytes per token as the engine stores it: float32 (4 bytes
        per value) or Q8_0 with KV compression on (34 bytes per 32 values)."""
        values = self.layer_count * self.head_count_kv * (self.key_length + self.value_length)
        return (values * 34 + 31) // 32 if quantized else values * 4

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
        layers = i(f"{arch}.block_count") or 0
        sliding = i(f"{arch}.attention.sliding_window") or 0
        local_layers = 0
        if sliding > 0 and layers > 0:
            pattern = kv.get(f"{arch}.attention.sliding_window_pattern")
            if isinstance(pattern, list) and pattern:
                local_layers = sum(1 for x in pattern[:layers] if x)
            else:
                period = pattern if isinstance(pattern, int) and not isinstance(pattern, bool) and pattern > 0 else 6
                local_layers = sum(1 for l in range(layers) if l % period != period - 1)
        return ModelTraits(
            architecture=arch,
            trained_context_length=i(f"{arch}.context_length") or 0,
            layer_count=layers,
            head_count_kv=head_count_kv,
            key_length=key_length,
            value_length=value_length,
            sliding_window=i(f"{arch}.attention.sliding_window") or 0,
            file_size_bytes=os.path.getsize(model_path),
            recommended_temperature=fl("general.sampling.temp"),
            recommended_top_k=top_k if top_k is not None and top_k > 0 else None,
            recommended_top_p=fl("general.sampling.top_p"),
            local_layer_count=local_layers,
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
    # The largest context that still fits this machine (trained length, RAM and, on a GPU, VIDEO MEMORY): what a caller may
    # grow the context to when a prompt needs more than context_size. Never below context_size.
    max_context_size: int = 0


def max_context_by_video_memory(hw: HardwareProfile, plan: ExecutionPlan, model: ModelTraits):
    """The context that fits the VIDEO MEMORY of the GPU, as ``(tokens, note)``, or ``(None, None)`` when it cannot be told.

    Weights are counted in video memory, then the KV cache per token, with SAFETY_MARGIN_MB left for the compute buffers and
    the desktop. MEASURED on a 6 GB card with a 4B model (3.5 GB of weights, 78 KB of KV per token): 16384 tokens run,
    40960 spill into shared memory and the prefill gets seven times slower with the GPU pinned at 100%.
    """
    if plan.backend != InferenceBackend.CUDA or not hw.gpus:
        return None, None
    kv_per_token = model.kv_bytes_per_token_for(plan.kv_compression)
    if kv_per_token <= 0:
        return None, None
    gpu = max(hw.gpus, key=lambda g: g["free_bytes"])
    free_mb = gpu["free_bytes"] // (1024 * 1024)
    model_mb = max(0, model.file_size_bytes) // (1024 * 1024)
    usable_mb = free_mb - VIDEO_MARGIN_MB - model_mb
    if usable_mb <= 0:
        return MIN_CONTEXT_SIZE, (
            f"video memory: {free_mb} MB free on {gpu['name']}, the weights alone are ~{model_mb} MB "
            f"(+{VIDEO_MARGIN_MB} MB margin): no room for a KV cache on the GPU, the smallest context is used")
    fit = model.max_positions_for(plan.kv_compression, usable_mb * 1024 * 1024)
    kv_at_64k = model.kv_bytes_for(plan.kv_compression, 65536) // 65536   # the local layers stay small: only the global ones grow
    return int(fit), (
        f"video memory: {free_mb} MB free on {gpu['name']}, weights ~{model_mb} MB, ~{kv_at_64k // 1024} KB of KV per "
        f"token: up to {fit} tokens of context fit")


def _fmt(v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}".rstrip("0").rstrip(".")


def compute_configuration(hw: HardwareProfile, plan: ExecutionPlan, model: ModelTraits,
                          target_context: int = TARGET_CONTEXT_SIZE) -> AutoConfiguration:
    """The pure decision, separated from probing so it can be tested."""
    notes = [
        f"backend {plan.backend.name}, {plan.thread_count} threads, RAM budget {plan.ram_budget_mb} MB "
        f"(cpu threads {hw.cpu_threads}, cuda devices {hw.cuda_device_count}, free RAM {hw.ram_free_mb} MB)"
    ]

    if target_context == TARGET_CONTEXT_SIZE:
        target_context = target_context_for(model, plan)   # the default target follows the cache layout
    context = max(MIN_CONTEXT_SIZE, int(target_context))
    max_context = None   # what could still fit, whatever the target
    if model.trained_context_length > 0:
        max_context = model.trained_context_length
        if model.trained_context_length < context:
            context = model.trained_context_length
            notes.append(f"context limited to the trained length {model.trained_context_length}")

    budget_mb = plan.ram_budget_mb if plan.ram_budget_mb > 0 else hw.ram_free_mb
    model_mb = max(0, model.file_size_bytes) // (1024 * 1024)
    kv_per_token = model.kv_bytes_per_token_for(plan.kv_compression)
    if budget_mb > 0 and kv_per_token > 0:
        free_for_kv_mb = budget_mb - model_mb - SAFETY_MARGIN_MB if budget_mb > model_mb + SAFETY_MARGIN_MB else 0
        by_memory = model.max_positions_for(plan.kv_compression, free_for_kv_mb * 1024 * 1024 // 2)
        max_context = by_memory if max_context is None else min(max_context, by_memory)
        if by_memory < context:
            context = max(MIN_CONTEXT_SIZE, by_memory)
            notes.append(f"context limited by memory: ~{kv_per_token // 1024} KB of KV cache per token "
                         f"({'Q8_0' if plan.kv_compression else 'float32'}), "
                         f"{free_for_kv_mb} MB available after the model")

    # The video memory of the GPU: the RAM budget above does not see it, and a context that does not fit it is not refused
    # by the driver, it is silently spilled into shared memory and every prefill crawls.
    by_video, video_note = max_context_by_video_memory(hw, plan, model)
    if by_video is not None:
        notes.append(video_note)
        max_context = by_video if max_context is None else min(max_context, by_video)
        if by_video < context:
            context = max(MIN_CONTEXT_SIZE, by_video)
            notes.append(f"context limited by video memory to {context} tokens")
    if context > MIN_CONTEXT_SIZE:
        context -= context % 1024
    context = max(MIN_CONTEXT_SIZE, int(context))
    if max_context is not None and max_context > MIN_CONTEXT_SIZE:
        max_context -= max_context % 1024
    max_context = context if max_context is None else max(context, int(max_context))

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
    return AutoConfiguration(hw, plan, model, sampling, context, max_tokens, notes, max_context)


def auto_configure(model_path: str, overrides: Optional[ExecutionPlan] = None,
                   target_context: int = TARGET_CONTEXT_SIZE) -> AutoConfiguration:
    """Configure ``model_path`` for the current machine."""
    traits = ModelTraits.read(model_path)
    hw = detect_hardware()
    plan = build_plan(model_path, overrides)
    return compute_configuration(hw, plan, traits, target_context)
