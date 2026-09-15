"""Public data types — mirrors C# HardwareProfile, ExecutionPlan, SamplingOptions, etc."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .enums import (
    InferenceBackend,
    ModelFormat,
    Quantization,
    SsdTierMode,
)
from . import _native as _nat


# ---------------------------------------------------------------------------
# HardwareProfile
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HardwareProfile:
    cpu_threads: int
    avx: bool
    avx2: bool
    avx512: bool
    neon: bool
    cuda_device_count: int
    ram_total_mb: int
    ram_free_mb: int
    intel_gpu_count: int
    axelera_device_count: int
    metal: bool
    vulkan: bool

    @staticmethod
    def from_native(hw: _nat.HwInfo) -> HardwareProfile:
        return HardwareProfile(
            cpu_threads=hw.cpu_threads,
            avx=hw.cpu_has_avx != 0,
            avx2=hw.cpu_has_avx2 != 0,
            avx512=hw.cpu_has_avx512 != 0,
            neon=hw.cpu_has_neon != 0,
            cuda_device_count=hw.cuda_device_count,
            ram_total_mb=hw.ram_total_mb,
            ram_free_mb=hw.ram_free_mb,
            intel_gpu_count=hw.intel_gpu_count,
            axelera_device_count=hw.axelera_device_count,
            metal=hw.has_metal != 0,
            vulkan=hw.has_vulkan != 0,
        )


# ---------------------------------------------------------------------------
# ExecutionPlan
# ---------------------------------------------------------------------------

@dataclass
class ExecutionPlan:
    backend: InferenceBackend = InferenceBackend.UNCONFIGURED
    format: ModelFormat = ModelFormat.UNKNOWN
    dense_quantization: Quantization = Quantization.Q8_0
    expert_quantization: Quantization = Quantization.Q4K
    thread_count: int = os.cpu_count() or 1
    ram_budget_mb: int = 0
    expert_cache_size: int = 256
    expert_prefetch: bool = True
    kv_compression: bool = True
    expert_pin_learning: bool = True
    prefetch_depth: int = 1
    batch_union: bool = True
    dual_ssd: bool = False
    ssd_tier: SsdTierMode = SsdTierMode.AUTO
    ssd_tier_cache_mb: int = 0

    def to_native(self) -> _nat.Plan:
        p = _nat.Plan()
        p.backend = int(self.backend)
        p.format = int(self.format)
        p.dense_quant = int(self.dense_quantization)
        p.expert_quant = int(self.expert_quantization)
        p.n_threads = self.thread_count
        p.ram_budget_mb = self.ram_budget_mb
        p.expert_cache_count = self.expert_cache_size
        p.expert_prefetch_enabled = 1 if self.expert_prefetch else 0
        p.kv_compression_enabled = 1 if self.kv_compression else 0
        p.expert_pin_enabled = 1 if self.expert_pin_learning else 0
        p.expert_prefetch_depth = self.prefetch_depth
        p.batch_union_enabled = 1 if self.batch_union else 0
        p.dual_ssd_enabled = 1 if self.dual_ssd else 0
        p.ssd_tier_mode = int(self.ssd_tier)
        p.ssd_tier_cache_mb = self.ssd_tier_cache_mb
        return p

    @staticmethod
    def from_native(p: _nat.Plan) -> ExecutionPlan:
        return ExecutionPlan(
            backend=InferenceBackend(p.backend),
            format=ModelFormat(p.format),
            dense_quantization=Quantization(p.dense_quant),
            expert_quantization=Quantization(p.expert_quant),
            thread_count=p.n_threads,
            ram_budget_mb=p.ram_budget_mb,
            expert_cache_size=p.expert_cache_count,
            expert_prefetch=p.expert_prefetch_enabled != 0,
            kv_compression=p.kv_compression_enabled != 0,
            expert_pin_learning=p.expert_pin_enabled != 0,
            prefetch_depth=p.expert_prefetch_depth,
            batch_union=p.batch_union_enabled != 0,
            dual_ssd=p.dual_ssd_enabled != 0,
            ssd_tier=SsdTierMode(p.ssd_tier_mode),
            ssd_tier_cache_mb=p.ssd_tier_cache_mb,
        )

    def __str__(self) -> str:
        return (
            f"backend={self.backend.name} format={self.format.name} "
            f"dense={self.dense_quantization.name} experts={self.expert_quantization.name} "
            f"threads={self.thread_count} ram={self.ram_budget_mb}MB "
            f"cache={self.expert_cache_size} pin={self.expert_pin_learning} "
            f"prefetchDepth={self.prefetch_depth} batchUnion={self.batch_union} "
            f"dualSsd={self.dual_ssd} kvCompression={self.kv_compression} "
            f"ssdTier={self.ssd_tier.name}"
        )


# ---------------------------------------------------------------------------
# SamplingOptions
# ---------------------------------------------------------------------------

@dataclass
class SamplingOptions:
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    penalty_repeat: float = 1.0
    penalty_frequency: float = 0.0
    penalty_presence: float = 0.0
    penalty_last_n: int = 64
    seed: int = 0

    def to_native(self) -> _nat.Sampling:
        s = _nat.Sampling()
        s.temperature = self.temperature
        s.top_k = self.top_k
        s.top_p = self.top_p
        s.penalty_repeat = self.penalty_repeat
        s.penalty_freq = self.penalty_frequency
        s.penalty_present = self.penalty_presence
        s.penalty_last_n = self.penalty_last_n
        s.seed = self.seed
        return s

    @staticmethod
    def from_native(s: _nat.Sampling) -> SamplingOptions:
        return SamplingOptions(
            temperature=s.temperature,
            top_k=s.top_k,
            top_p=s.top_p,
            penalty_repeat=s.penalty_repeat,
            penalty_frequency=s.penalty_freq,
            penalty_presence=s.penalty_present,
            penalty_last_n=s.penalty_last_n,
            seed=s.seed,
        )


# ---------------------------------------------------------------------------
# GenerateOptions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GenerateOptions:
    max_tokens: int = 512
    stop_sequences: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# VisionConfig (read-only, from native)
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class VisionConfig:
    embedding_dim: int
    patch_size: int
    image_size: int
    num_heads: int
    num_layers: int
    projection_dim: int
    has_encoder: bool

    @staticmethod
    def from_native(c: _nat.VisionConfig) -> VisionConfig:
        return VisionConfig(
            embedding_dim=c.embedding_dim,
            patch_size=c.patch_size,
            image_size=c.image_size,
            num_heads=c.num_heads,
            num_layers=c.num_layers,
            projection_dim=c.projection_dim,
            has_encoder=c.has_encoder != 0,
        )


# ---------------------------------------------------------------------------
# ToolCalling types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters_json_schema: str


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments_json: str
