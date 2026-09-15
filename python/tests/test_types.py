"""Unit tests for the data types and struct <-> python conversions."""

import pytest

from desireeia import (
    ExecutionPlan,
    HardwareProfile,
    InferenceBackend,
    ModelFormat,
    Quantization,
    SsdTierMode,
    SamplingOptions,
)
from desireeia import _native as _nat


class TestExecutionPlan:
    def test_defaults_match_csharp(self):
        plan = ExecutionPlan()
        assert plan.backend == InferenceBackend.UNCONFIGURED
        assert plan.format == ModelFormat.UNKNOWN
        assert plan.dense_quantization == Quantization.Q8_0
        assert plan.expert_quantization == Quantization.Q4K
        assert plan.expert_cache_size == 256
        assert plan.expert_prefetch is True
        assert plan.kv_compression is True
        assert plan.expert_pin_learning is True
        assert plan.prefetch_depth == 1
        assert plan.batch_union is True
        assert plan.dual_ssd is False
        assert plan.ssd_tier == SsdTierMode.AUTO

    def test_to_native_field_mapping(self):
        plan = ExecutionPlan(
            backend=InferenceBackend.CUDA,
            format=ModelFormat.GGUF,
            dense_quantization=Quantization.Q6K,
            expert_quantization=Quantization.Q4K,
            thread_count=4,
            ram_budget_mb=8192,
            expert_cache_size=128,
            expert_prefetch=False,
            kv_compression=False,
            expert_pin_learning=False,
            prefetch_depth=2,
            batch_union=False,
            dual_ssd=True,
            ssd_tier=SsdTierMode.OFF,
            ssd_tier_cache_mb=512,
        )
        n = plan.to_native()
        assert n.backend == int(InferenceBackend.CUDA)
        assert n.format == int(ModelFormat.GGUF)
        assert n.dense_quant == int(Quantization.Q6K)
        assert n.expert_quant == int(Quantization.Q4K)
        assert n.n_threads == 4
        assert n.ram_budget_mb == 8192
        assert n.expert_cache_count == 128
        assert n.expert_prefetch_enabled == 0
        assert n.kv_compression_enabled == 0
        assert n.expert_pin_enabled == 0
        assert n.expert_prefetch_depth == 2
        assert n.batch_union_enabled == 0
        assert n.dual_ssd_enabled == 1
        assert n.ssd_tier_mode == int(SsdTierMode.OFF)
        assert n.ssd_tier_cache_mb == 512

    def test_from_native_roundtrip(self):
        plan = ExecutionPlan(thread_count=3, dual_ssd=True, ssd_tier=SsdTierMode.ALWAYS)
        native = plan.to_native()
        back = ExecutionPlan.from_native(native)
        assert back == plan

    def test_string_repr(self):
        s = str(ExecutionPlan())
        assert "backend=" in s and "threads=" in s


class TestHardwareProfile:
    def test_from_native(self):
        hw = _nat.HwInfo()
        hw.cpu_threads = 8
        hw.cpu_has_avx2 = 1
        hw.cuda_device_count = 2
        hw.ram_total_mb = 32768
        hp = HardwareProfile.from_native(hw)
        assert hp.cpu_threads == 8
        assert hp.avx2 is True
        assert hp.cuda_device_count == 2
        assert hp.ram_total_mb == 32768

    def test_from_native_false_flags(self):
        hw = _nat.HwInfo()
        hp = HardwareProfile.from_native(hw)
        assert hp.avx is False
        assert hp.metal is False
        assert hp.vulkan is False


class TestSamplingOptions:
    def test_to_from_native_roundtrip(self):
        opts = SamplingOptions(
            temperature=0.8,
            top_k=50,
            top_p=0.9,
            penalty_repeat=1.1,
            penalty_frequency=0.5,
            penalty_presence=0.2,
            penalty_last_n=128,
            seed=42,
        )
        native = opts.to_native()
        back = SamplingOptions.from_native(native)
        assert back.temperature == pytest.approx(0.8)
        assert back.top_k == 50
        assert back.top_p == pytest.approx(0.9)
        assert back.penalty_repeat == pytest.approx(1.1)
        assert back.penalty_frequency == pytest.approx(0.5)
        assert back.penalty_presence == pytest.approx(0.2)
        assert back.penalty_last_n == 128
        assert back.seed == 42

    def test_defaults_are_greedy(self):
        opts = SamplingOptions()
        assert opts.temperature == 0.0
        assert opts.top_k == 40
        assert opts.top_p == pytest.approx(0.95)