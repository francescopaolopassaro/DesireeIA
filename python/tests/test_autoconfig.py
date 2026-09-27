"""Auto-configuration: same rules and numbers as the C# AutoConfigTests."""

import os

import pytest

from desireeia.autoconfig import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MIN_CONTEXT_SIZE,
    TARGET_CONTEXT_SIZE,
    ModelTraits,
    compute_configuration,
)
from desireeia.enums import InferenceBackend
from desireeia.types import ExecutionPlan, HardwareProfile


def _hw(free_mb=11_000):
    return HardwareProfile(16, True, True, False, False, 1, 32_000, free_mb, 0, 0, False, False)


def _plan(ram_mb=11_000, kv=False):
    return ExecutionPlan(backend=InferenceBackend.CUDA, thread_count=16, ram_budget_mb=ram_mb,
                         kv_compression=kv)


def _spark(trained=1_048_576, size=2_600_224_352):
    # Numbers from a real Spark-X2.5-4B-Q4_K_M.gguf header.
    return ModelTraits("spark2_5", trained, 36, 4, 256, 256, 512, size, 1.0, None, 0.95)


def test_sampling_is_never_greedy():
    c = compute_configuration(_hw(), _plan(), _spark())
    assert c.sampling.temperature == pytest.approx(DEFAULT_TEMPERATURE)
    assert c.sampling.top_k == DEFAULT_TOP_K
    assert c.sampling.top_p == pytest.approx(DEFAULT_TOP_P)
    assert c.sampling.temperature > 0


def test_comfortable_machine_gets_target_context_and_full_replies():
    # Q8_0 KV cache, the engine default since 0.1.2.
    c = compute_configuration(_hw(), _plan(kv=True), _spark())
    assert c.context_size == TARGET_CONTEXT_SIZE
    assert c.max_tokens == DEFAULT_MAX_TOKENS


def test_context_never_exceeds_trained_length():
    c = compute_configuration(_hw(), _plan(), _spark(trained=4096))
    assert c.context_size == 4096
    assert c.max_tokens == 1024


def test_tight_memory_shrinks_context_but_not_below_minimum():
    c = compute_configuration(_hw(3_000), _plan(3_000), _spark())
    assert c.context_size == MIN_CONTEXT_SIZE
    assert any("memory" in n for n in c.notes)


def test_moderate_memory_context_is_bounded_and_aligned():
    c = compute_configuration(_hw(6_000), _plan(6_000), _spark())
    assert MIN_CONTEXT_SIZE <= c.context_size < TARGET_CONTEXT_SIZE
    assert c.context_size % 1024 == 0


def test_kv_bytes_per_token_matches_engine_storage():
    # 36 layers * 4 kv heads * (256 + 256) = 73,728 values per token
    assert _spark().kv_bytes_per_token == 294_912               # float32
    assert _spark().kv_bytes_per_token_for(True) == 78_336      # Q8_0: 34 bytes / 32 values


def test_quantized_cache_allows_more_context_on_the_same_memory():
    f32 = compute_configuration(_hw(6_000), _plan(6_000, kv=False), _spark())
    q8 = compute_configuration(_hw(6_000), _plan(6_000, kv=True), _spark())
    assert q8.context_size > f32.context_size


def test_model_traits_read_real_model_when_available():
    path = os.environ.get("DESIREEIA_TEST_MODEL_PATH")
    if not path or not os.path.exists(path):
        pytest.skip("DESIREEIA_TEST_MODEL_PATH not set")
    t = ModelTraits.read(path)
    assert t.architecture
    assert t.layer_count > 0
    assert t.kv_bytes_per_token > 0
