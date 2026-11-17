"""The context follows the VIDEO MEMORY of the GPU, not just the RAM (mirror of the C# VideoMemoryFitTests).

MEASURED on a 6 GB laptop card with a 4B model (3.5 GB of weights, 78 KB of KV cache per token): a 9174-token prefill
takes 15.7 s at a context of 16384; at 40960 the KV cache and the weights (6.7 GB) overflow the card, the first layer
alone takes 80 s and the GPU stays at 100%.
"""

import dataclasses

from desireeia.autoconfig import (
    MIN_CONTEXT_SIZE,
    TARGET_CONTEXT_SIZE,
    ModelTraits,
    compute_configuration,
    max_context_by_video_memory,
)
from desireeia.enums import InferenceBackend
from desireeia.types import ExecutionPlan, HardwareProfile

MB = 1024 * 1024


def _hw(total_mb, free_mb, ram_free_mb=24_000):
    gpu = {"index": 0, "name": "Test GPU", "total_bytes": total_mb * MB, "free_bytes": free_mb * MB,
           "compute_capability": (8, 9), "multiprocessors": 24}
    return HardwareProfile(16, True, True, False, False, 1, 32_000, ram_free_mb, 0, 0, False, False, gpus=(gpu,))


def _cuda(kv=True):
    return ExecutionPlan(backend=InferenceBackend.CUDA, thread_count=16, ram_budget_mb=24_000, kv_compression=kv)


def _spark26(size=3_543_441_312):
    # Spark-X2.6-4B-Q4_K_M: 36 layers, 8 KV heads, key+value 128 each -> 78,336 bytes of Q8_0 KV per token.
    return ModelTraits("spark2_6", 1_048_576, 36, 8, 128, 128, 0, size, None, None, None)


def test_kv_bytes_per_token_matches_the_engine():
    assert _spark26().kv_bytes_per_token_for(True) == 78_336


def test_six_gigabyte_card_fits_the_target_but_not_much_more():
    c = compute_configuration(_hw(6141, 5800), _cuda(), _spark26())
    assert c.context_size == TARGET_CONTEXT_SIZE
    assert 16384 <= c.max_context_size < 40960
    assert any("video memory" in n for n in c.notes)


def test_the_real_card_5080_mb_free_runs_the_target_context():
    # What the driver reported with the desktop running: 5080 MB free of 6140. MEASURED: 16384 tokens ran at ~2100
    # tokens/s on it, so the fit must not refuse them (a 1 GB margin chose 8192 and starved the prompt).
    c = compute_configuration(_hw(6140, 5080), _cuda(), _spark26())
    assert c.context_size == TARGET_CONTEXT_SIZE
    assert 16384 <= c.max_context_size <= 18432


def test_big_card_allows_a_larger_context_when_asked_for():
    c = compute_configuration(_hw(24_576, 23_000), _cuda(), _spark26(), target_context=65_536)
    assert c.context_size == 65_536
    assert c.max_context_size >= 65_536


def test_big_card_keeps_the_default_target_but_reports_how_far_it_can_grow():
    c = compute_configuration(_hw(24_576, 23_000), _cuda(), _spark26())
    assert c.context_size == TARGET_CONTEXT_SIZE
    assert c.max_context_size > TARGET_CONTEXT_SIZE


def test_card_too_small_for_the_weights_falls_back_to_the_smallest_context():
    c = compute_configuration(_hw(4096, 3500), _cuda(), _spark26())
    assert c.context_size == MIN_CONTEXT_SIZE
    assert any("weights alone" in n for n in c.notes)


def test_without_a_gpu_probe_behaves_as_before():
    hw = dataclasses.replace(_hw(6141, 5800), gpus=())
    c = compute_configuration(hw, _cuda(), _spark26())
    assert c.context_size == TARGET_CONTEXT_SIZE
    assert not any("video memory" in n for n in c.notes)


def test_a_cpu_plan_ignores_the_video_memory():
    plan = ExecutionPlan(backend=InferenceBackend.CPU, thread_count=16, ram_budget_mb=24_000)
    assert max_context_by_video_memory(_hw(6141, 5800), plan, _spark26()) == (None, None)


def test_the_most_free_gpu_is_the_one_counted():
    small = {"index": 0, "name": "Small", "total_bytes": 6141 * MB, "free_bytes": 1000 * MB,
             "compute_capability": (8, 9), "multiprocessors": 24}
    big = {"index": 1, "name": "Big", "total_bytes": 24_576 * MB, "free_bytes": 23_000 * MB,
           "compute_capability": (8, 9), "multiprocessors": 80}
    hw = dataclasses.replace(_hw(6141, 1000), gpus=(small, big))
    fit, note = max_context_by_video_memory(hw, _cuda(), _spark26())
    assert fit > 100_000 and "Big" in note
