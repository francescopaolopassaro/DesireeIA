"""The quantized KV cache keeps only a window of positions in the local-attention layers (mirror of WindowedKvFitTests)."""
from desireeia.autoconfig import ModelTraits

MB = 1024 * 1024


def _spark(locals_=27):
    return ModelTraits("spark2_5", 1_048_576, 36, 4, 256, 256, 512, 3_543_441_312, None, None, None, local_layer_count=locals_)


def test_only_the_global_layers_grow_with_the_context():
    m = _spark()
    whole = m.kv_bytes_per_token_for(True)
    assert whole == 78_336
    per = whole // 36
    fixed = 27 * 1599 * per
    assert m.kv_bytes_for(True, 16384) == fixed + 9 * 16384 * per
    assert m.kv_bytes_for(True, 65536) == fixed + 9 * 65536 * per


def test_float_cache_and_models_without_local_layers_are_unchanged():
    assert _spark().kv_bytes_for(False, 4096) == _spark().kv_bytes_per_token_for(False) * 4096
    m = _spark(0)
    assert m.kv_bytes_for(True, 10_000) == m.kv_bytes_per_token_for(True) * 10_000


def test_max_positions_inverts_the_size():
    m = _spark()
    for budget_mb in (400, 900, 1500, 6000):
        p = m.max_positions_for(True, budget_mb * MB)
        assert m.kv_bytes_for(True, p) <= budget_mb * MB
        assert m.kv_bytes_for(True, p + 64) > budget_mb * MB
    assert 70_000 <= m.max_positions_for(True, 1500 * MB) <= 80_000


def test_the_default_target_follows_the_cache_layout():
    from desireeia.autoconfig import TARGET_CONTEXT_SIZE, WINDOWED_TARGET_CONTEXT_SIZE, target_context_for
    from desireeia.enums import InferenceBackend
    from desireeia.types import ExecutionPlan
    plan = ExecutionPlan(backend=InferenceBackend.CUDA, kv_compression=True)
    assert target_context_for(_spark(), plan) == WINDOWED_TARGET_CONTEXT_SIZE
    assert target_context_for(_spark(0), plan) == TARGET_CONTEXT_SIZE
    assert target_context_for(_spark(), ExecutionPlan(backend=InferenceBackend.CUDA, kv_compression=False)) == TARGET_CONTEXT_SIZE
