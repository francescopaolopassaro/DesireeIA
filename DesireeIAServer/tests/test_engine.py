"""Engine adapter mapping: settings -> plan and sampling options."""

from __future__ import annotations

import pytest

from desireeiaserver.config import Settings
from desireeiaserver.engine import (
    build_plan,
    engine_version,
    hardware_probe,
    sampling_options,
)


def test_engine_probes_with_fake_module(fake_desireeia):
    assert engine_version() == "9.9.9-fake"
    hw = hardware_probe()
    assert hw["cpu_threads"] == 8
    assert hw["cuda_device_count"] == 0


def test_build_plan_forwards_threads_and_ram(fake_desireeia):
    settings = Settings(threads=4, ram_budget_mb=4096)
    plan = build_plan("model.gguf", settings)
    assert plan.thread_count == 4
    assert plan.ram_budget_mb == 4096


def test_build_plan_backend_override(fake_desireeia):
    settings = Settings(backend="cuda")
    plan = build_plan("model.gguf", settings)
    assert plan.backend == fake_desireeia.InferenceBackend.CUDA


def test_sampling_options_mapping(fake_desireeia):
    settings = Settings(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repeat_penalty=1.2,
        repeat_last_n=32,
        frequency_penalty=0.1,
        presence_penalty=0.2,
        seed=42,
    )
    opts = sampling_options(settings)
    assert opts.temperature == 0.8
    assert opts.top_k == 50
    assert opts.top_p == 0.9
    assert opts.penalty_repeat == 1.2
    assert opts.penalty_last_n == 32
    assert opts.penalty_frequency == 0.1
    assert opts.penalty_presence == 0.2
    assert opts.seed == 42


def test_seed_negative_means_random(fake_desireeia):
    opts = sampling_options(Settings(seed=-1))
    assert opts.seed == 0


def test_unknown_backend_raises(fake_desireeia):
    with pytest.raises(ValueError):
        build_plan("model.gguf", Settings(backend="quantum"))