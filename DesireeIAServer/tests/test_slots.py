# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Tests for router mode, parallel replicas, and idle unload in slots.py."""

from __future__ import annotations

import threading
import time

import pytest

from desireeiaserver.config import Settings
from desireeiaserver.slots import ModelGroup, SlotBusy, SlotManager, SlotState
from desireeiaserver.token_stats import TokenTracker


def _write_gguf(path, name: str) -> None:
    (path / name).write_bytes(b"GGUF" + b"\x00" * 32)


# -- router mode: discovery -----------------------------------------------

def test_single_model_mode_registers_one_group(tmp_path, fake_desireeia):
    settings = Settings(model="some/path/model.gguf", data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    assert set(manager.groups) == {"model"}
    assert len(manager.slots) == 1


def test_router_mode_scans_models_folder(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "a.gguf")
    _write_gguf(models_dir, "b.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=5)
    manager = SlotManager(settings)
    assert set(manager.groups) == {"a", "b"}


def test_router_mode_skips_invalid_checkpoints(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "good.gguf")
    (models_dir / "bad.gguf").write_bytes(b"NOTG" + b"\x00" * 10)
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    assert set(manager.groups) == {"good"}


def test_models_max_caps_registration(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for n in ("a", "b", "c"):
        _write_gguf(models_dir, f"{n}.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=2)
    manager = SlotManager(settings)
    assert len(manager.groups) == 2


def test_empty_models_folder_yields_no_groups(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    assert manager.groups == {}


# -- resolve() -------------------------------------------------------------

def test_resolve_by_name(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "a.gguf")
    _write_gguf(models_dir, "b.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=5)
    manager = SlotManager(settings)
    assert manager.resolve("a").model_id == "a"
    assert manager.resolve("b").model_id == "b"


def test_resolve_none_with_single_model_ok(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    assert manager.resolve(None).model_id == "x"


def test_resolve_none_ambiguous_in_router_mode_raises(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "a.gguf")
    _write_gguf(models_dir, "b.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=5)
    manager = SlotManager(settings)
    from desireeiaserver.api.errors import NotFoundError
    with pytest.raises(NotFoundError) as exc_info:
        manager.resolve(None)
    assert exc_info.value.status == 400


def test_resolve_unknown_name_raises_404(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    from desireeiaserver.api.errors import NotFoundError
    with pytest.raises(NotFoundError) as exc_info:
        manager.resolve("nope")
    assert exc_info.value.status == 404


def test_resolve_empty_manager_raises_404(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    from desireeiaserver.api.errors import NotFoundError
    with pytest.raises(NotFoundError):
        manager.resolve(None)


# -- parallel replicas -------------------------------------------------------

def test_parallel_creates_n_replicas(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", parallel=3)
    manager = SlotManager(settings)
    group = manager.groups["x"]
    assert len(group.replicas) == 3
    assert len(manager.slots) == 3


def test_pick_replica_prefers_idle_over_busy(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", parallel=2)
    group = ModelGroup(settings, "x.gguf", model_id="x")
    group.replicas[0]._busy = True
    picked = group.pick_replica()
    assert picked is group.replicas[1]


def test_all_replicas_busy_raises_slot_busy(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", parallel=2)
    manager = SlotManager(settings)
    slot = manager.resolve("x")
    slot.ensure_loaded()
    # Manually mark busy, bypassing run(), to simulate "already generating"
    # without needing a real slow model.
    slot._busy = True
    try:
        gen = slot.prediction([("user", "hi")], _params()).run(threading.Event(), TokenTracker())
        with pytest.raises(SlotBusy):
            next(gen)
    finally:
        slot._busy = False


def test_second_replica_serves_while_first_is_busy(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", parallel=2)
    manager = SlotManager(settings)
    r0, r1 = manager.groups["x"].replicas
    r0.ensure_loaded()
    r0._busy = True
    try:
        picked = manager.resolve("x")
        assert picked is r1  # round-robin skips the busy one
        gen = picked.prediction([("user", "hi")], _params()).run(threading.Event(), TokenTracker())
        # Should not raise SlotBusy: r1 is free even though r0 isn't.
        list(gen)
    finally:
        r0._busy = False


# -- idle unload -------------------------------------------------------------

def test_sweep_idle_noop_when_disabled(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", sleep_idle_seconds=0)
    manager = SlotManager(settings)
    manager.resolve(None).ensure_loaded()
    assert manager.sweep_idle() == []
    assert manager.resolve(None).state == SlotState.LOADED


def test_sweep_idle_unloads_past_threshold(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", sleep_idle_seconds=1)
    manager = SlotManager(settings)
    slot = manager.resolve(None)
    slot.ensure_loaded()
    assert slot.state == SlotState.LOADED
    time.sleep(1.1)
    touched = manager.sweep_idle()
    assert touched == ["x"]
    assert slot.state == SlotState.UNLOADED


def test_sweep_idle_leaves_busy_replica_alone(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", sleep_idle_seconds=1)
    manager = SlotManager(settings)
    slot = manager.resolve(None)
    slot.ensure_loaded()
    slot._busy = True
    try:
        time.sleep(1.1)
        touched = manager.sweep_idle()
        assert touched == []
        assert slot.state == SlotState.LOADED
    finally:
        slot._busy = False


def test_touch_resets_idle_clock(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data", sleep_idle_seconds=1)
    manager = SlotManager(settings)
    slot = manager.resolve(None)
    slot.ensure_loaded()
    time.sleep(0.6)
    slot.touch()
    time.sleep(0.6)
    # 0.6s since the touch, under the 1s threshold: still not swept.
    assert manager.sweep_idle() == []
    assert slot.state == SlotState.LOADED


# -- refresh_from_folder -----------------------------------------------------

def test_refresh_from_folder_registers_new_checkpoint(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "a.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=5)
    manager = SlotManager(settings)
    assert set(manager.groups) == {"a"}

    _write_gguf(models_dir, "b.gguf")
    manager.refresh_from_folder()
    assert set(manager.groups) == {"a", "b"}


def test_refresh_from_folder_drops_deleted_checkpoint(tmp_path, fake_desireeia):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    _write_gguf(models_dir, "a.gguf")
    settings = Settings(models_dir=models_dir, data_dir=tmp_path / "data", models_max=5)
    manager = SlotManager(settings)
    assert set(manager.groups) == {"a"}

    (models_dir / "a.gguf").unlink()
    manager.refresh_from_folder()
    assert manager.groups == {}


def test_refresh_from_folder_noop_in_single_model_mode(tmp_path, fake_desireeia):
    settings = Settings(model="x.gguf", data_dir=tmp_path / "data")
    manager = SlotManager(settings)
    manager.refresh_from_folder()  # must not raise or change anything
    assert set(manager.groups) == {"x"}


def _params():
    from desireeiaserver.sampling import resolve
    return resolve({"stream": False})
