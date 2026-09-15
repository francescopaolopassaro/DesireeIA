# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Unit tests for models_store.py, independent of FastAPI/HTTP."""

from __future__ import annotations

import pytest

from desireeiaserver import models_store as store


def test_validate_filename_rejects_separators():
    for bad in ("../evil.gguf", "a/b.gguf", "a\\b.gguf", "", "..", "/etc/passwd"):
        with pytest.raises(store.InvalidFilename):
            store.validate_filename(bad)


def test_validate_filename_rejects_unsupported_extension():
    with pytest.raises(store.InvalidFilename):
        store.validate_filename("model.bin")


def test_validate_filename_accepts_supported():
    assert store.validate_filename("model.gguf") == "model.gguf"
    assert store.validate_filename("model.safetensors") == "model.safetensors"


def test_list_models_empty_dir(tmp_path):
    assert list(store.list_models(tmp_path / "does-not-exist")) == []
    (tmp_path / "models").mkdir()
    assert list(store.list_models(tmp_path / "models")) == []


def test_list_models_reports_valid_and_invalid(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    (d / "good.gguf").write_bytes(b"GGUF" + b"\x00" * 20)
    (d / "bad.gguf").write_bytes(b"NOTG" + b"\x00" * 20)
    (d / "not-a-checkpoint.txt").write_bytes(b"ignored")
    models = {m.name: m for m in store.list_models(d)}
    assert set(models) == {"good.gguf", "bad.gguf"}
    assert models["good.gguf"].valid is True
    assert models["bad.gguf"].valid is False


def test_delete_model_removes_file(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    f = d / "x.gguf"
    f.write_bytes(b"GGUF")
    store.delete_model(d, "x.gguf")
    assert not f.exists()


def test_delete_model_missing_raises(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        store.delete_model(d, "nope.gguf")


def test_delete_model_rejects_traversal(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    with pytest.raises(store.InvalidFilename):
        store.delete_model(d, "../escape.gguf")


class TestChunkedUpload:
    def test_valid_gguf(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        up = store.ChunkedUpload(d, "m.gguf", max_bytes=1_000_000)
        up.write(b"GG")
        up.write(b"UF" + b"\x00" * 50)
        result = up.finish()
        assert result.name == "m.gguf"
        assert result.valid is True
        assert (d / "m.gguf").exists()
        # no .partial files left behind
        assert list(d.glob(".upload-*.partial")) == []

    def test_valid_safetensors(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        import struct
        header = b'{"a":1}'
        blob = struct.pack("<Q", len(header)) + header + b"\x00" * 10
        up = store.ChunkedUpload(d, "m.safetensors", max_bytes=1_000_000)
        up.write(blob)
        result = up.finish()
        assert result.valid is True
        assert result.format == "safetensors"

    def test_bad_magic_raises_and_cleans_up(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        up = store.ChunkedUpload(d, "m.gguf", max_bytes=1_000_000)
        with pytest.raises(store.BadMagic):
            up.write(b"NOTGGUF!!")
        up.abort()
        assert list(d.iterdir()) == []

    def test_too_large_raises(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        up = store.ChunkedUpload(d, "m.gguf", max_bytes=10)
        with pytest.raises(store.FileTooLarge):
            up.write(b"GGUF" + b"\x00" * 20)
        up.abort()
        assert list(d.iterdir()) == []

    def test_too_short_for_magic_raises_on_finish(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        up = store.ChunkedUpload(d, "m.gguf", max_bytes=1_000_000)
        up.write(b"GG")
        with pytest.raises(store.BadMagic):
            up.finish()
        assert list(d.iterdir()) == []

    def test_invalid_filename_raises_immediately(self, tmp_path):
        d = tmp_path / "models"
        d.mkdir()
        with pytest.raises(store.InvalidFilename):
            store.ChunkedUpload(d, "../evil.gguf", max_bytes=1_000_000)
