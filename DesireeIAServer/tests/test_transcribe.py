# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Tests for POST /transcribe (voice input) and the MaxBodySizeMiddleware
exemption it and /models/upload need to accept a body larger than the
general per-request cap.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from desireeiaserver import transcription
from desireeiaserver.app import create_app
from desireeiaserver.config import Settings


def _settings(tmp_path, **overrides) -> Settings:
    defaults = dict(model="my-model.gguf", temperature=0.0,
                     models_dir=tmp_path / "models", data_dir=tmp_path / "data")
    defaults.update(overrides)
    return Settings(**defaults)


def _client(tmp_path, **overrides):
    app = create_app(_settings(tmp_path, **overrides))
    return TestClient(app, raise_server_exceptions=False)


def test_transcribe_disabled_by_default(tmp_path, fake_desireeia):
    with _client(tmp_path) as c:
        r = c.post("/transcribe", files={"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")})
        assert r.status_code == 400
        assert r.json()["error"]["type"] == "not_supported_error"


def test_transcribe_enabled_but_dependency_missing(tmp_path, fake_desireeia, monkeypatch):
    # faster-whisper is a core dependency (always installed) but this path
    # - a broken/partial install - must still fail cleanly rather than
    # crash, so it's exercised here by forcing available() to report False
    # regardless of what's actually importable in this environment.
    import desireeiaserver.transcription as transcription
    monkeypatch.setattr(transcription, "available", lambda: False)
    with _client(tmp_path, enable_whisper=True) as c:
        r = c.post("/transcribe", files={"audio": ("clip.webm", b"fake-audio-bytes", "audio/webm")})
        assert r.status_code == 400
        assert "not installed" in r.json()["error"]["message"]


def test_transcribe_rejects_empty_audio(tmp_path, fake_desireeia, monkeypatch):
    import desireeiaserver.transcription as transcription
    monkeypatch.setattr(transcription, "available", lambda: True)
    with _client(tmp_path, enable_whisper=True) as c:
        r = c.post("/transcribe", files={"audio": ("clip.webm", b"", "audio/webm")})
        assert r.status_code == 400
        assert r.json()["error"]["param"] == "audio"


def test_transcribe_exempt_from_general_body_size_cap(tmp_path, fake_desireeia):
    # A tiny max_request_mb would reject almost anything under the general
    # middleware - /transcribe must bypass it (it enforces its own,
    # separate MAX_AUDIO_BYTES instead) or a real voice note would 413
    # before ever reaching the disabled/missing-dependency checks above.
    big_audio = b"x" * (200 * 1024)  # bigger than max_request_mb below
    with _client(tmp_path, max_request_mb=0.05) as c:  # ~51KB cap
        r = c.post("/transcribe", files={"audio": ("clip.webm", big_audio, "audio/webm")})
        assert r.status_code != 413
        assert r.status_code == 400  # falls through to "disabled" - proves the body was accepted


def test_models_upload_exempt_from_general_body_size_cap(tmp_path, fake_desireeia):
    big_payload = b"GGUF" + b"\x00" * (200 * 1024)
    with _client(tmp_path, max_request_mb=0.05, models_dir=tmp_path / "models", models_max=5) as c:
        r = c.post("/models/upload?filename=big.gguf", content=big_payload)
        assert r.status_code != 413


def test_other_endpoints_still_capped(tmp_path, fake_desireeia):
    with _client(tmp_path, max_request_mb=0.05) as c:
        huge = "x" * (200 * 1024)
        r = c.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": huge}]})
        assert r.status_code == 413


# -- transcription.py: VAD-empty fallback ---------------------------------

@dataclass
class _FakeSegment:
    text: str


@dataclass
class _FakeInfo:
    language: str = "en"


class _FakeModel:
    """vad_filter=True finds nothing (as real VAD would for quiet/close
    audio it isn't confident about); vad_filter=False (the fallback) finds
    the actual speech - exercises transcription.py's retry path without
    needing a real audio file or a slow real model load."""

    def transcribe(self, audio_path, language=None, vad_filter=True):
        if vad_filter:
            return [], _FakeInfo()
        return [_FakeSegment("hello there")], _FakeInfo()


def test_transcribe_retries_without_vad_when_vad_finds_nothing(monkeypatch):
    monkeypatch.setattr(transcription, "_get_model", lambda model_size: _FakeModel())
    result = transcription.transcribe("fake-path.wav")
    assert result["text"] == "hello there"


class _AlwaysEmptyModel:
    def transcribe(self, audio_path, language=None, vad_filter=True):
        return [], _FakeInfo()


def test_transcribe_returns_empty_when_genuinely_no_speech(monkeypatch):
    monkeypatch.setattr(transcription, "_get_model", lambda model_size: _AlwaysEmptyModel())
    result = transcription.transcribe("fake-path.wav")
    assert result["text"] == ""


class _HallucinatesDotsModel:
    """vad_filter=True does NOT come back empty - it comes back with
    Whisper's classic punctuation-hallucination on silence/noise, which is
    a non-empty string. Reproduces the real bug: `if text:` alone never
    catches this, since the string genuinely isn't empty."""

    def transcribe(self, audio_path, language=None, vad_filter=True):
        return [_FakeSegment("... ... ...")], _FakeInfo()


def test_transcribe_treats_dots_hallucination_as_no_speech(monkeypatch):
    monkeypatch.setattr(transcription, "_get_model", lambda model_size: _HallucinatesDotsModel())
    result = transcription.transcribe("fake-path.wav")
    assert result["text"] == ""


class _HallucinatesThenRealSpeechModel:
    """First (VAD) pass hallucinates dots; second (no-VAD) pass finds the
    actual speech - the fallback must still trigger even though the first
    pass's result was non-empty."""

    def __init__(self):
        self.calls = 0

    def transcribe(self, audio_path, language=None, vad_filter=True):
        self.calls += 1
        if vad_filter:
            return [_FakeSegment("...")], _FakeInfo()
        return [_FakeSegment("turn on the lights")], _FakeInfo()


def test_transcribe_retries_past_dots_hallucination_to_find_real_speech(monkeypatch):
    fake = _HallucinatesThenRealSpeechModel()
    monkeypatch.setattr(transcription, "_get_model", lambda model_size: fake)
    result = transcription.transcribe("fake-path.wav")
    assert result["text"] == "turn on the lights"
    assert fake.calls == 2


@pytest.mark.parametrize("junk", ["...", "..", "-", "—", "♪", "  ...  ", ""])
def test_hallucinated_silence_detector_catches_known_junk_patterns(junk):
    assert transcription._looks_like_hallucinated_silence(junk) is True


@pytest.mark.parametrize("real", ["hello", "... but wait", "3...2...1", "ciao come stai"])
def test_hallucinated_silence_detector_leaves_real_text_alone(real):
    assert transcription._looks_like_hallucinated_silence(real) is False
