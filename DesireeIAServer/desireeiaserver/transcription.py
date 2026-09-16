# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Speech-to-text backing POST /transcribe (voice input in the web UI),
via faster-whisper. Optional: the server runs fully without this installed
- only the transcribe endpoint needs it, and it reports a clear error
instead of failing to start when the [audio] extra isn't present.

The model is loaded lazily (first real request, not at import time) and
cached in a module-level singleton, same rationale as engine.py's model
cache: a speech model is a multi-hundred-MB download/load, not something to
pay for on every call, but also not something this project wants paid for
on every server start when the feature might not be used that session.
"""

from __future__ import annotations

import re
import threading
from typing import Optional

_model = None
_model_lock = threading.Lock()
_loaded_size: Optional[str] = None

# Whisper's classic hallucination on silence/near-silence/noise input is
# NOT an empty string - it's repeated punctuation ("... ... ..." being the
# most common shape, but "..", "-", "♪" and similar also occur). A check
# for `if text:` never catches this, since the string is non-empty - it
# has to be recognized by content instead.
_NO_REAL_SPEECH_RE = re.compile(r"^[\s.\-–—…♪]*$")


def _looks_like_hallucinated_silence(text: str) -> bool:
    return bool(_NO_REAL_SPEECH_RE.match(text))


class WhisperUnavailable(Exception):
    """faster-whisper is not installed - see the [audio] extra."""


def available() -> bool:
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        return False
    return True


def _get_model(model_size: str):
    global _model, _loaded_size
    if _model is not None and _loaded_size == model_size:
        return _model
    with _model_lock:
        if _model is not None and _loaded_size == model_size:
            return _model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise WhisperUnavailable(
                "faster-whisper is not installed - install the [audio] extra "
                "(pip install desireeia-server[audio]) to enable voice input"
            ) from exc
        # device="auto" picks CUDA when a GPU is detected, but ctranslate2
        # (faster-whisper's backend) needs its OWN cuBLAS/cuDNN DLLs on
        # PATH, separate from whatever the DesireeIA CUDA backend links
        # against - on a machine that has the latter but not the former,
        # "auto" constructs the model fine and then blows up with
        # "Library cublas64_12.dll is not found" on the first real
        # transcription. A short voice note transcribes plenty fast on
        # CPU with the base/small model sizes this exposes, so CPU is the
        # reliable default rather than a GPU path this project can't
        # guarantee the DLLs for.
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")
        _loaded_size = model_size
        return _model


def transcribe(audio_path: str, model_size: str = "base", language: Optional[str] = None) -> dict:
    model = _get_model(model_size)
    # vad_filter=True runs Silero VAD first and only feeds Whisper the
    # segments it thinks actually contain speech. Without it, a short or
    # quiet browser recording (background noise, a beat of silence before
    # speech starts) is a well-documented trigger for Whisper hallucinating
    # repeated punctuation ("... ... ...") instead of transcribing nothing.
    segments, info = model.transcribe(audio_path, language=language, vad_filter=True)
    text = "".join(segment.text for segment in segments).strip()
    if text and not _looks_like_hallucinated_silence(text):
        return {"text": text, "language": info.language}

    # Either VAD found nothing it was confident was speech, or what it DID
    # feed Whisper came back as the punctuation-hallucination pattern -
    # both mean "no usable speech was found", not "here is the transcript".
    # A real quiet/close-talk recording can trip an over-eager VAD
    # threshold, so retry once without it rather than giving up
    # immediately: worst case this second pass hallucinates the same
    # pattern again and gets caught by the same check below.
    segments, info = model.transcribe(audio_path, language=language, vad_filter=False)
    text = "".join(segment.text for segment in segments).strip()
    if _looks_like_hallucinated_silence(text):
        text = ""
    return {"text": text, "language": info.language}
