# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""POST /transcribe: speech-to-text for the chat composer's microphone
button, via faster-whisper (see transcription.py). Off by default - see
Settings.enable_whisper / --enable-whisper - since it pulls in a separate
optional dependency and a model download distinct from the DesireeIA
engine itself.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import anyio
from fastapi import APIRouter, Request, UploadFile

from .. import transcription
from ..config import Settings
from .errors import InvalidRequestError, NotSupportedError

router = APIRouter(tags=["transcribe"])

MAX_AUDIO_BYTES = 25 * 1024 * 1024  # a voice note, not a podcast upload


@router.post("/transcribe")
async def transcribe(request: Request, audio: UploadFile):
    settings: Settings = request.app.state.settings
    if not settings.enable_whisper:
        raise NotSupportedError(
            "voice input is disabled on this server (start it with --enable-whisper to turn it on)"
        )
    if not transcription.available():
        raise NotSupportedError(
            "faster-whisper is not installed - install the [audio] extra "
            "(pip install desireeia-server[audio]) to enable voice input"
        )

    data = await audio.read()
    if len(data) > MAX_AUDIO_BYTES:
        raise InvalidRequestError(f"audio exceeds {MAX_AUDIO_BYTES} bytes", param="audio")
    if not data:
        raise InvalidRequestError("empty audio upload", param="audio")

    suffix = Path(audio.filename or "").suffix or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        result = await anyio.to_thread.run_sync(
            transcription.transcribe, tmp_path, settings.whisper_model
        )
    except transcription.WhisperUnavailable as exc:
        raise NotSupportedError(str(exc)) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return result
