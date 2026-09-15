"""VisionImageWrapper — manages native image data lifecycle.

Mirrors C# VisionImageWrapper.cs.
"""

from __future__ import annotations

import base64
from ctypes import c_char, c_void_p, cast, POINTER

from . import _native as _nat
from .enums import Error


class VisionImageWrapper:
    """Wrapper around a native VisionImage. Frees native memory on close."""

    __slots__ = ("_image", "_closed")

    def __init__(self, image: _nat.VisionImage) -> None:
        self._image = image
        self._closed = False

    # -- properties ----------------------------------------------------------

    @property
    def width(self) -> int:
        return self._image.width

    @property
    def height(self) -> int:
        return self._image.height

    @property
    def channels(self) -> int:
        return self._image.channels

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    @property
    def byte_count(self) -> int:
        return self.pixel_count * self.channels

    @property
    def mime_type(self) -> str:
        if self.channels == 1:
            return "image/grayscale"
        if self.channels == 3:
            return "image/rgb"
        if self.channels == 4:
            return "image/rgba"
        return "application/octet-stream"

    # -- data access ---------------------------------------------------------

    def to_bytes(self) -> bytes:
        if self._closed:
            raise ValueError("Image already closed")
        n = self.byte_count
        if self._image.data is None or n == 0:
            return b""
        buf = (c_char * n)()
        _nat.ctypes.memmove(buf, self._image.data, n)
        return bytes(buf)

    def to_base64(self) -> str:
        return base64.b64encode(self.to_bytes()).decode("ascii")

    def to_message_content(self, text_content: str | None = None) -> str:
        b64 = self.to_base64()
        content = f"[image:{self.width}x{self.height}@{self.channels}ch:{b64}]"
        if text_content:
            content += "\n" + text_content
        return content

    def get_native(self) -> _nat.VisionImage:
        if self._closed:
            raise ValueError("Image already closed")
        return self._image

    # -- context manager / cleanup -------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._image.data is not None:
            _nat.get_lib().desireeia_free_image(_nat.ctypes.byref(self._image))
            self._image.data = None

    def __enter__(self) -> VisionImageWrapper:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"{self.width}x{self.height}@{self.channels}ch"
        return f"<VisionImageWrapper {state}>"
