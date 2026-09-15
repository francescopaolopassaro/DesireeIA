"""LocalModel — primary user-facing class. Mirrors C# LocalModel.cs.

Wraps a native desireeia_ctx handle and exposes all inference operations.
Uses Python context manager protocol (with statement) instead of IDisposable.
"""

from __future__ import annotations

import ctypes
import threading
from ctypes import c_char, c_float, c_int32, c_uint32, c_uint64, c_void_p, POINTER, byref, cast
from typing import Callable, Generator, Iterator, List, Optional, Tuple

from . import _native as _nat
from .enums import Error, SpecialToken
from .generation import StopSequenceScanner
from .types import (
    ExecutionPlan,
    GenerateOptions,
    HardwareProfile,
    SamplingOptions,
    VisionConfig,
)
from .vision import VisionImageWrapper


class LocalModel:
    """Loaded model context. Implements context manager (close replaces Dispose).

    Usage::

        with LocalModel.load("model.gguf", plan) as model:
            tokens = model.tokenize("Hello")
            token = model.predict(tokens)
            print(model.token_piece(token))
    """

    __slots__ = ("_ctx", "_closed", "_model_path", "_plan", "_log_cb_ref", "_lock")

    def __init__(
        self,
        ctx: c_void_p,
        model_path: str,
        plan: ExecutionPlan,
        log_cb_ref: Optional[_nat.LOG_CB],
    ) -> None:
        self._ctx = ctx
        self._model_path = model_path
        self._plan = plan
        self._log_cb_ref = log_cb_ref  # prevent GC of the callback
        self._closed = False
        self._lock = threading.Lock()  # native ctx is NOT thread-safe

    # -- properties ----------------------------------------------------------

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def plan(self) -> ExecutionPlan:
        return self._plan

    # -- factory -------------------------------------------------------------

    @staticmethod
    def load(
        model_path: str,
        plan: ExecutionPlan,
        logger: Optional[Callable[[str], None]] = None,
    ) -> LocalModel:
        """Load a model from disk. Raises RuntimeError on failure."""
        lib = _nat.get_lib()
        cb_ref = None
        if logger is not None:
            def _cb(level: int, msg: c_char_p, user: c_void_p) -> None:
                if msg:
                    logger(msg.decode("utf-8", errors="replace"))
            cb_ref = _nat.LOG_CB(_cb)

        plan_native = plan.to_native()
        ctx = c_void_p()

        err = lib.desireeia_create(
            model_path.encode("utf-8"),
            byref(plan_native),
            cb_ref,
            None,
            byref(ctx),
        )
        if err != Error.OK:
            raise RuntimeError(f"Model load failed: {Error(err).name}")

        return LocalModel(ctx, model_path, plan, cb_ref)

    # -- guard ---------------------------------------------------------------

    def _check(self) -> None:
        if self._closed:
            raise ValueError("Model is closed")

    # -- core inference ------------------------------------------------------

    def predict(self, tokens: List[int]) -> int:
        """Feed tokens and get the next predicted token id."""
        self._check()
        arr = (c_int32 * len(tokens))(*tokens)
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_predict(
                self._ctx, arr, len(tokens), byref(out)
            )
        if err != Error.OK:
            raise RuntimeError(f"Predict failed: {Error(err).name}")
        return out.value

    def next_token(self) -> int:
        """Get the next token autoregressively."""
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_next_token(self._ctx, byref(out))
        if err != Error.OK:
            raise RuntimeError(f"Next token failed: {Error(err).name}")
        return out.value

    def context_size(self) -> int:
        """Maximum context length of the loaded model."""
        self._check()
        return _nat.get_lib().desireeia_context_size(self._ctx)

    # -- tokenizer -----------------------------------------------------------

    def tokenize(self, text: str, add_bos: bool = True) -> Optional[List[int]]:
        """Tokenize text. Returns None if no tokenizer is available."""
        self._check()
        lib = _nat.get_lib()
        text_bytes = text.encode("utf-8")
        count = c_uint64()

        err = lib.desireeia_tokenize(
            self._ctx, text_bytes, 1 if add_bos else 0,
            None, 0, byref(count)
        )
        if err == Error.NOT_SUPPORTED:
            return None
        if err != Error.OK:
            raise RuntimeError(f"Tokenize failed: {Error(err).name}")

        n = count.value
        if n == 0:
            return []

        ids = (c_int32 * n)()
        with self._lock:
            err = lib.desireeia_tokenize(
                self._ctx, text_bytes, 1 if add_bos else 0,
                ids, n, byref(count)
            )
        if err != Error.OK:
            raise RuntimeError(f"Tokenize failed: {Error(err).name}")
        return list(ids)

    def token_piece(self, token_id: int) -> Optional[str]:
        """Decode a single token id to text. Returns None on error."""
        self._check()
        buf = (c_char * 256)()
        with self._lock:
            err = _nat.get_lib().desireeia_token_piece(
                self._ctx, token_id, buf, 256
            )
        if err != Error.OK:
            return None
        raw = buf.value
        return raw.decode("utf-8", errors="replace") if raw else ""

    @property
    def has_tokenizer(self) -> bool:
        try:
            return self.tokenize("", add_bos=False) is not None
        except Exception:
            return False

    # -- embeddings (BERT) ---------------------------------------------------

    def embed(self, tokens: List[int]) -> Optional[List[List[float]]]:
        """BERT per-token embeddings. Returns None if model is not BERT."""
        self._check()
        if not tokens:
            return []

        lib = _nat.get_lib()
        arr = (c_int32 * len(tokens))(*tokens)
        out_len = c_uint64()
        out_dim = c_uint32()

        with self._lock:
            err = lib.desireeia_embed(
                self._ctx, arr, len(tokens),
                None, 0, byref(out_len), byref(out_dim)
            )
        if err == Error.NOT_SUPPORTED:
            return None
        if err != Error.OK:
            raise RuntimeError(f"Embed failed: {Error(err).name}")

        n = out_len.value
        dim = out_dim.value
        if n == 0:
            return []

        flat = (c_float * n)()
        with self._lock:
            err = lib.desireeia_embed(
                self._ctx, arr, len(tokens),
                flat, n, byref(out_len), byref(out_dim)
            )
        if err != Error.OK:
            raise RuntimeError(f"Embed failed: {Error(err).name}")

        rows = []
        for i in range(len(tokens)):
            row = list(flat[i * dim : (i + 1) * dim])
            rows.append(row)
        return rows

    # -- special tokens ------------------------------------------------------

    def special_token_id(self, which: SpecialToken) -> Optional[int]:
        """Get id of a special token. Returns None if undefined."""
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_special_token_id(
                self._ctx, int(which), byref(out)
            )
        if err != Error.OK:
            return None
        return out.value if out.value >= 0 else None

    @property
    def eos_id(self) -> Optional[int]:
        return self.special_token_id(SpecialToken.EOS)

    def is_end_of_generation(self, token_id: int) -> bool:
        """Check if token marks end of generation (EOS + chat-specific EOT)."""
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_is_eog_token(
                self._ctx, token_id, byref(out)
            )
        return err == Error.OK and out.value != 0

    # -- chat template -------------------------------------------------------

    def apply_chat_template(
        self,
        messages: List[Tuple[str, str]],
        add_assistant: bool = True,
    ) -> str:
        """Apply the model's chat template to a list of (role, content) pairs."""
        self._check()
        lib = _nat.get_lib()

        # Build parallel arrays of NUL-terminated UTF-8 pointers
        role_ptrs = (c_char_p * len(messages))()
        content_ptrs = (c_char_p * len(messages))()
        for i, (role, content) in enumerate(messages):
            role_ptrs[i] = role.encode("utf-8")
            content_ptrs[i] = content.encode("utf-8")

        needed = c_uint64()
        with self._lock:
            err = lib.desireeia_apply_chat_template(
                self._ctx, role_ptrs, content_ptrs, len(messages),
                1 if add_assistant else 0,
                None, 0, byref(needed)
            )
        if err != Error.OK:
            raise RuntimeError(f"Apply chat template failed: {Error(err).name}")

        buf = (c_char * (needed.value + 1))()
        with self._lock:
            err = lib.desireeia_apply_chat_template(
                self._ctx, role_ptrs, content_ptrs, len(messages),
                1 if add_assistant else 0,
                buf, len(buf), byref(needed)
            )
        if err != Error.OK:
            raise RuntimeError(f"Apply chat template failed: {Error(err).name}")

        return buf.value.decode("utf-8", errors="replace")

    # -- sampling ------------------------------------------------------------

    def set_sampling(self, options: SamplingOptions) -> None:
        """Set sampling parameters. Takes effect from the next token."""
        self._check()
        native = options.to_native()
        with self._lock:
            err = _nat.get_lib().desireeia_set_sampling(
                self._ctx, byref(native)
            )
        if err != Error.OK:
            raise RuntimeError(f"Set sampling failed: {Error(err).name}")

    def get_sampling(self) -> SamplingOptions:
        """Get current sampling parameters."""
        self._check()
        native = _nat.Sampling()
        with self._lock:
            err = _nat.get_lib().desireeia_get_sampling(
                self._ctx, byref(native)
            )
        if err != Error.OK:
            raise RuntimeError(f"Get sampling failed: {Error(err).name}")
        return SamplingOptions.from_native(native)

    # -- LoRA ----------------------------------------------------------------

    def load_lora_adapter(self, lora_gguf_path: str, scale: float = 1.0) -> None:
        self._check()
        with self._lock:
            err = _nat.get_lib().desireeia_load_lora_adapter(
                self._ctx, lora_gguf_path.encode("utf-8"), scale
            )
        if err != Error.OK:
            raise RuntimeError(f"Load LoRA adapter failed: {Error(err).name}")

    def clear_lora_adapters(self) -> None:
        self._check()
        with self._lock:
            err = _nat.get_lib().desireeia_clear_lora_adapters(self._ctx)
        if err != Error.OK:
            raise RuntimeError(f"Clear LoRA adapters failed: {Error(err).name}")

    # -- prerouter ------------------------------------------------------------

    def load_prerouter(self, path: str) -> None:
        self._check()
        with self._lock:
            err = _nat.get_lib().desireeia_load_prerouter(
                self._ctx, path.encode("utf-8")
            )
        if err != Error.OK:
            raise RuntimeError(f"Load prerouter failed: {Error(err).name}")

    def clear_prerouter(self) -> None:
        self._check()
        with self._lock:
            err = _nat.get_lib().desireeia_clear_prerouter(self._ctx)
        if err != Error.OK:
            raise RuntimeError(f"Clear prerouter failed: {Error(err).name}")

    def set_prerouter_heuristic(self, enabled: bool) -> None:
        self._check()
        with self._lock:
            err = _nat.get_lib().desireeia_set_prerouter_heuristic(
                self._ctx, 1 if enabled else 0
            )
        if err != Error.OK:
            raise RuntimeError(f"Set prerouter heuristic failed: {Error(err).name}")

    # -- streaming generation ------------------------------------------------

    def stream(
        self,
        prompt_tokens: List[int],
        options: Optional[GenerateOptions] = None,
    ) -> Generator[str, None, None]:
        """Streaming generation from tokenized prompt. Yields text pieces."""
        self._check()
        opts = options or GenerateOptions()
        scanner = StopSequenceScanner(opts.stop_sequences)

        token = self.predict(prompt_tokens)
        for _ in range(opts.max_tokens):
            if self.is_end_of_generation(token):
                break

            piece = self.token_piece(token) or ""
            emit, stopped = scanner.feed(piece)
            if emit:
                yield emit
            if stopped:
                return

            token = self.next_token()

        tail = scanner.flush()
        if tail:
            yield tail

    def chat_stream(
        self,
        messages: List[Tuple[str, str]],
        options: Optional[GenerateOptions] = None,
        add_assistant: bool = True,
    ) -> Generator[str, None, None]:
        """Streaming generation from a conversation. Applies chat template."""
        self._check()
        prompt = self.apply_chat_template(messages, add_assistant)
        ids = self.tokenize(prompt, add_bos=True)
        if ids is None:
            raise RuntimeError("Model has no recognized tokenizer")
        yield from self.stream(ids, options)

    # -- vision --------------------------------------------------------------

    @staticmethod
    def load_image(path: str, expected_channels: int = 3) -> VisionImageWrapper:
        """Load an image from disk. Caller must close() the wrapper when done."""
        lib = _nat.get_lib()
        img = _nat.VisionImage()
        err = lib.desireeia_load_image(
            path.encode("utf-8"), expected_channels, byref(img)
        )
        if err != Error.OK:
            raise RuntimeError(f"Failed to load image: {Error(err).name}")
        return VisionImageWrapper(img)

    @property
    def has_vision(self) -> bool:
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_has_vision(self._ctx, byref(out))
        return err == Error.OK and out.value != 0

    @property
    def vision_token_count(self) -> int:
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_vision_token_count(self._ctx, byref(out))
        return out.value if err == Error.OK else 0

    @property
    def vision_image_token(self) -> Optional[int]:
        self._check()
        out = c_int32()
        with self._lock:
            err = _nat.get_lib().desireeia_vision_image_token(self._ctx, byref(out))
        if err != Error.OK or out.value < 0:
            return None
        return out.value

    def encode_image(self, image: VisionImageWrapper) -> Tuple[Optional[List[float]], int]:
        """Encode image into embedding vectors. Returns (embeddings, dim)."""
        self._check()
        native_img = image.get_native()
        lib = _nat.get_lib()
        out_len = c_uint64()
        out_dim = c_uint32()

        with self._lock:
            err = lib.desireeia_vision_encode_ctx(
                self._ctx, byref(native_img),
                None, 0, byref(out_len), byref(out_dim)
            )
        if err == Error.NOT_SUPPORTED:
            return None, 0
        if err != Error.OK:
            raise RuntimeError(f"Vision encode failed: {Error(err).name}")

        n = out_len.value
        dim = out_dim.value
        if n == 0:
            return [], dim

        embd = (c_float * n)()
        with self._lock:
            err = lib.desireeia_vision_encode_ctx(
                self._ctx, byref(native_img),
                embd, n, byref(out_len), byref(out_dim)
            )
        if err != Error.OK:
            raise RuntimeError(f"Vision encode failed: {Error(err).name}")

        return list(embd), dim

    def predict_with_image(
        self,
        tokens: List[int],
        embd: List[float],
        image_token: int = -1,
    ) -> int:
        """Multimodal prefill: predict with vision embeddings injected."""
        self._check()
        if not self.has_vision:
            raise RuntimeError("Model has no vision encoder")

        lib = _nat.get_lib()
        tok_arr = (c_int32 * len(tokens))(*tokens)
        embd_arr = (c_float * len(embd))(*embd)
        out = c_int32()

        with self._lock:
            err = lib.desireeia_predict_image(
                self._ctx, tok_arr, len(tokens),
                embd_arr, len(embd), image_token, byref(out)
            )
        if err != Error.OK:
            raise RuntimeError(f"PredictWithImage failed: {Error(err).name}")
        return out.value

    @staticmethod
    def preprocess_image(image: VisionImageWrapper, target_size: int) -> List[float]:
        """Resize + normalize image to CHW float array."""
        native_img = image.get_native()
        lib = _nat.get_lib()
        out_len = c_uint64()

        err = lib.desireeia_vision_preprocess(
            byref(native_img), target_size,
            None, 0, byref(out_len)
        )
        if err != Error.OK:
            raise RuntimeError(f"Vision preprocess failed: {Error(err).name}")

        n = out_len.value
        if n == 0:
            return []

        pixels = (c_float * n)()
        err = lib.desireeia_vision_preprocess(
            byref(native_img), target_size,
            pixels, n, byref(out_len)
        )
        if err != Error.OK:
            raise RuntimeError(f"Vision preprocess failed: {Error(err).name}")

        return list(pixels)

    # -- context manager protocol --------------------------------------------

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._ctx and self._ctx.value:
            _nat.get_lib().desireeia_destroy(self._ctx)
            self._ctx = c_void_p()

    def __enter__(self) -> LocalModel:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else f"ctx={self._ctx.value:#x}"
        return f"<LocalModel path={self._model_path!r} {state}>"
