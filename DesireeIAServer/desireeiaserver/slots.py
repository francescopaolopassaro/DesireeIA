"""Slot model manager.

Single-model mode owns one `Slot`. A slot wraps a `LocalModel`, guards its
(non-thread-safe) native context during generation, loads lazily, and hides
the sample/generate plumbing from the routers.
"""

from __future__ import annotations

import threading
from typing import Iterator, List, Optional, Tuple

from . import engine as engine_mod
from .api.errors import NotFoundError, UnavailableError
from .config import Settings
from .generation import PredictionStep, build_prompt_ids, iter_prediction
from .sampling import CodegenParams
from .token_stats import TokenTracker


class SlotState(str):
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    FAILED = "failed"


class SlotBusy(UnavailableError):
    def __init__(self) -> None:
        super().__init__("server is busy: generation is already in progress on this slot",
                         status=503, code="slot_busy")


class Slot:
    def __init__(self, settings: Settings, model_path: str, model_id: Optional[str] = None) -> None:
        self._settings = settings
        self._model_path = model_path
        self._model_id = model_id or _default_model_id(model_path)
        self._state = SlotState.UNLOADED
        self._error: Optional[str] = None
        self._model = None
        self._default_sampling = None
        self._load_cond = threading.Condition()
        self._inference_lock = threading.Lock()
        self._gate = threading.Lock()
        self._busy = False

    # -- identity ------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def state(self) -> str:
        return self._state

    @property
    def error(self) -> Optional[str]:
        return self._error

    def matches(self, name: Optional[str]) -> bool:
        if not name:
            return True
        candidates = {
            self._model_id,
            self._alias_or_id(),
            str(self._model_path),
            self._model_path.replace("\\", "/"),
        }
        return name in candidates

    def _alias_or_id(self) -> str:
        alias = self._settings.alias
        return alias or self._model_id

    # -- loading -------------------------------------------------------------

    def ensure_loaded(self):
        with self._load_cond:
            while True:
                if self._state == SlotState.LOADED:
                    return self._model
                if self._state == SlotState.LOADING:
                    self._load_cond.wait()
                    continue
                if self._state == SlotState.FAILED:
                    raise UnavailableError(
                        f"model {self._model_id!r} failed to load: {self._error}",
                        status=503, code="model_load_failed",
                    )
                self._state = SlotState.LOADING
                self._error = None
                try:
                    self._load()
                except Exception as exc:
                    self._state = SlotState.FAILED
                    self._error = str(exc) or exc.__class__.__name__
                    self._load_cond.notify_all()
                    raise UnavailableError(
                        f"model {self._model_id!r} failed to load: {self._error}",
                        status=503, code="model_load_failed",
                    ) from exc
                self._state = SlotState.LOADED
                self._load_cond.notify_all()
                return self._model

    def _load(self) -> None:
        self._model = engine_mod.load_model(self._model_path, self._settings)
        self._default_sampling = engine_mod.sampling_options(self._settings)

    def unload(self) -> None:
        with self._load_cond:
            if self._model is not None:
                self._model.close()
                self._model = None
            self._default_sampling = None
            self._state = SlotState.UNLOADED

    # -- generation ----------------------------------------------------------

    def info(self) -> Optional[dict]:
        if self._state != SlotState.LOADED or self._model is None:
            return None
        model = self._model
        try:
            return {
                "id": self._model_id,
                "path": self._model_path,
                "context_size": model.context_size(),
                "has_tokenizer": model.has_tokenizer,
                "has_vision": model.has_vision,
                "backend": str(model.plan.backend.name),
                "threads": model.plan.thread_count,
            }
        except Exception:
            return {"id": self._model_id, "path": self._model_path}

    def prediction(
        self,
        messages: List[Tuple[str, str]],
        params: CodegenParams,
    ):
        return SlotPrediction(self, messages, params)


class SlotPrediction:
    def __init__(self, slot: Slot, messages: List[Tuple[str, str]], params: CodegenParams) -> None:
        self._slot = slot
        self._messages = messages
        self._params = params

    def _apply_sampling(self, model) -> None:
        options = engine_mod.sampling_options(self._slot._settings)
        target = options
        env = self._params
        if any((env.temperature, env.top_k, env.top_p, env.frequency_penalty,
                env.presence_penalty, env.repeat_penalty, env.repeat_last_n, env.seed)):
            desired = {
                "temperature": env.temperature,
                "top_k": env.top_k,
                "top_p": env.top_p,
                "frequency_penalty": env.frequency_penalty,
                "presence_penalty": env.presence_penalty,
                "repeat_penalty": env.repeat_penalty,
                "repeat_last_n": env.repeat_last_n,
                "seed": env.seed,
            }
            kwargs = {}
            for key, value in desired.items():
                if value is not None:
                    kwargs[key] = value
            from dataclasses import replace
            target = replace(target, **kwargs)
        model.set_sampling(target)

    def _build_generation(self, model) -> None:
        options = engine_mod.generate_options(
            self._slot._settings,
            max_tokens=self._params.generation_max_tokens(self._slot._settings),
            stop_sequences=self._params.stop,
        )
        return options

    def run(self, cancelled: threading.Event, tracker: TokenTracker) -> Iterator[PredictionStep]:
        model = self._slot.ensure_loaded()
        with self._slot._gate:
            if self._slot._busy:
                raise SlotBusy()
            self._slot._busy = True
        try:
            with self._slot._inference_lock:
                self._apply_sampling(model)
                generation = self._build_generation(model)
                try:
                    ids = build_prompt_ids(model, self._messages)
                    yield from iter_prediction(model, ids, generation, tracker, cancelled)
                finally:
                    if self._slot._default_sampling is not None:
                        try:
                            model.set_sampling(self._slot._default_sampling)
                        except Exception:
                            pass
        finally:
            with self._slot._gate:
                self._slot._busy = False


def _default_model_id(model_path: str) -> str:
    path = model_path.replace("\\", "/")
    segment = path.rsplit("/", 1)[-1]
    for extension in (".gguf", ".safetensors"):
        if segment.lower().endswith(extension):
            return segment[: -len(extension)]
    return segment or "model"


class SlotManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._slots: List[Slot] = []
        self._lock = threading.Lock()
        if settings.model:
            self._slots.append(Slot(settings, settings.model))

    def resolve(self, name: Optional[str]) -> Slot:
        with self._lock:
            if not self._slots:
                raise NotFoundError(
                    "no model is configured; start the server with --model or add "
                    "a checkpoint to the models folder",
                    status=404, code="model_not_found",
                )
            for slot in self._slots:
                if slot.matches(name):
                    return slot
        raise NotFoundError(
            f"model {name!r} not found in single-model mode (serving {self._slots[0].model_id!r})",
            status=404, code="model_not_found",
        )

    @property
    def slots(self) -> List[Slot]:
        return self._slots

    def shutdown(self) -> None:
        for slot in self._slots:
            slot.unload()

    def reconfigure(self, settings: Settings) -> None:
        with self._lock:
            for slot in self._slots:
                slot.unload()
            self._slots = []
            if settings.model:
                self._slots.append(Slot(settings, settings.model))