# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Slot model manager.

Two modes, chosen once at startup from Settings:

- Single-model mode (`--model` given): one checkpoint, known up front.
- Router mode (`--model` omitted): the models folder is scanned for
  checkpoints (models_store.list_models), up to `models_max` distinct
  ones are registered, and a request picks one by name.

Either way, a MODEL is a ModelGroup of `settings.parallel` independent
Slot replicas. This is not cosmetic: the native model context is NOT
thread-safe (a single mutex serializes every call into it — see the
`self._lock = threading.Lock()` around inference calls in
python/desireeia/model.py), so the only way to serve more than one
generation at once against the same checkpoint is more than one loaded
context. `parallel` is the knob for exactly that trade — each replica is
a full, independent load of the checkpoint, at full memory cost.

`sleep_idle_seconds` (> 0) unloads a replica that has sat idle past that
threshold, checked by a background sweep the app's lifespan starts.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Iterator, List, Optional, Tuple

from . import engine as engine_mod
from . import models_store
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
        super().__init__(
            "server is busy: every replica of this model is already generating "
            "(increase --parallel to serve more concurrent requests for it)",
            status=503, code="slot_busy",
        )


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
        # time.monotonic(), not wall-clock: sweep_idle() only ever compares
        # two readings of this same clock, so a system clock adjustment
        # (NTP sync, DST, the user changing the clock) can never make a
        # freshly-used replica look idle or vice versa. Set on construction
        # so a never-used replica has a real "since" rather than reading as
        # infinitely idle from epoch 0.
        self._last_used = time.monotonic()

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

    @property
    def is_busy(self) -> bool:
        """Best-effort, not a claim: used by ModelGroup.pick_replica() to
        prefer an idle-looking replica. The actual safe-against-races gate
        is `_gate`/`_busy` inside SlotPrediction.run() — two concurrent
        callers can both read is_busy False for the same replica and only
        one of them will actually get to run; the other gets SlotBusy from
        run() itself, not a corrupted double-generation."""
        return self._busy

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_used

    def touch(self) -> None:
        self._last_used = time.monotonic()

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
                    self.touch()
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
                self.touch()
                self._load_cond.notify_all()
                return self._model

    def _load(self) -> None:
        self._model = engine_mod.load_model(self._model_path, self._settings)
        self._default_sampling = engine_mod.sampling_options(self._settings)

    def unload(self) -> None:
        with self._load_cond:
            if self._busy:
                # Never unload out from under an in-flight generation: the
                # sweep just skips a busy replica this round and catches it
                # on the next pass once it's idle again.
                return
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
                # desireeia.types.SamplingOptions names these penalty_*, not
                # <name>_penalty like the OpenAI-shaped request body
                # (CodegenParams) they're being read from here — dataclasses
                # replace() with a kwarg it doesn't recognize raises
                # TypeError, so this mismatch failed EVERY request that set
                # any of these four fields. Never caught by a test because
                # the only existing sampling test sends temperature=0.0
                # (falsy) and nothing else, so the `any(...)` guard above
                # was never true there; found by actually driving the real
                # UI, whose sliders always send non-default values.
                "penalty_frequency": env.frequency_penalty,
                "penalty_presence": env.presence_penalty,
                "penalty_repeat": env.repeat_penalty,
                "penalty_last_n": env.repeat_last_n,
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
                self._slot.touch()
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
            self._slot.touch()


def _default_model_id(model_path: str) -> str:
    path = model_path.replace("\\", "/")
    segment = path.rsplit("/", 1)[-1]
    for extension in (".gguf", ".safetensors"):
        if segment.lower().endswith(extension):
            return segment[: -len(extension)]
    return segment or "model"


class ModelGroup:
    """`parallel` independent replicas of one checkpoint."""

    def __init__(self, settings: Settings, model_path: str, model_id: Optional[str] = None) -> None:
        replica_count = max(1, settings.parallel)
        self.replicas: List[Slot] = [
            Slot(settings, model_path, model_id) for _ in range(replica_count)
        ]
        self._rr = 0  # round-robin cursor, spreads picks across replicas

    @property
    def model_id(self) -> str:
        return self.replicas[0].model_id

    @property
    def model_path(self) -> str:
        return self.replicas[0].model_path

    def matches(self, name: Optional[str]) -> bool:
        return self.replicas[0].matches(name)

    def aggregate_state(self) -> str:
        states = {r.state for r in self.replicas}
        if SlotState.LOADED in states:
            return SlotState.LOADED
        if SlotState.LOADING in states:
            return SlotState.LOADING
        if states == {SlotState.FAILED}:
            return SlotState.FAILED
        return SlotState.UNLOADED

    def pick_replica(self) -> Slot:
        """Best-effort selection, not a hard claim (see Slot.is_busy) —
        prefers an idle replica that is ALREADY loaded (no cold-start cost),
        then an idle unloaded one, then just rotates round-robin if every
        replica looks busy (the caller's run() will raise SlotBusy if that
        turns out to still be true when it actually tries)."""
        n = len(self.replicas)
        order = [self.replicas[(self._rr + i) % n] for i in range(n)]
        self._rr = (self._rr + 1) % n
        for r in order:
            if not r.is_busy and r.state == SlotState.LOADED:
                return r
        for r in order:
            if not r.is_busy:
                return r
        return order[0]

    def unload_all(self) -> None:
        for r in self.replicas:
            r.unload()


class SlotManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.Lock()
        self._groups: Dict[str, ModelGroup] = {}
        self._router_mode = not bool(settings.model)
        if not self._router_mode:
            group = ModelGroup(settings, settings.model)
            self._groups[group.model_id] = group
        else:
            self._scan_folder(settings, log_skipped=True)

    # -- discovery (router mode) ---------------------------------------------

    def _scan_folder(self, settings: Settings, *, log_skipped: bool) -> None:
        """Populates self._groups from the models folder. Caller holds
        self._lock or is still inside __init__ (no concurrent access yet)."""
        found = list(models_store.list_models(settings.resolved_models_dir))
        skipped = []
        for m in found:
            if not m.valid:
                continue
            if m.name in self._groups:
                continue
            if len(self._groups) >= max(1, settings.models_max):
                skipped.append(m.name)
                continue
            path = str(settings.resolved_models_dir / m.name)
            group = ModelGroup(settings, path, model_id=_default_model_id(m.name))
            self._groups[group.model_id] = group
        if log_skipped and skipped:
            from .logging_setup import get_logger
            get_logger(__name__).warning(
                "models_max=%d reached: %d checkpoint(s) in %s not registered (%s)",
                settings.models_max, len(skipped), settings.resolved_models_dir,
                ", ".join(skipped),
            )

    def refresh_from_folder(self) -> None:
        """Re-scans the models folder (router mode only) and reconciles:
        newly appeared valid checkpoints are registered (up to models_max),
        checkpoints that disappeared (deleted via /models/{name} or by hand)
        are unloaded and dropped. Called after an upload/delete event so
        router mode doesn't need a full server restart to see a new file.
        No-op in single-model mode."""
        if not self._router_mode:
            return
        with self._lock:
            on_disk = {m.name for m in models_store.list_models(self._settings.resolved_models_dir)
                       if m.valid}
            gone = [mid for mid, g in self._groups.items()
                    if _basename_of(g.model_path) not in on_disk]
            for mid in gone:
                self._groups.pop(mid).unload_all()
            self._scan_folder(self._settings, log_skipped=True)

    # -- resolution ------------------------------------------------------------

    def resolve(self, name: Optional[str]) -> Slot:
        with self._lock:
            if not self._groups:
                raise NotFoundError(
                    "no model is configured; start the server with --model or add "
                    "a checkpoint to the models folder",
                    status=404, code="model_not_found",
                )
            if name:
                for group in self._groups.values():
                    if group.matches(name):
                        return group.pick_replica()
                raise NotFoundError(
                    f"model {name!r} not found (available: "
                    f"{', '.join(sorted(self._groups))})",
                    status=404, code="model_not_found",
                )
            if len(self._groups) == 1:
                return next(iter(self._groups.values())).pick_replica()
        # Router mode with more than one model and no name given: which one
        # was meant is genuinely ambiguous. Guessing (e.g. "the first one")
        # would silently run the wrong model for a caller who just forgot
        # the field — a clear 400 costs one retry, a wrong model costs trust.
        raise NotFoundError(
            f"multiple models are available ({', '.join(sorted(self._groups))}); "
            "the 'model' field is required",
            status=400, code="model_required",
        )

    @property
    def slots(self) -> List[Slot]:
        """Flat list of every replica across every model — the shape
        /health and the old /v1/models loop already expect. With
        parallel > 1 this repeats the same model_id once per replica by
        design; callers that want one row per DISTINCT model should use
        `groups` instead (see api/v1.py's list_models)."""
        out: List[Slot] = []
        for group in self._groups.values():
            out.extend(group.replicas)
        return out

    @property
    def groups(self) -> Dict[str, ModelGroup]:
        return dict(self._groups)

    def shutdown(self) -> None:
        for group in self._groups.values():
            group.unload_all()

    def reconfigure(self, settings: Settings) -> None:
        with self._lock:
            for group in self._groups.values():
                group.unload_all()
            self._groups = {}
            self._settings = settings
            self._router_mode = not bool(settings.model)
            if not self._router_mode:
                group = ModelGroup(settings, settings.model)
                self._groups[group.model_id] = group
            else:
                self._scan_folder(settings, log_skipped=True)

    # -- idle eviction ---------------------------------------------------------

    def sweep_idle(self) -> List[str]:
        """Unloads every LOADED, non-busy replica idle past
        settings.sleep_idle_seconds. Returns the model_ids touched, for
        logging. A no-op (returns []) when sleep_idle_seconds <= 0 — the
        default, so nothing changes for anyone who hasn't opted in."""
        threshold = self._settings.sleep_idle_seconds
        if threshold <= 0:
            return []
        touched: List[str] = []
        with self._lock:
            for group in self._groups.values():
                for replica in group.replicas:
                    if replica.state == SlotState.LOADED and not replica.is_busy \
                            and replica.idle_seconds > threshold:
                        replica.unload()
                        touched.append(replica.model_id)
        return touched


def _basename_of(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]
