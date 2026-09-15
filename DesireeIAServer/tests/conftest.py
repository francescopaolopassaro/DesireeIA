"""Shared fixtures: a fake `desireeia` module so tests run without the native engine."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest


@dataclass
class FakeHardware:
    cpu_threads: int = 8
    avx: bool = True
    avx2: bool = True
    avx512: bool = False
    neon: bool = False
    cuda_device_count: int = 0
    ram_total_mb: int = 16384
    ram_free_mb: int = 8192
    intel_gpu_count: int = 0
    axelera_device_count: int = 0
    metal: bool = False
    vulkan: bool = False


@dataclass
class FakeExecutionPlan:
    backend: int = 0
    thread_count: int = 1
    ram_budget_mb: int = 0


@dataclass
class FakeSamplingOptions:
    temperature: float = 0.0
    top_k: int = 40
    top_p: float = 0.95
    penalty_repeat: float = 1.0
    penalty_frequency: float = 0.0
    penalty_presence: float = 0.0
    penalty_last_n: int = 64
    seed: int = 0


@dataclass
class FakeGenerateOptions:
    max_tokens: int = 512
    stop_sequences: list = field(default=None)


class FakeLocalModel:
    """Deterministic token stream: "TokenA TokenB " then EOS (id 12)."""

    EOS = 12
    PIECES = {10: "TokenA ", 11: "TokenB ", 12: ""}

    def __init__(self, path: str, plan=None, logger=None) -> None:
        self.model_path = path
        self.plan = plan
        self._lock = threading.Lock()
        self._closed = False
        self._counter = 0
        self._sampling = None

    @staticmethod
    def load(model_path, plan, logger=None) -> "FakeLocalModel":
        return FakeLocalModel(model_path, plan, logger)

    def tokenize(self, text: str, add_bos: bool = True):
        return [1, 2, 3, 4]

    def token_piece(self, token_id: int):
        return self.PIECES.get(token_id, "?")

    def predict(self, tokens):
        return 10

    def next_token(self):
        with self._lock:
            value = [11, 12][min(self._counter, 1)]
            self._counter += 1
            return value

    def is_end_of_generation(self, token_id: int) -> bool:
        return token_id == self.EOS

    def apply_chat_template(self, messages, add_assistant: bool = True) -> str:
        return "TEMPLATE|" + "".join(f"{role}:{content}\n" for role, content in messages)

    def set_sampling(self, options) -> None:
        self._sampling = options

    def embed(self, tokens):
        return None

    def context_size(self) -> int:
        return 2048

    @property
    def has_tokenizer(self) -> bool:
        return True

    @property
    def has_vision(self) -> bool:
        return False

    def close(self) -> None:
        self._closed = True


class FakeDesireeiaModule:
    InferenceBackend = SimpleNamespace(
        UNCONFIGURED=0, CPU=1, CUDA=2, METAL=3, VULKAN=4, INTEL=5, AXELERA=6
    )
    ExecutionPlan = FakeExecutionPlan
    SamplingOptions = FakeSamplingOptions
    GenerateOptions = FakeGenerateOptions
    LocalModel = FakeLocalModel

    @staticmethod
    def version() -> str:
        return "9.9.9-fake"

    @staticmethod
    def detect_hardware() -> FakeHardware:
        return FakeHardware()

    @staticmethod
    def build_plan(model_path: str, overrides: FakeExecutionPlan | None = None):
        return overrides or FakeExecutionPlan()


@pytest.fixture
def anyio_backend():
    # Every other test here drives the app synchronously through
    # TestClient; this exists only for the handful of tests that need to
    # await something directly (see test_models_api.py's SSE route test
    # and its comment on why). asyncio is the only backend this project
    # uses anywhere else, so trio support is not exercised.
    return "asyncio"


@pytest.fixture
def fake_desireeia(monkeypatch):
    import desireeiaserver.engine as engine_mod

    engine_mod._desireeia = FakeDesireeiaModule
    engine_mod._import_error = None
    return FakeDesireeiaModule