"""Sampling map: OpenAI JSON -> engine knobs, with rejection of unsupported fields."""

from __future__ import annotations

import pytest

from desireeiaserver.api.errors import InvalidRequestError, NotSupportedError
from desireeiaserver.config import Settings
from desireeiaserver.sampling import resolve


def test_resolve_maps_fields():
    params = resolve({
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
        "frequency_penalty": 0.1,
        "presence_penalty": 0.2,
        "repeat_penalty": 1.2,
        "seed": 42,
        "stop": "END",
        "max_tokens": 100,
        "stream": True,
    })
    assert params.temperature == 0.8
    assert params.top_k == 50
    assert params.top_p == 0.9
    assert params.frequency_penalty == 0.1
    assert params.presence_penalty == 0.2
    assert params.repeat_penalty == 1.2
    assert params.seed == 42
    assert params.stop == ["END"]
    assert params.max_tokens == 100
    assert params.stream is True


def test_resolve_stop_list():
    assert resolve({"stop": ["a", "b"]}).stop == ["a", "b"]


def test_rejects_unsupported_sampling_fields():
    for field in ("min_p", "mirostat", "typical_p", "top_logprobs"):
        with pytest.raises(NotSupportedError):
            resolve({field: 0.1})


def test_rejects_logprobs():
    with pytest.raises(NotSupportedError):
        resolve({"logprobs": 1})


def test_rejects_multiple_choices():
    with pytest.raises(NotSupportedError):
        resolve({"n": 2})
    with pytest.raises(NotSupportedError):
        resolve({"best_of": 3})


def test_rejects_bad_types():
    with pytest.raises(InvalidRequestError):
        resolve({"temperature": "hot"})
    with pytest.raises(InvalidRequestError):
        resolve({"top_k": 3.5})
    with pytest.raises(InvalidRequestError):
        resolve({"n": 0})


def test_max_tokens_resolution():
    assert resolve({"max_tokens": 100}).generation_max_tokens(Settings()) == 100
    assert resolve({}).generation_max_tokens(Settings(n_predict=256)) == 256
    assert resolve({"max_tokens": 0}).generation_max_tokens(Settings()) == 2048


def test_n_predict_allowed_in_native_mode():
    params = resolve({"n_predict": 128}, allow_n_predict=True)
    assert params.n_predict == 128
    assert params.generation_max_tokens(Settings()) == 128