"""Map OpenAI JSON request fields to engine sampling knobs.

Fields the engine has no equivalent for are rejected with
`not_supported_error` instead of being silently ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from .config import Settings
from .api.errors import InvalidRequestError, NotSupportedError

UNSUPPORTED_FIELDS = (
    "min_p",
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "typical_p",
    "top_logprobs",
)

_BIG_CAP = 32768


@dataclass(frozen=True)
class CodegenParams:
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    repeat_penalty: Optional[float] = None
    repeat_last_n: Optional[int] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    max_tokens: Optional[int] = None
    n_predict: Optional[int] = None
    stream: bool = False
    echo: bool = False
    n: int = 1
    response_format: Optional[dict] = None

    def generation_max_tokens(self, settings: Settings) -> int:
        if self.max_tokens is not None and self.max_tokens > 0:
            return self.max_tokens
        if self.n_predict:
            return self.n_predict if self.n_predict > 0 else _BIG_CAP
        return settings.n_predict if settings.n_predict > 0 else 2048

    def sampling_tuple(self) -> Tuple[str, float, float]:
        # `is None`, not `or`: an explicit temperature 0 (greedy) must survive.
        return (0.7 if self.temperature is None else self.temperature,
                40 if self.top_k is None else self.top_k,
                0.9 if self.top_p is None else self.top_p)


def _optional_float(body: dict, key: str, default: Optional[float]) -> Optional[float]:
    if key not in body or body[key] is None:
        return default
    value = body[key]
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise InvalidRequestError(f"field {key!r} must be a number", param=key)
    return float(value)


def _optional_int(body: dict, key: str, default: Optional[int]) -> Optional[int]:
    if key not in body or body[key] is None:
        return default
    value = body[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise InvalidRequestError(f"field {key!r} must be an integer", param=key)
    return value


def _stop_sequences(body: dict) -> Optional[List[str]]:
    if "stop" not in body or body["stop"] is None:
        return None
    value = body["stop"]
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise InvalidRequestError("field 'stop' must be a string or a list of strings", param="stop")


def _response_format(body: dict) -> Optional[dict]:
    if "response_format" not in body or body["response_format"] is None:
        return None
    value = body["response_format"]
    if not isinstance(value, dict):
        raise InvalidRequestError("field 'response_format' must be an object", param="response_format")
    kind = value.get("type")
    if kind not in ("json_object", "json_schema"):
        raise NotSupportedError(
            f"response_format type {kind!r} is not supported (expected 'json_object' or 'json_schema')",
            param="response_format",
        )
    return value


def resolve(body, *, allow_n_predict: bool = False) -> CodegenParams:
    for key in UNSUPPORTED_FIELDS:
        if key in body:
            raise NotSupportedError(f"sampling field {key!r} is not supported by the DesireeIA engine", param=key)

    if "logprobs" in body and body["logprobs"]:
        raise NotSupportedError("the engine does not expose token probabilities; 'logprobs' is not supported", param="logprobs")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise InvalidRequestError("field 'stream' must be a boolean", param="stream")

    n = body.get("n", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise InvalidRequestError("field 'n' must be a positive integer", param="n")
    if n != 1:
        raise NotSupportedError("multiple choices ('n') are not supported by the engine", param="n")

    best_of = body.get("best_of", 1)
    if isinstance(best_of, bool) or not isinstance(best_of, int) or best_of < 1:
        raise InvalidRequestError("field 'best_of' must be a positive integer", param="best_of")
    if best_of != 1:
        raise NotSupportedError("multiple candidates ('best_of') are not supported by the engine", param="best_of")

    echo = body.get("echo", False)
    if not isinstance(echo, bool):
        raise InvalidRequestError("field 'echo' must be a boolean", param="echo")

    return CodegenParams(
        temperature=_optional_float(body, "temperature", None),
        top_k=_optional_int(body, "top_k", None),
        top_p=_optional_float(body, "top_p", None),
        frequency_penalty=_optional_float(body, "frequency_penalty", None),
        presence_penalty=_optional_float(body, "presence_penalty", None),
        repeat_penalty=_optional_float(body, "repeat_penalty", None),
        repeat_last_n=_optional_int(body, "repeat_last_n", None),
        seed=_optional_int(body, "seed", None),
        stop=_stop_sequences(body),
        max_tokens=_optional_int(body, "max_tokens", None),
        n_predict=_optional_int(body, "n_predict", None) if allow_n_predict else None,
        stream=stream,
        echo=echo,
        n=n,
        response_format=_response_format(body),
    )