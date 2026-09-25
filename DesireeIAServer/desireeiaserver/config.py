"""Server configuration: CLI parsing, env-var fallbacks, JSON config file.

Precedence (lowest to highest): built-in defaults -> environment variables ->
config file (`--config`, default `<data-dir>/config.json`) -> explicit
command-line flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

DESIREEIA_DATA_DIRNAME = ".desireeia"
DESIREEIA_MODELS_DIRNAME = "models"
DESIREEIA_CONFIG_FILENAME = "config.json"
SUPPORTED_BACKENDS = ("cpu", "cuda", "metal", "vulkan", "intel", "axelera")

SERIALIZED_KEYS = (
    "host", "port", "model", "alias", "models_dir", "models_max",
    "models_autoload", "sleep_idle_seconds", "parallel", "threads",
    "n_gpu_layers", "backend", "ram_budget_mb", "temperature", "top_k",
    "top_p", "repeat_penalty", "repeat_last_n", "frequency_penalty",
    "presence_penalty", "seed", "n_predict", "system_prompt", "log_level",
    "max_upload_mb", "max_request_mb", "whisper_model",
)


class _EnvDefault(argparse.Action):
    def __init__(self, env: Optional[str] = None, **kwargs):
        if env and os.environ.get(env):
            kwargs["default"] = os.environ[env]
        self._env = env
        super().__init__(**kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8080
    model: Optional[str] = None
    alias: Optional[str] = None
    data_dir: Optional[Path] = None
    models_dir: Optional[Path] = None
    models_max: int = 1
    models_autoload: bool = True
    sleep_idle_seconds: int = 0
    parallel: int = 1
    threads: Optional[int] = None
    n_gpu_layers: Optional[int] = None
    backend: Optional[str] = None
    ram_budget_mb: int = 0
    # Same defaults as desireeia.autoconfig (never greedy: greedy decoding
    # falls into repetitive loops on long generations). --temperature 0
    # still selects greedy explicitly.
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repeat_penalty: float = 1.0
    repeat_last_n: int = 64
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    seed: int = -1
    n_predict: int = 2048
    system_prompt: str = ""
    api_keys: tuple = ()
    cors_origins: tuple = ()
    log_level: str = "info"
    log_file: Optional[Path] = None
    ui_dir: Optional[Path] = None
    max_upload_mb: int = 100 * 1024
    max_request_mb: int = 8
    enable_python_tool: bool = False
    enable_whisper: bool = False
    whisper_model: str = "base"
    config_path: Optional[Path] = None

    @property
    def resolved_data_dir(self) -> Path:
        return self.data_dir or (Path.home() / DESIREEIA_DATA_DIRNAME)

    @property
    def resolved_models_dir(self) -> Path:
        return self.models_dir or (self.resolved_data_dir / DESIREEIA_MODELS_DIRNAME)

    @property
    def resolved_config_path(self) -> Path:
        return self.config_path or (self.resolved_data_dir / DESIREEIA_CONFIG_FILENAME)

    def ensure_models_dir(self) -> Path:
        directory = self.resolved_models_dir
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def to_dict(self) -> dict:
        result = {}
        for key in SERIALIZED_KEYS:
            value = getattr(self, key)
            result[key] = str(value) if isinstance(value, Path) else value
        return result


def _field_types(cls) -> dict:
    """Map dataclass field name -> concrete type, unwrapping Optional[...]."""
    hints = typing.get_type_hints(cls)
    result = {}
    for field in dataclasses.fields(cls):
        hint = hints[field.name]
        if typing.get_origin(hint) is typing.Union:
            members = [member for member in typing.get_args(hint) if member is not type(None)]
            hint = members[0] if len(members) == 1 else typing.Union[tuple(members)]
        result[field.name] = hint
    return result


_FIELD_TYPES = _field_types(Settings)


def _env_api_keys() -> List[str]:
    raw = os.environ.get("DESIREEIA_API_KEY")
    if not raw:
        return []
    return [key.strip() for key in raw.replace(",", ";").split(";") if key.strip()]


def _coerce(field_type, value):
    if isinstance(value, field_type):
        return value
    if field_type is Path:
        return Path(str(value))
    if field_type is bool:
        return str(value).lower() in ("1", "true", "yes", "on")
    if field_type is str:
        return str(value)
    return field_type(value)


def _apply_overrides(settings: Settings, overrides: dict) -> Settings:
    fields = dict(settings.__dict__)
    for key, value in overrides.items():
        if key not in fields or value is None:
            continue
        try:
            fields[key] = _coerce(_FIELD_TYPES.get(key, type(fields[key])), value)
        except (TypeError, ValueError):
            continue
    return Settings(**fields)


def load_config(path: Optional[Path], settings: Settings) -> Settings:
    if path is None or not path.is_file():
        return settings
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return settings
    if not isinstance(raw, dict):
        return settings
    return _apply_overrides(settings, raw)


def save_config(settings: Settings, path: Optional[Path] = None) -> dict:
    target = path or settings.resolved_config_path
    target.parent.mkdir(parents=True, exist_ok=True)
    data = settings.to_dict()
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return data


def parse_args(argv: Optional[List[str]] = None) -> Settings:
    parser = argparse.ArgumentParser(
        prog="desireeia-server",
        description="DesireeIA local LLM server: OpenAI-compatible API + web UI. "
        "The inference motor is always DesireeIA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    generic = parser.add_argument_group("config")
    generic.add_argument(
        "--config", type=Path, default=None, dest="config_path",
        help="JSON config file (default <data-dir>/config.json)")

    serving = parser.add_argument_group("serving")
    serving.add_argument("--host", action=_EnvDefault, env="DESIREEIA_HOST", default=None)
    serving.add_argument("--port", type=int, action=_EnvDefault, env="DESIREEIA_PORT", default=None)
    serving.add_argument("--api-key", action="append", dest="api_keys", default=[],
                         help="static API key (repeatable); env DESIREEIA_API_KEY")
    serving.add_argument("--cors-origin", action="append", dest="cors_origins", default=[],
                         help="allowed cross-origin caller (repeatable); none = no CORS headers added")

    model = parser.add_argument_group("model")
    model.add_argument("--model", action=_EnvDefault, env="DESIREEIA_MODEL", default=None)
    model.add_argument("--alias", default=None)
    model.add_argument("--models-dir", type=Path, action=_EnvDefault,
                       env="DESIREEIA_MODELS_DIR", default=None,
                       help="checkpoint folder (default <data-dir>/models, created automatically)")
    model.add_argument("--models-max", type=int, default=None, help="max simultaneously loaded models")
    model.add_argument("--models-autoload", "--no-models-autoload",
                       action=argparse.BooleanOptionalAction, default=None,
                       help="automatically load a listed model on request")
    model.add_argument("--sleep-idle-seconds", type=int, default=None,
                       help="unload models idle for N seconds (0 = disabled)")

    engine = parser.add_argument_group("engine")
    engine.add_argument("--backend", choices=SUPPORTED_BACKENDS, default=None)
    engine.add_argument("--threads", type=int, default=None)
    engine.add_argument("--n-gpu-layers", type=int, default=None)
    engine.add_argument("--ram-budget-mb", type=int, default=None)
    engine.add_argument("--parallel", type=int, default=None, help="number of concurrent slots")

    sampling = parser.add_argument_group("sampling")
    sampling.add_argument("--temperature", type=float, default=None, help="0 = greedy")
    sampling.add_argument("--top-k", type=int, default=None, help="0 = disabled")
    sampling.add_argument("--top-p", type=float, default=None, help="1 = disabled")
    sampling.add_argument("--repeat-penalty", type=float, default=None, help="1 = disabled")
    sampling.add_argument("--repeat-last-n", type=int, default=None)
    sampling.add_argument("--frequency-penalty", type=float, default=None)
    sampling.add_argument("--presence-penalty", type=float, default=None)
    sampling.add_argument("--seed", type=int, default=None, help="-1 = random")
    sampling.add_argument("--n-predict", type=int, default=None, help="default max tokens (2048)")
    sampling.add_argument("--system-prompt", default=None,
                          help="default system prompt for a new conversation")

    storage = parser.add_argument_group("storage and logging")
    storage.add_argument("--data-dir", type=Path, action=_EnvDefault,
                         env="DESIREEIA_DATA_DIR", default=None,
                         help="data root for sessions, files and downloads")
    storage.add_argument("--ui-dir", type=Path, action=_EnvDefault,
                         env="DESIREEIA_UI_DIR", default=None)
    storage.add_argument("--log-level", choices=("debug", "info", "warning", "error"), default=None)
    storage.add_argument("--log-file", type=Path, default=None)
    storage.add_argument("--max-upload-mb", type=int, default=None)
    storage.add_argument("--max-request-mb", type=int, default=None,
                         help="cap on a regular (non-checkpoint-upload) request body")

    tools = parser.add_argument_group("tools")
    tools.add_argument("--enable-python-tool", "--no-enable-python-tool",
                       action=argparse.BooleanOptionalAction, default=None,
                       help="expose POST /tools/run_python (sandboxed code execution for the run_python tool)")
    tools.add_argument("--enable-whisper", "--no-enable-whisper",
                       action=argparse.BooleanOptionalAction, default=None,
                       help="expose POST /transcribe (voice input) - requires the [audio] extra installed")
    tools.add_argument("--whisper-model", default=None,
                       help="faster-whisper model size/name (tiny, base, small, medium, large-v3, ...)")

    ns = parser.parse_args(argv)

    cli_data_dir = ns.data_dir or os.environ.get("DESIREEIA_DATA_DIR")
    effective_config_path = ns.config_path or (
        Path(cli_data_dir) if cli_data_dir else Path.home() / DESIREEIA_DATA_DIRNAME
    ) / DESIREEIA_CONFIG_FILENAME

    settings = load_config(effective_config_path, Settings())

    explicit = {
        key: getattr(ns, key)
        for key in SERIALIZED_KEYS + ("data_dir", "log_file", "ui_dir")
        if hasattr(ns, key) and getattr(ns, key) is not None
    }
    settings = _apply_overrides(settings, explicit)

    return _apply_overrides(settings, {
        "api_keys": tuple(ns.api_keys) + tuple(_env_api_keys()),
        "cors_origins": tuple(ns.cors_origins),
        "enable_python_tool": ns.enable_python_tool,
        "enable_whisper": ns.enable_whisper,
        "config_path": ns.config_path,
    })