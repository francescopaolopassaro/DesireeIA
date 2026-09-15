"""Settings and CLI parsing."""

from __future__ import annotations

from pathlib import Path

from desireeiaserver.config import Settings, parse_args


def test_defaults_resolve_models_dir_under_home():
    settings = Settings()
    assert settings.resolved_models_dir == (
        Path.home() / ".desireeia" / "models"
    )


def test_ensure_models_dir_creates_folder(tmp_path):
    target = tmp_path / "nested" / "models"
    settings = Settings(models_dir=target)
    created = settings.ensure_models_dir()
    assert created == target
    assert target.is_dir()


def test_parse_long_flags():
    settings = parse_args([
        "--model", "model.gguf",
        "--alias", "name",
        "--host", "0.0.0.0",
        "--port", "9000",
        "--temperature", "0.7",
        "--top-k", "50",
        "--backend", "cuda",
        "--parallel", "4",
        "--models-dir", r"C:\models",
        "--n-predict", "1024",
    ])
    assert settings.model == "model.gguf"
    assert settings.alias == "name"
    assert settings.host == "0.0.0.0"
    assert settings.port == 9000
    assert settings.temperature == 0.7
    assert settings.top_k == 50
    assert settings.backend == "cuda"
    assert settings.parallel == 4
    assert settings.n_predict == 1024
    assert settings.resolved_models_dir == Path(r"C:\models")


def test_models_autoload_default_on():
    assert Settings().models_autoload is True


def test_no_models_autoload_flag():
    settings = parse_args(["--no-models-autoload"])
    assert settings.models_autoload is False


def test_api_keys_repeatable(monkeypatch):
    monkeypatch.delenv("DESIREEIA_API_KEY", raising=False)
    settings = parse_args(["--api-key", "one", "--api-key", "two"])
    assert settings.api_keys == ("one", "two")


def test_api_keys_env(monkeypatch):
    monkeypatch.setenv("DESIREEIA_API_KEY", "a;b")
    settings = parse_args([])
    assert set(settings.api_keys) == {"a", "b"}


def test_help_exits_zero():
    import sys

    try:
        parse_args(["--help"])
    except SystemExit as exc:
        assert exc.code == 0
        return
    sys.exit("expected SystemExit")