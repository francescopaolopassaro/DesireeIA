"""Logging setup shared by the CLI and the app factory."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s - %(message)s"


def configure_logging(level: str = "info", log_file: Optional[Path] = None) -> None:
    handlers = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)