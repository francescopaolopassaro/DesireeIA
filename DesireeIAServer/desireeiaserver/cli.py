"""Console entry point: parse config, wire logging, run uvicorn."""

from __future__ import annotations

import logging
from typing import Optional

from .app import create_app
from .config import parse_args
from .logging_setup import configure_logging


def main(argv: Optional[list] = None) -> None:
    settings = parse_args(argv)
    configure_logging(settings.log_level, settings.log_file)
    logger = logging.getLogger("desireeiaserver")
    logger.info("models folder: %s", settings.ensure_models_dir())
    if settings.model:
        logger.info("single-model mode: %s", settings.model)
    else:
        logger.info("router mode: scanning %s", settings.resolved_models_dir)

    import uvicorn

    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level=settings.log_level)