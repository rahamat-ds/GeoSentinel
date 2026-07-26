"""Structured logging configuration.

Every log line is machine-parseable (JSON) and carries the
experiment_id, so logs from concurrent or historical runs can be
filtered and correlated with their experiment record. Human-readable
console output is a rendering choice on top of the same structured
events, not a separate code path.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import structlog


def configure_logging(
    *,
    log_dir: Path | None = None,
    level: int = logging.INFO,
    experiment_id: str | None = None,
) -> structlog.stdlib.BoundLogger:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(log_dir / f"{experiment_id or 'geosentinel'}.jsonl", encoding="utf-8")
        )

    logging.basicConfig(level=level, format="%(message)s", handlers=handlers, force=True)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    logger = structlog.get_logger("geosentinel")
    if experiment_id:
        structlog.contextvars.bind_contextvars(experiment_id=experiment_id)
    return logger