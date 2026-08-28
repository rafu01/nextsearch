"""Central logging configuration for nextsearch."""

from __future__ import annotations

import logging
import sys
from typing import TextIO


def setup_logging(
    level: str | int = logging.INFO,
    stream: TextIO = sys.stderr,
) -> None:
    """Configure root logging for the application.

    Safe to call multiple times — existing handlers are reused and the level
    is updated. Third-party libraries can be noisy, so this only configures
    the root handler; module-level loggers control their own level.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if not root.handlers:
        handler = logging.StreamHandler(stream)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        root.addHandler(handler)
