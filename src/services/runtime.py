"""Shared cached accessors for project config and run manager."""

from __future__ import annotations

from functools import lru_cache

from .config import load_project_config
from .run_manager import RunManager


@lru_cache(maxsize=1)
def get_project_config():
    return load_project_config()


@lru_cache(maxsize=1)
def get_run_manager():
    return RunManager(get_project_config())
