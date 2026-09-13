"""Resolve installed or source-tree simulation2d data files."""

from __future__ import annotations

from pathlib import Path


def package_share() -> Path:
    """Return the ament share directory, with a source-tree fallback."""
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("simulation2d"))
    except (ImportError, LookupError):
        return Path(__file__).resolve().parents[1]


def asset_path(name: str) -> Path:
    return package_share() / "assets" / name


def config_path(name: str) -> Path:
    return package_share() / "config" / name


def checkpoint_path(name: str = "sentry_ppo_00155.pt") -> Path:
    return package_share() / "checkpoints" / name
