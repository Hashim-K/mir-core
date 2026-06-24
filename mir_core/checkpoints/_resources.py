"""Helpers for packaged checkpoint resources."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


def immutable_mapping(mapping: Mapping[str, str]) -> Mapping[str, str]:
    """Return a read-only copy of a checkpoint metadata mapping."""
    return MappingProxyType(dict(mapping))


def checkpoint_names(filenames: Mapping[str, str]) -> tuple[str, ...]:
    """Return available checkpoint selector names."""
    return tuple(filenames)


def checkpoint_path(package: str, filenames: Mapping[str, str], name: str) -> Path:
    """Return a packaged checkpoint path by selector name."""
    try:
        filename = filenames[name]
    except KeyError as exc:
        available = ", ".join(filenames)
        raise KeyError(
            f"Unknown checkpoint '{name}' for {package}. Available: {available}"
        ) from exc

    resource = files(package).joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Packaged checkpoint is missing: {resource}")
    return Path(str(resource))
