"""Packaged baseline checkpoint paths."""

from __future__ import annotations

from pathlib import Path

from .beast import base_checkpoint_path as _beast_base_checkpoint_path
from .beast import baseline_checkpoint_names as _beast_baseline_checkpoint_names
from .beast import baseline_checkpoint_path as _beast_baseline_checkpoint_path
from .beatnet import base_checkpoint_path as _beatnet_base_checkpoint_path
from .beatnet import baseline_checkpoint_names as _beatnet_baseline_checkpoint_names
from .beatnet import baseline_checkpoint_path as _beatnet_baseline_checkpoint_path
from .beatnet_plus import base_checkpoint_path as _beatnet_plus_base_checkpoint_path
from .beatnet_plus import baseline_checkpoint_names as _beatnet_plus_baseline_checkpoint_names
from .beatnet_plus import baseline_checkpoint_path as _beatnet_plus_baseline_checkpoint_path
from .bocktcn import base_checkpoint_path as _bocktcn_base_checkpoint_path
from .bocktcn import baseline_checkpoint_names as _bocktcn_baseline_checkpoint_names
from .bocktcn import baseline_checkpoint_path as _bocktcn_baseline_checkpoint_path


def beatnet_base_checkpoint_path() -> Path:
    """Return the shipped BeatNet baseline checkpoint path."""
    return _beatnet_base_checkpoint_path()


def beatnet_baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a shipped BeatNet baseline checkpoint path by selector name."""
    return _beatnet_baseline_checkpoint_path(name)


def beatnet_baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BeatNet baseline checkpoint selector names."""
    return _beatnet_baseline_checkpoint_names()


def beatnet_plus_base_checkpoint_path() -> Path:
    """Return the shipped BeatNet+ baseline checkpoint path."""
    return _beatnet_plus_base_checkpoint_path()


def beatnet_plus_baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a shipped BeatNet+ baseline checkpoint path by selector name."""
    return _beatnet_plus_baseline_checkpoint_path(name)


def beatnet_plus_baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BeatNet+ baseline checkpoint selector names."""
    return _beatnet_plus_baseline_checkpoint_names()


def bocktcn_base_checkpoint_path() -> Path:
    """Return the shipped Bock TCN baseline checkpoint path."""
    return _bocktcn_base_checkpoint_path()


def bocktcn_baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a shipped Bock TCN baseline checkpoint path by selector name."""
    return _bocktcn_baseline_checkpoint_path(name)


def bocktcn_baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available Bock TCN baseline checkpoint selector names."""
    return _bocktcn_baseline_checkpoint_names()


def bock_tcn_base_checkpoint_path() -> Path:
    """Return the shipped Bock TCN baseline checkpoint path."""
    return bocktcn_base_checkpoint_path()


def bock_tcn_baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a shipped Bock TCN baseline checkpoint path by selector name."""
    return bocktcn_baseline_checkpoint_path(name)


def bock_tcn_baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available Bock TCN baseline checkpoint selector names."""
    return bocktcn_baseline_checkpoint_names()


def beast_base_checkpoint_path() -> Path:
    """Return the shipped BEAST baseline checkpoint path."""
    return _beast_base_checkpoint_path()


def beast_baseline_checkpoint_path(name: str = "baseline") -> Path:
    """Return a shipped BEAST baseline checkpoint path by selector name."""
    return _beast_baseline_checkpoint_path(name)


def beast_baseline_checkpoint_names() -> tuple[str, ...]:
    """Return available BEAST baseline checkpoint selector names."""
    return _beast_baseline_checkpoint_names()


__all__ = [
    "beast_base_checkpoint_path",
    "beast_baseline_checkpoint_names",
    "beast_baseline_checkpoint_path",
    "beatnet_base_checkpoint_path",
    "beatnet_baseline_checkpoint_names",
    "beatnet_baseline_checkpoint_path",
    "beatnet_plus_base_checkpoint_path",
    "beatnet_plus_baseline_checkpoint_names",
    "beatnet_plus_baseline_checkpoint_path",
    "bock_tcn_base_checkpoint_path",
    "bock_tcn_baseline_checkpoint_names",
    "bock_tcn_baseline_checkpoint_path",
    "bocktcn_base_checkpoint_path",
    "bocktcn_baseline_checkpoint_names",
    "bocktcn_baseline_checkpoint_path",
]
