"""Packaged baseline checkpoint paths."""

from __future__ import annotations

from pathlib import Path

from .beatnet import base_checkpoint_path as _beatnet_base_checkpoint_path


def beatnet_base_checkpoint_path() -> Path:
    """Return the shipped BeatNet baseline checkpoint path."""
    return _beatnet_base_checkpoint_path()


__all__ = ["beatnet_base_checkpoint_path"]
