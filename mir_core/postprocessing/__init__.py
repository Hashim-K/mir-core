"""Post-processing utilities for beat detection.

Provides Dynamic Bayesian Network (DBN) beat trackers, particle filter
cascade, and convenience functions for converting neural network
activation functions to discrete beat times.

Classes:
    DBNBeatTracker         — beat tracking from 1D activation (wraps madmom DBN).
    DBNDownbeatTracker     — joint beat+downbeat tracking from two activations.
    DBNBarTracker          — bar (meter) tracking from beat times + downbeat activations.
    ParticleFilterTracker  — particle filter cascade for joint beat/downbeat tracking.
    Heydari1DStateSpaceTracker — jump-reward inference on Heydari's compact 1D state space.

Convenience functions:
    detect_beats  — one-call beat detection (creates a DBNBeatTracker internally).
    detect_tempo  — tempo estimation from activation histogram via peak picking.
    peak_picking  — simple threshold + min-interval peak picking (no DBN).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "DBNBeatTracker": (".dbn", "DBNBeatTracker"),
    "DBNDownbeatTracker": (".dbn", "DBNDownbeatTracker"),
    "DBNBarTracker": (".dbn", "DBNBarTracker"),
    "ParticleFilterTracker": (".particle_filter", "ParticleFilterTracker"),
    "Heydari1DStateSpaceTracker": (
        ".state_space_1d",
        "Heydari1DStateSpaceTracker",
    ),
    "detect_beats": (".peak_picking", "detect_beats"),
    "detect_tempo": (".peak_picking", "detect_tempo"),
    "peak_picking": (".peak_picking", "peak_picking"),
}


def __getattr__(name: str) -> Any:
    """Load optional post-processing backends only when requested."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value

__all__ = [
    "DBNBeatTracker",
    "DBNDownbeatTracker",
    "DBNBarTracker",
    "ParticleFilterTracker",
    "Heydari1DStateSpaceTracker",
    "detect_beats",
    "detect_tempo",
    "peak_picking",
]
