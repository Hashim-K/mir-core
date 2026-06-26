"""
Post-processing utilities for beat detection.

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

from .dbn import DBNBeatTracker, DBNDownbeatTracker, DBNBarTracker
from .particle_filter import ParticleFilterTracker
from .peak_picking import detect_beats, detect_tempo, peak_picking
from .state_space_1d import Heydari1DStateSpaceTracker

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
