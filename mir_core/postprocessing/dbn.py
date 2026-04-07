"""
Dynamic Bayesian Network (DBN) beat trackers via madmom.

Classes:
    DBNBeatTracker     — beat tracking from 1D activation (wraps madmom DBN).
    DBNDownbeatTracker — joint beat+downbeat tracking from two activations.
    DBNBarTracker      — bar (meter) tracking from beat times + downbeat activations.
"""

from typing import Optional, Tuple, List

import numpy as np
import madmom


class DBNBeatTracker:
    """
    Dynamic Bayesian Network beat tracker.

    Wraps madmom's DBNBeatTrackingProcessor for converting
    beat activation functions to discrete beat times.

    Args:
        min_bpm: Minimum tempo in BPM
        max_bpm: Maximum tempo in BPM
        fps: Frames per second of activation function
        transition_lambda: Transition distribution concentration
        threshold: Beat detection threshold
        online: Whether to use online (causal) processing
    """

    def __init__(
        self,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        fps: int = 100,
        transition_lambda: float = 100.0,
        threshold: float = 0.05,
        online: bool = False,
    ):
        self.processor = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            transition_lambda=transition_lambda,
            threshold=threshold,
            online=online,
        )
        self.fps = fps

    def __call__(self, activations: np.ndarray) -> np.ndarray:
        """
        Detect beats from activation function.

        Args:
            activations: Beat activation function (1D array)

        Returns:
            Array of beat times in seconds
        """
        if activations.size <= 1:
            return np.array([])

        return self.processor(activations)


class DBNDownbeatTracker:
    """
    Dynamic Bayesian Network downbeat tracker.

    Uses combined beat and downbeat activations.

    Args:
        beats_per_bar: Possible time signatures (beats per bar)
        min_bpm: Minimum tempo in BPM
        max_bpm: Maximum tempo in BPM
        fps: Frames per second
        transition_lambda: Transition distribution concentration
    """

    def __init__(
        self,
        beats_per_bar: Optional[List[int]] = None,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        fps: int = 100,
        transition_lambda: float = 100.0,
    ):
        if beats_per_bar is None:
            beats_per_bar = [3, 4]
        self.processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=beats_per_bar,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            transition_lambda=transition_lambda,
        )
        self.fps = fps

    def __call__(
        self,
        beat_activations: np.ndarray,
        downbeat_activations: np.ndarray
    ) -> np.ndarray:
        """
        Detect downbeats from activations.

        Args:
            beat_activations: Beat activation function
            downbeat_activations: Downbeat activation function

        Returns:
            Array of (time, beat_position) tuples
        """
        # Combine activations: [beat-only, downbeat]
        combined = np.vstack((
            np.maximum(beat_activations - downbeat_activations, 0),
            downbeat_activations
        )).T

        return self.processor(combined)


class DBNBarTracker:
    """
    Dynamic Bayesian Network bar tracker.

    First tracks beats, then infers downbeat positions.

    Args:
        beats_per_bar: Possible time signatures
        meter_change_prob: Probability of meter change
        observation_weight: Weight for observations
    """

    def __init__(
        self,
        beats_per_bar: Tuple[int, ...] = (3, 4),
        meter_change_prob: float = 1e-3,
        observation_weight: float = 4.0,
    ):
        self.processor = madmom.features.downbeats.DBNBarTrackingProcessor(
            beats_per_bar=beats_per_bar,
            meter_change_prob=meter_change_prob,
            observation_weight=observation_weight,
        )

    def __call__(
        self,
        beat_times: np.ndarray,
        downbeat_activations: np.ndarray,
        fps: int = 100,
    ) -> np.ndarray:
        """
        Track bars from beat times and downbeat activations.

        Args:
            beat_times: Detected beat times in seconds
            downbeat_activations: Downbeat activation function
            fps: Frames per second

        Returns:
            Array of (time, beat_position) tuples
        """
        from scipy.ndimage import maximum_filter1d

        # Get downbeat activations at beat positions
        beat_idx = (beat_times * fps).astype(np.int32)
        beat_idx = np.clip(beat_idx, 0, len(downbeat_activations) - 1)

        # Widen activations
        db_act = maximum_filter1d(downbeat_activations, size=3)
        db_act_at_beats = db_act[beat_idx]

        # Combine beat times with downbeat activations
        bar_act = np.vstack((beat_times, db_act_at_beats)).T

        try:
            return self.processor(bar_act)
        except (IndexError, ValueError):
            return np.empty((0, 2))
