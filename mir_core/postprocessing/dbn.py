"""Dynamic Bayesian Network (DBN) beat trackers via madmom.

Classes:
    DBNBeatTracker     — beat tracking from 1D activation (wraps madmom DBN).
    DBNDownbeatTracker — joint beat+downbeat tracking from two activations.
    DBNBarTracker      — bar (meter) tracking from beat times + downbeat activations.
"""

from collections.abc import Sequence
from typing import TypeAlias

import madmom
import numpy as np

from mir_core.beats.schema import (
    EventActivations,
    require_event_activations,
    to_exclusive_beat_downbeat_activation_data,
)

NumericOrSequence: TypeAlias = float | int | Sequence[float] | Sequence[int]


class DBNBeatTracker:
    """
    Dynamic Bayesian Network beat tracker.

    Wraps madmom's DBNBeatTrackingProcessor for converting
    beat activation functions to discrete beat times.

    Args:
        min_bpm: Minimum tempo in BPM
        max_bpm: Maximum tempo in BPM
        fps: Frames per second of activation function
        num_tempi: Number of tempo states. ``None`` uses linear spacing.
        transition_lambda: Transition distribution concentration
        observation_lambda: Fraction of each beat interval assigned to beat states
        threshold: Beat detection threshold
        correct: Align decoded beats to local activation peaks
        online: Whether to use online (causal) processing
        num_threads: Decoder worker threads
    """

    def __init__(
        self,
        min_bpm: float = 55.0,
        max_bpm: float = 215.0,
        fps: int = 100,
        num_tempi: int | None = None,
        transition_lambda: float = 100.0,
        observation_lambda: int = 16,
        threshold: float = 0.05,
        correct: bool = True,
        online: bool = False,
        num_threads: int = 1,
    ):
        self.processor = madmom.features.beats.DBNBeatTrackingProcessor(
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            num_tempi=num_tempi,
            transition_lambda=transition_lambda,
            observation_lambda=observation_lambda,
            threshold=threshold,
            correct=correct,
            online=online,
            num_threads=num_threads,
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
        num_tempi: Number of tempo states, optionally specified per meter
        transition_lambda: Transition distribution concentration
        observation_lambda: Fraction of each beat interval assigned to beat states
        threshold: Trim leading/trailing regions below this activation level
        correct: Align decoded events to local activation peaks
        num_threads: Number of meters decoded in parallel
    """

    def __init__(
        self,
        beats_per_bar: Sequence[int] | None = None,
        min_bpm: NumericOrSequence = 55.0,
        max_bpm: NumericOrSequence = 215.0,
        fps: int = 100,
        num_tempi: int | Sequence[int] = 60,
        transition_lambda: NumericOrSequence = 100.0,
        observation_lambda: int = 16,
        threshold: float = 0.05,
        correct: bool = True,
        num_threads: int = 1,
    ):
        if beats_per_bar is None:
            beats_per_bar = [3, 4]
        beats_per_bar = list(beats_per_bar)
        self.processor = madmom.features.downbeats.DBNDownBeatTrackingProcessor(
            beats_per_bar=beats_per_bar,
            min_bpm=min_bpm,
            max_bpm=max_bpm,
            fps=fps,
            num_tempi=num_tempi,
            transition_lambda=transition_lambda,
            observation_lambda=observation_lambda,
            threshold=threshold,
            correct=correct,
            num_threads=num_threads,
        )
        self.fps = fps

    def __call__(
        self,
        activations: EventActivations[np.ndarray],
    ) -> np.ndarray:
        """
        Detect downbeats from canonical runtime-tagged activations.

        Args:
            activations: Canonical ``[all beats, downbeat]`` activation data.
                A bare ``(frames, 2)`` array is intentionally rejected because
                it cannot be distinguished from decoder-exclusive
                ``[beat-only, downbeat]`` data.

        Returns:
            Array of (time, beat_position) tuples
        """
        canonical = require_event_activations(activations)
        values = np.asarray(canonical.values, dtype=float)
        if values.ndim != 2:
            raise ValueError("DBN event activations must be a 2-dimensional array.")
        beat = np.asarray(canonical.all_beats, dtype=float)
        downbeat = np.asarray(canonical.downbeats, dtype=float)
        if not np.all(np.isfinite(beat)) or not np.all(np.isfinite(downbeat)):
            raise ValueError("DBN beat and downbeat activations must be finite.")
        if np.any(beat < 0) or np.any(beat > 1):
            raise ValueError("DBN beat activations must be probabilities in [0, 1].")
        if np.any(downbeat < 0) or np.any(downbeat > 1):
            raise ValueError(
                "DBN downbeat activations must be probabilities in [0, 1]."
            )

        exclusive = to_exclusive_beat_downbeat_activation_data(
            canonical,
            dtype=np.float64,
        )
        combined = exclusive.values
        if np.any(np.sum(combined, axis=1) > 1.0 + 1e-6):
            raise ValueError(
                "Exclusive DBN beat and downbeat probabilities must sum to at most 1."
            )

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
        beats_per_bar: tuple[int, ...] = (3, 4),
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
