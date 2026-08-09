"""Dynamic Bayesian Network (DBN) beat trackers via madmom.

Classes:
    DBNBeatTracker     — beat tracking from 1D activation (wraps madmom DBN).
    DBNDownbeatTracker — joint beat+downbeat tracking from two activations.
    CausalDBNDownbeatTracker — online joint beat+downbeat forward decoder.
    DBNBarTracker      — bar (meter) tracking from beat times + downbeat activations.
"""

import time
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
        fps: float = 100.0,
        num_tempi: int | None = None,
        transition_lambda: float = 100.0,
        observation_lambda: int = 16,
        threshold: float = 0.05,
        correct: bool = True,
        online: bool = False,
        num_threads: int = 1,
    ):
        fps = float(fps)
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
        fps: float = 100.0,
        num_tempi: int | Sequence[int] = 60,
        transition_lambda: NumericOrSequence = 100.0,
        observation_lambda: int = 16,
        threshold: float = 0.05,
        correct: bool = True,
        num_threads: int = 1,
    ):
        fps = float(fps)
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


class CausalDBNDownbeatTracker:
    """Causal joint beat/downbeat DBN using HMM forward filtering.

    Madmom's :class:`DBNDownBeatTrackingProcessor` only implements offline
    Viterbi decoding; passing an ``online`` keyword to it is silently ignored.
    This tracker builds the same bar state spaces and observation model inside
    one multi-meter HMM, then advances its forward distribution exactly once
    per activation frame. It never revises an emitted event and never reads a
    future activation.

    ``correct`` must remain false because local-peak correction requires future
    context. Likewise, offline leading/trailing activation trimming is not
    available; ``threshold`` must remain zero.
    """

    def __init__(
        self,
        beats_per_bar: Sequence[int] | None = None,
        min_bpm: NumericOrSequence = 55.0,
        max_bpm: NumericOrSequence = 215.0,
        fps: float = 100.0,
        num_tempi: int | Sequence[int] = 60,
        transition_lambda: NumericOrSequence = 100.0,
        observation_lambda: int = 16,
        threshold: float = 0.0,
        correct: bool = False,
        meter_change_prob: float | None = 1e-7,
    ) -> None:
        from madmom.features.beats_hmm import (
            BarStateSpace,
            BarTransitionModel,
            MultiPatternStateSpace,
            MultiPatternTransitionModel,
            RNNDownBeatTrackingObservationModel,
        )
        from madmom.ml.hmm import HiddenMarkovModel

        meters = tuple(int(value) for value in (beats_per_bar or (3, 4)))
        if not meters or any(value < 1 for value in meters):
            raise ValueError("beats_per_bar must contain positive integers")
        if len(set(meters)) != len(meters):
            raise ValueError("beats_per_bar must not contain duplicate meters")

        fps_f = float(fps)
        if not np.isfinite(fps_f) or fps_f <= 0.0:
            raise ValueError("fps must be a positive finite number")
        observation_lambda_i = int(observation_lambda)
        if observation_lambda_i < 1 or observation_lambda_i != observation_lambda:
            raise ValueError("observation_lambda must be a positive integer")
        threshold_f = float(threshold)
        if threshold_f != 0.0:
            raise ValueError(
                "Causal joint DBN does not support offline activation trimming; "
                "threshold must be 0"
            )
        if bool(correct):
            raise ValueError(
                "Causal joint DBN cannot use future-looking peak correction; "
                "correct must be false"
            )
        if meter_change_prob is not None:
            meter_change_prob = float(meter_change_prob)
            if (
                not np.isfinite(meter_change_prob)
                or meter_change_prob <= 0.0
                or meter_change_prob >= 1.0
            ):
                raise ValueError(
                    "meter_change_prob must be None or finite and strictly "
                    "between 0 and 1"
                )

        def expand_numeric(
            value: NumericOrSequence,
            *,
            name: str,
            integer: bool = False,
        ) -> list[float | int]:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                raw = list(value)
            else:
                raw = [value]
            if len(raw) == 1:
                raw *= len(meters)
            if len(raw) != len(meters):
                raise ValueError(
                    f"{name} must be scalar or contain one value per meter"
                )
            converted: list[float | int] = []
            for item in raw:
                numeric = float(item)
                if not np.isfinite(numeric) or numeric <= 0.0:
                    raise ValueError(f"{name} values must be positive and finite")
                if integer:
                    if not numeric.is_integer():
                        raise ValueError(f"{name} values must be integers")
                    converted.append(int(numeric))
                else:
                    converted.append(numeric)
            return converted

        minimum_bpms = expand_numeric(min_bpm, name="min_bpm")
        maximum_bpms = expand_numeric(max_bpm, name="max_bpm")
        tempi = expand_numeric(num_tempi, name="num_tempi", integer=True)
        transition_lambdas = expand_numeric(
            transition_lambda,
            name="transition_lambda",
        )
        if any(
            float(low) >= float(high)
            for low, high in zip(minimum_bpms, maximum_bpms, strict=True)
        ):
            raise ValueError("min_bpm must be lower than max_bpm for every meter")

        state_spaces = []
        transition_models = []
        for meter, low, high, tempo_count, transition in zip(
            meters,
            minimum_bpms,
            maximum_bpms,
            tempi,
            transition_lambdas,
            strict=True,
        ):
            state_space = BarStateSpace(
                meter,
                60.0 * fps_f / float(high),
                60.0 * fps_f / float(low),
                int(tempo_count),
            )
            state_spaces.append(state_space)
            transition_models.append(
                BarTransitionModel(state_space, float(transition))
            )

        self.state_space = MultiPatternStateSpace(state_spaces)
        self.transition_model = MultiPatternTransitionModel(
            transition_models,
            transition_prob=meter_change_prob,
        )
        self.observation_model = RNNDownBeatTrackingObservationModel(
            self.state_space,
            observation_lambda_i,
        )
        # Give every meter equal prior mass instead of every state. Otherwise
        # meters with larger state spaces receive an accidental prior advantage.
        initial_distribution = np.zeros(self.state_space.num_states, dtype=float)
        for pattern in range(len(meters)):
            indices = np.flatnonzero(self.state_space.state_patterns == pattern)
            initial_distribution[indices] = 1.0 / (len(meters) * len(indices))
        self.hmm = HiddenMarkovModel(
            self.transition_model,
            self.observation_model,
            initial_distribution,
        )
        self.beats_per_bar = meters
        self.fps = fps_f
        self.min_bpm = tuple(float(value) for value in minimum_bpms)
        self.max_bpm = tuple(float(value) for value in maximum_bpms)
        self.num_tempi = tuple(int(value) for value in tempi)
        self.transition_lambda = tuple(
            float(value) for value in transition_lambdas
        )
        self.observation_lambda = observation_lambda_i
        self.meter_change_prob = meter_change_prob
        self._minimum_separation_frames = max(
            1,
            int(np.floor(60.0 * fps_f / max(self.max_bpm))),
        )
        self.reset()

    def reset(self) -> None:
        """Reset all HMM and event-emission state for a new stream."""

        self.hmm.reset()
        self._frame_index = 0
        self._last_event_frame = -self._minimum_separation_frames

    @staticmethod
    def _validated_exclusive_values(
        activations: EventActivations[np.ndarray],
    ) -> np.ndarray:
        canonical = require_event_activations(activations)
        values = np.asarray(canonical.values, dtype=float)
        if values.ndim != 2:
            raise ValueError(
                "Causal joint DBN expects a 2-dimensional activation array"
            )
        if not canonical.downbeats_available:
            raise ValueError("Causal joint DBN requires downbeat activations")
        if not np.all(np.isfinite(values)):
            raise ValueError("Causal joint DBN activations must be finite")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                "Causal joint DBN activations must be probabilities in [0, 1]"
            )
        return to_exclusive_beat_downbeat_activation_data(
            canonical,
            dtype=np.float64,
        ).values

    def _process_exclusive_frame(self, frame: np.ndarray) -> np.ndarray:
        # Madmom takes logarithms of all three observation densities. Exact
        # zeros are valid probabilities but produce noisy divide-by-zero
        # warnings; epsilon clipping is numerically equivalent for decoding.
        epsilon = np.finfo(float).eps
        observation = np.clip(
            np.asarray(frame, dtype=float).reshape(2),
            epsilon,
            1.0 - epsilon,
        )
        total = float(np.sum(observation))
        if total >= 1.0:
            observation *= (1.0 - epsilon) / total
        forward = self.hmm.forward(
            observation.reshape(1, 2),
            reset=self._frame_index == 0,
        )[0]
        pattern_probabilities = np.bincount(
            np.asarray(self.state_space.state_patterns, dtype=int),
            weights=forward,
            minlength=len(self.beats_per_bar),
        )
        pattern = int(np.argmax(pattern_probabilities))
        pattern_states = np.flatnonzero(
            self.state_space.state_patterns == pattern
        )
        state = int(pattern_states[np.argmax(forward[pattern_states])])
        pointer = int(self.observation_model.pointers[state])
        current_frame = self._frame_index
        self._frame_index += 1

        if pointer == 0 or (
            current_frame - self._last_event_frame
            < self._minimum_separation_frames
        ):
            return np.empty((0, 2), dtype=float)

        self._last_event_frame = current_frame
        beat_number = int(np.floor(self.state_space.state_positions[state])) + 1
        meter = self.beats_per_bar[pattern]
        beat_number = min(max(beat_number, 1), meter)
        return np.asarray(
            [[current_frame / self.fps, float(beat_number)]],
            dtype=float,
        )

    def process_frame(
        self,
        activation: EventActivations[np.ndarray],
    ) -> np.ndarray:
        """Advance one activation frame and return zero or one new event."""

        values = self._validated_exclusive_values(activation)
        if len(values) != 1:
            raise ValueError("process_frame requires exactly one activation frame")
        return self._process_exclusive_frame(values[0])

    def process_with_event_timing(
        self,
        activations: EventActivations[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode a stream with exact source/emission frames and frame costs."""

        values = self._validated_exclusive_values(activations)
        self.reset()
        rows: list[np.ndarray] = []
        source_frames: list[int] = []
        emission_frames: list[int] = []
        frame_seconds = np.zeros(len(values), dtype=float)
        for frame_index, frame in enumerate(values):
            started = time.perf_counter()
            emitted = self._process_exclusive_frame(frame)
            frame_seconds[frame_index] = time.perf_counter() - started
            if len(emitted):
                rows.extend(emitted)
                source_frames.extend([frame_index] * len(emitted))
                emission_frames.extend([frame_index] * len(emitted))
        decoded = np.vstack(rows) if rows else np.empty((0, 2), dtype=float)
        return (
            decoded,
            np.asarray(source_frames, dtype=np.int64),
            np.asarray(emission_frames, dtype=np.int64),
            frame_seconds,
        )

    def process(
        self,
        activations: EventActivations[np.ndarray],
    ) -> np.ndarray:
        """Decode a complete activation stream without using future frames."""

        decoded, _sources, _emissions, _costs = self.process_with_event_timing(
            activations
        )
        return decoded

    __call__ = process


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
        fps: float = 100.0,
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
