"""Heydari jump-reward inference on the compact 1D state space.

Adapted from Mojtaba Heydari's MIT-licensed ``jump_reward_inference`` package:
``thesis-docs/literature/codebases/beat-detection/1d-statespace``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from mir_core.beats.schema import (
    ExclusiveBeatDownbeatActivations,
    require_exclusive_beat_downbeat_activations,
    to_exclusive_beat_downbeat_activation_data,
)


STATE_SPACE_1D_AT_MODE = "at"
STATE_SPACE_1D_SB_MODE = "sb"
STATE_SPACE_1D_AT_TYPE = "1d-ss-at"
STATE_SPACE_1D_SB_TYPE = "1d-ss-sb"

_STATE_SPACE_1D_MODE_ALIASES = {
    "at": STATE_SPACE_1D_AT_MODE,
    "activation": STATE_SPACE_1D_AT_MODE,
    "activation_threshold": STATE_SPACE_1D_AT_MODE,
    "threshold": STATE_SPACE_1D_AT_MODE,
    "threshold_crossing": STATE_SPACE_1D_AT_MODE,
    "1d-ss-at": STATE_SPACE_1D_AT_MODE,
    "1d_ss_at": STATE_SPACE_1D_AT_MODE,
    "sb": STATE_SPACE_1D_SB_MODE,
    "state": STATE_SPACE_1D_SB_MODE,
    "boundary": STATE_SPACE_1D_SB_MODE,
    "state_boundary": STATE_SPACE_1D_SB_MODE,
    "1d-ss-sb": STATE_SPACE_1D_SB_MODE,
    "1d_ss_sb": STATE_SPACE_1D_SB_MODE,
}


def normalize_state_space_1d_mode(value: object = STATE_SPACE_1D_AT_MODE) -> str:
    """Return the canonical ``at`` or ``sb`` 1D-SS mode identifier."""

    token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return _STATE_SPACE_1D_MODE_ALIASES[token]
    except KeyError as exc:
        raise ValueError(
            "mode must be 'at' (activation-triggered) or 'sb' "
            "(state-boundary)."
        ) from exc


def state_space_1d_type(value: object = STATE_SPACE_1D_AT_MODE) -> str:
    """Return the stable experiment label for a 1D-SS mode."""

    return (
        STATE_SPACE_1D_AT_TYPE
        if normalize_state_space_1d_mode(value) == STATE_SPACE_1D_AT_MODE
        else STATE_SPACE_1D_SB_TYPE
    )


@dataclass
class StateSpace1D:
    """Compact one-dimensional state space for one rhythmic hierarchy."""

    min_interval: int
    max_interval: int

    def __post_init__(self) -> None:
        self.min_interval = int(np.round(self.min_interval))
        self.max_interval = int(np.round(self.max_interval))
        if self.min_interval < 1 or self.max_interval < self.min_interval:
            raise ValueError("Invalid 1D state-space interval range.")
        self.first_states = np.array([0])
        self.last_states = np.array([self.max_interval - 1])
        self.num_states = self.max_interval
        self.state_intervals = np.array([self.max_interval] * self.max_interval)
        self.state_positions = np.linspace(0, 1, self.num_states, endpoint=False)


class BeatStateSpace1D(StateSpace1D):
    """Beat/tempo 1D state space with jump-back reward weights."""

    def __init__(
        self,
        min_interval: int,
        max_interval: int,
        alpha: float = 0.01,
        tempo: float | None = None,
        fps: float | None = None,
    ) -> None:
        super().__init__(min_interval, max_interval)
        self.jump_weights = np.concatenate(
            (
                np.zeros(self.min_interval),
                np.array([alpha] * (self.max_interval - self.min_interval)),
            )
        )
        if tempo and fps:
            index = round(60.0 * fps / tempo) - self.min_interval
            if 0 <= index < len(self.jump_weights):
                self.jump_weights[index] = 1 - alpha


class DownbeatStateSpace1D(StateSpace1D):
    """Downbeat/meter 1D state space with jump-back reward weights."""

    def __init__(
        self,
        min_beats_per_bar: int,
        max_beats_per_bar: int,
        alpha: float = 0.01,
        meter: Sequence[int] | None = None,
    ) -> None:
        super().__init__(min_beats_per_bar, max_beats_per_bar)
        self.jump_weights = np.concatenate(
            (
                np.zeros(self.min_interval - 1),
                np.array([alpha] * (self.max_interval - self.min_interval + 1)),
            )
        )
        meter_values = list(meter or [])
        if meter_values:
            index = int(meter_values[0]) - self.min_interval + 1
            if 0 <= index < len(self.jump_weights):
                self.jump_weights[index] = 1 - alpha


class ObservationModel1D:
    """Beat/downbeat observation masks from the Heydari 1D inference code."""

    def __init__(self, state_space: StateSpace1D, observation_lambda: str) -> None:
        if not observation_lambda:
            raise ValueError("observation_lambda must be a non-empty string.")
        mode = observation_lambda[0].upper()
        value = observation_lambda[1:]

        if mode == "B":
            width = int(value)
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            pointers[state_space.state_positions < 1.0 / width] = 2
            self.pointers = pointers
            return

        if mode == "N":
            width = int(value)
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            for offset in range(width):
                border = np.asarray(state_space.first_states) + offset
                pointers[border[1:]] = 1
                pointers[border[0]] = 2
            self.pointers = pointers
            return

        raise ValueError(
            "Heydari 1D state-space observation_lambda currently supports "
            "'B<n>' and 'N<n>' modes."
        )


def _beat_densities(
    activation: float,
    observation_model: ObservationModel1D,
    state_model: StateSpace1D,
) -> np.ndarray:
    densities = np.zeros(state_model.num_states, dtype=float)
    densities[observation_model.pointers == 2] = activation
    densities[observation_model.pointers == 0] = 0.03
    return densities


def _downbeat_densities(
    activations: np.ndarray,
    observation_model: ObservationModel1D,
    state_model: StateSpace1D,
) -> np.ndarray:
    densities = np.zeros(state_model.num_states, dtype=float)
    densities[observation_model.pointers == 2] = float(activations[1])
    densities[observation_model.pointers == 0] = 0.00002
    return densities


def _renormalize(values: np.ndarray, scale: float = 0.8) -> np.ndarray:
    maximum = float(np.max(values)) if values.size else 0.0
    if maximum <= 0.0:
        return values
    return scale * values / maximum


class Heydari1DStateSpaceTracker:
    """Joint beat/downbeat inference using Heydari's 1D state space.

    Input activations must be tagged mutually-exclusive beat-only/downbeat
    probabilities. The return value follows the reference package:
    ``(time_seconds, label, local_tempo_bpm, local_meter)``, where label ``1``
    marks downbeats and label ``2`` marks non-downbeat beats.

    ``mode="sb"`` uses the inferred state boundary as the event decision.
    ``mode="at"`` (the default) emits on the current all-beat activation
    crossing while retaining the state-space tempo and meter estimates. AT
    never backdates an event or waits for a later peak.

    ``beat_jump_threshold`` exposes the reference implementation's fixed
    ``0.7`` gate on learned beat/tempo jump weights. AT retains that reference
    value by default. The repaired SB mode defaults to ``0`` so all positive
    learned jump mass can reset the state; negative weights are still removed
    because they cannot represent transition probabilities. Supplying an
    explicit value reproduces or tunes either transition rule.
    """

    MIN_BPM = 55.0
    MAX_BPM = 215.0
    LAMBDA_B = 0.01
    LAMBDA_D = 0.01
    REFERENCE_BEAT_JUMP_THRESHOLD = 0.7
    SB_BEAT_JUMP_THRESHOLD = 0.0
    BEAT_JUMP_THRESHOLD = REFERENCE_BEAT_JUMP_THRESHOLD
    OBSERVATION_LAMBDA = "B56"
    DOWNBEAT_OBSERVATION_LAMBDA = "B60"
    MIN_BEATS_PER_BAR = 1
    MAX_BEATS_PER_BAR = 4
    AT_OFFSET = 0.0
    SB_OFFSET = 4.0
    OFFSET = AT_OFFSET
    IG_THRESHOLD = 0.4
    EVENT_ACTIVATION_THRESHOLD = 0.5
    DOWNBEAT_ACTIVATION_THRESHOLD = 0.4
    MIN_SEPARATION_FRACTION = 0.45

    def __init__(
        self,
        *,
        fps: int = 50,
        min_bpm: float = MIN_BPM,
        max_bpm: float = MAX_BPM,
        beats_per_bar: Sequence[int] | None = None,
        min_beats_per_bar: int = MIN_BEATS_PER_BAR,
        max_beats_per_bar: int = MAX_BEATS_PER_BAR,
        lambda_b: float = LAMBDA_B,
        lambda_d: float = LAMBDA_D,
        beat_jump_threshold: float | None = None,
        observation_lambda: str = OBSERVATION_LAMBDA,
        downbeat_observation_lambda: str = DOWNBEAT_OBSERVATION_LAMBDA,
        offset: float | None = None,
        ig_threshold: float = IG_THRESHOLD,
        min_separation_mode: str = "min_interval",
        min_separation_fraction: float = MIN_SEPARATION_FRACTION,
        mode: str = STATE_SPACE_1D_AT_MODE,
        event_trigger_mode: str | None = None,
        event_activation_threshold: float = EVENT_ACTIVATION_THRESHOLD,
        downbeat_activation_threshold: float = DOWNBEAT_ACTIVATION_THRESHOLD,
        peak_snap_window_frames: int = 0,
        peak_snap_mode: str = "center",
        peak_snap_threshold: float | None = None,
    ) -> None:
        lambda_b = float(lambda_b)
        lambda_d = float(lambda_d)
        for name, value in (("lambda_b", lambda_b), ("lambda_d", lambda_d)):
            if not np.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(
                    f"{name} must be finite and strictly between 0 and 1."
                )

        self.fps = int(fps)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.beats_per_bar = list(beats_per_bar or [])
        # ``event_trigger_mode`` is the pre-mode API.  It remains a supported
        # compatibility alias, but new configs should persist only ``mode``.
        selected_mode = normalize_state_space_1d_mode(
            event_trigger_mode if event_trigger_mode is not None else mode
        )
        self.mode = selected_mode
        self.event_trigger_mode = (
            "activation_threshold"
            if selected_mode == STATE_SPACE_1D_AT_MODE
            else "state_boundary"
        )
        self.offset = float(
            (
                self.AT_OFFSET
                if selected_mode == STATE_SPACE_1D_AT_MODE
                else self.SB_OFFSET
            )
            if offset is None
            else offset
        )
        self.ig_threshold = float(ig_threshold)
        self.min_separation_mode = str(min_separation_mode).lower()
        if self.min_separation_mode in {"min", "fixed"}:
            self.min_separation_mode = "min_interval"
        elif self.min_separation_mode in {"local", "tempo", "estimated"}:
            self.min_separation_mode = "local_tempo"
        self.min_separation_fraction = float(min_separation_fraction)
        self.beat_jump_threshold = float(
            (
                self.REFERENCE_BEAT_JUMP_THRESHOLD
                if selected_mode == STATE_SPACE_1D_AT_MODE
                else self.SB_BEAT_JUMP_THRESHOLD
            )
            if beat_jump_threshold is None
            else beat_jump_threshold
        )
        self.event_activation_threshold = float(event_activation_threshold)
        self.downbeat_activation_threshold = float(downbeat_activation_threshold)
        self.peak_snap_window_frames = int(peak_snap_window_frames)
        self.peak_snap_mode = str(peak_snap_mode).lower()
        if self.peak_snap_mode == "causal":
            self.peak_snap_mode = "past"
        self.peak_snap_threshold = (
            None if peak_snap_threshold is None else float(peak_snap_threshold)
        )
        if self.min_separation_mode not in {"min_interval", "local_tempo"}:
            raise ValueError(
                "min_separation_mode must be 'min_interval' or 'local_tempo'."
            )
        if not np.isfinite(self.min_separation_fraction) or not (
            0.0 < self.min_separation_fraction <= 1.0
        ):
            raise ValueError("min_separation_fraction must be in (0, 1].")
        if not np.isfinite(self.beat_jump_threshold) or not (
            0.0 <= self.beat_jump_threshold <= 1.0
        ):
            raise ValueError("beat_jump_threshold must be finite and in [0, 1].")
        for name, value in (
            ("event_activation_threshold", self.event_activation_threshold),
            ("downbeat_activation_threshold", self.downbeat_activation_threshold),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1].")
        if self.peak_snap_window_frames < 0:
            raise ValueError("peak_snap_window_frames must be >= 0.")
        if self.peak_snap_mode not in {"center", "past", "future"}:
            raise ValueError("peak_snap_mode must be 'center', 'past', or 'future'.")
        if (
            self.event_trigger_mode == "activation_threshold"
            and self.peak_snap_window_frames
        ):
            raise ValueError(
                "activation_threshold events are emitted immediately and cannot "
                "use retrospective peak snapping."
            )

        min_interval = round(60.0 * self.fps / self.max_bpm)
        max_interval = round(60.0 * self.fps / self.min_bpm)
        self.st = BeatStateSpace1D(
            min_interval=min_interval,
            max_interval=max_interval,
            alpha=lambda_b,
        )
        self.st2 = DownbeatStateSpace1D(
            min_beats_per_bar=int(min_beats_per_bar),
            max_beats_per_bar=int(max_beats_per_bar),
            alpha=lambda_d,
            meter=self.beats_per_bar,
        )
        self.om = ObservationModel1D(self.st, observation_lambda)
        self.om2 = ObservationModel1D(self.st2, downbeat_observation_lambda)
        self._initial_beat_jump_weights = self.st.jump_weights.copy()
        self._initial_downbeat_jump_weights = self.st2.jump_weights.copy()
        self.reset()

    def reset(self) -> None:
        """Reset all jump-reward and event state for a new causal stream."""

        self.st.jump_weights = self._initial_beat_jump_weights.copy()
        self.st2.jump_weights = self._initial_downbeat_jump_weights.copy()
        self._beat_distribution = np.ones(self.st.num_states, dtype=float) * 0.8
        if len(self._beat_distribution) > 5:
            self._beat_distribution[5] = 1.0
        self._down_distribution = np.ones(self.st2.num_states, dtype=float) * 0.8
        self._local_tempo = 0
        self._meter = 0
        # A short start-up refractory suppresses the common priming activation
        # at frame zero; there is no preceding audio from which a live system
        # could have made a reliable onset decision.
        self._last_boundary_time = 0.0
        self._last_emitted_event_time = float("-inf")
        self._suppressed_event_count = 0
        self._input_frame_index = 0
        self._active_frame_index = 0
        self._previous_event_activation_value = 0.0
        self._activation_history: list[np.ndarray] = []

    @property
    def suppressed_event_count(self) -> int:
        """Number of event candidates rejected by the output refractory gate."""

        return self._suppressed_event_count

    def _claim_event_time(
        self,
        event_time: float,
        minimum_separation: float,
    ) -> bool:
        """Claim a final timestamp if it is safely after the last output.

        Boundary separation is checked before peak snapping. Snapping can move
        two otherwise valid boundaries onto the same (or an earlier) activation
        peak, so the externally visible stream needs its own final gate. This
        mirrors the last-emitted-event guards in the causal DBN and particle
        filter trackers.
        """

        if event_time - self._last_emitted_event_time <= minimum_separation:
            self._suppressed_event_count += 1
            return False
        self._last_emitted_event_time = event_time
        return True

    @staticmethod
    def _validated_exclusive_values(
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> np.ndarray:
        supplied = require_exclusive_beat_downbeat_activations(activations)
        exclusive = to_exclusive_beat_downbeat_activation_data(
            supplied,
            dtype=np.float64,
        )
        values = exclusive.values
        if values.ndim != 2:
            raise ValueError(
                "Heydari1DStateSpaceTracker expects a 2-dimensional "
                "activation array."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("Heydari 1D activations must be finite.")
        if np.any(values < 0.0) or np.any(values > 1.0):
            raise ValueError(
                "Heydari 1D activations must be probabilities in [0, 1]."
            )
        if np.any(np.sum(values, axis=1) > 1.0 + 1e-6):
            raise ValueError(
                "Exclusive beat-only and downbeat probabilities must sum to "
                "at most 1."
            )
        return values

    def process_frame(
        self,
        activation: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> np.ndarray:
        """Advance one activation frame and return zero or one new event.

        This is the deployment API. It preserves the tracker distributions and
        jump rewards between calls and therefore has constant work per audio
        frame. Future/centred peak snapping is deliberately rejected because it
        cannot be emitted causally. Activation-triggered output requires a
        zero-frame window because even retrospective snapping would backdate
        the event seen by a live actuator.
        """

        values = self._validated_exclusive_values(activation)
        if len(values) != 1:
            raise ValueError("process_frame requires exactly one activation frame")
        if self.peak_snap_window_frames and self.peak_snap_mode != "past":
            raise ValueError(
                "process_frame requires peak_snap_mode='past' when peak snapping "
                "is enabled"
            )

        decoded, _source_frame, _emission_frame = self._process_exclusive_frame(
            values[0]
        )
        return decoded

    def _process_exclusive_frame(
        self,
        activation: np.ndarray,
    ) -> tuple[np.ndarray, int | None, int | None]:
        """Advance one already-validated frame through the causal decoder."""

        current_activations = np.asarray(activation, dtype=float).copy()
        source_frame_index = self._input_frame_index
        self._input_frame_index += 1
        raw_activation_value = float(np.max(current_activations))
        event_activation_value = float(np.sum(current_activations))
        previous_event_activation_value = self._previous_event_activation_value
        self._previous_event_activation_value = event_activation_value
        if self.peak_snap_window_frames:
            self._activation_history.append(current_activations)
        start_frame = int(self.offset * self.fps)
        if source_frame_index < start_frame:
            return np.empty((0, 4), dtype=float), None, None

        self._active_frame_index += 1
        frame_index = self._active_frame_index
        frame_period = 1.0 / self.fps
        activation_value = raw_activation_value
        if activation_value < self.ig_threshold:
            activation_value = 0.03

        if np.max(self.st.jump_weights) > 1:
            self.st.jump_weights = (
                0.7 * self.st.jump_weights / np.max(self.st.jump_weights)
            )
        beat_weight = self.st.jump_weights.copy()
        beat_jump_rewards1 = -self._beat_distribution * beat_weight
        beat_weight[beat_weight < self.beat_jump_threshold] = 0
        # Keep Python's left-to-right summation from the reference package.
        # The learned weights can be close enough that NumPy's pairwise
        # reduction changes the reported tempo argmax on some tracks.
        jump_back_mass = float(sum(self._beat_distribution * beat_weight))
        self._beat_distribution = np.roll(
            self._beat_distribution * (1 - beat_weight),
            1,
        )
        self._beat_distribution[0] += jump_back_mass

        if activation_value > self.ig_threshold:
            obs = _beat_densities(activation_value, self.om, self.st)
            old_distribution = self._beat_distribution.copy()
            self._beat_distribution = old_distribution * obs
            if np.min(self._beat_distribution) < 1e-5:
                self._beat_distribution = _renormalize(self._beat_distribution)
            beat_max = int(np.argmax(self._beat_distribution))
            beat_jump_rewards = self._beat_distribution - old_distribution
            beat_jump_rewards[: self.st.min_interval - 1] = 0
            max_negative_reward = float(np.max(-beat_jump_rewards))
            if max_negative_reward != 0:
                self.st.jump_weights += -4 * beat_jump_rewards / max_negative_reward
            self._local_tempo = round(
                self.fps * 60 / (np.argmax(self.st.jump_weights) + 1)
            )
        else:
            beat_jump_rewards1[: self.st.min_interval - 1] = 0
            self.st.jump_weights += 2 * beat_jump_rewards1
            self.st.jump_weights[: self.st.min_interval - 1] = 0
            beat_max = int(np.argmax(self._beat_distribution))

        boundary_time = frame_index * frame_period + self.offset
        event_time = boundary_time
        event_source_frame_index = source_frame_index
        peak_strength = activation_value
        if self.peak_snap_window_frames:
            lo = max(0, source_frame_index - self.peak_snap_window_frames)
            history = np.asarray(self._activation_history[lo : source_frame_index + 1])
            local_strength = np.max(history, axis=1)
            peak_offset = int(np.argmax(local_strength))
            peak_strength = float(local_strength[peak_offset])
            event_source_frame_index = lo + peak_offset
            event_time = (lo + peak_offset) * frame_period

        activation_crossing = (
            event_activation_value >= self.event_activation_threshold
            and previous_event_activation_value < self.event_activation_threshold
        )
        if self.event_trigger_mode == "activation_threshold":
            # This frame's activation is available now.  Its musical timestamp
            # is the frame timestamp, while the separately reported emission
            # frame captures the one-hop software availability delay used by
            # RT-F1 and the actuator benchmark.
            event_time = source_frame_index * frame_period
            event_source_frame_index = source_frame_index
            peak_strength = event_activation_value

        if self.min_separation_mode == "local_tempo":
            separation_interval = max(
                self.st.min_interval,
                int(np.argmax(self.st.jump_weights) + 1),
            )
        else:
            separation_interval = self.st.min_interval
        min_separation = (
            self.min_separation_fraction * frame_period * separation_interval
        )
        near_beat_boundary = beat_max < int(0.07 / frame_period) + 1
        event_triggered = (
            near_beat_boundary
            if self.event_trigger_mode == "state_boundary"
            else activation_crossing
        )
        enough_peak = self.event_trigger_mode == "activation_threshold" or (
            self.peak_snap_threshold is None
            or peak_strength >= self.peak_snap_threshold
        )
        if not (
            event_triggered
            and boundary_time - self._last_boundary_time > min_separation
            and enough_peak
        ):
            return np.empty((0, 4), dtype=float), None, None

        # Record the internal boundary even if peak snapping maps it onto a
        # previously emitted event. Otherwise the same boundary can be retried
        # on successive frames and create a cluster of duplicate candidates.
        self._last_boundary_time = boundary_time
        if not self._claim_event_time(event_time, min_separation):
            return np.empty((0, 4), dtype=float), None, None

        if np.max(self.st2.jump_weights) > 1:
            self.st2.jump_weights = (
                0.2 * self.st2.jump_weights / np.max(self.st2.jump_weights)
            )
        down_weight = self.st2.jump_weights.copy()
        down_jump_rewards1 = -self._down_distribution * down_weight
        down_weight[down_weight < 0.2] = 0
        down_jump_back_mass = float(sum(self._down_distribution * down_weight))
        self._down_distribution = np.roll(
            self._down_distribution * (1 - down_weight),
            1,
        )
        self._down_distribution[0] += down_jump_back_mass

        if current_activations[1] > 0.00002:
            obs2 = _downbeat_densities(current_activations, self.om2, self.st2)
            old_down_distribution = self._down_distribution.copy()
            self._down_distribution = old_down_distribution * obs2
            if np.min(self._down_distribution) < 1e-5:
                self._down_distribution = _renormalize(self._down_distribution)
            down_max = int(np.argmax(self._down_distribution))
            down_jump_rewards = self._down_distribution - old_down_distribution
            down_jump_rewards[: self.st2.max_interval - 1] = 0
            max_negative_reward = float(np.max(-down_jump_rewards))
            if max_negative_reward != 0:
                self.st2.jump_weights += (
                    -0.3 * down_jump_rewards / max_negative_reward
                )
            self._meter = int(np.argmax(self.st2.jump_weights) + 1)
        else:
            down_jump_rewards1[: self.st2.min_interval - 1] = 0
            self.st2.jump_weights += 2 * down_jump_rewards1
            self.st2.jump_weights[: self.st2.min_interval - 1] = 0
            down_max = int(np.argmax(self._down_distribution))

        if self.event_trigger_mode == "activation_threshold":
            label = (
                1.0
                if float(current_activations[1]) >= self.downbeat_activation_threshold
                else 2.0
            )
        else:
            label = 1.0 if down_max == int(self.st2.first_states[0]) else 2.0
        return (
            np.asarray(
                [[event_time, label, float(self._local_tempo), float(self._meter)]],
                dtype=float,
            ),
            event_source_frame_index,
            source_frame_index,
        )

    def __call__(
        self,
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> np.ndarray:
        return self.process(activations)

    def process(
        self,
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> np.ndarray:
        decoded, _, _ = self.process_with_emission_frames(activations)
        return decoded

    def process_with_emission_frames(
        self,
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode activations and expose causal emission frames and frame costs.

        Emission frames identify when each returned event became available to
        the caller. They can differ from event timestamps when peak snapping
        moves an event to an earlier activation peak.
        """
        decoded, _, emission_frames, frame_seconds = self.process_with_event_timing(
            activations
        )
        return decoded, emission_frames, frame_seconds

    def process_with_event_timing(
        self,
        activations: ExclusiveBeatDownbeatActivations[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode events with source-activation and callback-emission frames."""
        values = self._validated_exclusive_values(activations)
        self.reset()
        if values.size == 0:
            return (
                np.empty((0, 4), dtype=float),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=float),
            )

        # Causal batch replay and live frame-by-frame inference deliberately
        # share the exact same decoder and output-refractory path. Centred and
        # future snapping remain available only for explicit offline analysis.
        if not self.peak_snap_window_frames or self.peak_snap_mode == "past":
            output: list[np.ndarray] = []
            source_frames: list[int] = []
            emission_frames: list[int] = []
            frame_seconds = np.zeros(len(values), dtype=float)
            for frame_index, frame in enumerate(values):
                frame_started = time.perf_counter()
                decoded, source_frame, emission_frame = self._process_exclusive_frame(
                    frame
                )
                frame_seconds[frame_index] = time.perf_counter() - frame_started
                if len(decoded):
                    output.extend(decoded)
                    if source_frame is None or emission_frame is None:
                        raise RuntimeError("Emitted event is missing timing metadata")
                    source_frames.append(source_frame)
                    emission_frames.append(emission_frame)
            return (
                np.vstack(output) if output else np.empty((0, 4), dtype=float),
                np.asarray(source_frames, dtype=np.int64),
                np.asarray(emission_frames, dtype=np.int64),
                frame_seconds,
            )

        frame_period = 1.0 / self.fps
        start_frame = int(self.offset / frame_period)
        if start_frame >= len(values):
            return (
                np.empty((0, 4), dtype=float),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.zeros(len(values), dtype=float),
            )

        both_activations = values[start_frame:].copy()
        beat_activations = np.max(both_activations, axis=1)
        beat_activations[beat_activations < self.ig_threshold] = 0.03

        output: list[list[float]] = []
        source_frames: list[int] = []
        emission_frames: list[int] = []
        frame_seconds = np.zeros(len(values), dtype=float)
        beat_distribution = np.ones(self.st.num_states, dtype=float) * 0.8
        if len(beat_distribution) > 5:
            beat_distribution[5] = 1.0
        down_distribution = np.ones(self.st2.num_states, dtype=float) * 0.8
        local_tempo = 0
        meter = 0
        last_boundary_time = 0.0

        for frame_index, activation in enumerate(beat_activations, start=1):
            frame_started = time.perf_counter()
            source_frame_index = start_frame + frame_index - 1
            if np.max(self.st.jump_weights) > 1:
                self.st.jump_weights = 0.7 * self.st.jump_weights / np.max(self.st.jump_weights)
            beat_weight = self.st.jump_weights.copy()
            beat_jump_rewards1 = -beat_distribution * beat_weight
            beat_weight[beat_weight < self.beat_jump_threshold] = 0
            jump_back_mass = float(sum(beat_distribution * beat_weight))
            beat_distribution = np.roll(beat_distribution * (1 - beat_weight), 1)
            beat_distribution[0] += jump_back_mass

            if activation > self.ig_threshold:
                obs = _beat_densities(activation, self.om, self.st)
                old_distribution = beat_distribution.copy()
                beat_distribution = old_distribution * obs
                if np.min(beat_distribution) < 1e-5:
                    beat_distribution = _renormalize(beat_distribution)
                beat_max = int(np.argmax(beat_distribution))
                beat_jump_rewards = beat_distribution - old_distribution
                beat_jump_rewards[: self.st.min_interval - 1] = 0
                max_negative_reward = float(np.max(-beat_jump_rewards))
                if max_negative_reward != 0:
                    self.st.jump_weights += -4 * beat_jump_rewards / max_negative_reward
                local_tempo = round(self.fps * 60 / (np.argmax(self.st.jump_weights) + 1))
            else:
                beat_jump_rewards1[: self.st.min_interval - 1] = 0
                self.st.jump_weights += 2 * beat_jump_rewards1
                self.st.jump_weights[: self.st.min_interval - 1] = 0
                beat_max = int(np.argmax(beat_distribution))

            boundary_time = frame_index * frame_period + self.offset
            current_time = boundary_time
            event_source_frame_index = source_frame_index
            emission_frame_index = source_frame_index
            peak_strength = float(activation)
            if self.peak_snap_window_frames:
                center = start_frame + frame_index - 1
                if self.peak_snap_mode == "past":
                    lo = max(0, center - self.peak_snap_window_frames)
                    hi = min(len(values), center + 1)
                elif self.peak_snap_mode == "future":
                    lo = max(0, center)
                    hi = min(len(values), center + self.peak_snap_window_frames + 1)
                else:
                    lo = max(0, center - self.peak_snap_window_frames)
                    hi = min(len(values), center + self.peak_snap_window_frames + 1)
                if hi > lo:
                    local_strength = np.max(values[lo:hi], axis=1)
                    peak_offset = int(np.argmax(local_strength))
                    peak_strength = float(local_strength[peak_offset])
                    current_time = (lo + peak_offset) * frame_period
                    event_source_frame_index = lo + peak_offset
                    if self.peak_snap_mode in {"center", "future"}:
                        emission_frame_index = max(emission_frame_index, hi - 1)

            if self.min_separation_mode == "local_tempo":
                separation_interval = max(
                    self.st.min_interval,
                    int(np.argmax(self.st.jump_weights) + 1),
                )
            else:
                separation_interval = self.st.min_interval
            min_separation = 0.45 * frame_period * separation_interval
            near_beat_boundary = beat_max < int(0.07 / frame_period) + 1
            enough_peak = (
                self.peak_snap_threshold is None
                or peak_strength >= self.peak_snap_threshold
            )
            boundary_candidate = (
                near_beat_boundary
                and boundary_time - last_boundary_time > min_separation
                and enough_peak
            )
            if boundary_candidate:
                last_boundary_time = boundary_time
            if boundary_candidate and self._claim_event_time(
                current_time, min_separation
            ):
                if np.max(self.st2.jump_weights) > 1:
                    self.st2.jump_weights = 0.2 * self.st2.jump_weights / np.max(
                        self.st2.jump_weights
                    )
                down_weight = self.st2.jump_weights.copy()
                down_jump_rewards1 = -down_distribution * down_weight
                down_weight[down_weight < 0.2] = 0
                down_jump_back_mass = float(sum(down_distribution * down_weight))
                down_distribution = np.roll(down_distribution * (1 - down_weight), 1)
                down_distribution[0] += down_jump_back_mass

                current_activations = both_activations[frame_index - 1]
                if current_activations[1] > 0.00002:
                    obs2 = _downbeat_densities(current_activations, self.om2, self.st2)
                    old_down_distribution = down_distribution.copy()
                    down_distribution = old_down_distribution * obs2
                    if np.min(down_distribution) < 1e-5:
                        down_distribution = _renormalize(down_distribution)
                    down_max = int(np.argmax(down_distribution))
                    down_jump_rewards = down_distribution - old_down_distribution
                    down_jump_rewards[: self.st2.max_interval - 1] = 0
                    max_negative_reward = float(np.max(-down_jump_rewards))
                    if max_negative_reward != 0:
                        self.st2.jump_weights += -0.3 * down_jump_rewards / max_negative_reward
                    meter = int(np.argmax(self.st2.jump_weights) + 1)
                else:
                    down_jump_rewards1[: self.st2.min_interval - 1] = 0
                    self.st2.jump_weights += 2 * down_jump_rewards1
                    self.st2.jump_weights[: self.st2.min_interval - 1] = 0
                    down_max = int(np.argmax(down_distribution))

                label = 1.0 if down_max == int(self.st2.first_states[0]) else 2.0
                output.append([current_time, label, float(local_tempo), float(meter)])
                source_frames.append(event_source_frame_index)
                emission_frames.append(emission_frame_index)
            frame_seconds[source_frame_index] = time.perf_counter() - frame_started

        if not output:
            decoded = np.empty((0, 4), dtype=float)
        else:
            decoded = np.asarray(output, dtype=float)
        return (
            decoded,
            np.asarray(source_frames, dtype=np.int64),
            np.asarray(emission_frames, dtype=np.int64),
            frame_seconds,
        )
