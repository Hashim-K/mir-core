"""Heydari jump-reward inference on the compact 1D state space.

Adapted from Mojtaba Heydari's MIT-licensed ``jump_reward_inference`` package:
``thesis-docs/literature/codebases/beat-detection/1d-statespace``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


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

    Input activations must be shaped ``(frames, 2)`` with beat and downbeat
    probabilities. The return value follows the reference package:
    ``(time_seconds, label, local_tempo_bpm, local_meter)``, where label ``1``
    marks downbeats and label ``2`` marks non-downbeat beats.
    """

    MIN_BPM = 55.0
    MAX_BPM = 215.0
    LAMBDA_B = 0.01
    LAMBDA_D = 0.01
    OBSERVATION_LAMBDA = "B56"
    DOWNBEAT_OBSERVATION_LAMBDA = "B60"
    MIN_BEATS_PER_BAR = 1
    MAX_BEATS_PER_BAR = 4
    OFFSET = 4.0
    IG_THRESHOLD = 0.4

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
        observation_lambda: str = OBSERVATION_LAMBDA,
        downbeat_observation_lambda: str = DOWNBEAT_OBSERVATION_LAMBDA,
        offset: float = OFFSET,
        ig_threshold: float = IG_THRESHOLD,
        min_separation_mode: str = "min_interval",
        peak_snap_window_frames: int = 0,
        peak_snap_mode: str = "center",
        peak_snap_threshold: float | None = None,
    ) -> None:
        self.fps = int(fps)
        self.min_bpm = float(min_bpm)
        self.max_bpm = float(max_bpm)
        self.beats_per_bar = list(beats_per_bar or [])
        self.offset = float(offset)
        self.ig_threshold = float(ig_threshold)
        self.min_separation_mode = str(min_separation_mode).lower()
        if self.min_separation_mode in {"min", "fixed"}:
            self.min_separation_mode = "min_interval"
        elif self.min_separation_mode in {"local", "tempo", "estimated"}:
            self.min_separation_mode = "local_tempo"
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
        if self.peak_snap_window_frames < 0:
            raise ValueError("peak_snap_window_frames must be >= 0.")
        if self.peak_snap_mode not in {"center", "past", "future"}:
            raise ValueError("peak_snap_mode must be 'center', 'past', or 'future'.")

        min_interval = round(60.0 * self.fps / self.max_bpm)
        max_interval = round(60.0 * self.fps / self.min_bpm)
        self.st = BeatStateSpace1D(
            min_interval=min_interval,
            max_interval=max_interval,
            alpha=float(lambda_b),
        )
        self.st2 = DownbeatStateSpace1D(
            min_beats_per_bar=int(min_beats_per_bar),
            max_beats_per_bar=int(max_beats_per_bar),
            alpha=float(lambda_d),
            meter=self.beats_per_bar,
        )
        self.om = ObservationModel1D(self.st, observation_lambda)
        self.om2 = ObservationModel1D(self.st2, downbeat_observation_lambda)

    def __call__(self, activations: np.ndarray) -> np.ndarray:
        return self.process(activations)

    def process(self, activations: np.ndarray) -> np.ndarray:
        activations = np.asarray(activations, dtype=float)
        if activations.ndim != 2 or activations.shape[1] < 2:
            raise ValueError("Heydari1DStateSpaceTracker expects activations with shape (frames, 2).")
        if activations.size == 0:
            return np.empty((0, 4), dtype=float)

        frame_period = 1.0 / self.fps
        start_frame = int(self.offset / frame_period)
        if start_frame >= len(activations):
            return np.empty((0, 4), dtype=float)

        both_activations = activations[start_frame:, :2].copy()
        beat_activations = np.max(both_activations, axis=1)
        beat_activations[beat_activations < self.ig_threshold] = 0.03

        output: list[list[float]] = []
        beat_distribution = np.ones(self.st.num_states, dtype=float) * 0.8
        if len(beat_distribution) > 5:
            beat_distribution[5] = 1.0
        down_distribution = np.ones(self.st2.num_states, dtype=float) * 0.8
        local_tempo = 0
        meter = 0
        last_boundary_time = 0.0

        for frame_index, activation in enumerate(beat_activations, start=1):
            if np.max(self.st.jump_weights) > 1:
                self.st.jump_weights = 0.7 * self.st.jump_weights / np.max(self.st.jump_weights)
            beat_weight = self.st.jump_weights.copy()
            beat_jump_rewards1 = -beat_distribution * beat_weight
            beat_weight[beat_weight < 0.7] = 0
            jump_back_mass = float(np.sum(beat_distribution * beat_weight))
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
            peak_strength = float(activation)
            if self.peak_snap_window_frames:
                center = start_frame + frame_index - 1
                if self.peak_snap_mode == "past":
                    lo = max(0, center - self.peak_snap_window_frames)
                    hi = min(len(activations), center + 1)
                elif self.peak_snap_mode == "future":
                    lo = max(0, center)
                    hi = min(len(activations), center + self.peak_snap_window_frames + 1)
                else:
                    lo = max(0, center - self.peak_snap_window_frames)
                    hi = min(len(activations), center + self.peak_snap_window_frames + 1)
                if hi > lo:
                    local_strength = np.max(activations[lo:hi, :2], axis=1)
                    peak_offset = int(np.argmax(local_strength))
                    peak_strength = float(local_strength[peak_offset])
                    current_time = (lo + peak_offset) * frame_period

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
            if (
                near_beat_boundary
                and boundary_time - last_boundary_time > min_separation
                and enough_peak
            ):
                if np.max(self.st2.jump_weights) > 1:
                    self.st2.jump_weights = 0.2 * self.st2.jump_weights / np.max(
                        self.st2.jump_weights
                    )
                down_weight = self.st2.jump_weights.copy()
                down_jump_rewards1 = -down_distribution * down_weight
                down_weight[down_weight < 0.2] = 0
                down_jump_back_mass = float(np.sum(down_distribution * down_weight))
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
                if output and current_time <= output[-1][0]:
                    current_time = boundary_time
                output.append([current_time, label, float(local_tempo), float(meter)])
                last_boundary_time = boundary_time

        if not output:
            return np.empty((0, 4), dtype=float)
        return np.asarray(output, dtype=float)
