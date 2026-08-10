from __future__ import annotations

import warnings

import numpy as np
import pytest

from mir_core.beats.schema import (
    ActivationFormatMismatchError,
    EventActivations,
    ExclusiveBeatDownbeatActivations,
    to_exclusive_beat_downbeat_activation_data,
)
from mir_core.evaluation.metrics import compute_beat_metrics
from mir_core.postprocessing.state_space_1d import Heydari1DStateSpaceTracker


def _decoder_activations(values: np.ndarray) -> ExclusiveBeatDownbeatActivations:
    return to_exclusive_beat_downbeat_activation_data(
        EventActivations(values)
    )


def _collision_regression_activations() -> ExclusiveBeatDownbeatActivations:
    """Deterministic activations that exposed the selected sweep collision."""

    rng = np.random.default_rng(4)
    frame_count = 800
    values = np.empty((frame_count, 2), dtype=np.float32)
    values[:, 0] = rng.uniform(0.0, 0.08, frame_count)
    values[:, 1] = rng.uniform(0.0, 0.03, frame_count)
    pulse_frames = np.sort(
        rng.choice(np.arange(10, frame_count - 10), size=70, replace=False)
    )
    for pulse_index, frame in enumerate(pulse_frames):
        strength = rng.uniform(0.55, 0.99)
        if pulse_index % 4 == 0:
            values[frame, 1] = strength
            values[frame, 0] = min(values[frame, 0], 1.0 - strength)
        else:
            values[frame, 0] = strength
            values[frame, 1] = min(values[frame, 1], 1.0 - strength)
    return _decoder_activations(values)


def _selected_sweep_candidate_params(*, peak_snap_mode: str) -> dict[str, object]:
    return {
        "fps": 50,
        "min_bpm": 80,
        "max_bpm": 240,
        "beats_per_bar": [2, 4],
        "lambda_b": 0.005,
        "lambda_d": 0.1,
        "observation_lambda": "N4",
        "downbeat_observation_lambda": "B60",
        "offset": 0.0,
        "ig_threshold": 0.4,
        "min_separation_mode": "min_interval",
        "peak_snap_window_frames": 12,
        "peak_snap_mode": peak_snap_mode,
        "peak_snap_threshold": None,
    }


def test_heydari_1d_state_space_tracker_returns_event_rows() -> None:
    fps = 50
    activations = np.zeros((500, 2), dtype=np.float32)
    for index, frame in enumerate(range(200, 500, 25)):
        activations[frame, 0] = 0.95
        activations[frame, 1] = 0.9 if index % 4 == 0 else 0.05

    decoded = Heydari1DStateSpaceTracker(fps=fps)(
        _decoder_activations(activations)
    )

    assert decoded.ndim == 2
    assert decoded.shape[1] == 4
    assert decoded.shape[0] > 0
    assert np.all(decoded[:, 0] >= Heydari1DStateSpaceTracker.OFFSET)
    assert set(np.unique(decoded[:, 1])).issubset({1.0, 2.0})


def test_heydari_1d_state_space_tracker_handles_short_tracks() -> None:
    activations = np.zeros((10, 2), dtype=np.float32)

    decoded = Heydari1DStateSpaceTracker(fps=50)(
        _decoder_activations(activations)
    )

    assert decoded.shape == (0, 4)


@pytest.mark.parametrize("parameter", ["lambda_b", "lambda_d"])
@pytest.mark.parametrize(
    "value",
    [0.0, 1.0, -0.01, 60.0, float("nan"), float("inf"), float("-inf")],
)
def test_heydari_1d_rejects_invalid_jump_reward_lambda(
    parameter: str,
    value: float,
) -> None:
    with pytest.raises(
        ValueError,
        match=rf"{parameter} must be finite and strictly between 0 and 1",
    ):
        Heydari1DStateSpaceTracker(**{parameter: value})


def test_heydari_1d_state_space_tracker_supports_peak_snap_options() -> None:
    fps = 50
    activations = np.zeros((500, 2), dtype=np.float32)
    for index, frame in enumerate(range(200, 500, 25)):
        activations[frame, 0] = 0.95
        activations[frame, 1] = 0.9 if index % 4 == 0 else 0.05

    decoded = Heydari1DStateSpaceTracker(
        fps=fps,
        min_bpm=120,
        max_bpm=260,
        observation_lambda="N4",
        min_separation_mode="local",
        peak_snap_window_frames=4,
        peak_snap_mode="causal",
        peak_snap_threshold=0.3,
    )(_decoder_activations(activations))

    assert decoded.ndim == 2
    assert decoded.shape[1] == 4
    assert decoded.shape[0] > 0
    assert np.all(np.diff(decoded[:, 0]) > 0)


def test_heydari_1d_reports_emission_frames_and_frame_costs() -> None:
    fps = 50
    activations = np.zeros((500, 2), dtype=np.float32)
    for frame in range(200, 500, 25):
        activations[frame, 0] = 0.95

    tracker = Heydari1DStateSpaceTracker(
        fps=fps,
        peak_snap_window_frames=4,
        peak_snap_mode="future",
    )
    decoded, emission_frames, frame_seconds = tracker.process_with_emission_frames(
        _decoder_activations(activations)
    )

    assert len(decoded) == len(emission_frames)
    assert frame_seconds.shape == (len(activations),)
    assert np.all(frame_seconds >= 0.0)
    event_frames = np.rint(decoded[:, 0] * fps).astype(int)
    assert np.all(emission_frames >= event_frames)

    decoded, source_frames, emission_frames, _ = tracker.process_with_event_timing(
        _decoder_activations(activations)
    )
    assert len(decoded) == len(source_frames) == len(emission_frames)
    assert np.all(emission_frames >= source_frames)


def test_heydari_1d_selected_candidate_suppresses_post_snap_collisions() -> None:
    tracker = Heydari1DStateSpaceTracker(
        **_selected_sweep_candidate_params(peak_snap_mode="center")
    )

    decoded, source_frames, emission_frames, _ = tracker.process_with_event_timing(
        _collision_regression_activations()
    )

    minimum_separation = 0.45 * tracker.st.min_interval / tracker.fps
    assert len(decoded) > 0
    assert tracker.suppressed_event_count > 0
    assert np.all(np.diff(decoded[:, 0]) > minimum_separation)
    assert len(np.unique(decoded[:, 0])) == len(decoded)
    assert len(decoded) == len(source_frames) == len(emission_frames)
    assert np.all(emission_frames >= source_frames)


def test_heydari_1d_collision_regression_is_metric_warning_free() -> None:
    tracker = Heydari1DStateSpaceTracker(
        **_selected_sweep_candidate_params(peak_snap_mode="center")
    )
    decoded = tracker.process(_collision_regression_activations())
    annotations = np.arange(0.36, 16.0, 0.5)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        metrics = compute_beat_metrics(decoded[:, 0], annotations)

    assert np.isfinite(metrics["information_gain"])


def test_heydari_1d_collision_regression_has_exact_causal_replay() -> None:
    activations = _collision_regression_activations()
    params = _selected_sweep_candidate_params(peak_snap_mode="past")
    batch_tracker = Heydari1DStateSpaceTracker(**params)
    batch, source_frames, emission_frames, _ = (
        batch_tracker.process_with_event_timing(activations)
    )

    stream_tracker = Heydari1DStateSpaceTracker(**params)
    rows = []
    for frame_index in range(len(activations.values)):
        emitted = stream_tracker.process_frame(
            ExclusiveBeatDownbeatActivations(
                activations.values[frame_index : frame_index + 1],
                definition=activations.definition,
                downbeats_available=activations.downbeats_available,
            )
        )
        if len(emitted):
            rows.extend(emitted)
    streamed = np.asarray(rows, dtype=float).reshape(-1, 4)

    minimum_separation = 0.45 * batch_tracker.st.min_interval / batch_tracker.fps
    np.testing.assert_allclose(streamed, batch)
    assert batch_tracker.suppressed_event_count > 0
    assert (
        stream_tracker.suppressed_event_count
        == batch_tracker.suppressed_event_count
    )
    assert np.all(np.diff(batch[:, 0]) > minimum_separation)
    assert np.all(emission_frames >= source_frames)

    # Batch decoding resets all state, including the refractory gate and its
    # diagnostic counter, and therefore replays identically.
    first_suppressed = batch_tracker.suppressed_event_count
    replay = batch_tracker.process(activations)
    np.testing.assert_allclose(replay, batch)
    assert batch_tracker.suppressed_event_count == first_suppressed


def test_heydari_1d_process_frame_matches_causal_batch_decode() -> None:
    fps = 50
    activations = np.zeros((500, 2), dtype=np.float32)
    for index, frame in enumerate(range(200, 500, 25)):
        activations[frame, 0] = 0.95
        activations[frame, 1] = 0.9 if index % 4 == 0 else 0.05
    tagged = _decoder_activations(activations)

    batch = Heydari1DStateSpaceTracker(
        fps=fps,
        peak_snap_window_frames=4,
        peak_snap_mode="past",
    ).process(tagged)
    tracker = Heydari1DStateSpaceTracker(
        fps=fps,
        peak_snap_window_frames=4,
        peak_snap_mode="past",
    )
    rows = []
    for frame_index in range(len(tagged.values)):
        emitted = tracker.process_frame(
            ExclusiveBeatDownbeatActivations(
                tagged.values[frame_index : frame_index + 1],
                definition=tagged.definition,
                downbeats_available=tagged.downbeats_available,
            )
        )
        if len(emitted):
            rows.extend(emitted)
    streamed = np.asarray(rows, dtype=float).reshape(-1, 4)

    np.testing.assert_allclose(streamed, batch)


def test_heydari_1d_process_frame_reset_is_repeatable() -> None:
    tracker = Heydari1DStateSpaceTracker(fps=50, offset=0.0)
    activation = _decoder_activations(np.asarray([[0.95, 0.0]], dtype=np.float32))

    first = tracker.process_frame(activation)
    tracker.reset()
    second = tracker.process_frame(activation)

    np.testing.assert_allclose(second, first)


def test_heydari_1d_process_frame_rejects_future_peak_snap() -> None:
    tracker = Heydari1DStateSpaceTracker(
        peak_snap_window_frames=2,
        peak_snap_mode="future",
    )

    with pytest.raises(ValueError, match="peak_snap_mode='past'"):
        tracker.process_frame(
            _decoder_activations(np.asarray([[0.5, 0.1]], dtype=np.float32))
        )


def test_heydari_1d_rejects_untagged_two_channel_arrays() -> None:
    with pytest.raises(ActivationFormatMismatchError, match="untagged"):
        Heydari1DStateSpaceTracker().process(
            np.zeros((10, 2), dtype=np.float32)
        )
