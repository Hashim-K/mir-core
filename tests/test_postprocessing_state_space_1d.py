from __future__ import annotations

import numpy as np

from mir_core.postprocessing.state_space_1d import Heydari1DStateSpaceTracker


def test_heydari_1d_state_space_tracker_returns_event_rows() -> None:
    fps = 50
    activations = np.zeros((500, 2), dtype=np.float32)
    for index, frame in enumerate(range(200, 500, 25)):
        activations[frame, 0] = 0.95
        activations[frame, 1] = 0.9 if index % 4 == 0 else 0.05

    decoded = Heydari1DStateSpaceTracker(fps=fps)(activations)

    assert decoded.ndim == 2
    assert decoded.shape[1] == 4
    assert decoded.shape[0] > 0
    assert np.all(decoded[:, 0] >= Heydari1DStateSpaceTracker.OFFSET)
    assert set(np.unique(decoded[:, 1])).issubset({1.0, 2.0})


def test_heydari_1d_state_space_tracker_handles_short_tracks() -> None:
    activations = np.zeros((10, 2), dtype=np.float32)

    decoded = Heydari1DStateSpaceTracker(fps=50)(activations)

    assert decoded.shape == (0, 4)


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
    )(activations)

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
        activations
    )

    assert len(decoded) == len(emission_frames)
    assert frame_seconds.shape == (len(activations),)
    assert np.all(frame_seconds >= 0.0)
    event_frames = np.rint(decoded[:, 0] * fps).astype(int)
    assert np.all(emission_frames >= event_frames)

    decoded, source_frames, emission_frames, _ = tracker.process_with_event_timing(
        activations
    )
    assert len(decoded) == len(source_frames) == len(emission_frames)
    assert np.all(emission_frames >= source_frames)
