from __future__ import annotations

import numpy as np
import pytest

from mir_core.beats.schema import (
    ActivationFormatMismatchError,
    EventActivations,
    frame_class_activations_to_event_activations,
)
from mir_core.postprocessing import dbn

BEAST_FPS = 44100 / 1024


class _FakeProcessor:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.activations = None

    def __call__(self, activations):
        self.activations = np.asarray(activations)
        return np.asarray([[0.5, 1.0]])


def test_beat_tracker_exposes_accuracy_parameters(monkeypatch) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.beats,
        "DBNBeatTrackingProcessor",
        _FakeProcessor,
    )

    tracker = dbn.DBNBeatTracker(
        min_bpm=70,
        max_bpm=180,
        fps=BEAST_FPS,
        num_tempi=90,
        transition_lambda=50,
        observation_lambda=8,
        threshold=0.03,
        correct=False,
        online=True,
        num_threads=2,
    )

    assert tracker.processor.kwargs == {
        "min_bpm": 70,
        "max_bpm": 180,
        "fps": BEAST_FPS,
        "num_tempi": 90,
        "transition_lambda": 50,
        "observation_lambda": 8,
        "threshold": 0.03,
        "correct": False,
        "online": True,
        "num_threads": 2,
    }
    assert tracker.fps == pytest.approx(BEAST_FPS)


def test_downbeat_tracker_exposes_accuracy_parameters(monkeypatch) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )

    tracker = dbn.DBNDownbeatTracker(
        beats_per_bar=[2, 4],
        min_bpm=[65, 70],
        max_bpm=[160, 180],
        fps=BEAST_FPS,
        num_tempi=[30, 60],
        transition_lambda=[25, 100],
        observation_lambda=8,
        threshold=0.01,
        correct=False,
        num_threads=2,
    )

    assert tracker.processor.kwargs == {
        "beats_per_bar": [2, 4],
        "min_bpm": [65, 70],
        "max_bpm": [160, 180],
        "fps": BEAST_FPS,
        "num_tempi": [30, 60],
        "transition_lambda": [25, 100],
        "observation_lambda": 8,
        "threshold": 0.01,
        "correct": False,
        "num_threads": 2,
    }
    assert tracker.fps == pytest.approx(BEAST_FPS)


def test_real_madmom_dbn_processors_accept_fractional_fps() -> None:
    beat_tracker = dbn.DBNBeatTracker(
        fps=BEAST_FPS,
        num_tempi=60,
    )
    downbeat_tracker = dbn.DBNDownbeatTracker(
        beats_per_bar=[4],
        fps=BEAST_FPS,
        num_tempi=60,
    )

    assert beat_tracker.processor.fps == pytest.approx(BEAST_FPS)
    assert downbeat_tracker.processor.fps == pytest.approx(BEAST_FPS)


def test_downbeat_tracker_converts_canonical_activations_to_madmom_probabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(beats_per_bar=[2])
    activations = EventActivations(
        np.asarray([[0.9, 0.2], [0.9, 0.8]])
    )

    tracker(activations)

    assert np.allclose(
        tracker.processor.activations,
        [[0.7, 0.2], [0.1, 0.8]],
    )


def test_downbeat_tracker_preserves_frame_class_numerics_after_canonicalization(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(beats_per_bar=[2])
    frame_classes = np.asarray(
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
        dtype=np.float32,
    )

    events = frame_class_activations_to_event_activations(frame_classes)
    tracker(EventActivations(events))

    assert np.allclose(
        tracker.processor.activations,
        frame_classes[:, :2],
    )


def test_downbeat_tracker_rejects_invalid_probabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(beats_per_bar=[2])

    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        tracker(EventActivations(np.asarray([[1.2, 0.4]])))


def test_downbeat_tracker_rejects_untagged_two_channel_arrays(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(beats_per_bar=[2])

    with pytest.raises(ActivationFormatMismatchError, match="untagged"):
        tracker(np.asarray([[0.7, 0.2]]))


def _joint_pulse_activations(
    *,
    frames: int = 500,
    period: int = 25,
) -> EventActivations[np.ndarray]:
    values = np.full((frames, 2), 0.001, dtype=np.float64)
    for beat_index, frame in enumerate(range(period, frames, period)):
        values[frame, 0] = 0.98
        values[frame, 1] = 0.92 if beat_index % 4 == 0 else 0.02
    return EventActivations(values)


def test_causal_joint_dbn_emits_aligned_online_events() -> None:
    tracker = dbn.CausalDBNDownbeatTracker(
        beats_per_bar=[4],
        min_bpm=100,
        max_bpm=140,
        fps=50,
        num_tempi=20,
    )

    decoded, sources, emissions, frame_seconds = tracker.process_with_event_timing(
        _joint_pulse_activations()
    )

    assert decoded.ndim == 2
    assert decoded.shape[1] == 2
    assert len(decoded) > 0
    assert len(decoded) == len(sources) == len(emissions)
    assert np.array_equal(sources, emissions)
    assert decoded[:, 0] == pytest.approx(emissions / 50.0)
    assert set(np.unique(decoded[:, 1])).issubset({1.0, 2.0, 3.0, 4.0})
    minimum_separation_frames = int(np.floor(60.0 * 50 / 140))
    event_frames = np.rint(decoded[:, 0] * 50).astype(int)
    assert np.all(np.diff(event_frames) >= minimum_separation_frames)
    assert len(np.unique(event_frames)) == len(event_frames)
    assert frame_seconds.shape == (500,)
    assert np.all(frame_seconds >= 0.0)


def test_causal_joint_dbn_prefix_is_future_invariant() -> None:
    activations = _joint_pulse_activations(frames=500)
    prefix = EventActivations(activations.values[:300])
    kwargs = {
        "beats_per_bar": [3, 4],
        "min_bpm": 100,
        "max_bpm": 140,
        "fps": 50,
        "num_tempi": 20,
    }

    prefix_decoded = dbn.CausalDBNDownbeatTracker(**kwargs)(prefix)
    full_decoded = dbn.CausalDBNDownbeatTracker(**kwargs)(activations)

    assert full_decoded[full_decoded[:, 0] < 6.0] == pytest.approx(prefix_decoded)


def test_causal_joint_dbn_rejects_offline_corrections_and_trimming() -> None:
    with pytest.raises(ValueError, match="correct must be false"):
        dbn.CausalDBNDownbeatTracker(correct=True)
    with pytest.raises(ValueError, match="threshold must be 0"):
        dbn.CausalDBNDownbeatTracker(threshold=0.05)
