from __future__ import annotations

import numpy as np
import pytest

from mir_core.postprocessing import dbn


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
        fps=50,
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
        "fps": 50,
        "num_tempi": 90,
        "transition_lambda": 50,
        "observation_lambda": 8,
        "threshold": 0.03,
        "correct": False,
        "online": True,
        "num_threads": 2,
    }


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
        fps=50,
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
        "fps": 50,
        "num_tempi": [30, 60],
        "transition_lambda": [25, 100],
        "observation_lambda": 8,
        "threshold": 0.01,
        "correct": False,
        "num_threads": 2,
    }


def test_downbeat_tracker_preserves_exclusive_frame_class_probabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(
        beats_per_bar=[2],
        beat_activations_include_downbeats=False,
    )
    beat = np.asarray([0.7, 0.1])
    downbeat = np.asarray([0.2, 0.8])

    tracker(beat, downbeat)

    assert np.allclose(
        tracker.processor.activations,
        [[0.7, 0.2], [0.1, 0.8]],
    )


def test_downbeat_tracker_converts_inclusive_event_probabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(
        beats_per_bar=[2],
        beat_activations_include_downbeats=True,
    )

    tracker(np.asarray([0.9, 0.8]), np.asarray([0.2, 0.8]))

    assert np.allclose(
        tracker.processor.activations,
        [[0.7, 0.2], [0.0, 0.8]],
    )


def test_downbeat_tracker_rejects_invalid_exclusive_probabilities(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        dbn.madmom.features.downbeats,
        "DBNDownBeatTrackingProcessor",
        _FakeProcessor,
    )
    tracker = dbn.DBNDownbeatTracker(
        beats_per_bar=[2],
        beat_activations_include_downbeats=False,
    )

    with pytest.raises(ValueError, match="sum to at most 1"):
        tracker(np.asarray([0.8]), np.asarray([0.4]))
