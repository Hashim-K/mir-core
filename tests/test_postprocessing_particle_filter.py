import numpy as np
import pytest

from mir_core.beats.schema import (
    ActivationFormatMismatchError,
    BeatActivationFormat,
    BeatDataDefinition,
    ExclusiveBeatDownbeatActivations,
    ExclusiveBeatDownbeatChannel,
)
from mir_core.postprocessing import ParticleFilterTracker


def test_particle_filter_accepts_two_channel_activations_at_fractional_fps() -> None:
    fps = 44100 / 1024
    tracker = ParticleFilterTracker(
        fps=fps,
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    decoded = tracker.process(
        ExclusiveBeatDownbeatActivations(
            np.zeros((4, 2), dtype=np.float32)
        )
    )

    assert tracker.fps == pytest.approx(fps)
    assert tracker.T == pytest.approx(1 / fps)
    assert decoded.shape == (0, 2)


def test_particle_filter_rejects_untagged_two_channel_arrays() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    with pytest.raises(ActivationFormatMismatchError, match="untagged"):
        tracker.process(np.zeros((4, 2), dtype=np.float32))


def test_particle_filter_rejects_nonexclusive_probability_rows() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    with pytest.raises(ValueError, match="sum to at most 1"):
        tracker.process(
            ExclusiveBeatDownbeatActivations(
                np.asarray([[0.9, 0.2]], dtype=np.float32)
            )
        )


def test_particle_filter_normalizes_declared_exclusive_channel_order() -> None:
    tracker = ParticleFilterTracker(
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.exclusive_beat_downbeat,
        order=(
            ExclusiveBeatDownbeatChannel.downbeat,
            ExclusiveBeatDownbeatChannel.beat_only,
        ),
        names=("downbeat", "beat_only"),
    )

    tracker.process(
        ExclusiveBeatDownbeatActivations(
            np.asarray([[0.2, 0.7]], dtype=np.float32),
            definition=downbeat_first,
        )
    )

    assert np.allclose(tracker.both_activations, [[0.7, 0.2]])


def test_particle_filter_enforces_last_emitted_event_refractory() -> None:
    random_state = np.random.get_state()
    try:
        np.random.seed(7)
        fps = 50
        values = np.zeros((500, 2), dtype=np.float32)
        values[:, 0] = 0.9
        values[:, 1] = 0.05
        tracker = ParticleFilterTracker(
            fps=fps,
            min_bpm=80,
            max_bpm=240,
            particle_size=300,
            down_particle_size=60,
            num_tempi=60,
            offset=0,
        )

        decoded = tracker.process(ExclusiveBeatDownbeatActivations(values))
    finally:
        np.random.set_state(random_state)

    minimum_separation = 0.4 * tracker.T * np.min(tracker.st.state_intervals)
    assert len(decoded) > 1
    assert np.all(np.diff(decoded[:, 0]) > minimum_separation)
    assert len(np.unique(decoded[:, 0])) == len(decoded)
    assert set(np.unique(decoded[:, 1])).issubset({1.0, 2.0})
