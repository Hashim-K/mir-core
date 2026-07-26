import numpy as np
import pytest

from mir_core.postprocessing import ParticleFilterTracker


def test_particle_filter_accepts_two_channel_activations_at_fractional_fps() -> None:
    fps = 44100 / 1024
    tracker = ParticleFilterTracker(
        fps=fps,
        particle_size=100,
        down_particle_size=20,
        num_tempi=30,
    )

    decoded = tracker.process(np.zeros((4, 2), dtype=np.float32))

    assert tracker.fps == pytest.approx(fps)
    assert tracker.T == pytest.approx(1 / fps)
    assert decoded.shape == (0, 2)
