from __future__ import annotations

import numpy as np
import torch

from mir_core.beats.schema import (
    EVENT_ACTIVATION_DEFINITION,
    EVENT_CHANNEL_NAMES,
    FRAME_CLASS_DEFINITION,
    FRAME_CLASS_NAMES,
    BeatDataRepresentation,
    EventChannel,
    FrameClass,
    data_to_event_activations,
    event_activations_to_frame_class_activations,
    event_activations_to_frame_classes,
    frame_class_activations_to_event_activations,
    frame_classes_to_event_activations,
)
from mir_core.beats.tensor_converters import (
    event_activations_to_frame_class_activations as torch_event_to_frame_activations,
    event_activations_to_frame_classes as torch_event_to_frame_classes,
    frame_class_activations_to_event_activations as torch_frame_to_events,
)


def test_beat_schema_orders_are_explicit() -> None:
    assert int(FrameClass.beat) == 0
    assert int(FrameClass.downbeat) == 1
    assert int(FrameClass.non_beat) == 2
    assert FRAME_CLASS_NAMES == ("beat", "downbeat", "non_beat")

    assert int(EventChannel.beat) == 0
    assert int(EventChannel.downbeat) == 1
    assert EVENT_CHANNEL_NAMES == ("beat", "downbeat")

    assert FRAME_CLASS_DEFINITION.representation is BeatDataRepresentation.frame_class
    assert EVENT_ACTIVATION_DEFINITION.representation is BeatDataRepresentation.event_activation


def test_frame_classes_convert_to_event_activations() -> None:
    frame_classes = np.array([FrameClass.non_beat, FrameClass.beat, FrameClass.downbeat])

    activations = frame_classes_to_event_activations(frame_classes)

    assert activations.tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]


def test_event_activations_convert_to_frame_classes_and_probabilities() -> None:
    activations = np.array([[0.0, 0.0], [0.7, 0.1], [0.8, 0.9]], dtype=np.float32)

    classes = event_activations_to_frame_classes(activations)
    frame_activations = event_activations_to_frame_class_activations(activations)

    assert classes.tolist() == [
        int(FrameClass.non_beat),
        int(FrameClass.beat),
        int(FrameClass.downbeat),
    ]
    assert np.allclose(frame_activations.sum(axis=-1), np.ones(3))
    assert frame_activations[2].argmax() == int(FrameClass.downbeat)


def test_frame_class_probabilities_convert_to_event_activations() -> None:
    probs = np.array(
        [[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]],
        dtype=np.float32,
    )

    activations = frame_class_activations_to_event_activations(probs)

    assert np.allclose(activations, [[0.8, 0.1], [0.1, 0.7]])


def test_data_to_event_activations_respects_declared_definition() -> None:
    frame_labels = np.array([FrameClass.beat, FrameClass.downbeat, FrameClass.non_beat])
    channel_activations = np.array([[1.0, 0.0], [0.2, 0.9], [0.0, 0.0]])

    assert data_to_event_activations(frame_labels, FRAME_CLASS_DEFINITION).tolist() == [
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 0.0],
    ]
    assert np.allclose(
        data_to_event_activations(channel_activations, EVENT_ACTIVATION_DEFINITION),
        channel_activations,
    )


def test_torch_converters_match_schema_shapes() -> None:
    frame_probs = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]])

    event_activations = torch_frame_to_events(frame_probs)
    frame_activations = torch_event_to_frame_activations(event_activations)
    frame_classes = torch_event_to_frame_classes(event_activations)

    assert event_activations.shape == (1, 2, 2)
    assert frame_activations.shape == (1, 2, 3)
    assert frame_classes.tolist() == [[int(FrameClass.beat), int(FrameClass.downbeat)]]
