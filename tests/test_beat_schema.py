from __future__ import annotations

import numpy as np
import pytest
import torch

from mir_core.beats.schema import (
    EVENT_ACTIVATION_DEFINITION,
    EVENT_CHANNEL_NAMES,
    EVENT_CHANNEL_SEMANTIC_NAMES,
    EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
    FRAME_CLASS_DEFINITION,
    FRAME_CLASS_NAMES,
    FRAME_CLASS_SEMANTIC_NAMES,
    ActivationFormatMismatchError,
    BeatActivationFormat,
    BeatDataDefinition,
    BeatDataRepresentation,
    EventActivations,
    EventChannel,
    ExclusiveBeatDownbeatActivations,
    ExclusiveBeatDownbeatChannel,
    FrameClassActivations,
    FrameClass,
    beat_data_definition_to_dict,
    coerce_beat_data_definition,
    convert_beat_activations,
    data_to_event_activations,
    event_activation_data_from_channels,
    event_activations_to_frame_class_activations,
    event_activations_to_frame_classes,
    frame_class_activations_to_event_activations,
    frame_classes_to_event_activations,
    require_event_activations,
    to_exclusive_beat_downbeat_activation_data,
)
from mir_core.beats.tensor_converters import (
    convert_beat_activation_tensors,
    event_activations_to_frame_class_activations as torch_event_to_frame_activations,
    event_activations_to_frame_classes as torch_event_to_frame_classes,
    frame_class_activations_to_event_activations as torch_frame_to_events,
)


def test_beat_schema_orders_are_explicit() -> None:
    assert int(FrameClass.beat) == 0
    assert int(FrameClass.downbeat) == 1
    assert int(FrameClass.non_beat) == 2
    assert FRAME_CLASS_NAMES == ("beat", "downbeat", "non_beat")
    assert FRAME_CLASS_SEMANTIC_NAMES == (
        "beat_only",
        "downbeat",
        "no_beat",
    )

    assert int(EventChannel.beat) == 0
    assert int(EventChannel.downbeat) == 1
    assert EVENT_CHANNEL_NAMES == ("beat", "downbeat")
    assert EVENT_CHANNEL_SEMANTIC_NAMES == ("all_beats", "downbeat")

    assert FRAME_CLASS_DEFINITION.representation is BeatDataRepresentation.frame_class
    assert EVENT_ACTIVATION_DEFINITION.representation is BeatDataRepresentation.event_activation
    assert (
        EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION.representation
        is BeatActivationFormat.exclusive_beat_downbeat
    )


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

    assert np.allclose(activations, [[0.9, 0.1], [0.8, 0.7]])
    assert np.allclose(
        event_activations_to_frame_class_activations(activations),
        probs,
    )


def test_frame_class_probabilities_convert_explicitly_to_decoder_streams() -> None:
    probs = np.array([[0.7, 0.2, 0.1]], dtype=np.float32)

    activations = convert_beat_activations(
        FrameClassActivations(probs),
        BeatActivationFormat.exclusive_beat_downbeat,
    )

    assert isinstance(activations, ExclusiveBeatDownbeatActivations)
    assert np.allclose(activations.values, [[0.7, 0.2]])


def test_three_class_conversion_uses_declared_channel_order() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.frame_class,
        order=(FrameClass.downbeat, FrameClass.beat, FrameClass.non_beat),
        names=("downbeat", "beat_only", "no_beat"),
    )
    probs = FrameClassActivations(
        np.array([[0.2, 0.7, 0.1]], dtype=np.float32),
        definition=downbeat_first,
    )

    canonical = convert_beat_activations(
        probs,
        BeatActivationFormat.event_activation,
    )
    exclusive = convert_beat_activations(
        probs,
        BeatActivationFormat.exclusive_beat_downbeat,
    )

    assert isinstance(canonical, EventActivations)
    assert np.allclose(canonical.values, [[0.9, 0.2]])
    assert isinstance(exclusive, ExclusiveBeatDownbeatActivations)
    assert np.allclose(exclusive.values, [[0.7, 0.2]])


def test_reordered_definition_serialization_round_trip_preserves_semantics() -> None:
    definition = BeatDataDefinition(
        representation=BeatActivationFormat.frame_class,
        order=(FrameClass.downbeat, FrameClass.beat, FrameClass.non_beat),
        names=("downbeat", "beat_only", "no_beat"),
    )

    payload = beat_data_definition_to_dict(definition)

    assert payload == {
        "representation": "frame_class",
        "order": ["downbeat", "beat_only", "no_beat"],
        "names": ["downbeat", "beat_only", "no_beat"],
    }
    assert coerce_beat_data_definition(payload) == definition


def test_event_channels_can_be_stored_in_nondefault_order() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.event_activation,
        order=(EventChannel.downbeat, EventChannel.beat),
        names=("downbeat", "all_beats"),
    )

    activations = event_activation_data_from_channels(
        np.asarray([0.9], dtype=np.float32),
        np.asarray([0.2], dtype=np.float32),
        definition=downbeat_first,
    )

    assert np.allclose(activations.values, [[0.2, 0.9]])
    assert np.allclose(activations.all_beats, [0.9])
    assert np.allclose(activations.downbeats, [0.2])


def test_three_class_definition_rejects_missing_or_duplicate_channels() -> None:
    with pytest.raises(ValueError, match="exactly once"):
        BeatDataDefinition(
            representation=BeatActivationFormat.frame_class,
            order=(
                FrameClass.downbeat,
                FrameClass.beat,
                FrameClass.beat,
            ),
            names=("downbeat", "beat_only_a", "beat_only_b"),
        )
    with pytest.raises(ValueError, match="does not describe"):
        BeatDataDefinition(
            representation=BeatActivationFormat.frame_class,
            order=(
                FrameClass.downbeat,
                FrameClass.beat,
                FrameClass.non_beat,
            ),
            names=("beat_only", "downbeat", "no_beat"),
        )
    with pytest.raises(TypeError, match="FrameClass"):
        BeatDataDefinition(
            representation=BeatActivationFormat.frame_class,
            order=(
                EventChannel.beat,
                EventChannel.downbeat,
                FrameClass.non_beat,
            ),
            names=("beat_only", "downbeat", "no_beat"),
        )


def test_two_channel_formats_are_runtime_distinct() -> None:
    values = np.array([[0.7, 0.2]], dtype=np.float32)
    canonical = EventActivations(values)
    exclusive = ExclusiveBeatDownbeatActivations(values)

    assert require_event_activations(canonical) is canonical
    with pytest.raises(
        ActivationFormatMismatchError,
        match="exclusive_beat_downbeat",
    ):
        require_event_activations(exclusive)
    with pytest.raises(ActivationFormatMismatchError, match="bare arrays"):
        convert_beat_activations(
            values,
            BeatActivationFormat.event_activation,
        )


def test_canonical_to_decoder_conversion_subtracts_downbeats_once() -> None:
    canonical = EventActivations(
        np.array([[0.9, 0.2], [0.9, 0.8]], dtype=np.float32)
    )

    exclusive = to_exclusive_beat_downbeat_activation_data(
        canonical,
        dtype=np.float64,
    )

    assert np.allclose(exclusive.values, [[0.7, 0.2], [0.1, 0.8]])


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


def test_hard_frame_classes_use_declared_physical_order() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.frame_class,
        order=(FrameClass.downbeat, FrameClass.beat, FrameClass.non_beat),
        names=("downbeat", "beat_only", "no_beat"),
    )

    converted = frame_classes_to_event_activations(
        np.asarray([0, 1, 2]),
        definition=downbeat_first,
    )
    restored = event_activations_to_frame_classes(
        converted,
        target_definition=downbeat_first,
    )

    assert converted.tolist() == [
        [1.0, 1.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ]
    assert restored.tolist() == [0, 1, 2]


def test_torch_converters_match_schema_shapes() -> None:
    frame_probs = torch.tensor([[[0.8, 0.1, 0.1], [0.1, 0.7, 0.2]]])

    event_activations = torch_frame_to_events(frame_probs)
    frame_activations = torch_event_to_frame_activations(event_activations)
    frame_classes = torch_event_to_frame_classes(event_activations)

    assert event_activations.shape == (1, 2, 2)
    assert torch.allclose(
        event_activations,
        torch.tensor([[[0.9, 0.1], [0.8, 0.7]]]),
    )
    assert frame_activations.shape == (1, 2, 3)
    assert torch.allclose(frame_activations, frame_probs)
    assert frame_classes.tolist() == [[int(FrameClass.beat), int(FrameClass.downbeat)]]


def test_torch_tagged_converter_respects_reordered_frame_classes() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.frame_class,
        order=(FrameClass.downbeat, FrameClass.beat, FrameClass.non_beat),
        names=("downbeat", "beat_only", "no_beat"),
    )
    source = FrameClassActivations(
        torch.tensor([[[0.2, 0.7, 0.1]]]),
        definition=downbeat_first,
    )

    converted = convert_beat_activation_tensors(
        source,
        BeatActivationFormat.event_activation,
    )

    assert isinstance(converted, EventActivations)
    assert torch.allclose(converted.values, torch.tensor([[[0.9, 0.2]]]))


def test_torch_hard_classes_can_target_reordered_frame_classes() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.frame_class,
        order=(FrameClass.downbeat, FrameClass.beat, FrameClass.non_beat),
        names=("downbeat", "beat_only", "no_beat"),
    )

    classes = torch_event_to_frame_classes(
        torch.tensor([[[0.9, 0.2], [0.9, 0.8], [0.1, 0.0]]]),
        target_definition=downbeat_first,
    )

    assert classes.tolist() == [[1, 0, 2]]


def test_reordered_exclusive_channels_normalize_to_default_order() -> None:
    downbeat_first = BeatDataDefinition(
        representation=BeatActivationFormat.exclusive_beat_downbeat,
        order=(
            ExclusiveBeatDownbeatChannel.downbeat,
            ExclusiveBeatDownbeatChannel.beat_only,
        ),
        names=("downbeat", "beat_only"),
    )
    source = ExclusiveBeatDownbeatActivations(
        np.asarray([[0.2, 0.7]], dtype=np.float32),
        definition=downbeat_first,
    )

    normalized = to_exclusive_beat_downbeat_activation_data(source)

    assert normalized.definition == EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION
    assert np.allclose(normalized.values, [[0.7, 0.2]])
