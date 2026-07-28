"""Torch converters for runtime-tagged beat activation formats."""

from __future__ import annotations

from typing import Any

import torch

from .schema import (
    EVENT_ACTIVATION_DEFINITION,
    EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
    FRAME_CLASS_DEFINITION,
    ActivationFormatMismatchError,
    BeatActivationData,
    BeatActivationFormat,
    BeatDataDefinition,
    EventActivations,
    EventChannel,
    ExclusiveBeatDownbeatActivations,
    ExclusiveBeatDownbeatChannel,
    FrameClass,
    FrameClassActivations,
    coerce_beat_data_definition,
)


def convert_beat_activation_tensors(
    activations: BeatActivationData[torch.Tensor],
    target_format: BeatActivationFormat | str,
    *,
    target_definition: BeatDataDefinition | dict[str, Any] | None = None,
) -> BeatActivationData[torch.Tensor]:
    """Convert a tagged activation tensor without assuming channel order."""
    if not isinstance(
        activations,
        (
            FrameClassActivations,
            EventActivations,
            ExclusiveBeatDownbeatActivations,
        ),
    ):
        raise ActivationFormatMismatchError(
            "convert_beat_activation_tensors requires a tagged activation "
            "datatype; bare tensors do not identify their beat semantics."
        )
    if not torch.is_tensor(activations.values):
        raise TypeError("Tensor conversion requires torch.Tensor activation values.")

    target = BeatActivationFormat(target_format)
    if target_definition is None:
        if target is BeatActivationFormat.frame_class:
            resolved_target = FRAME_CLASS_DEFINITION
        elif target is BeatActivationFormat.event_activation:
            resolved_target = EVENT_ACTIVATION_DEFINITION
        else:
            resolved_target = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION
    else:
        resolved_target = coerce_beat_data_definition(target_definition)
        if resolved_target.representation is not target:
            raise ActivationFormatMismatchError(
                f"Target definition is {resolved_target.representation.value}, "
                f"not requested format {target.value}."
            )

    if isinstance(activations, FrameClassActivations):
        beat_only = activations.beat_only
        downbeat = activations.downbeats
        exclusive_downbeat = downbeat
        non_beat = activations.non_beats
        all_beats = beat_only + downbeat
    elif isinstance(activations, EventActivations):
        all_beats = activations.all_beats
        downbeat = activations.downbeats
        clipped_beat = all_beats.clamp(0.0, 1.0)
        clipped_downbeat = downbeat.clamp(0.0, 1.0)
        beat_only = torch.clamp(clipped_beat - clipped_downbeat, min=0.0)
        exclusive_downbeat = clipped_downbeat
        non_beat = torch.clamp(
            1.0 - torch.maximum(clipped_beat, clipped_downbeat),
            min=0.0,
        )
    else:
        beat_only = activations.beat_only
        downbeat = activations.downbeats
        exclusive_downbeat = downbeat
        all_beats = beat_only + downbeat
        non_beat = torch.clamp(1.0 - all_beats, min=0.0)

    if target is BeatActivationFormat.frame_class:
        channel_values = {
            FrameClass.beat: beat_only,
            FrameClass.downbeat: exclusive_downbeat,
            FrameClass.non_beat: non_beat,
        }
        converted = torch.stack(
            [channel_values[channel] for channel in resolved_target.order],
            dim=-1,
        )
        return FrameClassActivations(
            converted,
            definition=resolved_target,
            downbeats_available=activations.downbeats_available,
        )
    if target is BeatActivationFormat.event_activation:
        event_values = {
            EventChannel.beat: all_beats,
            EventChannel.downbeat: downbeat,
        }
        converted = torch.stack(
            [event_values[channel] for channel in resolved_target.order],
            dim=-1,
        )
        return EventActivations(
            converted,
            definition=resolved_target,
            downbeats_available=activations.downbeats_available,
        )

    exclusive_values = {
        ExclusiveBeatDownbeatChannel.beat_only: beat_only,
        ExclusiveBeatDownbeatChannel.downbeat: exclusive_downbeat,
    }
    converted = torch.stack(
        [exclusive_values[channel] for channel in resolved_target.order],
        dim=-1,
    )
    return ExclusiveBeatDownbeatActivations(
        converted,
        definition=resolved_target,
        downbeats_available=activations.downbeats_available,
    )


def to_event_activation_data(
    activations: BeatActivationData[torch.Tensor],
) -> EventActivations[torch.Tensor]:
    """Convert tagged tensor activations to canonical event semantics."""
    converted = convert_beat_activation_tensors(
        activations,
        BeatActivationFormat.event_activation,
    )
    assert isinstance(converted, EventActivations)
    return converted


def to_exclusive_beat_downbeat_activation_data(
    activations: BeatActivationData[torch.Tensor],
) -> ExclusiveBeatDownbeatActivations[torch.Tensor]:
    """Convert tagged tensor activations to joint-decoder semantics."""
    converted = convert_beat_activation_tensors(
        activations,
        BeatActivationFormat.exclusive_beat_downbeat,
    )
    assert isinstance(converted, ExclusiveBeatDownbeatActivations)
    return converted


def to_frame_class_activation_data(
    activations: BeatActivationData[torch.Tensor],
    *,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
) -> FrameClassActivations[torch.Tensor]:
    """Convert tagged tensor activations to exclusive frame classes."""
    converted = convert_beat_activation_tensors(
        activations,
        BeatActivationFormat.frame_class,
        target_definition=target_definition,
    )
    assert isinstance(converted, FrameClassActivations)
    return converted


def frame_class_activations_to_event_activations(
    frame_activations: torch.Tensor,
    *,
    definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
) -> torch.Tensor:
    """Convert exclusive class probabilities to canonical event activations."""
    source = FrameClassActivations(
        frame_activations,
        definition=coerce_beat_data_definition(definition),
    )
    return to_event_activation_data(source).values


def event_activations_to_frame_class_activations(
    event_activations: torch.Tensor,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
) -> torch.Tensor:
    """Convert canonical event probabilities to exclusive frame classes."""
    source = EventActivations(
        event_activations,
        definition=coerce_beat_data_definition(definition),
    )
    return to_frame_class_activation_data(
        source,
        target_definition=coerce_beat_data_definition(target_definition),
    ).values


def event_activations_to_exclusive_beat_downbeat_activations(
    event_activations: torch.Tensor,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
) -> torch.Tensor:
    """Convert canonical events to explicit joint-decoder probabilities."""
    source = EventActivations(
        event_activations,
        definition=coerce_beat_data_definition(definition),
    )
    converted = convert_beat_activation_tensors(
        source,
        BeatActivationFormat.exclusive_beat_downbeat,
        target_definition=coerce_beat_data_definition(target_definition),
    )
    assert isinstance(converted, ExclusiveBeatDownbeatActivations)
    return converted.values


def exclusive_beat_downbeat_activations_to_event_activations(
    activations: torch.Tensor,
    *,
    definition: BeatDataDefinition = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
) -> torch.Tensor:
    """Convert mutually-exclusive beat-only/downbeat probabilities to events."""
    source = ExclusiveBeatDownbeatActivations(
        activations,
        definition=coerce_beat_data_definition(definition),
    )
    return to_event_activation_data(source).values


def event_activations_to_frame_classes(
    event_activations: torch.Tensor,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    threshold: float = 0.5,
    downbeat_priority: bool = True,
) -> torch.Tensor:
    """Convert canonical event activations to hard frame-class labels."""
    event_data = EventActivations(
        event_activations,
        definition=coerce_beat_data_definition(definition),
    )
    resolved_target = coerce_beat_data_definition(target_definition)
    if not resolved_target.is_frame_class:
        raise ActivationFormatMismatchError(
            "event_activations_to_frame_classes target_definition must be "
            "frame_class."
        )
    beat_index = resolved_target.channel_index(FrameClass.beat)
    downbeat_index = resolved_target.channel_index(FrameClass.downbeat)
    non_beat_index = resolved_target.channel_index(FrameClass.non_beat)
    frame_classes = torch.full(
        event_data.values.shape[:-1],
        non_beat_index,
        dtype=torch.long,
        device=event_data.values.device,
    )
    beat_mask = event_data.all_beats >= threshold
    downbeat_mask = event_data.downbeats >= threshold
    if downbeat_priority:
        frame_classes[beat_mask] = beat_index
        frame_classes[downbeat_mask] = downbeat_index
    else:
        frame_classes[downbeat_mask] = downbeat_index
        frame_classes[beat_mask] = beat_index
    return frame_classes


__all__ = [
    "convert_beat_activation_tensors",
    "event_activations_to_exclusive_beat_downbeat_activations",
    "event_activations_to_frame_class_activations",
    "event_activations_to_frame_classes",
    "exclusive_beat_downbeat_activations_to_event_activations",
    "frame_class_activations_to_event_activations",
    "to_event_activation_data",
    "to_exclusive_beat_downbeat_activation_data",
    "to_frame_class_activation_data",
]
