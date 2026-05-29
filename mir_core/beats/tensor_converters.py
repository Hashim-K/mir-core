"""Torch tensor converters for the shared beat schema."""

from __future__ import annotations

import torch

from .schema import (
    BEAT_CHANNEL,
    DOWNBEAT_CHANNEL,
    NUM_EVENT_CHANNELS,
    NUM_FRAME_CLASSES,
    EventChannel,
    FrameClass,
)


def frame_class_activations_to_event_activations(
    frame_activations: torch.Tensor,
    *,
    downbeat_implies_beat: bool = False,
) -> torch.Tensor:
    """Convert class probabilities/logits to ``[beat, downbeat]`` activations."""
    if frame_activations.shape[-1] < NUM_FRAME_CLASSES:
        raise ValueError(
            "frame-class activations must have beat/downbeat/non_beat classes "
            "in the last axis"
        )

    beat = frame_activations[..., int(FrameClass.beat)]
    downbeat = frame_activations[..., int(FrameClass.downbeat)]
    if downbeat_implies_beat:
        beat = torch.maximum(beat, downbeat)
    return torch.stack((beat, downbeat), dim=-1)


def event_activations_to_frame_class_activations(
    event_activations: torch.Tensor,
) -> torch.Tensor:
    """Convert ``[beat, downbeat]`` activations to class-like probabilities."""
    if event_activations.shape[-1] < NUM_EVENT_CHANNELS:
        raise ValueError(
            "event activations must have beat/downbeat channels in the last axis"
        )

    beat = event_activations[..., int(EventChannel.beat)].clamp(0.0, 1.0)
    downbeat = event_activations[..., int(EventChannel.downbeat)].clamp(0.0, 1.0)
    beat_only = torch.clamp(beat - downbeat, min=0.0)
    non_beat = torch.clamp(1.0 - torch.maximum(beat, downbeat), min=0.0)
    return torch.stack((beat_only, downbeat, non_beat), dim=-1)


def event_activations_to_frame_classes(
    event_activations: torch.Tensor,
    *,
    threshold: float = 0.5,
    downbeat_priority: bool = True,
) -> torch.Tensor:
    """Convert event activations to hard frame-class labels."""
    frame_classes = torch.full(
        event_activations.shape[:-1],
        int(FrameClass.non_beat),
        dtype=torch.long,
        device=event_activations.device,
    )
    beat_mask = event_activations[..., BEAT_CHANNEL] >= threshold
    downbeat_mask = event_activations[..., DOWNBEAT_CHANNEL] >= threshold
    if downbeat_priority:
        frame_classes[beat_mask] = int(FrameClass.beat)
        frame_classes[downbeat_mask] = int(FrameClass.downbeat)
    else:
        frame_classes[downbeat_mask] = int(FrameClass.downbeat)
        frame_classes[beat_mask] = int(FrameClass.beat)
    return frame_classes


__all__ = [
    "event_activations_to_frame_class_activations",
    "event_activations_to_frame_classes",
    "frame_class_activations_to_event_activations",
]
