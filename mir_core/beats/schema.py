"""Shared beat/downbeat target schema.

The frame-class order follows the BeatNet/SpecTNT convention:
``[beat, downbeat, non_beat]``. The event-channel order follows the
two-channel activation convention: ``[beat, downbeat]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import IntEnum
from typing import Any

import numpy as np


class FrameClass(IntEnum):
    """Mutually-exclusive frame classes for cross-entropy targets."""

    beat = 0
    downbeat = 1
    non_beat = 2


class EventChannel(IntEnum):
    """Multi-label activation channels for beat/downbeat targets."""

    beat = 0
    downbeat = 1


class BeatDataRepresentation(Enum):
    """Representation used by model outputs or training targets."""

    frame_class = "frame_class"
    event_activation = "event_activation"


@dataclass(frozen=True)
class BeatDataDefinition:
    """Definition of beat data layout and ordering."""

    representation: BeatDataRepresentation
    order: tuple[IntEnum, ...]
    names: tuple[str, ...]

    @property
    def is_frame_class(self) -> bool:
        return self.representation is BeatDataRepresentation.frame_class

    @property
    def is_event_activation(self) -> bool:
        return self.representation is BeatDataRepresentation.event_activation


FRAME_CLASS_ORDER = (FrameClass.beat, FrameClass.downbeat, FrameClass.non_beat)
FRAME_CLASS_NAMES = tuple(frame_class.name for frame_class in FRAME_CLASS_ORDER)
NUM_FRAME_CLASSES = len(FRAME_CLASS_ORDER)

EVENT_CHANNEL_ORDER = (EventChannel.beat, EventChannel.downbeat)
EVENT_CHANNEL_NAMES = tuple(channel.name for channel in EVENT_CHANNEL_ORDER)
NUM_EVENT_CHANNELS = len(EVENT_CHANNEL_ORDER)

FRAME_CLASS_DEFINITION = BeatDataDefinition(
    representation=BeatDataRepresentation.frame_class,
    order=FRAME_CLASS_ORDER,
    names=FRAME_CLASS_NAMES,
)
EVENT_ACTIVATION_DEFINITION = BeatDataDefinition(
    representation=BeatDataRepresentation.event_activation,
    order=EVENT_CHANNEL_ORDER,
    names=EVENT_CHANNEL_NAMES,
)

BEAT_FRAME_CLASS = int(FrameClass.beat)
DOWNBEAT_FRAME_CLASS = int(FrameClass.downbeat)
NON_BEAT_FRAME_CLASS = int(FrameClass.non_beat)
PAD_FRAME_CLASS = FrameClass.non_beat

BEAT_CHANNEL = int(EventChannel.beat)
DOWNBEAT_CHANNEL = int(EventChannel.downbeat)

DOWNBEAT_POSITION = 1
UNKNOWN_ACTIVATION = -1.0


def coerce_beat_data_definition(value: Any) -> BeatDataDefinition:
    """Return a shared data definition from a definition, mapping, or name."""
    if isinstance(value, BeatDataDefinition):
        return value
    if isinstance(value, BeatDataRepresentation):
        representation = value
    elif isinstance(value, str):
        representation = BeatDataRepresentation(value)
    elif isinstance(value, dict):
        representation = BeatDataRepresentation(value["representation"])
    else:
        raise TypeError(f"Unsupported beat data definition: {value!r}")

    if representation is BeatDataRepresentation.frame_class:
        return FRAME_CLASS_DEFINITION
    if representation is BeatDataRepresentation.event_activation:
        return EVENT_ACTIVATION_DEFINITION
    raise ValueError(f"Unsupported beat data representation: {representation}")


def times_to_activation(
    times: Any,
    n_frames: int,
    fps: float,
    *,
    radius: int = 0,
    shoulder_value: float = 0.5,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert event times in seconds to a frame activation vector."""
    target = np.zeros(int(n_frames), dtype=dtype)
    if times is None:
        return target

    for time in np.asarray(times, dtype=float):
        center = int(round(float(time) * fps))
        for offset in range(-int(radius), int(radius) + 1):
            frame = center + offset
            if 0 <= frame < n_frames:
                value = 1.0 if offset == 0 else shoulder_value
                target[frame] = max(target[frame], value)
    return target


def annotation_to_frame_classes(
    annotation: Any,
    n_frames: int,
    fps: float,
    *,
    dtype: type[np.integer[Any]] = np.int64,
) -> np.ndarray:
    """Convert a beat annotation to mutually-exclusive frame classes."""
    targets = np.full(int(n_frames), int(FrameClass.non_beat), dtype=dtype)
    positions = getattr(annotation, "positions", None)

    for idx, time in enumerate(np.asarray(getattr(annotation, "times"), dtype=float)):
        frame = int(round(float(time) * fps))
        if 0 <= frame < n_frames:
            is_downbeat = positions is not None and int(positions[idx]) == DOWNBEAT_POSITION
            if is_downbeat:
                targets[frame] = int(FrameClass.downbeat)
            elif targets[frame] != int(FrameClass.downbeat):
                targets[frame] = int(FrameClass.beat)
    return targets


def annotation_to_event_activations(
    annotation: Any,
    n_frames: int,
    fps: float,
    *,
    radius: int = 0,
    shoulder_value: float = 0.5,
    include_downbeats_in_beat_channel: bool = True,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert a beat annotation to ``[beat, downbeat]`` activations."""
    positions = getattr(annotation, "positions", None)
    times = np.asarray(getattr(annotation, "times"), dtype=float)

    if include_downbeats_in_beat_channel or positions is None:
        beat_times = times
    else:
        beat_times = times[np.asarray(positions) != DOWNBEAT_POSITION]

    activations = np.zeros((int(n_frames), NUM_EVENT_CHANNELS), dtype=dtype)
    activations[:, BEAT_CHANNEL] = times_to_activation(
        beat_times,
        n_frames,
        fps,
        radius=radius,
        shoulder_value=shoulder_value,
        dtype=dtype,
    )

    downbeat_times = getattr(annotation, "downbeat_times", None)
    if downbeat_times is not None:
        activations[:, DOWNBEAT_CHANNEL] = times_to_activation(
            downbeat_times,
            n_frames,
            fps,
            radius=radius,
            shoulder_value=shoulder_value,
            dtype=dtype,
        )
    return activations


def frame_classes_to_event_activations(
    frame_classes: Any,
    *,
    dtype: type[np.floating[Any]] = np.float32,
    downbeat_implies_beat: bool = True,
) -> np.ndarray:
    """Convert frame-class labels to two-channel beat/downbeat activations."""
    classes = np.asarray(frame_classes, dtype=np.int64)
    activations = np.zeros(classes.shape + (NUM_EVENT_CHANNELS,), dtype=dtype)

    beat_mask = classes == int(FrameClass.beat)
    downbeat_mask = classes == int(FrameClass.downbeat)
    if downbeat_implies_beat:
        beat_mask = beat_mask | downbeat_mask

    activations[..., BEAT_CHANNEL] = beat_mask.astype(dtype)
    activations[..., DOWNBEAT_CHANNEL] = downbeat_mask.astype(dtype)
    return activations


def frame_class_activations_to_event_activations(
    frame_activations: Any,
    *,
    dtype: type[np.floating[Any]] = np.float32,
    downbeat_implies_beat: bool = False,
) -> np.ndarray:
    """Convert frame-class probabilities/logits to event-channel activations."""
    activations = np.asarray(frame_activations, dtype=dtype)
    if activations.shape[-1] < NUM_FRAME_CLASSES:
        raise ValueError(
            "frame-class activations must have beat/downbeat/non_beat classes "
            "in the last axis"
        )

    events = np.zeros(activations.shape[:-1] + (NUM_EVENT_CHANNELS,), dtype=dtype)
    events[..., BEAT_CHANNEL] = activations[..., int(FrameClass.beat)]
    if downbeat_implies_beat:
        events[..., BEAT_CHANNEL] = np.maximum(
            events[..., BEAT_CHANNEL],
            activations[..., int(FrameClass.downbeat)],
        )
    events[..., DOWNBEAT_CHANNEL] = activations[..., int(FrameClass.downbeat)]
    return events


def data_to_event_activations(
    data: Any,
    definition: BeatDataDefinition | BeatDataRepresentation | str | dict[str, Any],
    *,
    dtype: type[np.floating[Any]] = np.float32,
    downbeat_implies_beat: bool = False,
) -> np.ndarray:
    """Convert beat data in a declared representation to event activations."""
    resolved = coerce_beat_data_definition(definition)
    if resolved.is_event_activation:
        activations = np.asarray(data, dtype=dtype)
        if activations.shape[-1] < NUM_EVENT_CHANNELS:
            raise ValueError(
                "event activations must have beat/downbeat channels in the last axis"
            )
        return activations[..., :NUM_EVENT_CHANNELS]
    if resolved.is_frame_class:
        values = np.asarray(data)
        if (
            values.ndim > 0
            and values.shape[-1] == NUM_FRAME_CLASSES
            and not np.issubdtype(values.dtype, np.integer)
        ):
            return frame_class_activations_to_event_activations(
                values,
                dtype=dtype,
                downbeat_implies_beat=downbeat_implies_beat,
            )
        return frame_classes_to_event_activations(
            values,
            dtype=dtype,
            downbeat_implies_beat=True,
        )
    raise ValueError(f"Unsupported beat data representation: {resolved.representation}")


def event_activations_to_frame_class_activations(
    activations: Any,
    *,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert ``[beat, downbeat]`` activations to class-like probabilities."""
    acts = np.asarray(activations, dtype=dtype)
    if acts.shape[-1] < NUM_EVENT_CHANNELS:
        raise ValueError(
            "event activations must have beat/downbeat channels in the last axis"
        )

    beat = np.clip(acts[..., BEAT_CHANNEL], 0.0, 1.0)
    downbeat = np.clip(acts[..., DOWNBEAT_CHANNEL], 0.0, 1.0)
    beat_only = np.clip(beat - downbeat, 0.0, None)
    non_beat = np.clip(1.0 - np.maximum(beat, downbeat), 0.0, None)
    return np.stack((beat_only, downbeat, non_beat), axis=-1).astype(dtype)


def event_activations_to_frame_classes(
    activations: Any,
    *,
    threshold: float = 0.5,
    downbeat_priority: bool = True,
    dtype: type[np.integer[Any]] = np.int64,
) -> np.ndarray:
    """Convert ``[beat, downbeat]`` activations to frame-class labels."""
    acts = np.asarray(activations)
    if acts.shape[-1] < NUM_EVENT_CHANNELS:
        raise ValueError(
            "event activations must have beat/downbeat channels in the last axis"
        )

    frame_classes = np.full(acts.shape[:-1], int(FrameClass.non_beat), dtype=dtype)
    beat_mask = acts[..., BEAT_CHANNEL] >= threshold
    downbeat_mask = acts[..., DOWNBEAT_CHANNEL] >= threshold

    if downbeat_priority:
        frame_classes[beat_mask] = int(FrameClass.beat)
        frame_classes[downbeat_mask] = int(FrameClass.downbeat)
    else:
        frame_classes[downbeat_mask] = int(FrameClass.downbeat)
        frame_classes[beat_mask] = int(FrameClass.beat)
    return frame_classes


__all__ = [
    "BEAT_CHANNEL",
    "BEAT_FRAME_CLASS",
    "DOWNBEAT_CHANNEL",
    "DOWNBEAT_FRAME_CLASS",
    "DOWNBEAT_POSITION",
    "EVENT_ACTIVATION_DEFINITION",
    "EVENT_CHANNEL_NAMES",
    "EVENT_CHANNEL_ORDER",
    "FRAME_CLASS_DEFINITION",
    "FRAME_CLASS_NAMES",
    "FRAME_CLASS_ORDER",
    "NON_BEAT_FRAME_CLASS",
    "NUM_EVENT_CHANNELS",
    "NUM_FRAME_CLASSES",
    "PAD_FRAME_CLASS",
    "UNKNOWN_ACTIVATION",
    "BeatDataDefinition",
    "BeatDataRepresentation",
    "EventChannel",
    "FrameClass",
    "annotation_to_event_activations",
    "annotation_to_frame_classes",
    "coerce_beat_data_definition",
    "data_to_event_activations",
    "event_activations_to_frame_class_activations",
    "event_activations_to_frame_classes",
    "frame_class_activations_to_event_activations",
    "frame_classes_to_event_activations",
    "times_to_activation",
]
