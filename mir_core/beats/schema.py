"""Shared beat/downbeat target and activation schemas.

Three activation formats are deliberately distinct:

``frame_class``
    Mutually-exclusive probabilities for beat-only, downbeat, and non-beat.
    The channel order is carried by :class:`BeatDataDefinition` and may vary.
``event_activation``
    Canonical overlapping probabilities for all beats and downbeats.
``exclusive_beat_downbeat``
    Mutually-exclusive beat-only and downbeat probabilities expected by joint
    decoders such as madmom's downbeat DBN.

The last two formats both have two channels and cannot be identified safely
from shape or values. Runtime activation datatypes therefore carry the format
and channel order explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from enum import IntEnum
from typing import Any, Generic, TypeVar

import numpy as np

ArrayT = TypeVar("ArrayT")


class FrameClass(IntEnum):
    """Mutually-exclusive frame classes for cross-entropy targets.

    ``beat`` and ``non_beat`` remain aliases for checkpoint and target
    compatibility.
    """

    beat_only = 0
    beat = beat_only
    downbeat = 1
    no_beat = 2
    non_beat = no_beat


class EventChannel(IntEnum):
    """Multi-label channels where the beat channel includes downbeats."""

    all_beats = 0
    beat = all_beats
    downbeat = 1


class ExclusiveBeatDownbeatChannel(IntEnum):
    """Mutually-exclusive channels consumed by joint beat/downbeat decoders."""

    beat_only = 0
    downbeat = 1


class BeatDataRepresentation(str, Enum):
    """Activation format used by model outputs, caches, or decoders."""

    frame_class = "frame_class"
    event_activation = "event_activation"
    exclusive_beat_downbeat = "exclusive_beat_downbeat"


class BeatTargetSupervision(str, Enum):
    """Beat-related tasks for which an annotation supplies ground truth.

    ``beats_only`` means that every annotated event is known to be a beat, but
    its metrical position is unknown. It must not be interpreted as the
    mutually-exclusive :class:`FrameClass.beat_only` class.
    """

    beats_only = "beats_only"
    beats_and_downbeats = "beats_and_downbeats"

    @property
    def downbeats_available(self) -> bool:
        """Whether downbeat targets are valid for this supervision mode."""
        return self is BeatTargetSupervision.beats_and_downbeats

    @classmethod
    def from_downbeats_available(
        cls,
        available: bool,
    ) -> BeatTargetSupervision:
        """Construct an explicit supervision mode from annotation availability."""
        if not isinstance(available, bool):
            raise TypeError("Downbeat availability must be a boolean.")
        return cls.beats_and_downbeats if available else cls.beats_only


# Public semantic name for new code; retain BeatDataRepresentation as an alias
# used by existing configs and imports.
BeatActivationFormat = BeatDataRepresentation


@dataclass(frozen=True)
class BeatDataDefinition:
    """Definition of an activation format and its channel ordering."""

    representation: BeatDataRepresentation
    order: tuple[IntEnum, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.representation is BeatDataRepresentation.frame_class:
            expected_type: type[IntEnum] = FrameClass
            expected_values = {
                int(FrameClass.beat),
                int(FrameClass.downbeat),
                int(FrameClass.non_beat),
            }
            allowed_names = {
                int(FrameClass.beat): {"beat", "beat_only"},
                int(FrameClass.downbeat): {"downbeat"},
                int(FrameClass.non_beat): {"non_beat", "no_beat"},
            }
        elif self.representation is BeatDataRepresentation.event_activation:
            expected_type = EventChannel
            expected_values = {
                int(EventChannel.beat),
                int(EventChannel.downbeat),
            }
            allowed_names = {
                int(EventChannel.beat): {"beat", "beats", "all_beat", "all_beats"},
                int(EventChannel.downbeat): {"downbeat", "downbeats"},
            }
        elif (
            self.representation
            is BeatDataRepresentation.exclusive_beat_downbeat
        ):
            expected_type = ExclusiveBeatDownbeatChannel
            expected_values = {
                int(ExclusiveBeatDownbeatChannel.beat_only),
                int(ExclusiveBeatDownbeatChannel.downbeat),
            }
            allowed_names = {
                int(ExclusiveBeatDownbeatChannel.beat_only): {
                    "beat",
                    "beat_only",
                },
                int(ExclusiveBeatDownbeatChannel.downbeat): {
                    "downbeat",
                    "downbeats",
                },
            }
        else:
            raise ValueError(
                f"Unsupported beat data representation: {self.representation!r}"
            )

        if any(type(channel) is not expected_type for channel in self.order):
            raise TypeError(
                f"{self.representation.value} order must contain "
                f"{expected_type.__name__} members."
            )
        if (
            len(self.order) != len(expected_values)
            or {int(channel) for channel in self.order} != expected_values
        ):
            raise ValueError(
                f"{self.representation.value} order must contain every channel "
                "exactly once."
            )
        if any(not isinstance(name, str) for name in self.names):
            raise TypeError("Beat data definition names must be strings.")
        if len(self.names) != len(self.order) or len(set(self.names)) != len(
            self.names
        ):
            raise ValueError(
                "Beat data definition names must uniquely label every channel."
            )
        for channel, name in zip(self.order, self.names, strict=True):
            normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized not in allowed_names[int(channel)]:
                raise ValueError(
                    f"Channel name {name!r} does not describe {channel.name!r} "
                    f"in {self.representation.value} order."
                )

    @property
    def is_frame_class(self) -> bool:
        return self.representation is BeatDataRepresentation.frame_class

    @property
    def is_event_activation(self) -> bool:
        return self.representation is BeatDataRepresentation.event_activation

    @property
    def is_exclusive_beat_downbeat(self) -> bool:
        return (
            self.representation
            is BeatDataRepresentation.exclusive_beat_downbeat
        )

    def channel_index(self, channel: IntEnum) -> int:
        """Return the physical array index for a semantic channel."""
        for index, candidate in enumerate(self.order):
            if type(candidate) is type(channel) and int(candidate) == int(channel):
                return index
        raise ValueError(
            f"{channel!r} is not present in {self.representation.value} order "
            f"{self.order!r}."
        )


FRAME_CLASS_ORDER = (
    FrameClass.beat_only,
    FrameClass.downbeat,
    FrameClass.no_beat,
)
# Legacy/public model class labels retained for checkpoint-facing outputs.
FRAME_CLASS_NAMES = ("beat", "downbeat", "non_beat")
FRAME_CLASS_SEMANTIC_NAMES = tuple(
    frame_class.name for frame_class in FRAME_CLASS_ORDER
)
NUM_FRAME_CLASSES = len(FRAME_CLASS_ORDER)

EVENT_CHANNEL_ORDER = (EventChannel.all_beats, EventChannel.downbeat)
EVENT_CHANNEL_NAMES = ("beat", "downbeat")
EVENT_CHANNEL_SEMANTIC_NAMES = tuple(
    channel.name for channel in EVENT_CHANNEL_ORDER
)
NUM_EVENT_CHANNELS = len(EVENT_CHANNEL_ORDER)

EXCLUSIVE_BEAT_DOWNBEAT_ORDER = (
    ExclusiveBeatDownbeatChannel.beat_only,
    ExclusiveBeatDownbeatChannel.downbeat,
)
EXCLUSIVE_BEAT_DOWNBEAT_NAMES = tuple(
    channel.name for channel in EXCLUSIVE_BEAT_DOWNBEAT_ORDER
)
NUM_EXCLUSIVE_BEAT_DOWNBEAT_CHANNELS = len(EXCLUSIVE_BEAT_DOWNBEAT_ORDER)

FRAME_CLASS_DEFINITION = BeatDataDefinition(
    representation=BeatDataRepresentation.frame_class,
    order=FRAME_CLASS_ORDER,
    names=FRAME_CLASS_SEMANTIC_NAMES,
)
EVENT_ACTIVATION_DEFINITION = BeatDataDefinition(
    representation=BeatDataRepresentation.event_activation,
    order=EVENT_CHANNEL_ORDER,
    names=EVENT_CHANNEL_SEMANTIC_NAMES,
)
EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION = BeatDataDefinition(
    representation=BeatDataRepresentation.exclusive_beat_downbeat,
    order=EXCLUSIVE_BEAT_DOWNBEAT_ORDER,
    names=EXCLUSIVE_BEAT_DOWNBEAT_NAMES,
)

BEAT_FRAME_CLASS = int(FrameClass.beat)
DOWNBEAT_FRAME_CLASS = int(FrameClass.downbeat)
NON_BEAT_FRAME_CLASS = int(FrameClass.non_beat)
PAD_FRAME_CLASS = FrameClass.non_beat

BEAT_CHANNEL = int(EventChannel.beat)
DOWNBEAT_CHANNEL = int(EventChannel.downbeat)

DOWNBEAT_POSITION = 1
UNKNOWN_ACTIVATION = -1.0


class ActivationFormatMismatchError(TypeError):
    """Raised when an activation is untagged or has the wrong runtime format."""


def _validate_activation_layout(
    values: Any,
    definition: BeatDataDefinition,
) -> None:
    shape = getattr(values, "shape", None)
    if shape is None:
        raise TypeError("Activation values must expose an array-like shape.")
    if len(shape) == 0 or int(shape[-1]) != len(definition.order):
        raise ValueError(
            f"{definition.representation.value} activations must have exactly "
            f"{len(definition.order)} channels in the last axis according to "
            f"{definition.names}; got shape {tuple(shape)}."
        )


@dataclass(frozen=True)
class FrameClassActivations(Generic[ArrayT]):
    """Tagged mutually-exclusive beat-only/downbeat/non-beat activations."""

    values: ArrayT
    definition: BeatDataDefinition = FRAME_CLASS_DEFINITION
    downbeats_available: bool = True

    def __post_init__(self) -> None:
        if not self.definition.is_frame_class:
            raise ActivationFormatMismatchError(
                "FrameClassActivations requires a frame_class definition."
            )
        _validate_activation_layout(self.values, self.definition)

    @property
    def format(self) -> BeatActivationFormat:
        return BeatActivationFormat.frame_class

    def channel(self, channel: FrameClass) -> Any:
        return self.values[..., self.definition.channel_index(channel)]

    @property
    def beat_only(self) -> Any:
        return self.channel(FrameClass.beat)

    @property
    def downbeats(self) -> Any:
        return self.channel(FrameClass.downbeat)

    @property
    def non_beats(self) -> Any:
        return self.channel(FrameClass.non_beat)


@dataclass(frozen=True)
class EventActivations(Generic[ArrayT]):
    """Tagged canonical ``[all beats, downbeat]`` activations."""

    values: ArrayT
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION
    downbeats_available: bool = True

    def __post_init__(self) -> None:
        if not self.definition.is_event_activation:
            raise ActivationFormatMismatchError(
                "EventActivations requires an event_activation definition."
            )
        _validate_activation_layout(self.values, self.definition)

    @property
    def format(self) -> BeatActivationFormat:
        return BeatActivationFormat.event_activation

    def channel(self, channel: EventChannel) -> Any:
        return self.values[..., self.definition.channel_index(channel)]

    @property
    def all_beats(self) -> Any:
        return self.channel(EventChannel.beat)

    @property
    def beats(self) -> Any:
        """Compatibility alias for the canonical all-beat stream."""
        return self.all_beats

    @property
    def downbeats(self) -> Any:
        return self.channel(EventChannel.downbeat)


@dataclass(frozen=True)
class ExclusiveBeatDownbeatActivations(Generic[ArrayT]):
    """Tagged mutually-exclusive ``[beat-only, downbeat]`` activations."""

    values: ArrayT
    definition: BeatDataDefinition = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION
    downbeats_available: bool = True

    def __post_init__(self) -> None:
        if not self.definition.is_exclusive_beat_downbeat:
            raise ActivationFormatMismatchError(
                "ExclusiveBeatDownbeatActivations requires an "
                "exclusive_beat_downbeat definition."
            )
        _validate_activation_layout(self.values, self.definition)

    @property
    def format(self) -> BeatActivationFormat:
        return BeatActivationFormat.exclusive_beat_downbeat

    def channel(self, channel: ExclusiveBeatDownbeatChannel) -> Any:
        return self.values[..., self.definition.channel_index(channel)]

    @property
    def beat_only(self) -> Any:
        return self.channel(ExclusiveBeatDownbeatChannel.beat_only)

    @property
    def downbeats(self) -> Any:
        return self.channel(ExclusiveBeatDownbeatChannel.downbeat)


BeatActivationData = (
    FrameClassActivations[ArrayT]
    | EventActivations[ArrayT]
    | ExclusiveBeatDownbeatActivations[ArrayT]
)


def require_event_activations(value: Any) -> EventActivations[Any]:
    """Require a canonical runtime-tagged activation value."""
    if isinstance(value, EventActivations):
        return value
    if isinstance(
        value,
        (FrameClassActivations, ExclusiveBeatDownbeatActivations),
    ):
        raise ActivationFormatMismatchError(
            "Expected canonical EventActivations "
            f"({BeatActivationFormat.event_activation.value}), got "
            f"{value.format.value}. Convert it explicitly before this boundary."
        )
    raise ActivationFormatMismatchError(
        "Expected canonical EventActivations, but received untagged values. "
        "A two-channel array cannot reveal whether its first channel is "
        "all-beats or beat-only."
    )


def require_exclusive_beat_downbeat_activations(
    value: Any,
) -> ExclusiveBeatDownbeatActivations[Any]:
    """Require a decoder-exclusive runtime-tagged activation value."""
    if isinstance(value, ExclusiveBeatDownbeatActivations):
        return value
    if isinstance(value, (FrameClassActivations, EventActivations)):
        raise ActivationFormatMismatchError(
            "Expected ExclusiveBeatDownbeatActivations "
            f"({BeatActivationFormat.exclusive_beat_downbeat.value}), got "
            f"{value.format.value}. Convert it explicitly before this boundary."
        )
    raise ActivationFormatMismatchError(
        "Expected ExclusiveBeatDownbeatActivations, but received untagged "
        "values. A two-channel array cannot reveal whether its first channel "
        "is all-beats or beat-only."
    )


def coerce_beat_data_definition(value: Any) -> BeatDataDefinition:
    """Return a data definition from a definition, mapping, or format name.

    Mappings may include an explicit ``order`` (enum members, names, or
    integer values). This is required when a three-class producer uses a
    non-default channel order.
    """
    if isinstance(value, BeatDataDefinition):
        return value
    if isinstance(value, BeatDataRepresentation):
        representation = value
    elif isinstance(value, str):
        representation = BeatDataRepresentation(value)
    elif isinstance(value, dict):
        representation = BeatDataRepresentation(
            value.get("representation", value.get("format"))
        )
    else:
        raise TypeError(f"Unsupported beat data definition: {value!r}")

    if representation is BeatDataRepresentation.frame_class:
        default = FRAME_CLASS_DEFINITION
        channel_type: type[IntEnum] = FrameClass
        aliases = {
            "beat": FrameClass.beat,
            "beat_only": FrameClass.beat,
            "downbeat": FrameClass.downbeat,
            "non_beat": FrameClass.non_beat,
            "no_beat": FrameClass.non_beat,
        }
    elif representation is BeatDataRepresentation.event_activation:
        default = EVENT_ACTIVATION_DEFINITION
        channel_type = EventChannel
        aliases = {
            "beat": EventChannel.beat,
            "beats": EventChannel.beat,
            "all_beat": EventChannel.beat,
            "all_beats": EventChannel.beat,
            "downbeat": EventChannel.downbeat,
            "downbeats": EventChannel.downbeat,
        }
    elif representation is BeatDataRepresentation.exclusive_beat_downbeat:
        default = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION
        channel_type = ExclusiveBeatDownbeatChannel
        aliases = {
            "beat": ExclusiveBeatDownbeatChannel.beat_only,
            "beat_only": ExclusiveBeatDownbeatChannel.beat_only,
            "downbeat": ExclusiveBeatDownbeatChannel.downbeat,
            "downbeats": ExclusiveBeatDownbeatChannel.downbeat,
        }
    else:
        raise ValueError(f"Unsupported beat data representation: {representation}")

    if not isinstance(value, dict):
        return default

    raw_order = value.get("order")
    if raw_order is None and "names" in value:
        raw_order = value["names"]
    if raw_order is None:
        return default

    order: list[IntEnum] = []
    for raw_channel in raw_order:
        if type(raw_channel) is channel_type:
            channel = raw_channel
        elif isinstance(raw_channel, str):
            normalized = raw_channel.strip().lower().replace("-", "_").replace(
                " ", "_"
            )
            try:
                channel = aliases[normalized]
            except KeyError as exc:
                raise ValueError(
                    f"Unknown {representation.value} channel {raw_channel!r}."
                ) from exc
        elif isinstance(raw_channel, IntEnum):
            raise TypeError(
                f"{representation.value} order cannot contain "
                f"{type(raw_channel).__name__} members."
            )
        else:
            channel = channel_type(int(raw_channel))
        order.append(channel)

    raw_names = value.get("names")
    names = (
        tuple(str(name) for name in raw_names)
        if raw_names is not None
        else tuple(channel.name for channel in order)
    )
    return BeatDataDefinition(
        representation=representation,
        order=tuple(order),
        names=names,
    )


def beat_data_definition_to_dict(
    definition: BeatDataDefinition,
) -> dict[str, Any]:
    """Serialize a definition without losing channel-order metadata."""
    resolved = coerce_beat_data_definition(definition)
    return {
        "representation": resolved.representation.value,
        "order": [channel.name for channel in resolved.order],
        "names": list(resolved.names),
    }


def event_activation_data_from_channels(
    all_beats: Any,
    downbeats: Any | None = None,
    *,
    definition: BeatDataDefinition | dict[str, Any] = EVENT_ACTIVATION_DEFINITION,
    dtype: Any = None,
) -> EventActivations[np.ndarray]:
    """Build tagged event data from semantic streams in the declared order."""
    resolved = coerce_beat_data_definition(definition)
    if not resolved.is_event_activation:
        raise ActivationFormatMismatchError(
            "event_activation_data_from_channels requires an "
            "event_activation definition."
        )
    beat_values = np.asarray(all_beats, dtype=dtype)
    downbeats_available = downbeats is not None
    downbeat_values = (
        np.asarray(downbeats, dtype=dtype)
        if downbeats_available
        else np.zeros_like(beat_values)
    )
    if beat_values.shape != downbeat_values.shape:
        raise ValueError(
            "Canonical all-beat and downbeat streams must have equal shape."
        )
    semantic_values = {
        EventChannel.beat: beat_values,
        EventChannel.downbeat: downbeat_values,
    }
    values = np.stack(
        [semantic_values[channel] for channel in resolved.order],
        axis=-1,
    )
    return EventActivations(
        values,
        definition=resolved,
        downbeats_available=downbeats_available,
    )


def convert_beat_activations(
    activations: BeatActivationData[Any],
    target_format: BeatActivationFormat | str,
    *,
    target_definition: BeatDataDefinition | dict[str, Any] | None = None,
    dtype: type[np.floating[Any]] = np.float32,
) -> BeatActivationData[np.ndarray]:
    """Convert tagged NumPy activation data to another explicit format.

    The source datatype is mandatory: two-channel canonical and exclusive
    arrays are numerically indistinguishable in the general case. Channel
    selection always uses the source definition, so arbitrary permutations
    such as ``[downbeat, beat-only, non-beat]`` are handled correctly.
    """
    if not isinstance(
        activations,
        (
            FrameClassActivations,
            EventActivations,
            ExclusiveBeatDownbeatActivations,
        ),
    ):
        raise ActivationFormatMismatchError(
            "convert_beat_activations requires a tagged activation datatype; "
            "bare arrays do not identify their beat semantics."
        )

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
        beat_only = np.asarray(activations.beat_only, dtype=dtype)
        downbeat = np.asarray(activations.downbeats, dtype=dtype)
        exclusive_downbeat = downbeat
        non_beat = np.asarray(activations.non_beats, dtype=dtype)
        all_beats = beat_only + downbeat
    elif isinstance(activations, EventActivations):
        all_beats = np.asarray(activations.all_beats, dtype=dtype)
        downbeat = np.asarray(activations.downbeats, dtype=dtype)
        clipped_beat = np.clip(all_beats, 0.0, 1.0)
        clipped_downbeat = np.clip(downbeat, 0.0, 1.0)
        beat_only = np.clip(clipped_beat - clipped_downbeat, 0.0, None)
        exclusive_downbeat = clipped_downbeat
        non_beat = np.clip(
            1.0 - np.maximum(clipped_beat, clipped_downbeat),
            0.0,
            None,
        )
    else:
        beat_only = np.asarray(activations.beat_only, dtype=dtype)
        downbeat = np.asarray(activations.downbeats, dtype=dtype)
        exclusive_downbeat = downbeat
        all_beats = beat_only + downbeat
        non_beat = np.clip(1.0 - all_beats, 0.0, None)

    if target is BeatActivationFormat.frame_class:
        channel_values = {
            FrameClass.beat: beat_only,
            FrameClass.downbeat: exclusive_downbeat,
            FrameClass.non_beat: non_beat,
        }
        converted = np.stack(
            [channel_values[channel] for channel in resolved_target.order],
            axis=-1,
        ).astype(dtype, copy=False)
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
        converted = np.stack(
            [event_values[channel] for channel in resolved_target.order],
            axis=-1,
        ).astype(dtype, copy=False)
        return EventActivations(
            converted,
            definition=resolved_target,
            downbeats_available=activations.downbeats_available,
        )

    exclusive_values = {
        ExclusiveBeatDownbeatChannel.beat_only: beat_only,
        ExclusiveBeatDownbeatChannel.downbeat: exclusive_downbeat,
    }
    converted = np.stack(
        [exclusive_values[channel] for channel in resolved_target.order],
        axis=-1,
    ).astype(dtype, copy=False)
    return ExclusiveBeatDownbeatActivations(
        converted,
        definition=resolved_target,
        downbeats_available=activations.downbeats_available,
    )


def to_event_activation_data(
    activations: BeatActivationData[Any],
    *,
    dtype: type[np.floating[Any]] = np.float32,
) -> EventActivations[np.ndarray]:
    """Convert tagged NumPy activations to canonical event semantics."""
    converted = convert_beat_activations(
        activations,
        BeatActivationFormat.event_activation,
        dtype=dtype,
    )
    assert isinstance(converted, EventActivations)
    return converted


def to_exclusive_beat_downbeat_activation_data(
    activations: BeatActivationData[Any],
    *,
    dtype: type[np.floating[Any]] = np.float32,
) -> ExclusiveBeatDownbeatActivations[np.ndarray]:
    """Convert tagged NumPy activations to joint-decoder semantics."""
    converted = convert_beat_activations(
        activations,
        BeatActivationFormat.exclusive_beat_downbeat,
        dtype=dtype,
    )
    assert isinstance(converted, ExclusiveBeatDownbeatActivations)
    return converted


def to_frame_class_activation_data(
    activations: BeatActivationData[Any],
    *,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> FrameClassActivations[np.ndarray]:
    """Convert tagged NumPy activations to three exclusive frame classes."""
    converted = convert_beat_activations(
        activations,
        BeatActivationFormat.frame_class,
        target_definition=target_definition,
        dtype=dtype,
    )
    assert isinstance(converted, FrameClassActivations)
    return converted


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
    definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    dtype: type[np.integer[Any]] = np.int64,
) -> np.ndarray:
    """Convert a beat annotation to mutually-exclusive frame classes."""
    resolved = coerce_beat_data_definition(definition)
    if not resolved.is_frame_class:
        raise ActivationFormatMismatchError(
            "annotation_to_frame_classes requires a frame_class definition."
        )
    beat_index = resolved.channel_index(FrameClass.beat)
    downbeat_index = resolved.channel_index(FrameClass.downbeat)
    non_beat_index = resolved.channel_index(FrameClass.non_beat)
    targets = np.full(int(n_frames), non_beat_index, dtype=dtype)
    positions = getattr(annotation, "positions", None)

    for idx, time in enumerate(np.asarray(getattr(annotation, "times"), dtype=float)):
        frame = int(round(float(time) * fps))
        if 0 <= frame < n_frames:
            is_downbeat = positions is not None and int(positions[idx]) == DOWNBEAT_POSITION
            if is_downbeat:
                targets[frame] = downbeat_index
            elif targets[frame] != downbeat_index:
                targets[frame] = beat_index
    return targets


def annotation_to_event_activations(
    annotation: Any,
    n_frames: int,
    fps: float,
    *,
    radius: int = 0,
    shoulder_value: float = 0.5,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert an annotation to canonical ``[all beats, downbeat]`` data."""
    times = np.asarray(getattr(annotation, "times"), dtype=float)

    beat_activations = times_to_activation(
        times,
        n_frames,
        fps,
        radius=radius,
        shoulder_value=shoulder_value,
        dtype=dtype,
    )

    downbeat_times = getattr(annotation, "downbeat_times", None)
    downbeat_activations = (
        times_to_activation(
            downbeat_times,
            n_frames,
            fps,
            radius=radius,
            shoulder_value=shoulder_value,
            dtype=dtype,
        )
        if downbeat_times is not None
        else np.zeros_like(beat_activations)
    )
    return event_activation_data_from_channels(
        beat_activations,
        downbeat_activations,
        definition=definition,
        dtype=dtype,
    ).values


def frame_classes_to_event_activations(
    frame_classes: Any,
    *,
    definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert hard classes to canonical ``[all beats, downbeat]`` data."""
    resolved = coerce_beat_data_definition(definition)
    if not resolved.is_frame_class:
        raise ActivationFormatMismatchError(
            "frame_classes_to_event_activations requires a frame_class "
            "definition."
        )
    classes = np.asarray(frame_classes, dtype=np.int64)
    activations = np.zeros(classes.shape + (NUM_EVENT_CHANNELS,), dtype=dtype)

    beat_mask = classes == resolved.channel_index(FrameClass.beat)
    downbeat_mask = classes == resolved.channel_index(FrameClass.downbeat)
    beat_mask = beat_mask | downbeat_mask

    activations[..., BEAT_CHANNEL] = beat_mask.astype(dtype)
    activations[..., DOWNBEAT_CHANNEL] = downbeat_mask.astype(dtype)
    return activations


def frame_class_activations_to_event_activations(
    frame_activations: Any,
    *,
    definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert exclusive frame classes to canonical event activations.

    Frame classes are mutually exclusive, so the probability of any beat is
    always ``P(beat_only) + P(downbeat)``. Joint decoders should instead use
    :func:`to_exclusive_beat_downbeat_activation_data`.
    """
    source = FrameClassActivations(
        np.asarray(frame_activations, dtype=dtype),
        definition=coerce_beat_data_definition(definition),
    )
    return to_event_activation_data(source, dtype=dtype).values


def data_to_event_activations(
    data: Any,
    definition: BeatDataDefinition | BeatDataRepresentation | str | dict[str, Any],
    *,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert beat data in a declared representation to event activations."""
    resolved = coerce_beat_data_definition(definition)
    if resolved.is_event_activation:
        source: BeatActivationData[Any] = EventActivations(
            np.asarray(data, dtype=dtype),
            definition=resolved,
        )
        return to_event_activation_data(source, dtype=dtype).values
    if resolved.is_frame_class:
        values = np.asarray(data)
        if (
            values.ndim > 0
            and values.shape[-1] == NUM_FRAME_CLASSES
            and not np.issubdtype(values.dtype, np.integer)
        ):
            return frame_class_activations_to_event_activations(
                values,
                definition=resolved,
                dtype=dtype,
            )
        return frame_classes_to_event_activations(
            values,
            definition=resolved,
            dtype=dtype,
        )
    if resolved.is_exclusive_beat_downbeat:
        source = ExclusiveBeatDownbeatActivations(
            np.asarray(data, dtype=dtype),
            definition=resolved,
        )
        return to_event_activation_data(source, dtype=dtype).values
    raise ValueError(f"Unsupported beat data representation: {resolved.representation}")


def event_activations_to_frame_class_activations(
    activations: Any,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert canonical ``[all beats, downbeat]`` activations to classes."""
    source = EventActivations(
        np.asarray(activations, dtype=dtype),
        definition=coerce_beat_data_definition(definition),
    )
    return to_frame_class_activation_data(
        source,
        target_definition=coerce_beat_data_definition(target_definition),
        dtype=dtype,
    ).values


def event_activations_to_exclusive_beat_downbeat_activations(
    activations: Any,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert canonical events to explicit joint-decoder probabilities."""
    source = EventActivations(
        np.asarray(activations, dtype=dtype),
        definition=coerce_beat_data_definition(definition),
    )
    converted = convert_beat_activations(
        source,
        BeatActivationFormat.exclusive_beat_downbeat,
        target_definition=coerce_beat_data_definition(target_definition),
        dtype=dtype,
    )
    assert isinstance(converted, ExclusiveBeatDownbeatActivations)
    return converted.values


def exclusive_beat_downbeat_activations_to_event_activations(
    activations: Any,
    *,
    definition: BeatDataDefinition = EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION,
    dtype: type[np.floating[Any]] = np.float32,
) -> np.ndarray:
    """Convert mutually-exclusive beat-only/downbeat probabilities to events."""
    source = ExclusiveBeatDownbeatActivations(
        np.asarray(activations, dtype=dtype),
        definition=coerce_beat_data_definition(definition),
    )
    return to_event_activation_data(source, dtype=dtype).values


def event_activations_to_frame_classes(
    activations: Any,
    *,
    definition: BeatDataDefinition = EVENT_ACTIVATION_DEFINITION,
    target_definition: BeatDataDefinition = FRAME_CLASS_DEFINITION,
    threshold: float = 0.5,
    downbeat_priority: bool = True,
    dtype: type[np.integer[Any]] = np.int64,
) -> np.ndarray:
    """Convert canonical activations to hard frame-class labels."""
    event_data = EventActivations(
        np.asarray(activations),
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

    frame_classes = np.full(
        event_data.values.shape[:-1],
        non_beat_index,
        dtype=dtype,
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
    "ActivationFormatMismatchError",
    "BEAT_CHANNEL",
    "BEAT_FRAME_CLASS",
    "DOWNBEAT_CHANNEL",
    "DOWNBEAT_FRAME_CLASS",
    "DOWNBEAT_POSITION",
    "EVENT_ACTIVATION_DEFINITION",
    "EVENT_CHANNEL_NAMES",
    "EVENT_CHANNEL_ORDER",
    "EVENT_CHANNEL_SEMANTIC_NAMES",
    "EXCLUSIVE_BEAT_DOWNBEAT_DEFINITION",
    "EXCLUSIVE_BEAT_DOWNBEAT_NAMES",
    "EXCLUSIVE_BEAT_DOWNBEAT_ORDER",
    "FRAME_CLASS_DEFINITION",
    "FRAME_CLASS_NAMES",
    "FRAME_CLASS_ORDER",
    "FRAME_CLASS_SEMANTIC_NAMES",
    "NON_BEAT_FRAME_CLASS",
    "NUM_EVENT_CHANNELS",
    "NUM_EXCLUSIVE_BEAT_DOWNBEAT_CHANNELS",
    "NUM_FRAME_CLASSES",
    "PAD_FRAME_CLASS",
    "UNKNOWN_ACTIVATION",
    "BeatActivationData",
    "BeatActivationFormat",
    "BeatDataDefinition",
    "BeatDataRepresentation",
    "BeatTargetSupervision",
    "EventChannel",
    "EventActivations",
    "ExclusiveBeatDownbeatActivations",
    "ExclusiveBeatDownbeatChannel",
    "FrameClass",
    "FrameClassActivations",
    "annotation_to_event_activations",
    "annotation_to_frame_classes",
    "beat_data_definition_to_dict",
    "coerce_beat_data_definition",
    "convert_beat_activations",
    "data_to_event_activations",
    "event_activations_to_exclusive_beat_downbeat_activations",
    "event_activations_to_frame_class_activations",
    "event_activations_to_frame_classes",
    "event_activation_data_from_channels",
    "exclusive_beat_downbeat_activations_to_event_activations",
    "frame_class_activations_to_event_activations",
    "frame_classes_to_event_activations",
    "require_event_activations",
    "require_exclusive_beat_downbeat_activations",
    "times_to_activation",
    "to_event_activation_data",
    "to_exclusive_beat_downbeat_activation_data",
    "to_frame_class_activation_data",
]
