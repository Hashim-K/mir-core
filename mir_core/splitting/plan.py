"""Deterministic, group-aware dataset split plans.

The split engine is deliberately unaware of audio profiles, feature caches,
models, and training tasks.  It receives canonical track records and produces
one immutable train/validation/test assignment for every dataset and fold.
Downstream systems can therefore share memberships while choosing their own
audio representation after the split has been selected.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mir_core.utils.hashing import stable_digest

SPLIT_PLAN_SCHEMA_VERSION = "mir-core.split-plan/v1"
SPLIT_ALGORITHM_VERSION = "group-stratified-greedy/v1"
SplitRole = Literal["train", "validation", "test"]
SPLIT_ROLES: tuple[SplitRole, ...] = ("train", "validation", "test")


def _require_non_empty(value: Any, field: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field} must be a non-empty string.")
    return text


def _normalise_strata(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    items = value.items() if isinstance(value, Mapping) else value
    normalised: list[tuple[str, str]] = []
    for raw_key, raw_value in items:
        key = _require_non_empty(raw_key, "strata key")
        item_value = _require_non_empty(raw_value, f"strata[{key!r}]")
        normalised.append((key, item_value))
    if len({key for key, _ in normalised}) != len(normalised):
        raise ValueError("strata keys must be unique.")
    return tuple(sorted(normalised))


@dataclass(frozen=True)
class SplitRecord:
    """Canonical membership record consumed by the split engine.

    ``uid`` identifies a logical song independently of its audio profile.
    ``group_id`` identifies correlated tracks (artist, recording session,
    source song, or another leakage boundary) that must remain in one role.
    ``duration_seconds`` is canonical metadata for downstream budget selection;
    it is not used to decide split membership.
    """

    uid: str
    dataset_id: str
    group_id: str
    strata: tuple[tuple[str, str], ...] = ()
    duration_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "uid", _require_non_empty(self.uid, "uid"))
        object.__setattr__(
            self,
            "dataset_id",
            _require_non_empty(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(
            self,
            "group_id",
            _require_non_empty(self.group_id, "group_id"),
        )
        object.__setattr__(self, "strata", _normalise_strata(self.strata))
        if self.duration_seconds is not None:
            duration = float(self.duration_seconds)
            if not math.isfinite(duration) or duration < 0:
                raise ValueError(
                    "duration_seconds must be finite and non-negative when set."
                )
            object.__setattr__(self, "duration_seconds", duration)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitRecord":
        """Create a record from a JSON-style mapping."""
        allowed = {
            "uid",
            "dataset_id",
            "group_id",
            "strata",
            "duration_seconds",
        }
        unexpected = set(value) - allowed
        if unexpected:
            raise ValueError(f"Unexpected SplitRecord fields: {sorted(unexpected)}.")
        return cls(
            uid=value.get("uid", ""),
            dataset_id=value.get("dataset_id", ""),
            group_id=value.get("group_id", ""),
            strata=value.get("strata", ()),
            duration_seconds=value.get("duration_seconds"),
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "uid": self.uid,
            "dataset_id": self.dataset_id,
            "group_id": self.group_id,
            "strata": dict(self.strata),
        }
        if self.duration_seconds is not None:
            payload["duration_seconds"] = self.duration_seconds
        return payload

    @property
    def stratum_tokens(self) -> tuple[str, ...]:
        return tuple(f"{key}={value}" for key, value in self.strata)


@dataclass(frozen=True)
class FoldAssignment:
    """One dataset's role memberships for one fold."""

    fold_index: int
    train_uids: tuple[str, ...]
    validation_uids: tuple[str, ...]
    test_uids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.fold_index < 0:
            raise ValueError("fold_index must be non-negative.")
        for field in ("train_uids", "validation_uids", "test_uids"):
            values = tuple(
                sorted(_require_non_empty(uid, field) for uid in getattr(self, field))
            )
            if len(values) != len(set(values)):
                raise ValueError(f"{field} contains duplicate UIDs.")
            object.__setattr__(self, field, values)

    def uids(self, role: SplitRole) -> tuple[str, ...]:
        if role == "train":
            return self.train_uids
        if role == "validation":
            return self.validation_uids
        if role == "test":
            return self.test_uids
        raise ValueError(f"Unknown split role {role!r}.")

    @property
    def role_membership_hashes(self) -> dict[str, str]:
        return {role: stable_digest(list(self.uids(role))) for role in SPLIT_ROLES}

    @property
    def membership_hash(self) -> str:
        return stable_digest(
            {
                "fold_index": self.fold_index,
                "roles": {role: list(self.uids(role)) for role in SPLIT_ROLES},
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_index": self.fold_index,
            "roles": {role: list(self.uids(role)) for role in SPLIT_ROLES},
            "role_membership_hashes": self.role_membership_hashes,
            "membership_hash": self.membership_hash,
        }


@dataclass(frozen=True)
class DatasetSplitPlan:
    """All fold assignments for one dataset."""

    dataset_id: str
    folds: tuple[FoldAssignment, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dataset_id",
            _require_non_empty(self.dataset_id, "dataset_id"),
        )
        object.__setattr__(
            self,
            "folds",
            tuple(sorted(self.folds, key=lambda fold: fold.fold_index)),
        )

    def fold(self, fold_index: int) -> FoldAssignment:
        if fold_index < 0 or fold_index >= len(self.folds):
            raise IndexError(f"Dataset {self.dataset_id!r} has no fold {fold_index}.")
        return self.folds[fold_index]


@dataclass(frozen=True)
class SplitPlan:
    """Versioned, immutable split artifact shared by all experiment suites."""

    seed: int
    n_folds: int
    validation_fraction: float
    records: tuple[SplitRecord, ...]
    datasets: tuple[DatasetSplitPlan, ...]
    schema_version: str = SPLIT_PLAN_SCHEMA_VERSION
    algorithm_version: str = SPLIT_ALGORITHM_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SPLIT_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported split plan schema {self.schema_version!r}; "
                f"expected {SPLIT_PLAN_SCHEMA_VERSION!r}."
            )
        if self.algorithm_version != SPLIT_ALGORITHM_VERSION:
            raise ValueError(
                f"Unsupported split algorithm {self.algorithm_version!r}; "
                f"expected {SPLIT_ALGORITHM_VERSION!r}."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2.")
        fraction = float(self.validation_fraction)
        if not 0 <= fraction < 1:
            raise ValueError(
                "validation_fraction must be greater than or equal to 0 "
                "and less than 1."
            )
        object.__setattr__(self, "validation_fraction", fraction)

        records = tuple(
            sorted(self.records, key=lambda record: (record.dataset_id, record.uid))
        )
        if not records:
            raise ValueError("A split plan requires at least one record.")
        uids = [record.uid for record in records]
        if len(uids) != len(set(uids)):
            duplicates = sorted(
                uid for uid, count in Counter(uids).items() if count > 1
            )
            raise ValueError(
                f"Split record UIDs must be globally unique: {duplicates}."
            )
        object.__setattr__(self, "records", records)

        datasets = tuple(sorted(self.datasets, key=lambda dataset: dataset.dataset_id))
        dataset_ids = [dataset.dataset_id for dataset in datasets]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise ValueError("Dataset split IDs must be unique.")
        object.__setattr__(self, "datasets", datasets)
        self._validate_memberships()

    @property
    def records_hash(self) -> str:
        return stable_digest([record.to_dict() for record in self.records])

    @property
    def membership_hash(self) -> str:
        return stable_digest(self._membership_payload())

    @property
    def plan_hash(self) -> str:
        return stable_digest(self._payload_without_plan_hash())

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(dataset.dataset_id for dataset in self.datasets)

    def dataset(self, dataset_id: str) -> DatasetSplitPlan:
        for dataset in self.datasets:
            if dataset.dataset_id == dataset_id:
                return dataset
        raise KeyError(f"Unknown split-plan dataset {dataset_id!r}.")

    def fold(self, dataset_id: str, fold_index: int) -> FoldAssignment:
        return self.dataset(dataset_id).fold(fold_index)

    def uids(
        self,
        dataset_id: str,
        fold_index: int,
        role: SplitRole,
    ) -> tuple[str, ...]:
        return self.fold(dataset_id, fold_index).uids(role)

    def records_for(
        self,
        dataset_id: str,
        fold_index: int,
        role: SplitRole,
    ) -> tuple[SplitRecord, ...]:
        wanted = set(self.uids(dataset_id, fold_index, role))
        return tuple(record for record in self.records if record.uid in wanted)

    def _records_by_dataset(self) -> dict[str, tuple[SplitRecord, ...]]:
        grouped: dict[str, list[SplitRecord]] = defaultdict(list)
        for record in self.records:
            grouped[record.dataset_id].append(record)
        return {
            dataset_id: tuple(dataset_records)
            for dataset_id, dataset_records in grouped.items()
        }

    def _validate_memberships(self) -> None:
        records_by_dataset = self._records_by_dataset()
        if set(records_by_dataset) != set(self.dataset_ids):
            missing = sorted(set(records_by_dataset) - set(self.dataset_ids))
            extra = sorted(set(self.dataset_ids) - set(records_by_dataset))
            raise ValueError(
                "Dataset plans must exactly match record datasets; "
                f"missing={missing}, extra={extra}."
            )

        for dataset in self.datasets:
            if len(dataset.folds) != self.n_folds:
                raise ValueError(
                    f"Dataset {dataset.dataset_id!r} has {len(dataset.folds)} "
                    f"folds; expected {self.n_folds}."
                )
            if tuple(fold.fold_index for fold in dataset.folds) != tuple(
                range(self.n_folds)
            ):
                raise ValueError(
                    f"Dataset {dataset.dataset_id!r} fold indexes must be "
                    f"0..{self.n_folds - 1}."
                )

            dataset_records = records_by_dataset[dataset.dataset_id]
            expected_uids = {record.uid for record in dataset_records}
            group_by_uid = {record.uid: record.group_id for record in dataset_records}
            test_counts: Counter[str] = Counter()
            for fold in dataset.folds:
                role_sets = {role: set(fold.uids(role)) for role in SPLIT_ROLES}
                union = set().union(*role_sets.values())
                if union != expected_uids:
                    missing = sorted(expected_uids - union)
                    extra = sorted(union - expected_uids)
                    raise ValueError(
                        f"Dataset {dataset.dataset_id!r} fold "
                        f"{fold.fold_index} does not partition its records; "
                        f"missing={missing}, extra={extra}."
                    )
                for left_index, left in enumerate(SPLIT_ROLES):
                    for right in SPLIT_ROLES[left_index + 1 :]:
                        overlap = role_sets[left] & role_sets[right]
                        if overlap:
                            raise ValueError(
                                f"Dataset {dataset.dataset_id!r} fold "
                                f"{fold.fold_index} roles {left!r} and "
                                f"{right!r} overlap: {sorted(overlap)}."
                            )
                role_by_group: dict[str, str] = {}
                for role, role_uids in role_sets.items():
                    for uid in role_uids:
                        group_id = group_by_uid[uid]
                        previous = role_by_group.setdefault(group_id, role)
                        if previous != role:
                            raise ValueError(
                                f"Dataset {dataset.dataset_id!r} group "
                                f"{group_id!r} crosses {previous!r} and "
                                f"{role!r} in fold {fold.fold_index}."
                            )
                test_counts.update(role_sets["test"])

            incorrect = sorted(uid for uid in expected_uids if test_counts[uid] != 1)
            if incorrect:
                raise ValueError(
                    f"Every UID must be test exactly once across folds for "
                    f"dataset {dataset.dataset_id!r}: {incorrect}."
                )

        group_by_uid = {record.uid: record.group_id for record in self.records}
        for fold_index in range(self.n_folds):
            role_by_group: dict[str, str] = {}
            for dataset in self.datasets:
                fold = dataset.fold(fold_index)
                for role in SPLIT_ROLES:
                    for uid in fold.uids(role):
                        group_id = group_by_uid[uid]
                        previous = role_by_group.setdefault(group_id, role)
                        if previous != role:
                            raise ValueError(
                                f"Global group {group_id!r} crosses "
                                f"{previous!r} and {role!r} in fold "
                                f"{fold_index}."
                            )

    def _membership_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "algorithm_version": self.algorithm_version,
            "seed": self.seed,
            "n_folds": self.n_folds,
            "validation_fraction": self.validation_fraction,
            "datasets": {
                dataset.dataset_id: {
                    "folds": [
                        {
                            "fold_index": fold.fold_index,
                            "roles": {
                                role: list(fold.uids(role)) for role in SPLIT_ROLES
                            },
                        }
                        for fold in dataset.folds
                    ]
                }
                for dataset in self.datasets
            },
        }

    def _payload_without_plan_hash(self) -> dict[str, Any]:
        records_by_dataset = self._records_by_dataset()
        return {
            **self._membership_payload(),
            "records": [record.to_dict() for record in self.records],
            "records_hash": self.records_hash,
            "membership_hash": self.membership_hash,
            "datasets": {
                dataset.dataset_id: {
                    "records_hash": stable_digest(
                        [
                            record.to_dict()
                            for record in records_by_dataset[dataset.dataset_id]
                        ]
                    ),
                    "folds": [fold.to_dict() for fold in dataset.folds],
                }
                for dataset in self.datasets
            },
        }

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-compatible artifact payload."""
        return {
            **self._payload_without_plan_hash(),
            "plan_hash": self.plan_hash,
        }

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        *,
        expected_hash: str | None = None,
    ) -> "SplitPlan":
        """Load and fully validate a split-plan artifact."""
        try:
            records = tuple(
                SplitRecord.from_mapping(record) for record in value["records"]
            )
            datasets = tuple(
                DatasetSplitPlan(
                    dataset_id=dataset_id,
                    folds=tuple(
                        FoldAssignment(
                            fold_index=int(fold["fold_index"]),
                            train_uids=tuple(fold["roles"]["train"]),
                            validation_uids=tuple(fold["roles"]["validation"]),
                            test_uids=tuple(fold["roles"]["test"]),
                        )
                        for fold in dataset_value["folds"]
                    ),
                )
                for dataset_id, dataset_value in value["datasets"].items()
            )
            plan = cls(
                seed=value["seed"],
                n_folds=int(value["n_folds"]),
                validation_fraction=float(value["validation_fraction"]),
                records=records,
                datasets=datasets,
                schema_version=value["schema_version"],
                algorithm_version=value["algorithm_version"],
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(f"Malformed split-plan payload: {exc}.") from exc

        canonical = plan.to_dict()
        if dict(value) != canonical:
            for hash_field in (
                "records_hash",
                "membership_hash",
                "plan_hash",
            ):
                if value.get(hash_field) != canonical[hash_field]:
                    raise ValueError(
                        f"Split-plan {hash_field} does not match its contents."
                    )
            raise ValueError("Split-plan payload is not canonical or is inconsistent.")
        if expected_hash is not None and plan.plan_hash != expected_hash:
            raise ValueError(
                f"Split-plan hash {plan.plan_hash!r} does not match expected "
                f"hash {expected_hash!r}."
            )
        return plan


@dataclass(frozen=True)
class _Group:
    group_id: str
    records: tuple[SplitRecord, ...]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def uids(self) -> tuple[str, ...]:
        return tuple(record.uid for record in self.records)

    @property
    def dataset_ids(self) -> tuple[str, ...]:
        return tuple(sorted({record.dataset_id for record in self.records}))

    @property
    def balance_counts(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for record in self.records:
            counts[f"dataset={record.dataset_id}"] += 1
            for token in record.stratum_tokens:
                counts[f"stratum={record.dataset_id}:{token}"] += 1
        return counts


def _derived_rank(*parts: Any) -> int:
    return int(stable_digest(list(parts))[:16], 16)


def _make_groups(records: Sequence[SplitRecord]) -> tuple[_Group, ...]:
    grouped: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in records:
        grouped[record.group_id].append(record)
    return tuple(
        _Group(
            group_id=group_id,
            records=tuple(
                sorted(
                    group_records,
                    key=lambda record: (record.dataset_id, record.uid),
                )
            ),
        )
        for group_id, group_records in sorted(grouped.items())
    )


def _group_order(
    groups: Sequence[_Group],
    *,
    seed: int,
    dataset_id: str,
    purpose: str,
) -> list[_Group]:
    total_strata: Counter[str] = Counter()
    for group in groups:
        total_strata.update(group.balance_counts)

    def priority(group: _Group) -> tuple[Any, ...]:
        rarity = min(
            (total_strata[token] for token in group.balance_counts),
            default=sum(item.size for item in groups),
        )
        return (
            rarity,
            -group.size,
            _derived_rank(seed, dataset_id, purpose, group.group_id),
            group.group_id,
        )

    return sorted(groups, key=priority)


def _addition_delta(
    *,
    current_size: int,
    current_strata: Counter[str],
    group: _Group,
    target_size: float,
    target_strata: Mapping[str, float],
) -> float:
    size_scale = max(target_size, 1.0)
    before = ((current_size - target_size) / size_scale) ** 2
    after = ((current_size + group.size - target_size) / size_scale) ** 2
    delta = after - before
    if not target_strata:
        return delta

    stratum_delta = 0.0
    for token, amount in group.balance_counts.items():
        target = target_strata[token]
        scale = max(target, 1.0)
        current = current_strata[token]
        stratum_delta += ((current + amount - target) / scale) ** 2 - (
            (current - target) / scale
        ) ** 2
    return delta + stratum_delta / len(target_strata)


def _assign_test_groups(
    groups: Sequence[_Group],
    *,
    seed: int,
    component_id: str,
    n_folds: int,
) -> tuple[tuple[_Group, ...], ...]:
    total_size = sum(group.size for group in groups)
    total_strata: Counter[str] = Counter()
    for group in groups:
        total_strata.update(group.balance_counts)
    target_size = total_size / n_folds
    target_strata = {token: count / n_folds for token, count in total_strata.items()}
    bins: list[list[_Group]] = [[] for _ in range(n_folds)]
    bin_sizes = [0] * n_folds
    bin_strata = [Counter() for _ in range(n_folds)]

    for group in _group_order(
        groups,
        seed=seed,
        dataset_id=component_id,
        purpose="test",
    ):
        empty_bins = [index for index, items in enumerate(bins) if not items]
        candidates = empty_bins or list(range(n_folds))
        selected = min(
            candidates,
            key=lambda index: (
                _addition_delta(
                    current_size=bin_sizes[index],
                    current_strata=bin_strata[index],
                    group=group,
                    target_size=target_size,
                    target_strata=target_strata,
                ),
                bin_sizes[index],
                _derived_rank(
                    seed,
                    component_id,
                    "test-bin",
                    group.group_id,
                    index,
                ),
                index,
            ),
        )
        bins[selected].append(group)
        bin_sizes[selected] += group.size
        bin_strata[selected].update(group.balance_counts)
    return tuple(tuple(items) for items in bins)


def _select_validation_groups(
    groups: Sequence[_Group],
    *,
    seed: int,
    component_id: str,
    fold_index: int,
    validation_fraction: float,
) -> tuple[_Group, ...]:
    if validation_fraction == 0:
        return ()

    target_size = sum(group.size for group in groups) * validation_fraction
    total_strata: Counter[str] = Counter()
    for group in groups:
        total_strata.update(group.balance_counts)
    target_strata = {
        token: count * validation_fraction for token, count in total_strata.items()
    }
    remaining = _group_order(
        groups,
        seed=seed,
        dataset_id=component_id,
        purpose=f"validation:{fold_index}",
    )
    selected: list[_Group] = []
    selected_size = 0
    selected_strata: Counter[str] = Counter()
    remaining_dataset_group_counts: Counter[str] = Counter(
        dataset_id for group in groups for dataset_id in group.dataset_ids
    )
    insufficient = sorted(
        dataset_id
        for dataset_id, count in remaining_dataset_group_counts.items()
        if count < 2
    )
    if insufficient:
        raise ValueError(
            f"Fold {fold_index} needs at least two non-test groups for every "
            f"dataset; insufficient={insufficient}."
        )

    def distance(size: int, strata: Counter[str]) -> float:
        scale = max(target_size, 1.0)
        result = ((size - target_size) / scale) ** 2
        if target_strata:
            result += sum(
                ((strata[token] - target) / max(target, 1.0)) ** 2
                for token, target in target_strata.items()
            ) / len(target_strata)
        return result

    def can_select(group: _Group) -> bool:
        return all(
            remaining_dataset_group_counts[dataset_id] > 1
            for dataset_id in group.dataset_ids
        )

    def mark_selected(group: _Group) -> None:
        nonlocal selected_size
        selected.append(group)
        remaining.remove(group)
        selected_size += group.size
        selected_strata.update(group.balance_counts)
        for dataset_id in group.dataset_ids:
            remaining_dataset_group_counts[dataset_id] -= 1

    def candidate_key(group: _Group) -> tuple[Any, ...]:
        return (
            distance(
                selected_size + group.size,
                selected_strata + group.balance_counts,
            ),
            _derived_rank(
                seed,
                component_id,
                "validation-choice",
                fold_index,
                group.group_id,
            ),
            group.group_id,
        )

    for dataset_id in sorted(remaining_dataset_group_counts):
        if any(dataset_id in group.dataset_ids for group in selected):
            continue
        candidates = [
            group
            for group in remaining
            if dataset_id in group.dataset_ids and can_select(group)
        ]
        if not candidates:
            raise ValueError(
                f"Fold {fold_index} cannot assign validation data for "
                f"dataset {dataset_id!r} while retaining training data."
            )
        chosen = min(candidates, key=candidate_key)
        mark_selected(chosen)

    current_distance = distance(selected_size, selected_strata)
    while remaining:
        candidates = [group for group in remaining if can_select(group)]
        if not candidates:
            break
        best = min(candidates, key=candidate_key)
        candidate_distance = distance(
            selected_size + best.size,
            selected_strata + best.balance_counts,
        )
        if candidate_distance >= current_distance:
            break
        mark_selected(best)
        current_distance = candidate_distance
    return tuple(selected)


def _uids(
    groups: Iterable[_Group],
    *,
    dataset_id: str,
) -> tuple[str, ...]:
    return tuple(
        sorted(
            record.uid
            for group in groups
            for record in group.records
            if record.dataset_id == dataset_id
        )
    )


def _dataset_components(
    groups: Sequence[_Group],
) -> tuple[tuple[tuple[str, ...], tuple[_Group, ...]], ...]:
    parents: dict[str, str] = {}

    def find(dataset_id: str) -> str:
        parents.setdefault(dataset_id, dataset_id)
        while parents[dataset_id] != dataset_id:
            parents[dataset_id] = parents[parents[dataset_id]]
            dataset_id = parents[dataset_id]
        return dataset_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            smaller, larger = sorted((left_root, right_root))
            parents[larger] = smaller

    for group in groups:
        for dataset_id in group.dataset_ids:
            find(dataset_id)
        for dataset_id in group.dataset_ids[1:]:
            union(group.dataset_ids[0], dataset_id)

    datasets_by_root: dict[str, set[str]] = defaultdict(set)
    for dataset_id in parents:
        datasets_by_root[find(dataset_id)].add(dataset_id)

    components = []
    for dataset_ids in datasets_by_root.values():
        component_groups = tuple(
            group for group in groups if set(group.dataset_ids) & dataset_ids
        )
        components.append((tuple(sorted(dataset_ids)), component_groups))
    return tuple(sorted(components))


def build_split_plan(
    records: Iterable[SplitRecord | Mapping[str, Any]],
    *,
    seed: int,
    n_folds: int,
    validation_fraction: float,
) -> SplitPlan:
    """Build a deterministic split plan for every input dataset.

    Each disconnected dataset is split independently.  Datasets that share a
    ``group_id`` form a connected component and are allocated together so the
    shared group can never cross roles.  Adding an unrelated dataset therefore
    cannot change an existing dataset's memberships.
    """
    canonical_records = tuple(
        record if isinstance(record, SplitRecord) else SplitRecord.from_mapping(record)
        for record in records
    )
    uids = [record.uid for record in canonical_records]
    if len(uids) != len(set(uids)):
        duplicates = sorted(uid for uid, count in Counter(uids).items() if count > 1)
        raise ValueError(f"Split record UIDs must be globally unique: {duplicates}.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer.")
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    fraction = float(validation_fraction)
    if not 0 <= fraction < 1:
        raise ValueError(
            "validation_fraction must be greater than or equal to 0 " "and less than 1."
        )

    grouped_records: dict[str, list[SplitRecord]] = defaultdict(list)
    for record in canonical_records:
        grouped_records[record.dataset_id].append(record)
    if not grouped_records:
        raise ValueError("A split plan requires at least one record.")

    groups = _make_groups(canonical_records)
    dataset_group_counts: Counter[str] = Counter(
        dataset_id for group in groups for dataset_id in group.dataset_ids
    )
    insufficient = sorted(
        dataset_id
        for dataset_id, count in dataset_group_counts.items()
        if count < n_folds
    )
    if insufficient:
        details = {
            dataset_id: dataset_group_counts[dataset_id] for dataset_id in insufficient
        }
        raise ValueError(
            f"Datasets have fewer independent groups than n_folds={n_folds}: "
            f"{details}."
        )

    assignments: dict[tuple[str, int], dict[str, set[str]]] = {
        (dataset_id, fold_index): {role: set() for role in SPLIT_ROLES}
        for dataset_id in grouped_records
        for fold_index in range(n_folds)
    }
    for component_datasets, component_groups in _dataset_components(groups):
        component_id = "|".join(component_datasets)
        test_folds = _assign_test_groups(
            component_groups,
            seed=seed,
            component_id=component_id,
            n_folds=n_folds,
        )
        all_group_ids = {group.group_id for group in component_groups}
        group_lookup = {group.group_id: group for group in component_groups}
        for fold_index, test_groups in enumerate(test_folds):
            test_group_ids = {group.group_id for group in test_groups}
            remaining_groups = tuple(
                group_lookup[group_id]
                for group_id in sorted(all_group_ids - test_group_ids)
            )
            validation_groups = _select_validation_groups(
                remaining_groups,
                seed=seed,
                component_id=component_id,
                fold_index=fold_index,
                validation_fraction=fraction,
            )
            validation_group_ids = {group.group_id for group in validation_groups}
            train_groups = tuple(
                group
                for group in remaining_groups
                if group.group_id not in validation_group_ids
            )
            roles = {
                "train": train_groups,
                "validation": validation_groups,
                "test": test_groups,
            }
            for dataset_id in component_datasets:
                for role, role_groups in roles.items():
                    assignments[(dataset_id, fold_index)][role].update(
                        _uids(role_groups, dataset_id=dataset_id)
                    )

    dataset_plans: list[DatasetSplitPlan] = []
    for dataset_id in sorted(grouped_records):
        folds = tuple(
            FoldAssignment(
                fold_index=fold_index,
                train_uids=tuple(assignments[(dataset_id, fold_index)]["train"]),
                validation_uids=tuple(
                    assignments[(dataset_id, fold_index)]["validation"]
                ),
                test_uids=tuple(assignments[(dataset_id, fold_index)]["test"]),
            )
            for fold_index in range(n_folds)
        )
        dataset_plans.append(DatasetSplitPlan(dataset_id=dataset_id, folds=folds))

    return SplitPlan(
        seed=seed,
        n_folds=n_folds,
        validation_fraction=fraction,
        records=canonical_records,
        datasets=tuple(dataset_plans),
    )


def write_split_plan(path: str | Path, plan: SplitPlan) -> Path:
    """Write a canonical split-plan JSON artifact."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_split_plan(
    path: str | Path,
    *,
    expected_hash: str | None = None,
) -> SplitPlan:
    """Load and validate a split-plan JSON artifact."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read split plan {source}: {exc}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"Split plan {source} must contain a JSON object.")
    return SplitPlan.from_dict(payload, expected_hash=expected_hash)


__all__ = [
    "SPLIT_ALGORITHM_VERSION",
    "SPLIT_PLAN_SCHEMA_VERSION",
    "SPLIT_ROLES",
    "DatasetSplitPlan",
    "FoldAssignment",
    "SplitPlan",
    "SplitRecord",
    "SplitRole",
    "build_split_plan",
    "load_split_plan",
    "write_split_plan",
]
