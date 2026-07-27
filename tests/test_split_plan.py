from __future__ import annotations

import copy

import pytest

from mir_core.splitting import (
    SPLIT_ALGORITHM_VERSION,
    SplitPlan,
    SplitRecord,
    build_split_plan,
    load_split_plan,
    write_split_plan,
)


def _records(
    dataset_id: str,
    *,
    groups: int = 10,
    tracks_per_group: int = 2,
) -> list[SplitRecord]:
    records: list[SplitRecord] = []
    for group_index in range(groups):
        for track_index in range(tracks_per_group):
            records.append(
                SplitRecord(
                    uid=(f"{dataset_id}:group-{group_index}:" f"track-{track_index}"),
                    dataset_id=dataset_id,
                    group_id=f"{dataset_id}:artist-{group_index}",
                    strata={
                        "label": "latin" if group_index % 2 else "other",
                        "source": f"source-{group_index % 3}",
                    },
                    duration_seconds=30.0 + group_index,
                )
            )
    return records


def test_build_split_plan_is_group_isolated_and_complete() -> None:
    records = _records("primary")
    plan = build_split_plan(
        records,
        seed=42,
        n_folds=5,
        validation_fraction=0.2,
    )

    assert plan.algorithm_version == SPLIT_ALGORITHM_VERSION
    assert plan.dataset_ids == ("primary",)
    record_by_uid = {record.uid: record for record in records}
    test_counts = {record.uid: 0 for record in records}
    for fold_index in range(5):
        fold = plan.fold("primary", fold_index)
        role_sets = {
            role: set(fold.uids(role)) for role in ("train", "validation", "test")
        }
        assert set().union(*role_sets.values()) == set(record_by_uid)
        assert not (role_sets["train"] & role_sets["validation"])
        assert not (role_sets["train"] & role_sets["test"])
        assert not (role_sets["validation"] & role_sets["test"])
        assert role_sets["train"]
        assert role_sets["validation"]
        assert role_sets["test"]

        roles_by_group: dict[str, set[str]] = {}
        for role, uids in role_sets.items():
            for uid in uids:
                group_id = record_by_uid[uid].group_id
                roles_by_group.setdefault(group_id, set()).add(role)
                if role == "test":
                    test_counts[uid] += 1
        assert all(len(roles) == 1 for roles in roles_by_group.values())

    assert set(test_counts.values()) == {1}


def test_build_split_plan_is_input_order_invariant() -> None:
    records = _records("primary") + _records("background")
    forward = build_split_plan(
        records,
        seed=17,
        n_folds=5,
        validation_fraction=0.1,
    )
    reverse = build_split_plan(
        reversed(records),
        seed=17,
        n_folds=5,
        validation_fraction=0.1,
    )

    assert forward.to_dict() == reverse.to_dict()


def test_dataset_memberships_are_independent() -> None:
    primary = _records("primary")
    primary_only = build_split_plan(
        primary,
        seed=91,
        n_folds=5,
        validation_fraction=0.2,
    )
    combined = build_split_plan(
        primary + _records("new-dataset", groups=12),
        seed=91,
        n_folds=5,
        validation_fraction=0.2,
    )

    assert [fold.to_dict() for fold in primary_only.dataset("primary").folds] == [
        fold.to_dict() for fold in combined.dataset("primary").folds
    ]


def test_group_shared_across_datasets_never_crosses_roles() -> None:
    first = _records("first")
    second = _records("second")
    shared_group = "shared-artist"
    for records in (first, second):
        for index in (0, 1):
            original = records[index]
            records[index] = SplitRecord(
                uid=original.uid,
                dataset_id=original.dataset_id,
                group_id=shared_group,
                strata=original.strata,
                duration_seconds=original.duration_seconds,
            )

    plan = build_split_plan(
        first + second,
        seed=42,
        n_folds=5,
        validation_fraction=0.2,
    )

    shared_uids = {
        record.uid for record in first + second if record.group_id == shared_group
    }
    for fold_index in range(5):
        roles = {}
        for dataset_id in ("first", "second"):
            fold = plan.fold(dataset_id, fold_index)
            for role in ("train", "validation", "test"):
                for uid in set(fold.uids(role)) & shared_uids:
                    roles[uid] = role
        assert len(set(roles.values())) == 1


def test_seed_changes_membership() -> None:
    records = _records("primary", groups=20, tracks_per_group=1)
    first = build_split_plan(
        records,
        seed=1,
        n_folds=5,
        validation_fraction=0.2,
    )
    second = build_split_plan(
        records,
        seed=2,
        n_folds=5,
        validation_fraction=0.2,
    )

    assert first.membership_hash != second.membership_hash


def test_split_plan_algorithm_golden_membership_and_hashes() -> None:
    records = [
        SplitRecord(
            uid=f"d:{index}",
            dataset_id="d",
            group_id=f"d:g{index}",
            strata={"label": "a" if index % 2 else "b"},
            duration_seconds=10 + index,
        )
        for index in range(9)
    ]

    plan = build_split_plan(
        records,
        seed=7,
        n_folds=3,
        validation_fraction=0.25,
    )

    assert [
        (fold.train_uids, fold.validation_uids, fold.test_uids)
        for fold in plan.dataset("d").folds
    ] == [
        (
            ("d:1", "d:5", "d:6", "d:8"),
            ("d:4", "d:7"),
            ("d:0", "d:2", "d:3"),
        ),
        (
            ("d:2", "d:3", "d:4", "d:5", "d:8"),
            ("d:0",),
            ("d:1", "d:6", "d:7"),
        ),
        (
            ("d:2", "d:3", "d:6", "d:7"),
            ("d:0", "d:1"),
            ("d:4", "d:5", "d:8"),
        ),
    ]
    assert (
        plan.records_hash
        == "667663afd9d3ed3e4d312b6834153fafcbdd72b932840aa27267e5181a285d36"
    )
    assert (
        plan.membership_hash
        == "bd74c9c1c1b8d9aa5095bde82be04dd49036c77420a131444bf6ba86b70ec307"
    )
    assert (
        plan.plan_hash
        == "2afb50c64453d12e8f5f1a7eef387c425e7750aaf73410dc697b1d275c87b629"
    )


def test_duration_changes_plan_hash_but_not_membership_hash() -> None:
    records = _records("primary")
    first = build_split_plan(
        records,
        seed=5,
        n_folds=5,
        validation_fraction=0.2,
    )
    changed_records = list(records)
    original = changed_records[0]
    changed_records[0] = SplitRecord(
        uid=original.uid,
        dataset_id=original.dataset_id,
        group_id=original.group_id,
        strata=original.strata,
        duration_seconds=999.0,
    )
    second = build_split_plan(
        changed_records,
        seed=5,
        n_folds=5,
        validation_fraction=0.2,
    )

    assert first.membership_hash == second.membership_hash
    assert first.records_hash != second.records_hash
    assert first.plan_hash != second.plan_hash


def test_split_plan_round_trip_and_expected_hash(tmp_path) -> None:
    plan = build_split_plan(
        _records("primary"),
        seed=11,
        n_folds=5,
        validation_fraction=0.2,
    )
    path = write_split_plan(tmp_path / "nested" / "split-plan.json", plan)

    loaded = load_split_plan(path, expected_hash=plan.plan_hash)

    assert loaded == plan
    assert loaded.to_dict() == plan.to_dict()
    with pytest.raises(ValueError, match="does not match expected"):
        load_split_plan(path, expected_hash="0" * 64)


def test_split_plan_rejects_tampered_payload() -> None:
    plan = build_split_plan(
        _records("primary"),
        seed=11,
        n_folds=5,
        validation_fraction=0.2,
    )
    payload = copy.deepcopy(plan.to_dict())
    payload["records"][0]["duration_seconds"] = 12345.0

    with pytest.raises(ValueError, match="records_hash"):
        SplitPlan.from_dict(payload)


def test_validation_can_be_disabled() -> None:
    plan = build_split_plan(
        _records("primary"),
        seed=3,
        n_folds=5,
        validation_fraction=0.0,
    )

    assert all(not fold.validation_uids for fold in plan.dataset("primary").folds)


def test_too_few_independent_groups_fails() -> None:
    records = _records("primary", groups=4)

    with pytest.raises(ValueError, match="fewer independent groups"):
        build_split_plan(
            records,
            seed=42,
            n_folds=5,
            validation_fraction=0.1,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("uid", "", "uid"),
        ("dataset_id", "", "dataset_id"),
        ("group_id", "", "group_id"),
        ("duration_seconds", -1, "duration_seconds"),
    ],
)
def test_split_record_rejects_invalid_values(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs: dict[str, object] = {
        "uid": "dataset:track",
        "dataset_id": "dataset",
        "group_id": "group",
        "duration_seconds": 1.0,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=message):
        SplitRecord(**kwargs)


def test_duplicate_uids_fail_even_across_datasets() -> None:
    records = _records("primary")
    duplicate = SplitRecord(
        uid=records[0].uid,
        dataset_id="other",
        group_id="other-group",
    )

    with pytest.raises(ValueError, match="globally unique"):
        build_split_plan(
            records + [duplicate],
            seed=42,
            n_folds=5,
            validation_fraction=0.1,
        )
