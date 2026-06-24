from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest

from mir_core.checkpoints import (
    beast_base_checkpoint_path,
    beast_baseline_checkpoint_names,
    beast_baseline_checkpoint_path,
    beatnet_base_checkpoint_path,
    beatnet_baseline_checkpoint_names,
    beatnet_baseline_checkpoint_path,
    beatnet_plus_base_checkpoint_path,
    beatnet_plus_baseline_checkpoint_names,
    beatnet_plus_baseline_checkpoint_path,
    bock_tcn_base_checkpoint_path,
    bock_tcn_baseline_checkpoint_names,
    bock_tcn_baseline_checkpoint_path,
    bocktcn_base_checkpoint_path,
    bocktcn_baseline_checkpoint_names,
    bocktcn_baseline_checkpoint_path,
)
from mir_core.checkpoints.beast import (
    BASELINE_CHECKPOINT_FILENAMES as BEAST_FILENAMES,
    BASELINE_CHECKPOINT_SHA256 as BEAST_SHA256,
)
from mir_core.checkpoints.beatnet import (
    BASELINE_CHECKPOINT_FILENAMES as BEATNET_FILENAMES,
    BASELINE_CHECKPOINT_SHA256 as BEATNET_SHA256,
    BASE_CHECKPOINT_FILENAME as BEATNET_BASE_FILENAME,
    BASE_CHECKPOINT_SHA256 as BEATNET_BASE_SHA256,
)
from mir_core.checkpoints.beatnet_plus import (
    BASELINE_CHECKPOINT_FILENAMES as BEATNET_PLUS_FILENAMES,
    BASELINE_CHECKPOINT_SHA256 as BEATNET_PLUS_SHA256,
)
from mir_core.checkpoints.bocktcn import (
    BASELINE_CHECKPOINT_FILENAMES as BOCKTCN_FILENAMES,
    BASELINE_CHECKPOINT_SHA256 as BOCKTCN_SHA256,
)


CheckpointPathFn = Callable[[str], Path]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_packaged_beatnet_baseline_checkpoint_matches_expected_hash() -> None:
    path = beatnet_base_checkpoint_path()

    assert path.name == BEATNET_BASE_FILENAME
    assert path.is_file()
    assert _sha256(path) == BEATNET_BASE_SHA256


@pytest.mark.parametrize(
    ("path_fn", "filenames", "sha256"),
    [
        (beatnet_baseline_checkpoint_path, BEATNET_FILENAMES, BEATNET_SHA256),
        (
            beatnet_plus_baseline_checkpoint_path,
            BEATNET_PLUS_FILENAMES,
            BEATNET_PLUS_SHA256,
        ),
        (beast_baseline_checkpoint_path, BEAST_FILENAMES, BEAST_SHA256),
        (bocktcn_baseline_checkpoint_path, BOCKTCN_FILENAMES, BOCKTCN_SHA256),
    ],
)
def test_packaged_baseline_checkpoints_match_expected_hashes(
    path_fn: CheckpointPathFn,
    filenames: Mapping[str, str],
    sha256: Mapping[str, str],
) -> None:
    assert set(filenames) == set(sha256)
    assert "baseline" in filenames

    for name, filename in filenames.items():
        path = path_fn(name)

        assert path.name == filename
        assert path.is_file()
        assert _sha256(path) == sha256[name]


def test_base_checkpoint_helpers_return_default_baselines() -> None:
    assert beatnet_base_checkpoint_path() == beatnet_baseline_checkpoint_path()
    assert beatnet_plus_base_checkpoint_path() == beatnet_plus_baseline_checkpoint_path()
    assert beast_base_checkpoint_path() == beast_baseline_checkpoint_path()
    assert bocktcn_base_checkpoint_path() == bocktcn_baseline_checkpoint_path()
    assert bock_tcn_base_checkpoint_path() == bocktcn_base_checkpoint_path()
    assert bock_tcn_baseline_checkpoint_path("baseline_alt0") == bocktcn_baseline_checkpoint_path(
        "baseline_alt0"
    )


def test_top_level_baseline_checkpoint_name_helpers() -> None:
    assert beatnet_baseline_checkpoint_names() == tuple(BEATNET_FILENAMES)
    assert beatnet_plus_baseline_checkpoint_names() == tuple(BEATNET_PLUS_FILENAMES)
    assert beast_baseline_checkpoint_names() == tuple(BEAST_FILENAMES)
    assert bocktcn_baseline_checkpoint_names() == tuple(BOCKTCN_FILENAMES)
    assert bock_tcn_baseline_checkpoint_names() == bocktcn_baseline_checkpoint_names()


def test_unknown_baseline_checkpoint_name_fails_clearly() -> None:
    with pytest.raises(KeyError, match="baseline_alt0"):
        beast_baseline_checkpoint_path("baseline_alt0")
