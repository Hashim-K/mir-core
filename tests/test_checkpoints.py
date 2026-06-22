from __future__ import annotations

import hashlib

from mir_core.checkpoints import beatnet_base_checkpoint_path
from mir_core.checkpoints.beatnet import (
    BASE_CHECKPOINT_FILENAME,
    BASE_CHECKPOINT_SHA256,
)


def test_packaged_beatnet_baseline_checkpoint_matches_expected_hash() -> None:
    path = beatnet_base_checkpoint_path()

    assert path.name == BASE_CHECKPOINT_FILENAME
    assert path.is_file()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == BASE_CHECKPOINT_SHA256
