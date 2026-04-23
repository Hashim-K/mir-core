# tests/test_utils_hashing.py
"""Tests for mir_core.utils.hashing."""
from mir_core.utils.hashing import canonical_json, stable_digest, stable_hash


def test_stable_hash_is_deterministic():
    # Pin the output so algorithm or serialisation regressions are caught.
    assert stable_hash({"a": 1, "b": 2}) == "43258cff783fe703"


def test_stable_hash_is_order_independent():
    config_a = {"a": 1, "b": 2}
    config_b = {"b": 2, "a": 1}
    assert stable_hash(config_a) == stable_hash(config_b)


def test_stable_hash_default_length():
    result = stable_hash({"x": 1})
    assert len(result) == 16
    assert all(c in "0123456789abcdef" for c in result)


def test_stable_hash_custom_length():
    full = stable_hash({"x": 1}, length=64)
    short = stable_hash({"x": 1}, length=8)
    assert short == full[:8]


def test_stable_digest_full_length():
    result = stable_digest({"x": 1})
    assert len(result) == 64  # full SHA-256 hex


def test_canonical_json_sorts_keys():
    a = canonical_json({"b": 2, "a": 1})
    b = canonical_json({"a": 1, "b": 2})
    assert a == b


def test_different_configs_produce_different_hashes():
    assert stable_hash({"a": 1}) != stable_hash({"a": 2})
