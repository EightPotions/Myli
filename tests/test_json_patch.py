"""Tests for Myli's dependency-free RFC 6902 implementation."""

from __future__ import annotations

import unittest

from myli import JsonPatchError, apply_json_patch


def test_add_remove_replace_and_escaped_pointers() -> None:
    original = {
        "title": "Old",
        "elements": [{"id": "one"}],
        "a/b": {"~key": False},
    }

    result = apply_json_patch(
        original,
        [
            {"op": "replace", "path": "/title", "value": "New"},
            {"op": "add", "path": "/elements/-", "value": {"id": "two"}},
            {"op": "remove", "path": "/elements/0"},
            {"op": "replace", "path": "/a~1b/~0key", "value": True},
        ],
    )

    assert result == {
        "title": "New",
        "elements": [{"id": "two"}],
        "a/b": {"~key": True},
    }
    assert original == {
        "title": "Old",
        "elements": [{"id": "one"}],
        "a/b": {"~key": False},
    }


def test_move_copy_and_test_follow_rfc_operation_order() -> None:
    result = apply_json_patch(
        {"items": ["a", "b", "c"], "meta": {}},
        [
            {"op": "test", "path": "/items/1", "value": "b"},
            {"op": "move", "from": "/items/0", "path": "/items/2"},
            {"op": "copy", "from": "/items/0", "path": "/meta/first"},
        ],
    )

    assert result == {"items": ["b", "c", "a"], "meta": {"first": "b"}}


def test_root_replacement_is_supported() -> None:
    assert apply_json_patch(
        {"old": True},
        [{"op": "replace", "path": "", "value": {"new": True}}],
    ) == {"new": True}


def test_unrecognized_operation_members_are_ignored_per_rfc_6902() -> None:
    assert apply_json_patch(
        {"title": "Old"},
        [
            {
                "op": "replace",
                "path": "/title",
                "value": "New",
                "comment": "ignored extension member",
            }
        ],
    ) == {"title": "New"}


def test_failed_test_and_invalid_array_indexes_are_rejected() -> None:
    with unittest.TestCase().assertRaisesRegex(JsonPatchError, "test did not match"):
        apply_json_patch(
            {"enabled": True},
            [{"op": "test", "path": "/enabled", "value": 1}],
        )

    with unittest.TestCase().assertRaisesRegex(JsonPatchError, "array index '01'"):
        apply_json_patch(
            {"items": ["a"]},
            [{"op": "remove", "path": "/items/01"}],
        )


def test_move_cannot_target_its_own_child() -> None:
    with unittest.TestCase().assertRaisesRegex(JsonPatchError, "cannot be a child"):
        apply_json_patch(
            {"node": {"child": {}}},
            [{"op": "move", "from": "/node", "path": "/node/child/moved"}],
        )


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
