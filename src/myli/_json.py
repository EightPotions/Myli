"""Strict JSON helpers shared by Myli's trust boundaries."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any, NoReturn


def strict_json_loads(value: str) -> Any:
    """Parse JSON while rejecting JavaScript-style non-finite constants."""

    parsed = json.loads(value, parse_constant=_reject_non_finite_constant)
    validate_json_value(parsed)
    return parsed


def validate_json_value(value: Any, *, path: str = "$") -> None:
    """Require values that can be represented by interoperable JSON."""

    _validate_json_value(value, path=path, ancestors=set())


def canonical_json_value(value: Any) -> Any:
    """Return an isolated JSON value with every object ordered by key."""

    validate_json_value(value)
    return _canonical_json_value(value)


def canonical_json_dumps(value: Any) -> str:
    """Serialize strict JSON with stable object ordering and no whitespace."""

    return json.dumps(
        canonical_json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def first_json_difference(left: Any, right: Any, *, pointer: str = "") -> str | None:
    """Return the first RFC 6901 pointer where two JSON values differ."""

    if isinstance(left, Mapping) and isinstance(right, Mapping):
        for key in sorted(set(left).union(right)):
            child_pointer = f"{pointer}/{_pointer_token(key)}"
            if key not in left or key not in right:
                return child_pointer
            difference = first_json_difference(left[key], right[key], pointer=child_pointer)
            if difference is not None:
                return difference
        return None

    if isinstance(left, list) and isinstance(right, list):
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_pointer = f"{pointer}/{index}"
            difference = first_json_difference(left_item, right_item, pointer=child_pointer)
            if difference is not None:
                return difference
        if len(left) != len(right):
            return f"{pointer}/{min(len(left), len(right))}"
        return None

    if isinstance(left, (Mapping, list)) or isinstance(right, (Mapping, list)):
        return pointer

    if json.dumps(
        left,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ) == json.dumps(
        right,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ):
        return None
    return pointer


def _validate_json_value(
    value: Any,
    *,
    path: str,
    ancestors: set[int],
) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite number.")
        return

    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a circular reference.")
        ancestors.add(identity)
        try:
            for key, item in value.items():
                if not isinstance(key, str):
                    raise TypeError(f"{path} contains a non-string object key.")
                _validate_json_value(
                    item,
                    path=f"{path}/{_pointer_token(key)}",
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return

    if isinstance(value, list):
        identity = id(value)
        if identity in ancestors:
            raise ValueError(f"{path} contains a circular reference.")
        ancestors.add(identity)
        try:
            for index, item in enumerate(value):
                _validate_json_value(
                    item,
                    path=f"{path}/{index}",
                    ancestors=ancestors,
                )
        finally:
            ancestors.remove(identity)
        return

    raise TypeError(f"{path} contains unsupported JSON value {type(value).__name__}.")


def _reject_non_finite_constant(value: str) -> NoReturn:
    raise ValueError(f"Non-finite number {value} is not valid JSON.")


def _canonical_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _canonical_json_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json_value(item) for item in value]
    return value


def _pointer_token(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")
