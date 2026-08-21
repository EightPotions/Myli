"""Bounded, dependency-free RFC 6902 JSON Patch implementation."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._json import validate_json_value
from .errors import JsonPatchError


_ARRAY_INDEX_PATTERN = re.compile(r"0|[1-9][0-9]*")
_OPERATIONS = {"add", "remove", "replace", "move", "copy", "test"}
_REQUIRED_MEMBERS = {
    "add": {"op", "path", "value"},
    "remove": {"op", "path"},
    "replace": {"op", "path", "value"},
    "move": {"op", "path", "from"},
    "copy": {"op", "path", "from"},
    "test": {"op", "path", "value"},
}


@dataclass(frozen=True, slots=True)
class JsonPatchLimits:
    """Resource limits applied before and during patch evaluation."""

    max_operations: int = 1_000
    max_patch_bytes: int = 1_048_576
    max_pointer_depth: int = 128
    max_value_bytes: int = 524_288
    max_document_bytes: int = 5_242_880

    def __post_init__(self) -> None:
        for name in (
            "max_operations",
            "max_patch_bytes",
            "max_pointer_depth",
            "max_value_bytes",
            "max_document_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


def apply_json_patch(
    document: Any,
    patch: Any,
    *,
    limits: JsonPatchLimits | None = None,
    max_operations: int | None = None,
    max_patch_bytes: int | None = None,
    max_pointer_depth: int | None = None,
    max_value_bytes: int | None = None,
    max_document_bytes: int | None = None,
) -> Any:
    """Apply an RFC 6902 patch to a deep copy under explicit safety limits.

    Root operations, escaped pointers, and all six RFC operations are supported.
    Neither the input document nor values from the patch are ever mutated.
    """

    resolved = limits or JsonPatchLimits()
    resolved = JsonPatchLimits(
        max_operations=resolved.max_operations if max_operations is None else max_operations,
        max_patch_bytes=resolved.max_patch_bytes if max_patch_bytes is None else max_patch_bytes,
        max_pointer_depth=(resolved.max_pointer_depth if max_pointer_depth is None else max_pointer_depth),
        max_value_bytes=resolved.max_value_bytes if max_value_bytes is None else max_value_bytes,
        max_document_bytes=(resolved.max_document_bytes if max_document_bytes is None else max_document_bytes),
    )
    if not isinstance(patch, list):
        raise JsonPatchError("A JSON Patch must be an array of operations.")
    try:
        validate_json_value(document)
        validate_json_value(patch)
    except (TypeError, ValueError, RecursionError) as exc:
        raise JsonPatchError(f"JSON Patch inputs must contain valid JSON values: {exc}") from exc
    if len(patch) > resolved.max_operations:
        raise JsonPatchError(f"JSON Patch exceeds the {resolved.max_operations}-operation limit.")
    if _json_size(patch) > resolved.max_patch_bytes:
        raise JsonPatchError(f"JSON Patch exceeds the {resolved.max_patch_bytes}-byte limit.")
    if _json_size(document) > resolved.max_document_bytes:
        raise JsonPatchError(f"Input document exceeds the {resolved.max_document_bytes}-byte limit.")

    try:
        result = copy.deepcopy(document)
    except RecursionError as exc:
        raise JsonPatchError("Input document nesting is too deep.") from exc
    for index, operation in enumerate(patch):
        try:
            _validate_operation_limits(operation, resolved)
            result = _apply_operation(result, operation, max_pointer_depth=resolved.max_pointer_depth)
            validate_json_value(result)
            if _json_size(result) > resolved.max_document_bytes:
                raise JsonPatchError(f"resulting document exceeds the {resolved.max_document_bytes}-byte limit.")
        except JsonPatchError as exc:
            raise JsonPatchError(f"Patch operation {index} is invalid: {exc}") from exc
        except (TypeError, ValueError, RecursionError) as exc:
            raise JsonPatchError(f"Patch operation {index} produced invalid JSON: {exc}") from exc
    return result


def _validate_operation_limits(operation: Any, limits: JsonPatchLimits) -> None:
    if not isinstance(operation, Mapping):
        raise JsonPatchError("the operation must be an object.")
    if "value" in operation and _json_size(operation["value"]) > limits.max_value_bytes:
        raise JsonPatchError(f"value exceeds the {limits.max_value_bytes}-byte limit.")


def _apply_operation(document: Any, operation: Mapping[str, Any], *, max_pointer_depth: int) -> Any:
    name = operation.get("op")
    if not isinstance(name, str) or name not in _OPERATIONS:
        raise JsonPatchError("op must be one of add, remove, replace, move, copy, or test.")
    missing = _REQUIRED_MEMBERS[name].difference(operation)
    if missing:
        raise JsonPatchError(f"{name} is missing required members: {', '.join(sorted(missing))}.")

    path = _parse_pointer(operation["path"], member="path", max_depth=max_pointer_depth)
    if name == "add":
        return _add(document, path, operation["value"])
    if name == "remove":
        return _remove(document, path)[0]
    if name == "replace":
        _get(document, path)
        return _replace(document, path, operation["value"])
    if name == "test":
        if not _json_equal(_get(document, path), operation["value"]):
            raise JsonPatchError("test did not match the value at path.")
        return document

    source = _parse_pointer(operation["from"], member="from", max_depth=max_pointer_depth)
    if name == "copy":
        return _add(document, path, copy.deepcopy(_get(document, source)))
    if len(path) > len(source) and path[: len(source)] == source:
        raise JsonPatchError("move path cannot be a child of from.")
    document, value = _remove(document, source)
    return _add(document, path, value)


def _parse_pointer(value: Any, *, member: str, max_depth: int) -> tuple[str, ...]:
    if not isinstance(value, str):
        raise JsonPatchError(f"{member} must be a JSON Pointer string.")
    if value == "":
        return ()
    if not value.startswith("/"):
        raise JsonPatchError(f"{member} must be empty or start with '/'.")

    encoded_tokens = value[1:].split("/")
    if len(encoded_tokens) > max_depth:
        raise JsonPatchError(f"{member} exceeds the {max_depth}-segment pointer-depth limit.")
    tokens: list[str] = []
    for encoded in encoded_tokens:
        token = ""
        cursor = 0
        while cursor < len(encoded):
            character = encoded[cursor]
            if character != "~":
                token += character
                cursor += 1
                continue
            if cursor + 1 >= len(encoded) or encoded[cursor + 1] not in {"0", "1"}:
                raise JsonPatchError(f"{member} contains an invalid '~' escape.")
            token += "~" if encoded[cursor + 1] == "0" else "/"
            cursor += 2
        tokens.append(token)
    return tuple(tokens)


def _get(document: Any, path: tuple[str, ...]) -> Any:
    target = document
    for token in path:
        if isinstance(target, dict):
            if token not in target:
                raise JsonPatchError(f"path member {token!r} does not exist.")
            target = target[token]
        elif isinstance(target, list):
            target = target[_array_index(token, len(target))]
        else:
            raise JsonPatchError("path traverses a scalar value.")
    return target


def _parent(document: Any, path: tuple[str, ...]) -> tuple[Any, str]:
    if not path:
        raise JsonPatchError("the root path has no parent.")
    return _get(document, path[:-1]), path[-1]


def _add(document: Any, path: tuple[str, ...], value: Any) -> Any:
    value = copy.deepcopy(value)
    if not path:
        return value
    parent, token = _parent(document, path)
    if isinstance(parent, dict):
        parent[token] = value
    elif isinstance(parent, list):
        if token == "-":
            parent.append(value)
        else:
            parent.insert(_array_index(token, len(parent), allow_end=True), value)
    else:
        raise JsonPatchError("add path has a scalar parent.")
    return document


def _remove(document: Any, path: tuple[str, ...]) -> tuple[Any, Any]:
    if not path:
        return None, document
    parent, token = _parent(document, path)
    if isinstance(parent, dict):
        if token not in parent:
            raise JsonPatchError(f"path member {token!r} does not exist.")
        value = parent.pop(token)
    elif isinstance(parent, list):
        value = parent.pop(_array_index(token, len(parent)))
    else:
        raise JsonPatchError("remove path has a scalar parent.")
    return document, value


def _replace(document: Any, path: tuple[str, ...], value: Any) -> Any:
    value = copy.deepcopy(value)
    if not path:
        return value
    parent, token = _parent(document, path)
    if isinstance(parent, dict):
        if token not in parent:
            raise JsonPatchError(f"path member {token!r} does not exist.")
        parent[token] = value
    elif isinstance(parent, list):
        parent[_array_index(token, len(parent))] = value
    else:
        raise JsonPatchError("replace path has a scalar parent.")
    return document


def _array_index(token: str, length: int, *, allow_end: bool = False) -> int:
    if not _ARRAY_INDEX_PATTERN.fullmatch(token):
        raise JsonPatchError(f"array index {token!r} is invalid.")
    try:
        index = int(token)
    except ValueError as exc:
        raise JsonPatchError(f"array index {token!r} is too large.") from exc
    maximum = length if allow_end else length - 1
    if index > maximum:
        raise JsonPatchError(f"array index {token!r} is out of bounds.")
    return index


def _json_equal(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return isinstance(left, bool) and isinstance(right, bool) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left == right
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_json_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _json_equal(left_value, right_value) for left_value, right_value in zip(left, right)
        )
    return left == right


def _json_size(value: Any) -> int:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as exc:
        raise JsonPatchError(f"value is not strict JSON: {exc}") from exc
    return len(encoded)
