"""Pure validation and serialization helpers used by harness modules."""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .._json import canonical_json_dumps, canonical_json_value, validate_json_value
from ..contracts import Asset, FailureMode
from ..errors import ConfigurationError, ToolExecutionError


def _validate_timeout(value: Any, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive.")


def _required_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be text.")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{name} cannot be empty.")
    return stripped


def _validated_vision_review(value: Any) -> str | dict[str, Any]:
    if isinstance(value, str):
        return _required_string(value, name="vision review")
    if not isinstance(value, Mapping):
        raise TypeError("vision review must be text or a JSON object.")
    review = copy.deepcopy(dict(value))
    validate_json_value(review)
    return review


def _optional_metadata_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def _nonnegative_token_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _usage_token_count(
    usage: Mapping[str, Any],
    *names: str,
    nested: Sequence[tuple[str, str]] = (),
) -> int | None:
    for name in names:
        count = _nonnegative_token_count(usage.get(name))
        if count is not None:
            return count
    for container_name, name in nested:
        container = usage.get(container_name)
        if isinstance(container, Mapping):
            count = _nonnegative_token_count(container.get(name))
            if count is not None:
                return count
    return None


def _optional_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ConfigurationError(f"{name} must be text.")
    return value.strip()


def _required_decision_message(value: Any) -> str:
    try:
        return _required_string(value, name="middleware decision message")
    except (TypeError, ValueError) as exc:
        raise ToolExecutionError(str(exc)) from exc


def _coerce_failure_mode(value: Any, *, name: str) -> FailureMode:
    try:
        return value if isinstance(value, FailureMode) else FailureMode(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} failure_mode must be return_error or raise.") from exc


def _serialize_tool_result(value: Any) -> tuple[Any, int]:
    serialized = canonical_json_value(value)
    content = canonical_json_dumps(serialized)
    return serialized, len(content.encode("utf-8"))


def _resolve_json_pointer(value: Any, pointer: str, *, max_depth: int) -> Any:
    """Resolve one RFC 6901 pointer without mutating the source value."""

    if pointer == "":
        return copy.deepcopy(value)
    if not pointer.startswith("/"):
        raise ValueError("json_pointer must be empty or start with '/'.")
    encoded_tokens = pointer[1:].split("/")
    if len(encoded_tokens) > max_depth:
        raise ValueError(f"json_pointer exceeds the {max_depth}-segment pointer-depth limit.")

    target = value
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
                raise ValueError("json_pointer contains an invalid '~' escape.")
            token += "~" if encoded[cursor + 1] == "0" else "/"
            cursor += 2

        if isinstance(target, Mapping):
            if token not in target:
                raise ValueError(f"json_pointer member {token!r} does not exist.")
            target = target[token]
            continue
        if isinstance(target, list):
            if not re.fullmatch(r"0|[1-9][0-9]*", token):
                raise ValueError(f"json_pointer array index {token!r} is invalid.")
            try:
                index = int(token)
            except ValueError as exc:
                raise ValueError(f"json_pointer array index {token!r} is too large.") from exc
            if index >= len(target):
                raise ValueError(f"json_pointer array index {token!r} is out of bounds.")
            target = target[index]
            continue
        raise ValueError("json_pointer traverses a scalar value.")
    return copy.deepcopy(target)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return canonical_json_dumps(value)


def _same_asset(left: Asset, right: Asset) -> bool:
    return (
        left.id == right.id
        and left.uri == right.uri
        and left.kind == right.kind
        and left.description == right.description
        and _canonical_json(dict(left.metadata)) == _canonical_json(dict(right.metadata))
        and _canonical_json(dict(left.provenance)) == _canonical_json(dict(right.provenance))
    )


def _concise(message: str, *, limit: int = 500) -> str:
    cleaned = " ".join(message.split())
    return cleaned if len(cleaned) <= limit else f"{cleaned[: limit - 1]}…"
