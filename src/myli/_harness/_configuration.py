"""Provider, tool, capability, and callback validation."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Collection, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from ..contracts import (
    AgentTool,
    AssetSearchProvider,
    EventHandler,
    FailureMode,
    ModelResponse,
    RunEvent,
    ToolCall,
)
from ..errors import (
    ConfigurationError,
    ModelProtocolError,
)
from ._helpers import (
    _coerce_failure_mode,
    _validate_timeout,
)
from ._types import (
    ASSET_SEARCH_TOOL_PREFIX,
    COMMIT_RENDER_TOOL_NAME,
    INSPECT_ASSET_TOOL_NAME,
    RENDER_TOOL_NAME,
    RETRIEVE_EVIDENCE_TOOL_NAME,
    SEARCH_NAME_PATTERN,
    TOOL_NAME_PATTERN,
    _ToolConfig,
)


class ConfigurationMixin:
    def _validate_search_providers(
        self,
        providers: Sequence[AssetSearchProvider],
    ) -> dict[str, AssetSearchProvider]:
        validated: dict[str, AssetSearchProvider] = {}
        provider_names: set[str] = set()
        for provider in providers:
            name = getattr(provider, "name", None)
            description = getattr(provider, "description", None)
            if not isinstance(name, str) or not SEARCH_NAME_PATTERN.fullmatch(name):
                raise ConfigurationError("Asset provider names must match ^[a-z][a-z0-9_]{0,31}$.")
            if name in provider_names:
                raise ConfigurationError(f"Duplicate asset provider name: {name}.")
            provider_names.add(name)
            if not isinstance(description, str) or not description.strip():
                raise ConfigurationError(f"Asset provider {name} requires a description.")
            if not callable(getattr(provider, "search", None)):
                raise ConfigurationError(f"Asset provider {name} must implement async search().")
            _coerce_failure_mode(
                getattr(provider, "failure_mode", FailureMode.RETURN_ERROR),
                name=f"asset provider {name}",
            )
            tool_name = f"{ASSET_SEARCH_TOOL_PREFIX}{name}"
            validated[tool_name] = provider
        return validated

    def _validate_failure_mode_targets(self) -> None:
        available = {*self._tools, *self._search_providers}
        if self.renderer is not None and self.vision_model is not None:
            available.update({RENDER_TOOL_NAME, COMMIT_RENDER_TOOL_NAME})
        if self._search_providers and self.vision_model is not None:
            available.add(INSPECT_ASSET_TOOL_NAME)
        if any(config.model_view is not None for config in self._tools.values()):
            available.add(RETRIEVE_EVIDENCE_TOOL_NAME)
        unknown = sorted(set(self._failure_modes).difference(available))
        if unknown:
            raise ConfigurationError("tool_failure_modes contains unavailable tool names: " + ", ".join(unknown))

    def _validate_tools(self, tools: Sequence[AgentTool]) -> dict[str, _ToolConfig]:
        reserved = {
            RENDER_TOOL_NAME,
            COMMIT_RENDER_TOOL_NAME,
            INSPECT_ASSET_TOOL_NAME,
            RETRIEVE_EVIDENCE_TOOL_NAME,
            *self._search_providers,
        }
        validated: dict[str, _ToolConfig] = {}
        for tool in tools:
            try:
                name = tool.name
                description = tool.description
                input_schema = copy.deepcopy(dict(tool.input_schema))
                required_capabilities = tool.required_capabilities
                max_calls_per_run = tool.max_calls_per_run
                timeout_seconds = tool.timeout_seconds
                max_result_bytes = tool.max_result_bytes
                execute = tool.execute
            except (AttributeError, TypeError, ValueError) as exc:
                raise ConfigurationError(f"Agent tools must implement the complete protocol: {exc}") from exc
            if (
                not isinstance(name, str)
                or not TOOL_NAME_PATTERN.fullmatch(name)
                or name in reserved
                or name in validated
            ):
                raise ConfigurationError(f"Agent tool name is invalid, duplicate, or reserved: {name!r}.")
            if not isinstance(description, str) or not description.strip():
                raise ConfigurationError(f"Agent tool {name} requires a description.")
            try:
                Draft202012Validator.check_schema(input_schema)
                schema_validator = Draft202012Validator(input_schema)
            except SchemaError as exc:
                raise ConfigurationError(f"Agent tool {name} input_schema is invalid: {exc.message}") from exc
            if not callable(execute):
                raise ConfigurationError(f"Agent tool {name} execute must be callable.")
            model_view = getattr(tool, "model_view", None)
            if model_view is not None and not callable(model_view):
                raise ConfigurationError(f"Agent tool {name} model_view must be callable.")
            capabilities = self._validate_capabilities(required_capabilities)
            if isinstance(max_calls_per_run, bool) or not isinstance(max_calls_per_run, int) or max_calls_per_run < 1:
                raise ConfigurationError(f"Agent tool {name} max_calls_per_run must be positive.")
            _validate_timeout(timeout_seconds, name=f"{name}.timeout_seconds")
            if isinstance(max_result_bytes, bool) or not isinstance(max_result_bytes, int) or max_result_bytes < 1:
                raise ConfigurationError(f"Agent tool {name} max_result_bytes must be positive.")
            failure_mode = _coerce_failure_mode(
                getattr(tool, "failure_mode", FailureMode.RETURN_ERROR),
                name=f"tool {name}",
            )
            parallel_safe = getattr(tool, "parallel_safe", False)
            if not isinstance(parallel_safe, bool):
                raise ConfigurationError(f"Agent tool {name} parallel_safe must be boolean.")
            validated[name] = _ToolConfig(
                tool=tool,
                model_view=model_view,
                input_schema=input_schema,
                validator=schema_validator,
                required_capabilities=capabilities,
                max_calls_per_run=max_calls_per_run,
                timeout_seconds=float(timeout_seconds),
                max_result_bytes=max_result_bytes,
                failure_mode=failure_mode,
                parallel_safe=parallel_safe,
            )
        return validated

    @staticmethod
    def _validate_capabilities(capabilities: Collection[str]) -> frozenset[str]:
        if isinstance(capabilities, (str, bytes)) or not isinstance(
            capabilities,
            Collection,
        ):
            raise ConfigurationError("capabilities must be a collection of strings.")
        resolved: set[str] = set()
        for capability in capabilities:
            if not isinstance(capability, str) or not capability.strip():
                raise ConfigurationError("capabilities must contain non-empty strings.")
            resolved.add(capability.strip())
        return frozenset(resolved)

    @staticmethod
    def _validate_tool_calls(calls: Sequence[ToolCall]) -> None:
        if any(not isinstance(call, ToolCall) for call in calls):
            raise ModelProtocolError("tool_calls must contain ToolCall instances.")
        ids = [call.id for call in calls]
        if any(not isinstance(call_id, str) or not call_id for call_id in ids):
            raise ModelProtocolError("Tool call IDs must be non-empty strings.")
        if len(ids) != len(set(ids)):
            raise ModelProtocolError("Tool call IDs must be unique within a step.")
        for call in calls:
            if not isinstance(call.name, str) or not call.name:
                raise ModelProtocolError("Tool call names must be non-empty strings.")
            if not isinstance(call.arguments, Mapping):
                raise ModelProtocolError("Tool call arguments must be JSON objects.")

    @staticmethod
    def _validate_model_response(response: ModelResponse) -> None:
        if response.content is not None and not isinstance(response.content, str):
            raise ModelProtocolError("ModelResponse.content must be text or None.")
        if not isinstance(response.tool_calls, tuple):
            raise ModelProtocolError("ModelResponse.tool_calls must be a tuple.")
        if not isinstance(response.metadata, Mapping):
            raise ModelProtocolError("ModelResponse.metadata must be a mapping.")
        if response.reasoning_content is not None and not isinstance(
            response.reasoning_content,
            str,
        ):
            raise ModelProtocolError("ModelResponse.reasoning_content must be text or None.")

    @staticmethod
    def _validate_failure_modes(
        values: Mapping[str, FailureMode | str],
    ) -> dict[str, FailureMode]:
        resolved: dict[str, FailureMode] = {}
        for name, mode in values.items():
            if not isinstance(name, str) or not name:
                raise ConfigurationError("tool_failure_modes keys must be tool names.")
            resolved[name] = _coerce_failure_mode(mode, name=f"tool {name}")
        return resolved

    @staticmethod
    async def _emit(handler: EventHandler | None, event: RunEvent) -> None:
        if handler is not None:
            await handler(event)

    @staticmethod
    async def _emit_terminal(handler: EventHandler | None, event: RunEvent) -> None:
        if handler is None:
            return
        try:
            await handler(event)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Preserve the original run failure or cancellation. Callback failures
            # during ordinary progress remain application-visible and fail the run.
            return
