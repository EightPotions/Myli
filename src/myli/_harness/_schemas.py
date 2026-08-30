"""Model-facing prompts and JSON schemas."""

from __future__ import annotations

import copy
from typing import Any

from ..contracts import (
    TDesign,
    ToolDefinition,
)
from ._types import (
    COMMIT_RENDER_TOOL_NAME,
    INSPECT_ASSET_TOOL_NAME,
    RENDER_TOOL_NAME,
    RETRIEVE_EVIDENCE_TOOL_NAME,
    _RunState,
)


class SchemaMixin:
    def _build_tool_definitions(self) -> tuple[ToolDefinition, ...]:
        definitions: list[ToolDefinition] = []
        if self.renderer is not None and self.vision_model is not None:
            definitions.append(
                ToolDefinition(
                    name=RENDER_TOOL_NAME,
                    description=(
                        "Render a validated current or proposed design and receive "
                        "visual feedback. Patch is relative to the current design unless "
                        "base_render_ref selects a previously rendered candidate. "
                        "Optionally ask one or more open questions about the rendered image. "
                        "A successful call returns a render_ref that commit_render can select "
                        "without rendering again."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "base_render_ref": {"type": "string", "minLength": 1},
                            "patch": self._json_patch_schema(),
                            "questions": self._vision_questions_schema(),
                        },
                        "required": ["patch"],
                        "additionalProperties": False,
                    },
                )
            )
            definitions.append(
                ToolDefinition(
                    name=COMMIT_RENDER_TOOL_NAME,
                    description=(
                        "Select a previously successful render as the final proposal without "
                        "running the renderer or visual review again. This does not persist the "
                        "document. After committing, prefer a null final patch."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "render_ref": {"type": "string", "minLength": 1},
                        },
                        "required": ["render_ref"],
                        "additionalProperties": False,
                    },
                )
            )
        for tool_name in sorted(self._search_providers):
            provider = self._search_providers[tool_name]
            definitions.append(
                ToolDefinition(
                    name=tool_name,
                    description=provider.description,
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "minLength": 1},
                            "limit": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": self.limits.max_search_results,
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                )
            )
        if self._search_providers and self.vision_model is not None:
            definitions.append(
                ToolDefinition(
                    name=INSPECT_ASSET_TOOL_NAME,
                    description=(
                        "Lazily resolve and visually inspect a discovered asset preview. "
                        "Optionally ask one or more open questions about the preview."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "asset_ref": {"type": "string", "minLength": 1},
                            "questions": self._vision_questions_schema(),
                        },
                        "required": ["asset_ref"],
                        "additionalProperties": False,
                    },
                )
            )
        if self._compact_render_context or any(config.model_view is not None for config in self._tools.values()):
            definitions.append(
                ToolDefinition(
                    name=RETRIEVE_EVIDENCE_TOOL_NAME,
                    description=(
                        "Retrieve omitted data from compacted or projected tool evidence. Use only "
                        "an exact evidence_ref returned in this run. Omit json_pointer to "
                        "retrieve the complete result, or provide an RFC 6901 JSON Pointer "
                        "to retrieve one bounded subtree."
                    ),
                    input_schema={
                        "type": "object",
                        "properties": {
                            "evidence_ref": {"type": "string", "minLength": 1},
                            "json_pointer": {
                                "type": "string",
                                "maxLength": self.limits.max_evidence_pointer_chars,
                            },
                        },
                        "required": ["evidence_ref"],
                        "additionalProperties": False,
                    },
                )
            )
        for tool_name in sorted(self._tools):
            config = self._tools[tool_name]
            definitions.append(
                ToolDefinition(
                    name=config.tool.name,
                    description=config.tool.description,
                    input_schema=config.input_schema,
                )
            )
        return tuple(definitions)

    def _tool_definitions_for(
        self,
        state: _RunState[TDesign],
    ) -> tuple[ToolDefinition, ...]:
        return tuple(
            ToolDefinition(
                name=definition.name,
                description=definition.description,
                input_schema=copy.deepcopy(dict(definition.input_schema)),
            )
            for definition in self._tool_definitions
            if (
                definition.name != RETRIEVE_EVIDENCE_TOOL_NAME
                or state.compacted_render_context
                or any(
                    outcome.evidence_ref is not None and outcome.tool_name != RENDER_TOOL_NAME
                    for outcome in state.outcomes
                )
            )
            and (
                definition.name not in self._tools
                or self._tools[definition.name].required_capabilities.issubset(state.capabilities)
            )
        )

    def _build_output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "message": {"type": "string", "minLength": 1},
                "patch": {
                    "anyOf": [self._json_patch_schema(), {"type": "null"}],
                },
            },
            "required": ["message", "patch"],
            "additionalProperties": False,
        }

    def _json_patch_schema(self) -> dict[str, Any]:
        def operation(
            name: str,
            *,
            value: bool = False,
            source: bool = False,
        ) -> dict[str, Any]:
            properties: dict[str, Any] = {
                "op": {"type": "string", "const": name},
                "path": {"type": "string"},
            }
            required = ["op", "path"]
            if value:
                properties["value"] = {}
                required.append("value")
            if source:
                properties["from"] = {"type": "string"}
                required.append("from")
            return {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            }

        schema: dict[str, Any] = {
            "type": "array",
            "items": {
                "anyOf": [
                    operation("add", value=True),
                    operation("remove"),
                    operation("replace", value=True),
                    operation("move", source=True),
                    operation("copy", source=True),
                    operation("test", value=True),
                ]
            },
        }
        if self.limits.max_patch_operations is not None:
            schema["maxItems"] = self.limits.max_patch_operations
        return schema

    def _vision_questions_schema(self) -> dict[str, Any]:
        return {
            "type": "array",
            "minItems": 1,
            "maxItems": self.limits.max_vision_questions,
            "items": {
                "type": "string",
                "minLength": 1,
                "maxLength": self.limits.max_vision_question_chars,
            },
        }
