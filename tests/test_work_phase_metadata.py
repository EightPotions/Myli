"""Focused coverage for validation diagnostics and tool work-phase metadata."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Mapping
from typing import Any

from myli import (
    Allow,
    DesignSpec,
    FailureMode,
    HarnessLimits,
    ModelRequest,
    ModelResponse,
    Myli,
    ToolCall,
    ToolContext,
    ToolOutcome,
)


SIMPLE_SPEC = DesignSpec[dict[str, Any]](
    name="document",
    schema={"type": "object"},
    validator=lambda value: value,
    serializer=lambda value: value,
)


class SequenceModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        return next(self._responses)


class ContextTool:
    name = "context_probe"
    description = "Record the context for a test call."
    input_schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }
    required_capabilities = frozenset()
    max_calls_per_run = 3
    timeout_seconds = 1.0
    max_result_bytes = 1024
    failure_mode = FailureMode.RETURN_ERROR
    parallel_safe = True

    def __init__(self) -> None:
        self.contexts: list[ToolContext] = []

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> Any:
        self.contexts.append(context)
        return {"value": arguments["value"]}


class ContextMiddleware:
    def __init__(self) -> None:
        self.before_contexts: list[ToolContext] = []
        self.after_contexts: list[ToolContext] = []

    async def before_call(self, call: ToolCall, context: ToolContext) -> Allow:
        del call
        self.before_contexts.append(context)
        return Allow()

    async def after_call(
        self,
        call: ToolCall,
        outcome: ToolOutcome,
        context: ToolContext,
    ) -> None:
        del call, outcome
        self.after_contexts.append(context)


def test_candidate_round_trip_error_reports_escaped_json_pointer() -> None:
    def normalizing_validator(value: dict[str, Any]) -> dict[str, Any]:
        value["sections"][0]["a/b"]["~label"] = "normalized"
        return value

    spec = DesignSpec[dict[str, Any]](
        name="nested document",
        schema={"type": "object"},
        validator=normalizing_validator,
        serializer=lambda value: value,
    )

    with unittest.TestCase().assertRaisesRegex(
        ValueError,
        r"JSON Pointer '/sections/0/a~1b/~0label'",
    ):
        spec.validate_candidate(
            {"sections": [{"a/b": {"~label": "model value"}}]},
        )


def test_tool_context_exposes_model_step_and_batch_metadata() -> None:
    tool = ContextTool()
    middleware = ContextMiddleware()
    model = SequenceModel(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="call-1", name=tool.name, arguments={"value": 1}),
                    ToolCall(id="call-2", name=tool.name, arguments={"value": 2}),
                )
            ),
            ModelResponse(tool_calls=(ToolCall(id="call-3", name=tool.name, arguments={"value": 3}),)),
            ModelResponse(content=json.dumps({"message": "Done.", "patch": None})),
        ]
    )
    harness = Myli(
        design_spec=SIMPLE_SPEC,
        main_model=model,
        tools=[tool],
        middleware=[middleware],
        parallel_tool_calls=True,
    )

    asyncio.run(harness.run(request="Probe the contexts.", design={}))

    contexts = sorted(
        middleware.before_contexts,
        key=lambda context: (context.model_step or 0, context.tool_batch_index or 0),
    )
    assert [context.model_step for context in contexts] == [0, 0, 1]
    assert [context.tool_batch_index for context in contexts] == [0, 1, 0]
    assert [context.tool_batch_size for context in contexts] == [2, 2, 1]
    assert contexts[0].tool_batch_id == contexts[1].tool_batch_id
    assert contexts[1].tool_batch_id != contexts[2].tool_batch_id
    assert contexts[0].model_step_id == contexts[0].step_id
    assert contexts[0].tool_batch_id == contexts[0].batch_id
    assert [context.is_first_in_tool_batch for context in contexts] == [True, False, True]
    assert (
        sorted(
            tool.contexts,
            key=lambda context: (context.model_step or 0, context.tool_batch_index or 0),
        )
        == contexts
    )
    assert sorted(context.tool_batch_index for context in middleware.after_contexts) == [0, 0, 1]


def test_retry_prompt_carries_all_successive_validation_failures() -> None:
    model = SequenceModel(
        [
            ModelResponse(content=json.dumps({"message": "First.", "patch": None, "extra": True})),
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "Second.",
                        "patch": [{"op": "add", "path": "/value", "value": 1}],
                    }
                )
            ),
            ModelResponse(content=json.dumps({"message": "Third.", "patch": None})),
        ]
    )
    harness = Myli(
        design_spec=SIMPLE_SPEC,
        main_model=model,
        limits=HarnessLimits(max_validation_retries=2),
    )

    result = asyncio.run(harness.run(request="Give advice only.", design={}))

    assert result.message == "Third."
    first_retry = model.requests[1].messages[-1].content or ""
    second_retry = model.requests[2].messages[-1].content or ""
    assert "contain exactly message and patch" in first_retry
    assert "contain exactly message and patch" in second_retry
    assert "Editing is disabled; patch must be null" in second_retry
    assert second_retry.index("contain exactly message and patch") < second_retry.index("Editing is disabled")


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
