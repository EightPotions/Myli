"""Behavioral tests for the provider-neutral Myli harness."""

from __future__ import annotations

import asyncio
import json
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import Any

from myli import (
    Asset,
    CandidateContext,
    ConfigurationError,
    DefaultModelContextPolicy,
    DesignSpec,
    FailureMode,
    HarnessLimits,
    InputArtifact,
    LiteLLMMainModel,
    LiteLLMVisionModel,
    Message,
    ModelContext,
    ModelProtocolError,
    ModelRequest,
    ModelResponse,
    Myli,
    RenderedArtifact,
    ToolCall,
    ToolContext,
    ToolExecutionError,
    VisualReviewRequest,
    apply_json_patch,
)


def validate_design(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"background", "elements"}:
        raise ValueError("design requires background and elements")
    if not isinstance(value["background"], str):
        raise ValueError("background must be text")
    if not isinstance(value["elements"], list):
        raise ValueError("elements must be a list")
    return {
        "background": value["background"],
        "elements": [dict(element) for element in value["elements"]],
    }


DESIGN_SPEC = DesignSpec[dict[str, Any]](
    name="poster",
    schema={
        "type": "object",
        "properties": {
            "background": {"type": "string"},
            "elements": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["background", "elements"],
        "additionalProperties": False,
    },
    validator=validate_design,
    serializer=lambda design: design,
)


ResponseFactory = Callable[[ModelRequest], ModelResponse]


class FakeMainAgent:
    def __init__(self, responses: list[ModelResponse | Exception | ResponseFactory]) -> None:
        self._responses = iter(responses)
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        if callable(response):
            response = response(request)
        return response


class RecordingModelContextPolicy:
    def __init__(self) -> None:
        self.messages: list[tuple[Message, ...]] = []
        self.contexts: list[ModelContext] = []

    def prepare(
        self,
        messages: tuple[Message, ...],
        context: ModelContext,
    ) -> tuple[Message, ...]:
        self.messages.append(tuple(messages))
        self.contexts.append(context)
        return tuple(messages)


class FakeRenderer:
    def __init__(self) -> None:
        self.designs: list[dict[str, Any]] = []

    async def render(self, design: dict[str, Any]) -> RenderedArtifact:
        self.designs.append(design)
        return RenderedArtifact(b"rendered-poster", "image/png")


class FakeVisionAgent:
    def __init__(self) -> None:
        self.reviews: list[tuple[RenderedArtifact, str]] = []

    async def review(self, request: VisualReviewRequest) -> str:
        self.reviews.append((request.images[0].artifact, request.prompt))
        return "The title has clear hierarchy; increase the lower image contrast."


class FakeStructuredVisionAgent:
    async def review(
        self,
        request: VisualReviewRequest,
    ) -> dict[str, Any]:
        del request
        return {
            "summary": "The title is aligned.",
            "differences": [],
            "ready_to_commit": True,
        }


class FakeSearch:
    def __init__(self, name: str, assets: list[Asset]) -> None:
        self.name = name
        self.description = f"Search the {name} image collection."
        self.assets = assets
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, *, limit: int) -> list[Asset]:
        self.queries.append((query, limit))
        return self.assets[:limit]


class FakeAgentTool:
    def __init__(
        self,
        *,
        name: str = "lookup_brand",
        required_capabilities: frozenset[str] = frozenset({"brand.read"}),
        max_calls_per_run: int = 1,
        timeout_seconds: float = 1,
        max_result_bytes: int = 1024,
        result: Any = None,
        delay: float = 0,
    ) -> None:
        self.name = name
        self.description = "Look up an approved brand value."
        self.input_schema = {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        }
        self.required_capabilities = required_capabilities
        self.max_calls_per_run = max_calls_per_run
        self.timeout_seconds = timeout_seconds
        self.max_result_bytes = max_result_bytes
        self.result = {"value": "navy"} if result is None else result
        self.delay = delay
        self.calls: list[dict[str, Any]] = []

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        del context
        self.calls.append(dict(arguments))
        if self.delay:
            await asyncio.sleep(self.delay)
        return self.result


class RenderArtifactCaptureTool:
    name = "inspect_render_artifact"
    description = "Inspect one successful render artifact."
    input_schema = {
        "type": "object",
        "properties": {"render_ref": {"type": "string"}},
        "required": ["render_ref"],
        "additionalProperties": False,
    }
    required_capabilities = frozenset({"render.inspect"})
    max_calls_per_run = 1
    timeout_seconds = 1.0
    max_result_bytes = 1024
    failure_mode = FailureMode.RETURN_ERROR
    parallel_safe = False

    def __init__(self) -> None:
        self.artifacts: list[RenderedArtifact] = []

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        artifact = context.rendered_artifacts[arguments["render_ref"]]
        self.artifacts.append(artifact)
        return {"media_type": artifact.media_type, "size": len(artifact.data)}


class InputArtifactCaptureTool:
    name = "inspect_input_artifact"
    description = "Inspect one application-provided input artifact."
    input_schema = {
        "type": "object",
        "properties": {"artifact_id": {"type": "string"}},
        "required": ["artifact_id"],
        "additionalProperties": False,
    }
    required_capabilities = frozenset({"input.inspect"})
    max_calls_per_run = 2
    timeout_seconds = 1.0
    max_result_bytes = 1024
    failure_mode = FailureMode.RETURN_ERROR
    parallel_safe = False

    def __init__(self) -> None:
        self.loaded: list[RenderedArtifact] = []
        self.registered: list[InputArtifact] = []

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> Any:
        artifact_id = arguments["artifact_id"]
        self.registered.append(context.input_artifacts[artifact_id])
        artifact = await context.load_input_artifact(artifact_id)
        self.loaded.append(artifact)
        return {"media_type": artifact.media_type, "size": len(artifact.data)}


CURRENT_DESIGN = {"background": "#ffffff", "elements": []}


def normalize_background_in_place(value: dict[str, Any]) -> dict[str, Any]:
    value["background"] = value["background"].strip()
    return value


def searched_asset_ref(request: ModelRequest, *, source: str, asset_id: str) -> str:
    for message in reversed(request.messages):
        if message.role != "tool" or message.content is None:
            continue
        payload = json.loads(message.content)
        if payload.get("source") != source:
            continue
        for asset in payload["assets"]:
            if asset["id"] == asset_id:
                return asset["asset_ref"]
    raise AssertionError(f"No asset {source}:{asset_id} was present in the model request.")


def latest_render_ref(request: ModelRequest) -> str:
    for message in reversed(request.messages):
        if message.role != "tool" or message.content is None:
            continue
        payload = json.loads(message.content)
        render_ref = payload.get("render_ref")
        if isinstance(render_ref, str):
            return render_ref
    raise AssertionError("No successful render was present in the model request.")


def first_render_ref(request: ModelRequest) -> str:
    for message in request.messages:
        if message.role != "tool" or message.content is None:
            continue
        payload = json.loads(message.content)
        render_ref = payload.get("render_ref")
        if isinstance(render_ref, str):
            return render_ref
    raise AssertionError("No successful render was present in the model request.")


def commit_latest_render(request: ModelRequest) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="commit-render",
                name="commit_render",
                arguments={"render_ref": latest_render_ref(request)},
            ),
        )
    )


def commit_first_render(request: ModelRequest) -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="commit-render",
                name="commit_render",
                arguments={"render_ref": first_render_ref(request)},
            ),
        )
    )


def test_agent_can_return_advice_without_editing() -> None:
    model = FakeMainAgent(
        [ModelResponse(content=json.dumps({"message": "Increase contrast around the title.", "patch": None}))]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="What should I improve?", design=CURRENT_DESIGN))

    assert result.message == "Increase contrast around the title."
    assert result.design is None
    assert result.changed is False
    assert result.patch is None
    assert [tool.name for tool in model.requests[0].tools] == ["render_design", "commit_render"]
    assert model.requests[0].output_schema["required"] == ["message", "patch"]


def test_asset_inspection_limit_is_configurable() -> None:
    assert HarnessLimits().max_asset_inspections == 6
    assert HarnessLimits(max_asset_inspections=2).max_asset_inspections == 2
    assert HarnessLimits(max_vision_questions=3).max_vision_questions == 3


def test_model_protocol_error_is_retried_with_a_correction_prompt() -> None:
    model = FakeMainAgent(
        [
            ModelProtocolError("Model tool render_design returned invalid JSON arguments."),
            ModelResponse(content=json.dumps({"message": "Recovered.", "patch": None})),
        ]
    )
    events = []

    async def capture_event(event) -> None:
        events.append(event)

    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(
        harness.run(
            request="Inspect the design.",
            design=CURRENT_DESIGN,
            on_event=capture_event,
        )
    )

    assert result.message == "Recovered."
    assert len(model.requests) == 2
    correction = model.requests[1].messages[-1].content or ""
    assert "invalid JSON arguments" in correction
    assert "strict JSON objects" in correction
    assert [event.kind for event in events if event.kind.startswith("model.")][:3] == [
        "model.started",
        "model.failed",
        "model.started",
    ]
    assert [call.success for call in result.model_calls] == [False, True]
    assert [call.retry_count for call in result.model_calls] == [0, 1]
    assert result.usage.request_count == 2
    assert result.usage.failed_requests == 1


def test_model_protocol_retry_budget_is_bounded() -> None:
    model = FakeMainAgent(
        [
            ModelProtocolError("First invalid response."),
            ModelProtocolError("Second invalid response."),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        limits=HarnessLimits(max_model_response_retries=1),
    )

    with unittest.TestCase().assertRaisesRegex(
        ModelProtocolError,
        "exhausted the model-response retry budget",
    ):
        asyncio.run(harness.run(request="Inspect the design.", design=CURRENT_DESIGN))

    assert len(model.requests) == 2


def test_agent_can_render_review_and_return_a_validated_edit() -> None:
    candidate = {"background": "#f5efe6", "elements": []}
    render_call = ToolCall(
        id="render-1",
        name="render_design",
        arguments={
            "patch": [{"op": "replace", "path": "/background", "value": "#f5efe6"}],
            "questions": [
                "Is the title still the strongest element?",
                "Does the warmer background reduce contrast?",
            ],
        },
    )
    model = FakeMainAgent(
        [
            ModelResponse(tool_calls=(render_call,)),
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I warmed the background.",
                        "patch": [
                            {
                                "op": "replace",
                                "path": "/background",
                                "value": "#f5efe6",
                            }
                        ],
                    }
                )
            ),
        ]
    )
    renderer = FakeRenderer()
    vision = FakeVisionAgent()
    events = []

    async def capture_event(event) -> None:
        events.append(event)

    harness = Myli(
        main_model=model,
        vision_model=vision,
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )
    result = asyncio.run(
        harness.run(
            request="Make the background warmer and inspect it.",
            design=CURRENT_DESIGN,
            can_edit=True,
            on_event=capture_event,
        )
    )

    assert result.design == candidate
    assert result.changed is True
    assert result.patch == ({"op": "replace", "path": "/background", "value": "#f5efe6"},)
    assert renderer.designs == [candidate]
    assert vision.reviews[0][0].data == b"rendered-poster"
    assert "Make the background warmer" in vision.reviews[0][1]
    assert "Is the title still the strongest element?" in vision.reviews[0][1]
    assert "Does the warmer background reduce contrast?" in vision.reviews[0][1]
    assert json.loads(model.requests[1].messages[-1].content)["visual_review"]
    assert [event.kind for event in events] == [
        "run.started",
        "model.started",
        "model.completed",
        "tool.started",
        "tool.completed",
        "model.started",
        "model.completed",
        "validation.started",
        "validation.completed",
        "run.completed",
    ]
    assert all(event.run_id == result.run_id for event in events)
    assert events[0].step_id == events[-1].step_id == ""
    assert all(event.step_id == result.traces[0].step_id for event in events[1:5])
    assert all(event.step_id == result.traces[1].step_id for event in events[5:-1])
    assert all(event.safe_message == event.message for event in events)
    assert all(event.elapsed_seconds >= 0 for event in events)


def test_run_accounts_for_main_and_vision_model_usage() -> None:
    main_responses = iter(
        [
            {
                "model": "openai/main-test",
                "_hidden_params": {"custom_llm_provider": "openai"},
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 3},
                    "completion_tokens_details": {"reasoning_tokens": 1},
                },
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "id": "render-usage",
                                    "function": {
                                        "name": "render_design",
                                        "arguments": '{"patch":[]}',
                                    },
                                }
                            ]
                        }
                    }
                ],
            },
            {
                "model": "openai/main-test",
                "_hidden_params": {"custom_llm_provider": "openai"},
                "usage": {"input_tokens": 11, "output_tokens": 3},
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "message": "Review complete.",
                                    "patch": None,
                                }
                            )
                        }
                    }
                ],
            },
        ]
    )

    async def main_completion(**kwargs):
        del kwargs
        return next(main_responses)

    async def vision_completion(**kwargs):
        del kwargs
        return {
            "model": "anthropic/vision-test",
            "_hidden_params": {"custom_llm_provider": "anthropic"},
            "usage": {
                "input_tokens": 20,
                "output_tokens": 4,
                "cache_read_input_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 2},
            },
            "choices": [{"message": {"content": "The layout is balanced."}}],
        }

    harness = Myli(
        main_model=LiteLLMMainModel(model="openai/main-test", completion=main_completion),
        vision_model=LiteLLMVisionModel(model="anthropic/vision-test", completion=vision_completion),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Review the current design.", design=CURRENT_DESIGN))

    assert [call.purpose for call in result.model_calls] == [
        "main",
        "render_review",
        "main",
    ]
    assert [call.model for call in result.model_calls] == [
        "openai/main-test",
        "anthropic/vision-test",
        "openai/main-test",
    ]
    assert [call.provider for call in result.model_calls] == ["openai", "anthropic", "openai"]
    assert all(call.success for call in result.model_calls)
    assert all(call.latency_seconds >= 0 for call in result.model_calls)
    assert result.model_calls[0].cached_tokens == 3
    assert result.model_calls[0].reasoning_tokens == 1
    assert result.model_calls[1].cached_tokens == 5
    assert result.model_calls[1].reasoning_tokens == 2
    assert result.traces[0].model_calls == result.model_calls[:2]
    assert result.usage.input_tokens == 41
    assert result.usage.output_tokens == 9
    assert result.usage.cached_tokens == 8
    assert result.usage.reasoning_tokens == 3
    assert result.usage.request_count == 3
    assert result.usage.successful_requests == 3
    assert result.usage.failed_requests == 0
    assert result.usage.latency_seconds == sum(call.latency_seconds for call in result.model_calls)


def test_failed_vision_requests_are_traced_and_retries_are_counted() -> None:
    class FailingOnceVisionAgent:
        model = "provider/vision-test"
        provider = "provider"

        def __init__(self) -> None:
            self.calls = 0

        async def review(self, request: VisualReviewRequest) -> str:
            del request
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("temporary vision failure")
            return "The second review succeeded."

    render_calls = (
        ToolCall(id="render-failed", name="render_design", arguments={"patch": []}),
        ToolCall(id="render-retried", name="render_design", arguments={"patch": []}),
    )
    model = FakeMainAgent(
        [
            ModelResponse(tool_calls=(render_calls[0],)),
            ModelResponse(tool_calls=(render_calls[1],)),
            ModelResponse(content=json.dumps({"message": "Recovered.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FailingOnceVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Retry the visual review.", design=CURRENT_DESIGN))

    vision_calls = [call for call in result.model_calls if call.purpose == "render_review"]
    assert [call.success for call in vision_calls] == [False, True]
    assert [call.retry_count for call in vision_calls] == [0, 1]
    assert all(call.model == "provider/vision-test" for call in vision_calls)
    assert result.usage.request_count == 5
    assert result.usage.failed_requests == 1
    assert result.usage.retry_count == 1


def test_structured_vision_review_remains_an_object_in_tool_results() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-structured",
                        name="render_design",
                        arguments={"patch": []},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "Review complete.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeStructuredVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Inspect it.", design=CURRENT_DESIGN))

    review = result.tool_outcomes[0].result["visual_review"]
    assert review == {
        "summary": "The title is aligned.",
        "differences": [],
        "ready_to_commit": True,
    }


def test_application_tool_can_access_a_successful_render_by_reference() -> None:
    render_call = ToolCall(
        id="render-1",
        name="render_design",
        arguments={"patch": []},
    )

    def inspect_latest(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="inspect-render-1",
                    name="inspect_render_artifact",
                    arguments={"render_ref": latest_render_ref(request)},
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(tool_calls=(render_call,)),
            inspect_latest,
            ModelResponse(content=json.dumps({"message": "I inspected the render.", "patch": None})),
        ]
    )
    capture = RenderArtifactCaptureTool()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        tools=(capture,),
    )

    result = asyncio.run(
        harness.run(
            request="Inspect the current render.",
            design=CURRENT_DESIGN,
            capabilities={"render.inspect"},
        )
    )

    assert result.message == "I inspected the render."
    assert capture.artifacts == [RenderedArtifact(b"rendered-poster", "image/png")]


def test_application_tool_can_load_registered_inputs_once_per_run() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="inspect-input-1",
                        name="inspect_input_artifact",
                        arguments={"artifact_id": "attachment-1"},
                    ),
                    ToolCall(
                        id="inspect-input-2",
                        name="inspect_input_artifact",
                        arguments={"artifact_id": "attachment-1"},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "I inspected the attachment.", "patch": None})),
        ]
    )
    load_calls = 0

    async def load_attachment() -> RenderedArtifact:
        nonlocal load_calls
        load_calls += 1
        return RenderedArtifact(
            b"attached-image",
            "image/png",
            {"width": 320, "height": 180},
        )

    capture = InputArtifactCaptureTool()
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=(capture,),
    )
    result = asyncio.run(
        harness.run(
            request="Use my attachment as the visual reference.",
            design=CURRENT_DESIGN,
            capabilities={"input.inspect"},
            input_artifacts=(
                InputArtifact(
                    id="attachment-1",
                    kind="attached_image",
                    description="Latest user attachment 1.",
                    metadata={"attachment_number": 1, "message_scope": "latest"},
                    loader=load_attachment,
                ),
            ),
        )
    )

    assert result.message == "I inspected the attachment."
    assert load_calls == 1
    assert [artifact.data for artifact in capture.loaded] == [
        b"attached-image",
        b"attached-image",
    ]
    assert capture.registered[0].run_id == result.run_id
    prompt = model.requests[0].messages[-1].content
    assert '"id":"attachment-1"' in prompt
    assert '"kind":"attached_image"' in prompt
    assert '"message_scope":"latest"' in prompt


def test_agent_can_commit_a_render_and_return_null_final_patch() -> None:
    committed = {"background": "#f5efe6", "elements": []}
    previewed = {"background": "#000000", "elements": []}
    committed_patch = [{"op": "replace", "path": "/background", "value": "#f5efe6"}]
    policy_contexts: list[CandidateContext] = []

    def capture_policy_context(
        candidate: dict[str, Any],
        current: dict[str, Any],
        context: CandidateContext,
    ) -> None:
        del candidate, current
        policy_contexts.append(context)

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-selected",
                        name="render_design",
                        arguments={"patch": committed_patch},
                    ),
                )
            ),
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-preview",
                        name="render_design",
                        arguments={
                            "patch": [
                                {
                                    "op": "replace",
                                    "path": "/background",
                                    "value": "#000000",
                                }
                            ]
                        },
                    ),
                )
            ),
            commit_first_render,
            ModelResponse(content=json.dumps({"message": "I selected the warm render.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
        candidate_policies=[capture_policy_context],
    )

    result = asyncio.run(
        harness.run(
            request="Preview two backgrounds and use the warmer one.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    assert renderer.designs == [committed, previewed]
    assert result.design == committed
    assert result.changed is True
    assert result.patch == tuple(committed_patch)
    assert result.traces[0].tool_outcomes[0].result["render_ref"] == "render:1"
    assert result.traces[1].tool_outcomes[0].result["render_ref"] == "render:2"
    assert result.traces[2].tool_outcomes[0].result == {
        "render_ref": "render:1",
        "committed": True,
    }
    assert [context.phase for context in policy_contexts] == ["render", "render", "final"]
    assert [outcome.call_id for outcome in policy_contexts[-1].tool_outcomes] == [
        "render-selected",
        "render-preview",
        "commit-render",
    ]


def test_agent_can_increment_a_render_and_commit_the_composed_patch() -> None:
    first_patch = [
        {
            "op": "add",
            "path": "/elements/-",
            "value": {"kind": "text", "text": "Draft"},
        }
    ]
    revision_patch = [
        {
            "op": "replace",
            "path": "/elements/0/text",
            "value": "Final",
        }
    ]

    def revise_latest_render(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="render-revision",
                    name="render_design",
                    arguments={
                        "base_render_ref": latest_render_ref(request),
                        "patch": revision_patch,
                    },
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-draft",
                        name="render_design",
                        arguments={"patch": first_patch},
                    ),
                )
            ),
            revise_latest_render,
            commit_latest_render,
            ModelResponse(content=json.dumps({"message": "I selected the revision.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )

    result = asyncio.run(
        harness.run(
            request="Draft a title, revise it, and use the revision.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    assert renderer.designs == [
        {
            "background": "#ffffff",
            "elements": [{"kind": "text", "text": "Draft"}],
        },
        {
            "background": "#ffffff",
            "elements": [{"kind": "text", "text": "Final"}],
        },
    ]
    assert result.design == renderer.designs[-1]
    assert result.patch == tuple(first_patch + revision_patch)
    assert apply_json_patch(CURRENT_DESIGN, list(result.patch)) == result.design
    render_schema = model.requests[0].tools[0].input_schema
    assert render_schema["properties"]["base_render_ref"] == {
        "type": "string",
        "minLength": 1,
    }


def test_incremental_render_rejects_an_unknown_base_without_rendering() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-revision",
                        name="render_design",
                        arguments={
                            "base_render_ref": "render:missing",
                            "patch": [],
                        },
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "The base was unavailable.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )

    result = asyncio.run(harness.run(request="Revise that render.", design=CURRENT_DESIGN))

    assert renderer.designs == []
    outcome = result.tool_outcomes[0]
    assert outcome.status == "rejected"
    assert "does not identify a successful render" in (outcome.message or "")


def test_incremental_render_enforces_limits_on_the_composed_patch() -> None:
    first_patch = [{"op": "replace", "path": "/background", "value": "#f5efe6"}]

    def revise_latest_render(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="render-revision",
                    name="render_design",
                    arguments={
                        "base_render_ref": latest_render_ref(request),
                        "patch": [
                            {
                                "op": "add",
                                "path": "/elements/-",
                                "value": {"kind": "text", "text": "Title"},
                            }
                        ],
                    },
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-draft",
                        name="render_design",
                        arguments={"patch": first_patch},
                    ),
                )
            ),
            revise_latest_render,
            ModelResponse(content=json.dumps({"message": "The revision was too large.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
        limits=HarnessLimits(max_patch_operations=1),
    )

    result = asyncio.run(
        harness.run(
            request="Revise the first render.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    assert renderer.designs == [{"background": "#f5efe6", "elements": []}]
    outcome = result.tool_outcomes[1]
    assert outcome.status == "rejected"
    assert "exceeds the 1-operation limit" in (outcome.message or "")


def test_matching_final_patch_keeps_the_committed_render_patch() -> None:
    committed_patch = [{"op": "replace", "path": "/background", "value": "#f5efe6"}]
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-selected",
                        name="render_design",
                        arguments={"patch": committed_patch},
                    ),
                )
            ),
            commit_latest_render,
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I selected the warm render.",
                        "patch": [
                            {"op": "test", "path": "/background", "value": "#ffffff"},
                            *committed_patch,
                        ],
                    }
                )
            ),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Warm the background.", design=CURRENT_DESIGN, can_edit=True))

    assert result.design == {"background": "#f5efe6", "elements": []}
    assert result.patch == tuple(committed_patch)


def test_conflicting_final_patch_after_commit_uses_validation_retry() -> None:
    committed_patch = [{"op": "replace", "path": "/background", "value": "#f5efe6"}]
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-selected",
                        name="render_design",
                        arguments={"patch": committed_patch},
                    ),
                )
            ),
            commit_latest_render,
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I selected a dark background.",
                        "patch": [{"op": "replace", "path": "/background", "value": "#000000"}],
                    }
                )
            ),
            ModelResponse(content=json.dumps({"message": "I selected the warm render.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Warm the background.", design=CURRENT_DESIGN, can_edit=True))

    assert result.design == {"background": "#f5efe6", "elements": []}
    assert result.patch == tuple(committed_patch)
    assert "conflicts with the committed render" in result.traces[2].validation_failures[0]
    assert "previous final response was invalid" in (model.requests[3].messages[-1].content or "")


def test_uncommitted_render_does_not_affect_run_result() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-preview",
                        name="render_design",
                        arguments={
                            "patch": [
                                {
                                    "op": "replace",
                                    "path": "/background",
                                    "value": "#f5efe6",
                                }
                            ]
                        },
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "I only previewed it.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Preview a warm background.", design=CURRENT_DESIGN, can_edit=True))

    assert result.design is None
    assert result.changed is False
    assert result.patch is None


def test_render_commit_is_rejected_when_editing_is_disabled() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-diagnostic",
                        name="render_design",
                        arguments={"patch": []},
                    ),
                )
            ),
            commit_latest_render,
            ModelResponse(content=json.dumps({"message": "I did not change it.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )

    result = asyncio.run(harness.run(request="Review the current design.", design=CURRENT_DESIGN))

    assert renderer.designs == [CURRENT_DESIGN]
    assert result.design is None
    assert result.patch is None
    outcome = result.traces[1].tool_outcomes[0]
    assert outcome.status == "rejected"
    assert "cannot be committed" in (outcome.message or "")


def test_invalid_vision_questions_are_rejected_before_rendering() -> None:
    render_call = ToolCall(
        id="render-1",
        name="render_design",
        arguments={"patch": [], "questions": ["   "]},
    )
    model = FakeMainAgent(
        [
            ModelResponse(tool_calls=(render_call,)),
            ModelResponse(content=json.dumps({"message": "The questions were invalid.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    vision = FakeVisionAgent()
    harness = Myli(
        main_model=model,
        vision_model=vision,
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )

    result = asyncio.run(harness.run(request="Inspect the design.", design=CURRENT_DESIGN))

    outcome = result.traces[0].tool_outcomes[0]
    assert outcome.status == "rejected"
    assert "questions[0] cannot be empty" in (outcome.message or "")
    assert renderer.designs == []
    assert vision.reviews == []


def test_custom_agent_tool_enforces_capabilities_and_per_run_limit() -> None:
    tool = FakeAgentTool()
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-1",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                    ToolCall(
                        id="lookup-2",
                        name="lookup_brand",
                        arguments={"key": "secondary"},
                    ),
                ),
                metadata={
                    "provider_response_id": "response-1",
                    "model": "provider/model",
                    "finish_reason": "tool_calls",
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                },
                reasoning_content="contains sensitive planning",
            ),
            ModelResponse(
                content=json.dumps({"message": "Use navy.", "patch": None}),
            ),
        ]
    )
    steps = []

    def redact(trace):
        return replace(
            trace,
            response=replace(trace.response, reasoning_content="[redacted]"),
        )

    async def capture_step(trace) -> None:
        steps.append(trace)

    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        tools=[tool],
        trace_redactor=redact,
    )
    result = asyncio.run(
        harness.run(
            request="Which brand color should I use?",
            design=CURRENT_DESIGN,
            capabilities={"brand.read"},
            on_step=capture_step,
        )
    )

    assert [definition.name for definition in model.requests[0].tools][-1] == ("lookup_brand")
    assert tool.calls == [{"key": "primary"}]
    assert result.traces[0].tool_outcomes[0].result == {"value": "navy"}
    assert result.traces[0].tool_outcomes[1].status == "rejected"
    assert "call budget" in (result.traces[0].tool_outcomes[1].message or "")
    assert result.traces[0].provider_response_id == "response-1"
    assert result.traces[0].model_name == "provider/model"
    assert result.traces[0].finish_reason == "tool_calls"
    assert result.traces[0].token_usage == {
        "prompt_tokens": 12,
        "completion_tokens": 4,
    }
    assert result.traces[0].reasoning_content == "[redacted]"
    assert result.traces[0].model_latency_seconds is not None
    assert result.traces[0].tool_latency_seconds is not None
    assert all(item.latency_seconds is not None for item in result.traces[0].tool_outcomes)
    assert steps == list(result.traces)


def test_custom_agent_tool_can_limit_only_the_model_visible_result() -> None:
    complete_result = {
        "summary": "Use navy.",
        "records": [
            {"id": "brand-primary", "value": "navy", "audit_revision": 17},
            {"id": "brand-secondary", "value": "cream", "audit_revision": 9},
        ],
    }

    class ProjectedAgentTool(FakeAgentTool):
        def __init__(self) -> None:
            super().__init__(required_capabilities=frozenset(), result=complete_result)
            self.model_contexts: list[ToolContext] = []

        def model_view(self, result: Any, context: ToolContext) -> Any:
            self.model_contexts.append(context)
            result["records"].clear()
            return {"summary": result["summary"]}

    policy_contexts: list[CandidateContext] = []

    def capture_policy(
        candidate: dict[str, Any],
        current: dict[str, Any],
        context: CandidateContext,
    ) -> None:
        del candidate, current
        policy_contexts.append(context)

    def retrieve_evidence_values(request: ModelRequest) -> ModelResponse:
        projected = json.loads(request.messages[-1].content or "null")
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="retrieve-secondary",
                    name="retrieve_evidence",
                    arguments={
                        "evidence_ref": projected["evidence_ref"],
                        "json_pointer": "/records/1/value",
                    },
                ),
                ToolCall(
                    id="retrieve-complete",
                    name="retrieve_evidence",
                    arguments={"evidence_ref": projected["evidence_ref"]},
                ),
            )
        )

    tool = ProjectedAgentTool()
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-projected",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            retrieve_evidence_values,
            ModelResponse(content=json.dumps({"message": "Use navy.", "patch": []})),
        ]
    )
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=[tool],
        candidate_policies=[capture_policy],
    )

    result = asyncio.run(
        harness.run(
            request="Look up the brand.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    projected_message = model.requests[1].messages[-1]
    assert projected_message.role == "tool"
    projected = json.loads(projected_message.content or "null")
    evidence_ref = projected.pop("evidence_ref")
    assert projected == {"summary": "Use navy."}
    assert evidence_ref.startswith(f"{result.run_id}:evidence:")
    retrieved_messages = model.requests[2].messages[-2:]
    assert json.loads(retrieved_messages[0].content or "null") == {
        "evidence_ref": evidence_ref,
        "json_pointer": "/records/1/value",
        "value": "cream",
    }
    assert json.loads(retrieved_messages[1].content or "null") == {
        "evidence_ref": evidence_ref,
        "json_pointer": "",
        "value": complete_result,
    }
    assert [definition.name for definition in model.requests[0].tools] == [
        "lookup_brand",
    ]
    assert [definition.name for definition in model.requests[1].tools] == [
        "retrieve_evidence",
        "lookup_brand",
    ]
    assert result.tool_outcomes[0].result == complete_result
    assert result.tool_outcomes[0].evidence_ref == evidence_ref
    assert result.tool_outcomes[0].to_dict()["evidence_ref"] == evidence_ref
    assert result.traces[0].tool_outcomes[0].result == complete_result
    assert policy_contexts[-1].tool_outcomes[0].result == complete_result
    assert tool.model_contexts[0].run_id == result.run_id


def test_evidence_retrieval_is_run_scoped_and_bounded() -> None:
    class ProjectedAgentTool(FakeAgentTool):
        def __init__(self) -> None:
            super().__init__(
                required_capabilities=frozenset(),
                max_calls_per_run=1,
                max_result_bytes=4096,
                result={"blob": ["x" * 1000, "small"]},
            )

        def model_view(self, result: Any, context: ToolContext) -> Any:
            del result, context
            return {"summary": "Large evidence is available on demand."}

    def retrieve_bounded_values(request: ModelRequest) -> ModelResponse:
        evidence_ref = json.loads(request.messages[-1].content or "null")["evidence_ref"]
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="retrieve-too-large",
                    name="retrieve_evidence",
                    arguments={"evidence_ref": evidence_ref},
                ),
                ToolCall(
                    id="retrieve-small",
                    name="retrieve_evidence",
                    arguments={
                        "evidence_ref": evidence_ref,
                        "json_pointer": "/blob/1",
                    },
                ),
                ToolCall(
                    id="retrieve-foreign",
                    name="retrieve_evidence",
                    arguments={"evidence_ref": "another-run:evidence:1"},
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-large",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            retrieve_bounded_values,
            ModelResponse(content=json.dumps({"message": "Retrieved.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=[ProjectedAgentTool()],
        limits=HarnessLimits(
            max_evidence_retrievals=2,
            max_evidence_result_bytes=256,
        ),
    )

    result = asyncio.run(harness.run(request="Inspect the evidence.", design=CURRENT_DESIGN))

    assert [outcome.status for outcome in result.tool_outcomes] == [
        "succeeded",
        "rejected",
        "succeeded",
        "rejected",
    ]
    assert "use json_pointer" in (result.tool_outcomes[1].message or "")
    assert result.tool_outcomes[2].result["value"] == "small"
    assert "from this run" in (result.tool_outcomes[3].message or "")


def test_complete_model_view_does_not_create_an_evidence_reference() -> None:
    class IdentityViewAgentTool(FakeAgentTool):
        def model_view(self, result: Any, context: ToolContext) -> Any:
            del context
            return result

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-complete-view",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "Use navy.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=[IdentityViewAgentTool(required_capabilities=frozenset())],
    )

    result = asyncio.run(harness.run(request="Look up the brand.", design=CURRENT_DESIGN))

    assert json.loads(model.requests[1].messages[-1].content or "null") == {"value": "navy"}
    assert result.tool_outcomes[0].evidence_ref is None


def test_custom_agent_tool_denies_missing_capabilities() -> None:
    tool = FakeAgentTool()
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-denied",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "Access unavailable.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        tools=[tool],
    )

    result = asyncio.run(harness.run(request="Look up the brand.", design=CURRENT_DESIGN))

    assert tool.calls == []
    assert "lookup_brand" not in {definition.name for definition in model.requests[0].tools}
    assert result.traces[0].tool_outcomes[0].status == "rejected"
    assert "brand.read" in (result.traces[0].tool_outcomes[0].message or "")


def test_custom_agent_tool_enforces_timeout_and_result_size() -> None:
    slow = FakeAgentTool(
        name="slow_lookup",
        required_capabilities=frozenset(),
        timeout_seconds=0.001,
        delay=0.02,
    )
    large = FakeAgentTool(
        name="large_lookup",
        required_capabilities=frozenset(),
        max_result_bytes=4,
        result={"value": "too large"},
    )
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(id="slow-1", name="slow_lookup", arguments={"key": "x"}),
                    ToolCall(id="large-1", name="large_lookup", arguments={"key": "x"}),
                )
            ),
            ModelResponse(content=json.dumps({"message": "No result.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        tools=[slow, large],
    )

    result = asyncio.run(harness.run(request="Look it up.", design=CURRENT_DESIGN))

    assert [item.status for item in result.traces[0].tool_outcomes] == ["timed_out", "failed"]
    assert "timed out" in (result.traces[0].tool_outcomes[0].message or "")
    assert "byte limit" in (result.traces[0].tool_outcomes[1].message or "")


def test_custom_agent_tool_failure_raises_stable_execution_error() -> None:
    source_error = LookupError("brand service unavailable")

    class FailingAgentTool(FakeAgentTool):
        async def execute(self, arguments: dict[str, Any], context: ToolContext) -> Any:
            del context
            self.calls.append(dict(arguments))
            raise source_error

    tool = FailingAgentTool(required_capabilities=frozenset())
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="lookup-failed",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            )
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        tools=[tool],
        tool_failure_modes={"lookup_brand": FailureMode.RAISE},
    )

    with unittest.TestCase().assertRaisesRegex(ToolExecutionError, "Tool lookup_brand failed") as raised:
        asyncio.run(harness.run(request="Look it up.", design=CURRENT_DESIGN))

    assert raised.exception.__cause__ is source_error


def test_main_prompt_is_configurable_and_preserves_app_guidance() -> None:
    model = FakeMainAgent([ModelResponse(content=json.dumps({"message": "Reviewed.", "patch": None}))])
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        main_prompt="Custom main system prompt.",
        instructions="Use the house spacing scale.",
    )

    asyncio.run(harness.run(request="Review this.", design=CURRENT_DESIGN))

    prompt = model.requests[0].messages[0].content or ""
    assert prompt.startswith("Custom main system prompt.")
    assert "Use the house spacing scale." in prompt


def test_multiple_asset_search_tools_and_preview_inspection_share_approved_assets() -> None:
    photo = Asset(
        id="photo-1",
        uri="asset://photo-1",
        kind="photo",
        description="Warm editorial portrait",
        preview=RenderedArtifact(b"photo-preview", "image/jpeg"),
    )
    icon = Asset(
        id="icon-1",
        uri="asset://icon-1",
        kind="icon",
        description="Minimal sun icon",
    )
    stock = FakeSearch("stock", [photo])
    icons = FakeSearch("icons", [icon])
    final_design = {
        "background": "#ffffff",
        "elements": [{"type": "image", "uri": photo.uri}],
    }

    def inspect_photo(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="inspect-1",
                    name="inspect_asset",
                    arguments={
                        "asset_ref": searched_asset_ref(
                            request,
                            source="stock",
                            asset_id="photo-1",
                        )
                    },
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="search-1",
                        name="search_assets_stock",
                        arguments={"query": "warm editorial portrait", "limit": 2},
                    ),
                    ToolCall(
                        id="search-2",
                        name="search_assets_icons",
                        arguments={"query": "minimal sun", "limit": 1},
                    ),
                )
            ),
            inspect_photo,
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I added the selected portrait.",
                        "patch": [
                            {
                                "op": "add",
                                "path": "/elements/-",
                                "value": {"type": "image", "uri": photo.uri},
                            }
                        ],
                    }
                )
            ),
        ]
    )

    def require_searched_images(
        candidate: dict[str, Any],
        current: dict[str, Any],
        context: CandidateContext,
    ) -> None:
        del current
        approved = {asset.uri for asset in context.approved_assets}
        for element in candidate["elements"]:
            if element.get("type") == "image" and element.get("uri") not in approved:
                raise ValueError("image URI was not returned by search")

    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        asset_search_providers=[stock, icons],
        candidate_policies=[require_searched_images],
    )
    result = asyncio.run(
        harness.run(
            request="Find and add a warm editorial portrait.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    assert result.design == final_design
    assert {asset.uri for asset in result.approved_assets} == {
        "asset://photo-1",
        "asset://icon-1",
    }
    assert stock.queries == [("warm editorial portrait", 2)]
    assert icons.queries == [("minimal sun", 1)]
    tool_names = [tool.name for tool in model.requests[0].tools]
    assert tool_names == [
        "render_design",
        "commit_render",
        "search_assets_stock",
        "search_assets_icons",
        "inspect_asset",
    ]
    assert [call.purpose for call in result.model_calls] == [
        "main",
        "main",
        "asset_inspection",
        "main",
    ]


def test_asset_search_supports_non_image_assets_and_visual_previews() -> None:
    font = Asset(
        id="font-1",
        uri="asset://font-1",
        kind="font",
        description="A friendly geometric sans",
        preview=RenderedArtifact(b"font-specimen", "image/png"),
        metadata={"family": "Example Sans"},
    )
    palette = Asset(
        id="palette-1",
        uri="asset://palette-1",
        kind="palette",
        description="Warm earth tones",
    )
    library = FakeSearch("library", [font, palette])

    def inspect_font(request: ModelRequest) -> ModelResponse:
        return ModelResponse(
            tool_calls=(
                ToolCall(
                    id="inspect-font",
                    name="inspect_asset",
                    arguments={
                        "asset_ref": searched_asset_ref(
                            request,
                            source="library",
                            asset_id="font-1",
                        ),
                        "questions": ["Would this remain legible in a compact navigation bar?"],
                    },
                ),
            )
        )

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="search-assets",
                        name="search_assets_library",
                        arguments={"query": "friendly warm brand system", "limit": 2},
                    ),
                )
            ),
            inspect_font,
            ModelResponse(content=json.dumps({"message": "The font and palette fit.", "patch": None})),
        ]
    )
    vision = FakeVisionAgent()
    harness = Myli(
        main_model=model,
        vision_model=vision,
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        asset_search_providers=[library],
    )

    result = asyncio.run(harness.run(request="Find a friendly brand system.", design=CURRENT_DESIGN))

    assert [tool.name for tool in model.requests[0].tools] == [
        "render_design",
        "commit_render",
        "search_assets_library",
        "inspect_asset",
    ]
    search_result = result.traces[0].tool_outcomes[0].result
    assert [item["kind"] for item in search_result["assets"]] == ["font", "palette"]
    assert search_result["assets"][0]["preview_available"] is True
    assert search_result["assets"][1]["preview_available"] is False
    assert [(asset.id, asset.uri) for asset in result.approved_assets] == [
        (font.id, font.uri),
        (palette.id, palette.uri),
    ]
    assert all(asset.run_id == result.run_id for asset in result.approved_assets)
    assert "discovered font preview" in vision.reviews[0][1]
    assert "Would this remain legible in a compact navigation bar?" in vision.reviews[0][1]


def test_editing_disabled_retries_a_model_that_returns_a_patch() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I changed it.",
                        "patch": [
                            {
                                "op": "replace",
                                "path": "/background",
                                "value": "#000000",
                            }
                        ],
                    }
                )
            ),
            ModelResponse(content=json.dumps({"message": "Use a darker background for contrast.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(
        harness.run(
            request="How could this have more contrast?",
            design=CURRENT_DESIGN,
            can_edit=False,
        )
    )

    assert result.design is None
    assert len(model.requests) == 2
    assert result.traces[0].validation_failures == ("Editing is disabled; patch must be null.",)
    assert "previous final response was invalid" in (model.requests[1].messages[-1].content or "")


def test_editing_disabled_allows_only_unchanged_diagnostic_renders() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="render-forbidden",
                        name="render_design",
                        arguments={
                            "patch": [
                                {
                                    "op": "replace",
                                    "path": "/background",
                                    "value": "#000000",
                                }
                            ]
                        },
                    ),
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "A darker background would add contrast.",
                        "patch": None,
                    }
                )
            ),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=renderer,
    )

    result = asyncio.run(
        harness.run(
            request="Would a dark background work?",
            design=CURRENT_DESIGN,
            can_edit=False,
        )
    )

    assert result.changed is False
    assert renderer.designs == []
    assert result.traces[0].tool_outcomes[0].status == "rejected"
    assert "changed candidate cannot be rendered" in (result.traces[0].tool_outcomes[0].message or "")


def test_candidate_policy_can_reject_an_unsearched_asset() -> None:
    candidate = {
        "background": "#ffffff",
        "elements": [{"type": "image", "uri": "invented://image"}],
    }
    model = FakeMainAgent(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "Added it.",
                        "patch": [
                            {
                                "op": "add",
                                "path": "/elements/-",
                                "value": candidate["elements"][0],
                            }
                        ],
                    }
                )
            )
        ]
    )

    def reject_images(
        design: dict[str, Any],
        current: dict[str, Any],
        context: CandidateContext,
    ) -> None:
        del current
        approved = {asset.uri for asset in context.approved_assets}
        if any(element.get("uri") not in approved for element in design["elements"]):
            raise ValueError("unapproved image asset")

    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
        candidate_policies=[reject_images],
        limits=HarnessLimits(max_validation_retries=0),
    )

    with unittest.TestCase().assertRaises(ModelProtocolError) as error:
        asyncio.run(
            harness.run(
                request="Add an image.",
                design=CURRENT_DESIGN,
                can_edit=True,
            )
        )

    assert "validation retry budget" in str(error.exception)
    assert "unapproved image asset" in str(error.exception.__cause__)


def test_patch_schemas_do_not_embed_the_complete_design_schema() -> None:
    spec = DesignSpec[dict[str, Any]](
        name="card",
        schema={
            "$defs": {
                "element": {
                    "type": "object",
                    "properties": {"text": {"type": "string"}},
                    "required": ["text"],
                }
            },
            "type": "object",
            "properties": {
                "element": {"$ref": "#/$defs/element"},
            },
            "required": ["element"],
        },
        validator=lambda value: value,
        serializer=lambda value: value,
    )
    harness = Myli(
        main_model=FakeMainAgent([]),
        vision_model=FakeVisionAgent(),
        design_spec=spec,
        renderer=FakeRenderer(),
    )

    render_schema = harness._tool_definitions[0].input_schema
    assert set(render_schema["properties"]) == {"base_render_ref", "patch", "questions"}
    assert render_schema["properties"]["questions"]["maxItems"] == 8
    assert "$defs" not in render_schema
    commit_schema = harness._tool_definitions[1].input_schema
    assert commit_schema == {
        "type": "object",
        "properties": {"render_ref": {"type": "string", "minLength": 1}},
        "required": ["render_ref"],
        "additionalProperties": False,
    }
    assert set(harness._output_schema["properties"]) == {"message", "patch"}
    assert "$defs" not in harness._output_schema


def test_invalid_patch_is_retried_without_mutating_the_current_design() -> None:
    model = FakeMainAgent(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "Changed it.",
                        "patch": [{"op": "replace", "path": "/missing", "value": True}],
                    }
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I warmed the background.",
                        "patch": [
                            {
                                "op": "replace",
                                "path": "/background",
                                "value": "#f5efe6",
                            }
                        ],
                    }
                )
            ),
        ]
    )
    current = {"background": "#ffffff", "elements": []}
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(harness.run(request="Warm the background.", design=current, can_edit=True))

    assert current == CURRENT_DESIGN
    assert result.design == {"background": "#f5efe6", "elements": []}
    assert "path member 'missing' does not exist" in result.traces[0].validation_failures[0]


def test_normalized_final_patch_is_retried_to_preserve_patch_consistency() -> None:
    normalizing_spec = replace(
        DESIGN_SPEC,
        validator=normalize_background_in_place,
    )
    model = FakeMainAgent(
        [
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I warmed the background.",
                        "patch": [
                            {
                                "op": "replace",
                                "path": "/background",
                                "value": "#f5efe6 ",
                            }
                        ],
                    }
                )
            ),
            ModelResponse(
                content=json.dumps(
                    {
                        "message": "I warmed the background.",
                        "patch": [
                            {
                                "op": "replace",
                                "path": "/background",
                                "value": "#f5efe6",
                            }
                        ],
                    }
                )
            ),
        ]
    )
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=normalizing_spec,
        renderer=FakeRenderer(),
    )

    result = asyncio.run(
        harness.run(
            request="Warm the background.",
            design=CURRENT_DESIGN,
            can_edit=True,
        )
    )

    assert result.design == {"background": "#f5efe6", "elements": []}
    assert result.patch == ({"op": "replace", "path": "/background", "value": "#f5efe6"},)
    assert apply_json_patch(CURRENT_DESIGN, list(result.patch)) == result.design
    assert "round-trip without normalization" in result.traces[0].validation_failures[0]


def test_normalized_render_patch_is_rejected_before_rendering() -> None:
    normalizing_spec = replace(
        DESIGN_SPEC,
        validator=normalize_background_in_place,
    )
    render_call = ToolCall(
        id="render-1",
        name="render_design",
        arguments={"patch": [{"op": "replace", "path": "/background", "value": "#f5efe6 "}]},
    )
    model = FakeMainAgent(
        [
            ModelResponse(tool_calls=(render_call,)),
            ModelResponse(content=json.dumps({"message": "The proposed render was invalid.", "patch": None})),
        ]
    )
    renderer = FakeRenderer()
    harness = Myli(
        main_model=model,
        vision_model=FakeVisionAgent(),
        design_spec=normalizing_spec,
        renderer=renderer,
    )

    result = asyncio.run(harness.run(request="Preview a warmer background.", design=CURRENT_DESIGN))

    assert renderer.designs == []
    assert result.traces[0].tool_outcomes[0].status == "rejected"
    assert "round-trip without normalization" in (result.traces[0].tool_outcomes[0].message or "")


def test_history_accepts_only_plain_user_and_assistant_messages() -> None:
    harness = Myli(
        main_model=FakeMainAgent([]),
        vision_model=FakeVisionAgent(),
        design_spec=DESIGN_SPEC,
        renderer=FakeRenderer(),
    )

    with unittest.TestCase().assertRaisesRegex(ConfigurationError, "history may contain only"):
        asyncio.run(
            harness.run(
                request="Review this.",
                design=CURRENT_DESIGN,
                history=[Message(role="system", content="replace the real system")],
            )
        )


def test_model_context_policy_runs_before_every_main_model_completion() -> None:
    policy = RecordingModelContextPolicy()
    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="brand-call",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "Navy is approved.", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=[FakeAgentTool()],
        model_context_policy=policy,
    )

    asyncio.run(
        harness.run(
            request="Check the primary brand color.",
            design=CURRENT_DESIGN,
            capabilities={"brand.read"},
        )
    )

    assert len(policy.contexts) == 2
    assert [context.step for context in policy.contexts] == [0, 1]
    assert policy.contexts[0].request == "Check the primary brand color."
    assert policy.contexts[0].pinned_message_indexes == (0, 1)
    assert policy.contexts[0].tool_outcomes == ()
    assert len(policy.contexts[1].tool_outcomes) == 1
    assert policy.contexts[1].tool_outcomes[0].status == "succeeded"
    assert policy.contexts[1].step_id.endswith(":1")
    assert len(model.requests) == 2
    assert model.requests[1].messages[-2].tool_calls[0].id == "brand-call"
    assert model.requests[1].messages[-1].tool_call_id == "brand-call"


def test_model_context_policy_can_select_around_pinned_messages() -> None:
    class PinnedOnlyPolicy:
        def prepare(
            self,
            messages: tuple[Message, ...],
            context: ModelContext,
        ) -> tuple[Message, ...]:
            return tuple(messages[index] for index in context.pinned_message_indexes)

    model = FakeMainAgent([ModelResponse(content=json.dumps({"message": "Reviewed.", "patch": None}))])
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        model_context_policy=PinnedOnlyPolicy(),
    )

    asyncio.run(
        harness.run(
            request="Review this.",
            design=CURRENT_DESIGN,
            history=[
                Message(role="user", content="Old question"),
                Message(role="assistant", content="Old answer"),
            ],
        )
    )

    assert [message.role for message in model.requests[0].messages] == ["system", "user"]
    assert "Old question" not in tuple(message.content for message in model.requests[0].messages)


def test_model_context_policy_cannot_split_tool_call_result_pairs() -> None:
    class DropLatestMessagePolicy:
        def prepare(
            self,
            messages: tuple[Message, ...],
            context: ModelContext,
        ) -> tuple[Message, ...]:
            if context.step == 1:
                return tuple(messages[:-1])
            return tuple(messages)

    model = FakeMainAgent(
        [
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="brand-call",
                        name="lookup_brand",
                        arguments={"key": "primary"},
                    ),
                )
            ),
            ModelResponse(content=json.dumps({"message": "unused", "patch": None})),
        ]
    )
    harness = Myli(
        main_model=model,
        design_spec=DESIGN_SPEC,
        tools=[FakeAgentTool()],
        model_context_policy=DropLatestMessagePolicy(),
    )

    with unittest.TestCase().assertRaisesRegex(ConfigurationError, "required tool results"):
        asyncio.run(
            harness.run(
                request="Check the primary brand color.",
                design=CURRENT_DESIGN,
                capabilities={"brand.read"},
            )
        )

    assert len(model.requests) == 1


def test_default_model_context_policy_is_safe_and_public() -> None:
    harness = Myli(
        main_model=FakeMainAgent([]),
        design_spec=DESIGN_SPEC,
    )

    assert isinstance(harness.model_context_policy, DefaultModelContextPolicy)


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
