"""Offline tests for Myli-owned model clients."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest

from myli import (
    DesignSpec,
    FunctionRenderer,
    LiteLLMMainModel,
    LiteLLMVisionModel,
    Message,
    ModelProtocolError,
    ModelRequest,
    Myli,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RenderedArtifact,
    ToolCall,
    ToolDefinition,
    VisualReviewImage,
    VisualReviewRequest,
)


def model_request() -> ModelRequest:
    prior_call = ToolCall(
        id="prior-1",
        name="render_design",
        arguments={"patch": []},
    )
    return ModelRequest(
        messages=(
            Message(role="system", content="System instructions"),
            Message(role="user", content="Improve the poster"),
            Message(role="assistant", tool_calls=(prior_call,)),
            Message(
                role="tool",
                content='{"visual_review":"balanced"}',
                tool_call_id="prior-1",
            ),
        ),
        tools=(
            ToolDefinition(
                name="render_design",
                description="Render a design",
                input_schema={
                    "type": "object",
                    "properties": {"patch": {"type": "array"}},
                    "required": ["patch"],
                },
            ),
        ),
        output_schema={
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "patch": {"type": ["array", "null"]},
            },
            "required": ["message", "patch"],
        },
    )


def test_main_client_maps_messages_tools_schema_and_tool_response() -> None:
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {
            "id": "response-1",
            "model": "anthropic/claude-test",
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "content": None,
                        "reasoning_content": "I should search before editing.",
                        "tool_calls": [
                            {
                                "id": "search-1",
                                "type": "function",
                                "function": {
                                    "name": "search_assets_stock",
                                    "arguments": '{"query":"calm landscape","limit":2}',
                                },
                            }
                        ],
                    },
                }
            ],
        }

    client = LiteLLMMainModel(
        model="anthropic/claude-test",
        base_url="https://models.example.test/v1",
        api_key="main-secret",
        options={"temperature": 0.2, "timeout": 30},
        completion=completion,
    )
    response = asyncio.run(client.complete(model_request()))

    call = calls[0]
    assert call["model"] == "anthropic/claude-test"
    assert call["api_base"] == "https://models.example.test/v1"
    assert call["api_key"] == "main-secret"
    assert call["temperature"] == 0.2
    assert call["stream"] is False
    assert call["messages"][2]["tool_calls"][0]["function"]["arguments"] == ('{"patch":[]}')
    assert call["messages"][3] == {
        "role": "tool",
        "content": '{"visual_review":"balanced"}',
        "tool_call_id": "prior-1",
    }
    assert call["tools"][0]["function"]["name"] == "render_design"
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["strict"] is True
    assert call["response_format"]["json_schema"]["schema"] == (model_request().output_schema)
    assert response.tool_calls == (
        ToolCall(
            id="search-1",
            name="search_assets_stock",
            arguments={"query": "calm landscape", "limit": 2},
        ),
    )
    assert response.metadata["provider_response_id"] == "response-1"
    assert response.metadata["usage"]["prompt_tokens"] == 10
    assert response.reasoning_content == "I should search before editing."


def test_main_client_constructs_semantically_equal_requests_identically() -> None:
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"message":"Done.","patch":null}'}}]}

    def request(*, reverse: bool) -> ModelRequest:
        schema_items = [
            ("type", "object"),
            ("properties", {"z": {"type": "number"}, "a": {"type": "string"}}),
            ("required", ["a"]),
        ]
        if reverse:
            schema_items.reverse()
        tools = [
            ToolDefinition("z_tool", "Z tool", dict(schema_items)),
            ToolDefinition("a_tool", "A tool", dict(reversed(schema_items))),
        ]
        if reverse:
            tools.reverse()
        return ModelRequest(
            messages=(
                Message(role="system", content="Stable system instructions."),
                Message(
                    role="assistant",
                    tool_calls=(ToolCall("call-1", "a_tool", {"z": 1, "a": 2}),),
                ),
                Message(role="tool", content='{"z":1,"a":{"d":4,"c":3}}', tool_call_id="call-1"),
            ),
            tools=tuple(tools),
            output_schema=dict(schema_items),
        )

    client = LiteLLMMainModel(model="provider/test", completion=completion)
    asyncio.run(client.complete(request(reverse=False)))
    asyncio.run(client.complete(request(reverse=True)))

    assert calls[0] == calls[1]
    assert [tool["function"]["name"] for tool in calls[0]["tools"]] == ["a_tool", "z_tool"]
    assert calls[0]["messages"][1]["tool_calls"][0]["function"]["arguments"] == '{"a":2,"z":1}'
    assert calls[0]["messages"][2]["content"] == '{"a":{"c":3,"d":4},"z":1}'
    assert list(calls[0]["response_format"]["json_schema"]["schema"]) == ["properties", "required", "type"]


def test_litellm_cache_controls_and_provider_reported_usage_are_exposed() -> None:
    calls = []
    responses = iter(
        [
            {
                "_hidden_params": {"cache_hit": True},
                "usage": {
                    "prompt_tokens": 20,
                    "prompt_tokens_details": {
                        "cached_tokens": 8,
                        "cache_write_tokens": 5,
                    },
                },
                "choices": [{"message": {"content": '{"message":"Cached.","patch":null}'}}],
            },
            {
                "usage": {
                    "prompt_tokens": 20,
                    "prompt_tokens_details": {"cached_tokens": 7},
                },
                "choices": [{"message": {"content": '{"message":"Unknown.","patch":null}'}}],
            },
        ]
    )

    async def completion(**kwargs):
        calls.append(kwargs)
        return next(responses)

    client = LiteLLMMainModel(
        model="anthropic/test",
        cache_control_injection_points=(
            {"role": "system", "location": "message"},
            {"location": "tool_config"},
        ),
        completion=completion,
    )
    reported = asyncio.run(client.complete(model_request()))
    unreported = asyncio.run(client.complete(model_request()))

    expected_controls = [
        {"location": "message", "role": "system"},
        {"location": "tool_config"},
    ]
    assert calls[0]["cache_control_injection_points"] == expected_controls
    assert calls[1]["cache_control_injection_points"] == expected_controls
    assert reported.metadata["cache_hit"] is True
    assert reported.metadata["cache_usage"] == {
        "read_input_tokens": 8,
        "write_input_tokens": 5,
        "cache_hit": True,
    }
    assert "cache_hit" not in unreported.metadata
    assert unreported.metadata["cache_usage"] == {"read_input_tokens": 7}


def test_main_client_supports_json_and_text_modes() -> None:
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps({"message": "Looks balanced.", "patch": None})},
                }
            ]
        }

    json_client = LiteLLMMainModel(
        model="openai/test",
        output_mode="json",
        completion=completion,
    )
    text_client = LiteLLMMainModel(
        model="ollama/test",
        output_mode="text",
        completion=completion,
    )

    json_response = asyncio.run(json_client.complete(model_request()))
    asyncio.run(text_client.complete(model_request()))

    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
    assert json.loads(json_response.content)["patch"] is None


def test_main_client_rejects_invalid_tool_argument_json() -> None:
    async def completion(**kwargs):
        del kwargs
        return {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "bad-1",
                                "function": {
                                    "name": "render_design",
                                    "arguments": "not-json",
                                },
                            }
                        ]
                    }
                }
            ]
        }

    client = LiteLLMMainModel(model="openai/test", completion=completion)

    with unittest.TestCase().assertRaisesRegex(ModelProtocolError, "invalid JSON arguments"):
        asyncio.run(client.complete(model_request()))


def test_main_client_normalizes_unexpected_response_parsing_failures() -> None:
    source_error = LookupError("broken provider response object")

    class BrokenResponse:
        @property
        def choices(self):
            raise source_error

    async def completion(**kwargs):
        del kwargs
        return BrokenResponse()

    client = LiteLLMMainModel(model="provider/test", completion=completion)

    with unittest.TestCase().assertRaisesRegex(ModelProtocolError, "could not be normalized") as raised:
        asyncio.run(client.complete(model_request()))
    assert raised.exception.__cause__ is source_error


def test_model_client_translates_provider_failures_to_stable_errors() -> None:
    class RateLimitFailure(Exception):
        status_code = 429

    cases = (
        (ConnectionError("connection refused"), ProviderConnectionError),
        (TimeoutError("request expired"), ProviderTimeoutError),
        (RateLimitFailure("retry later"), ProviderRateLimitError),
        (ValueError("provider rejected request"), ProviderError),
    )
    test_case = unittest.TestCase()

    for source_error, expected_type in cases:

        async def completion(_error=source_error, **kwargs):
            del kwargs
            raise _error

        client = LiteLLMMainModel(model="provider/test", completion=completion)
        with test_case.subTest(expected_type=expected_type.__name__):
            with test_case.assertRaises(expected_type) as raised:
                asyncio.run(client.complete(model_request()))
            assert raised.exception.__cause__ is source_error


def test_model_client_does_not_rewrap_stable_myli_errors() -> None:
    source_error = ProviderRateLimitError("already normalized")

    async def completion(**kwargs):
        del kwargs
        raise source_error

    client = LiteLLMMainModel(model="provider/test", completion=completion)

    with unittest.TestCase().assertRaises(ProviderRateLimitError) as raised:
        asyncio.run(client.complete(model_request()))
    assert raised.exception is source_error


def test_vision_client_sends_an_in_memory_data_url() -> None:
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Strong hierarchy. "},
                            {"type": "text", "text": "Increase contrast."},
                        ]
                    }
                }
            ]
        }

    vision = LiteLLMVisionModel(
        model="openai/vision-test",
        base_url="https://vision.example.test/v1",
        api_key="vision-secret",
        options={"max_tokens": 500},
        completion=completion,
    )
    request = VisualReviewRequest(
        images=(
            VisualReviewImage(
                RenderedArtifact(b"test-png", "image/png"),
                label="Rendered poster",
            ),
        ),
        prompt="Review this poster.",
    )
    review = asyncio.run(vision.review(request))

    call = calls[0]
    content = call["messages"][1]["content"]
    assert content[0] == {
        "type": "text",
        "text": "Rendered poster. Original dimensions are unavailable.",
    }
    assert content[1]["image_url"]["url"] == ("data:image/png;base64," + base64.b64encode(b"test-png").decode("ascii"))
    assert content[2] == {"type": "text", "text": "Review this poster."}
    assert call["max_tokens"] == 500
    assert call["api_base"] == "https://vision.example.test/v1"
    assert call["api_key"] == "vision-secret"
    assert call["stream"] is False
    assert review == "Strong hierarchy. Increase contrast."
    assert request.purpose == "comparison"


def test_vision_client_sends_multiple_labeled_images_and_validates_schema() -> None:
    calls = []

    async def completion(**kwargs):
        calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "summary": "The title is too low.",
                                "ready_to_commit": False,
                            }
                        )
                    }
                }
            ]
        }

    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "ready_to_commit": {"type": "boolean"},
        },
        "required": ["summary", "ready_to_commit"],
        "additionalProperties": False,
    }
    vision = LiteLLMVisionModel(
        model="openai/vision-test",
        completion=completion,
    )

    review = asyncio.run(
        vision.review(
            VisualReviewRequest(
                images=(
                    VisualReviewImage(
                        RenderedArtifact(
                            b"reference",
                            "image/png",
                            {"width": 320, "height": 180},
                        ),
                        label="Image 1: reference",
                        detail="original",
                    ),
                    VisualReviewImage(
                        RenderedArtifact(
                            b"render",
                            "image/png",
                            {"width": 720, "height": 720},
                        ),
                        label="Image 2: reconstruction",
                        detail="original",
                    ),
                ),
                prompt="Compare them directly.",
                system_prompt="Return evidence only.",
                response_schema=schema,
                response_schema_name="design_comparison",
            )
        )
    )

    call = calls[0]
    assert call["messages"][0]["content"] == "Return evidence only."
    content = call["messages"][1]["content"]
    assert len([item for item in content if item["type"] == "image_url"]) == 2
    assert "320 by 180 pixels" in content[0]["text"]
    assert content[1]["image_url"]["detail"] == "original"
    assert "720 by 720 pixels" in content[2]["text"]
    assert call["response_format"]["json_schema"] == {
        "name": "design_comparison",
        "schema": schema,
        "strict": True,
    }
    assert review == {
        "summary": "The title is too low.",
        "ready_to_commit": False,
    }


def test_vision_client_rejects_structured_output_that_misses_required_evidence() -> None:
    async def completion(**kwargs):
        del kwargs
        return {"choices": [{"message": {"content": json.dumps({"summary": "Incomplete output."})}}]}

    vision = LiteLLMVisionModel(
        model="openai/vision-test",
        completion=completion,
    )
    request = VisualReviewRequest(
        images=(VisualReviewImage(RenderedArtifact(b"image", "image/png")),),
        prompt="Compare the evidence.",
        response_schema={
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "ready_to_commit": {"type": "boolean"},
            },
            "required": ["summary", "ready_to_commit"],
            "additionalProperties": False,
        },
    )

    with unittest.TestCase().assertRaisesRegex(
        ModelProtocolError,
        "requested JSON schema",
    ):
        asyncio.run(vision.review(request))


def test_myli_constructor_builds_model_clients_from_generic_settings() -> None:
    async def render(design: dict) -> RenderedArtifact:
        del design
        return RenderedArtifact(b"png", "image/png")

    spec = DesignSpec[dict](
        name="poster",
        schema={"type": "object"},
        validator=lambda value: value,
        serializer=lambda value: value,
    )
    harness = Myli(
        model="openai/main-test",
        vision_model="openai/vision-test",
        base_url="https://main.example.test/v1",
        api_key="main-secret",
        vision_base_url="https://vision.example.test/v1",
        vision_api_key="vision-secret",
        model_options={
            "temperature": 0.1,
            "cache_control_injection_points": [
                {"role": "system", "location": "message"},
            ],
        },
        vision_options={"max_tokens": 300},
        main_prompt="Custom main prompt.",
        vision_prompt="Custom vision prompt.",
        design_spec=spec,
        renderer=FunctionRenderer(render),
    )

    assert isinstance(harness.main_model, LiteLLMMainModel)
    assert isinstance(harness.vision_model, LiteLLMVisionModel)
    assert not hasattr(harness, "main_agent")
    assert not hasattr(harness, "vision_agent")
    assert harness.main_model.model == "openai/main-test"
    assert harness.main_model.base_url == "https://main.example.test/v1"
    assert harness.main_model.options == {
        "temperature": 0.1,
        "cache_control_injection_points": [
            {"location": "message", "role": "system"},
        ],
    }
    assert harness.main_model.cache_control_injection_points == ({"location": "message", "role": "system"},)
    assert harness.vision_model.model == "openai/vision-test"
    assert harness.vision_model.base_url == "https://vision.example.test/v1"
    assert harness.vision_model.options == {"max_tokens": 300}
    assert harness.main_prompt == "Custom main prompt."
    assert harness.vision_model.system_prompt == "Custom vision prompt."


def test_myli_reuses_main_model_endpoint_for_vision_by_default() -> None:
    async def render(design: dict) -> RenderedArtifact:
        del design
        return RenderedArtifact(b"png", "image/png")

    spec = DesignSpec[dict](
        name="poster",
        schema={"type": "object"},
        validator=lambda value: value,
        serializer=lambda value: value,
    )
    harness = Myli(
        model="ollama/llava",
        base_url="http://localhost:11434",
        design_spec=spec,
        renderer=FunctionRenderer(render),
    )

    assert isinstance(harness.vision_model, LiteLLMVisionModel)
    assert harness.vision_model.model == "ollama/llava"
    assert harness.vision_model.base_url == "http://localhost:11434"


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
