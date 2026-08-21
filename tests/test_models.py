"""Offline tests for Myli-owned model clients."""

from __future__ import annotations

import asyncio
import base64
import json
import unittest

from myli import (
    DesignSpec,
    FunctionRenderer,
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
)
from myli._models import _MainModelClient, _VisionModelClient


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

    client = _MainModelClient(
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

    json_client = _MainModelClient(
        model="openai/test",
        output_mode="json",
        completion=completion,
    )
    text_client = _MainModelClient(
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

    client = _MainModelClient(model="openai/test", completion=completion)

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

    client = _MainModelClient(model="provider/test", completion=completion)

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

        client = _MainModelClient(model="provider/test", completion=completion)
        with test_case.subTest(expected_type=expected_type.__name__):
            with test_case.assertRaises(expected_type) as raised:
                asyncio.run(client.complete(model_request()))
            assert raised.exception.__cause__ is source_error


def test_model_client_does_not_rewrap_stable_myli_errors() -> None:
    source_error = ProviderRateLimitError("already normalized")

    async def completion(**kwargs):
        del kwargs
        raise source_error

    client = _MainModelClient(model="provider/test", completion=completion)

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

    vision = _VisionModelClient(
        model="openai/vision-test",
        base_url="https://vision.example.test/v1",
        api_key="vision-secret",
        options={"max_tokens": 500},
        completion=completion,
    )
    review = asyncio.run(
        vision.review(
            RenderedArtifact(b"test-png", "image/png"),
            prompt="Review this poster.",
        )
    )

    call = calls[0]
    content = call["messages"][1]["content"]
    assert content[0] == {"type": "text", "text": "Review this poster."}
    assert content[1]["image_url"]["url"] == ("data:image/png;base64," + base64.b64encode(b"test-png").decode("ascii"))
    assert call["max_tokens"] == 500
    assert call["api_base"] == "https://vision.example.test/v1"
    assert call["api_key"] == "vision-secret"
    assert call["stream"] is False
    assert review == "Strong hierarchy. Increase contrast."


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
        model_options={"temperature": 0.1},
        vision_options={"max_tokens": 300},
        main_prompt="Custom main prompt.",
        vision_prompt="Custom vision prompt.",
        design_spec=spec,
        renderer=FunctionRenderer(render),
    )

    assert isinstance(harness._main_client, _MainModelClient)
    assert isinstance(harness._vision_client, _VisionModelClient)
    assert not hasattr(harness, "main_agent")
    assert not hasattr(harness, "vision_agent")
    assert harness._main_client.model == "openai/main-test"
    assert harness._main_client.base_url == "https://main.example.test/v1"
    assert harness._main_client.options == {"temperature": 0.1}
    assert harness._vision_client.model == "openai/vision-test"
    assert harness._vision_client.base_url == "https://vision.example.test/v1"
    assert harness._vision_client.options == {"max_tokens": 300}
    assert harness.main_prompt == "Custom main prompt."
    assert harness._vision_client.system_prompt == "Custom vision prompt."


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

    assert harness._vision_client.model == "ollama/llava"
    assert harness._vision_client.base_url == "http://localhost:11434"


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
