"""Events, tracing, and model-context preparation."""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Mapping, Sequence
from typing import Any

from .._json import validate_json_value
from .._models import (
    _capture_model_metadata,
)
from ..contracts import (
    EventHandler,
    Message,
    ModelCallPurpose,
    ModelCallTrace,
    ModelContext,
    RunEvent,
    StepHandler,
    StepTrace,
    TDesign,
    ToolCall,
    ToolContext,
    ToolOutcome,
    VisualReviewRequest,
)
from ..errors import (
    ConfigurationError,
    ModelProtocolError,
    RunLimitExceeded,
)
from ._helpers import (
    _canonical_json,
    _nonnegative_token_count,
    _optional_metadata_text,
    _usage_token_count,
    _validated_vision_review,
)
from ._types import (
    _RunState,
)


class RuntimeMixin:
    async def _tool_started(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        step: int,
        step_id: str,
        on_event: EventHandler | None,
    ) -> None:
        await self._emit(
            on_event,
            RunEvent(
                kind="tool.started",
                run_id=state.run_id,
                step_id=step_id,
                step=step,
                tool_name=call.name,
                message="Running a tool.",
            ),
        )

    async def _tool_finished(
        self,
        outcome: ToolOutcome,
        *,
        step: int,
        step_id: str,
        on_event: EventHandler | None,
    ) -> None:
        if outcome.status == "deferred":
            kind = "tool.deferred"
            message = "The tool was deferred."
        elif outcome.succeeded:
            kind = "tool.completed"
            message = "The tool completed."
        else:
            kind = "tool.failed"
            message = "The tool did not complete."
        await self._emit(
            on_event,
            RunEvent(
                kind=kind,
                run_id=outcome.run_id,
                step_id=step_id,
                step=step,
                tool_name=outcome.tool_name,
                message=message,
                elapsed_seconds=outcome.latency_seconds or 0.0,
            ),
        )

    async def _record_step(
        self,
        traces: list[StepTrace],
        trace: StepTrace,
        handler: StepHandler | None,
    ) -> None:
        if self.trace_redactor is not None:
            redacted = self.trace_redactor(trace)
            if inspect.isawaitable(redacted):
                redacted = await redacted
            if not isinstance(redacted, StepTrace):
                raise TypeError("trace_redactor must return StepTrace.")
            trace = redacted
        traces.append(trace)
        if handler is not None:
            await handler(trace)

    async def _review_with_trace(
        self,
        request: VisualReviewRequest,
        *,
        state: _RunState[TDesign],
        context: ToolContext,
    ) -> str | dict[str, Any]:
        """Run one internal vision request and retain provider-neutral accounting."""

        index = self._reserve_model_call(state)
        started = time.perf_counter()
        captured_metadata: dict[str, Any] = {}
        retry_count = state.vision_retry_pending.get(request.purpose, 0)
        try:
            with _capture_model_metadata() as captured_metadata:
                review = await asyncio.wait_for(
                    self.vision_model.review(request),  # type: ignore[union-attr]
                    timeout=self.limits.vision_timeout_seconds,
                )
            metadata = getattr(review, "metadata", None)
            if isinstance(metadata, Mapping):
                captured_metadata.update(metadata)
            validated = _validated_vision_review(review)
        except BaseException:
            self._record_model_call(
                state,
                index=index,
                step=context.model_step,
                step_id=context.model_step_id or "",
                purpose=request.purpose,
                client=self.vision_model,
                metadata=captured_metadata,
                latency_seconds=time.perf_counter() - started,
                success=False,
                retry_count=retry_count,
            )
            state.vision_retry_pending[request.purpose] = 1
            raise
        self._record_model_call(
            state,
            index=index,
            step=context.model_step,
            step_id=context.model_step_id or "",
            purpose=request.purpose,
            client=self.vision_model,
            metadata=captured_metadata,
            latency_seconds=time.perf_counter() - started,
            success=True,
            retry_count=retry_count,
        )
        state.vision_retry_pending[request.purpose] = 0
        return validated

    def _reserve_model_call(self, state: _RunState[Any]) -> int:
        limit = self.limits.max_total_provider_tokens
        if limit is not None and state.total_provider_tokens >= limit:
            raise RunLimitExceeded(f"The total provider token budget of {limit} tokens is exhausted.")
        index = state.next_model_call_index
        state.next_model_call_index += 1
        return index

    def _record_model_call(
        self,
        state: _RunState[Any],
        *,
        index: int,
        step: int | None,
        step_id: str,
        purpose: ModelCallPurpose,
        client: Any,
        metadata: Mapping[str, Any],
        latency_seconds: float,
        success: bool,
        retry_count: int,
    ) -> None:
        usage = metadata.get("usage")
        usage = usage if isinstance(usage, Mapping) else {}
        model = _optional_metadata_text(metadata.get("model")) or _optional_metadata_text(
            getattr(client, "model", None)
        )
        provider = _optional_metadata_text(metadata.get("provider")) or _optional_metadata_text(
            getattr(client, "provider", None)
        )
        if provider is None and model is not None and "/" in model:
            provider = model.partition("/")[0] or None
        reported_retries = _nonnegative_token_count(metadata.get("retry_count")) or 0
        cache_usage = metadata.get("cache_usage")
        cache_usage = cache_usage if isinstance(cache_usage, Mapping) else {}
        reported_cache_hit = metadata.get("cache_hit")
        if not isinstance(reported_cache_hit, bool):
            reported_cache_hit = cache_usage.get("cache_hit")
        if not isinstance(reported_cache_hit, bool):
            reported_cache_hit = None
        cached_tokens = _usage_token_count(
            usage,
            "cached_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
            "prompt_cache_hit_tokens",
            nested=(
                ("input_tokens_details", "cached_tokens"),
                ("prompt_tokens_details", "cached_tokens"),
            ),
        )
        if cached_tokens is None:
            cached_tokens = _nonnegative_token_count(cache_usage.get("read_input_tokens"))
        cache_write_tokens = _usage_token_count(
            usage,
            "cache_creation_input_tokens",
            "cache_write_tokens",
            "cache_creation_input_token_count",
            nested=(
                ("input_tokens_details", "cache_write_tokens"),
                ("prompt_tokens_details", "cache_write_tokens"),
            ),
        )
        if cache_write_tokens is None:
            cache_write_tokens = _nonnegative_token_count(cache_usage.get("write_input_tokens"))
        trace = ModelCallTrace(
            index=index,
            run_id=state.run_id,
            step_id=step_id,
            step=step,
            purpose=purpose,
            model=model,
            provider=provider,
            latency_seconds=latency_seconds,
            success=success,
            retry_count=retry_count + reported_retries,
            input_tokens=_usage_token_count(
                usage,
                "input_tokens",
                "prompt_tokens",
            ),
            output_tokens=_usage_token_count(
                usage,
                "output_tokens",
                "completion_tokens",
            ),
            cached_tokens=cached_tokens,
            cache_write_tokens=cache_write_tokens,
            cache_hit=reported_cache_hit,
            reasoning_tokens=_usage_token_count(
                usage,
                "reasoning_tokens",
                "thoughts_token_count",
                nested=(
                    ("output_tokens_details", "reasoning_tokens"),
                    ("completion_tokens_details", "reasoning_tokens"),
                ),
            ),
        )
        state.model_calls.append(trace)
        reported_total = trace.total_tokens
        if reported_total is None:
            reported_total = _usage_token_count(usage, "total_tokens")
        state.total_provider_tokens += reported_total or 0
        limit = self.limits.max_total_provider_tokens
        if limit is not None and state.total_provider_tokens > limit:
            raise RunLimitExceeded(f"The run exceeded the total provider token budget of {limit} tokens.")

    def _record_tool_calls(
        self,
        state: _RunState[Any],
        calls: Sequence[ToolCall],
    ) -> None:
        limit = self.limits.max_repeated_tool_calls
        if limit is None:
            return
        for call in calls:
            try:
                arguments = _canonical_json(dict(call.arguments))
            except (TypeError, ValueError):
                # Invalid JSON arguments are rejected by ordinary tool dispatch.
                continue
            signature = (call.name, arguments)
            count = state.tool_call_counts.get(signature, 0) + 1
            state.tool_call_counts[signature] = count
            if count > limit:
                raise RunLimitExceeded(
                    f"The repeated tool call limit of {limit} identical calls was exceeded for {call.name}."
                )

    def _record_tool_result(self, state: _RunState[Any], outcome: ToolOutcome) -> None:
        limit = self.limits.max_total_tool_result_bytes
        if limit is None:
            return
        if not outcome.succeeded or outcome.result_size_bytes is None:
            return
        total = state.total_tool_result_bytes + outcome.result_size_bytes
        if total > limit:
            raise RunLimitExceeded(f"The run exceeded the total tool result budget of {limit} bytes.")
        state.total_tool_result_bytes = total

    def _record_candidate(
        self,
        state: _RunState[Any],
        document: Mapping[str, Any],
        *,
        phase: str,
    ) -> None:
        limit = self.limits.max_identical_candidates
        if limit is None:
            return
        signature = (phase, _canonical_json(document))
        count = state.candidate_counts.get(signature, 0) + 1
        state.candidate_counts[signature] = count
        if count > limit:
            raise RunLimitExceeded(f"The identical candidate limit of {limit} was exceeded during {phase}.")

    def _record_render_progress(
        self,
        state: _RunState[Any],
        *,
        base_document: Mapping[str, Any],
        candidate_document: Mapping[str, Any],
    ) -> None:
        limit = self.limits.max_consecutive_noop_renders
        if limit is None:
            return
        if _canonical_json(candidate_document) == _canonical_json(base_document):
            state.consecutive_noop_renders += 1
        else:
            state.consecutive_noop_renders = 0
        if state.consecutive_noop_renders > limit:
            raise RunLimitExceeded(f"The consecutive no-op render limit of {limit} was exceeded.")

    @staticmethod
    def _ordered_model_calls(state: _RunState[Any]) -> tuple[ModelCallTrace, ...]:
        return tuple(sorted(state.model_calls, key=lambda call: call.index if call.index is not None else -1))

    @classmethod
    def _model_calls_for_step(
        cls,
        state: _RunState[Any],
        step_id: str,
    ) -> tuple[ModelCallTrace, ...]:
        return tuple(call for call in cls._ordered_model_calls(state) if call.step_id == step_id)

    def _validate_components(self) -> None:
        if not callable(getattr(self.main_model, "complete", None)):
            raise ConfigurationError("main_model must implement async complete().")
        if not callable(getattr(self.model_context_policy, "prepare", None)):
            raise ConfigurationError("model_context_policy must implement prepare().")
        if self.vision_model is not None and not callable(getattr(self.vision_model, "review", None)):
            raise ConfigurationError("vision_model must implement async review().")
        if self.renderer is not None and not callable(getattr(self.renderer, "render", None)):
            raise ConfigurationError("renderer must implement async render().")
        for middleware in self.middleware:
            if not callable(getattr(middleware, "before_call", None)) or not callable(
                getattr(middleware, "after_call", None)
            ):
                raise ConfigurationError("Tool middleware must implement before_call() and after_call().")
        for policy in self.candidate_policies:
            if not callable(policy) and not callable(getattr(policy, "validate", None)):
                raise ConfigurationError("Candidate policies must be callable or implement validate().")

    def _prepare_model_messages(
        self,
        messages: Sequence[Message],
        *,
        state: _RunState[TDesign],
        step: int,
        step_id: str,
        run_prompt_index: int,
    ) -> tuple[Message, ...]:
        pinned_indexes = (0, run_prompt_index)
        context = ModelContext(
            run_id=state.run_id,
            request=state.request,
            step=step,
            step_id=step_id,
            can_edit=state.can_edit,
            capabilities=state.capabilities,
            tool_outcomes=copy.deepcopy(tuple(state.outcomes)),
            pinned_message_indexes=pinned_indexes,
        )
        source = copy.deepcopy(tuple(messages))
        try:
            prepared = self.model_context_policy.prepare(source, context)
        except Exception as exc:
            raise ConfigurationError("model_context_policy.prepare() failed.") from exc
        if inspect.isawaitable(prepared):
            if inspect.iscoroutine(prepared):
                prepared.close()
            raise ConfigurationError("model_context_policy.prepare() must be synchronous.")
        if isinstance(prepared, (str, bytes)) or not isinstance(prepared, Sequence):
            raise ConfigurationError("model_context_policy.prepare() must return a sequence of Message values.")

        prepared_messages = tuple(prepared)
        self._validate_prepared_messages(
            prepared_messages,
            source=source,
            pinned_indexes=pinned_indexes,
        )
        if self._compact_render_context and prepared_messages != source:
            state.compacted_render_context = True
        return copy.deepcopy(prepared_messages)

    @classmethod
    def _validate_prepared_messages(
        cls,
        messages: Sequence[Message],
        *,
        source: Sequence[Message],
        pinned_indexes: Sequence[int],
    ) -> None:
        if any(not isinstance(message, Message) for message in messages):
            raise ConfigurationError("model_context_policy.prepare() must return only Message values.")
        if not messages:
            raise ConfigurationError("model_context_policy.prepare() cannot return an empty message sequence.")
        if messages[0] != source[pinned_indexes[0]]:
            raise ConfigurationError("model_context_policy.prepare() must retain the system prompt first.")
        for index in pinned_indexes[1:]:
            if source[index] not in messages:
                raise ConfigurationError("model_context_policy.prepare() removed a pinned message.")

        pending_call_ids: set[str] = set()
        seen_call_ids: set[str] = set()
        for message in messages:
            if message.content is not None and not isinstance(message.content, str):
                raise ConfigurationError("Prepared message content must be text or None.")
            if message.role not in {"system", "user", "assistant", "tool"}:
                raise ConfigurationError("Prepared messages contain an unsupported role.")
            if not isinstance(message.tool_calls, tuple):
                raise ConfigurationError("Prepared message tool_calls must be a tuple.")

            if pending_call_ids and message.role != "tool":
                raise ConfigurationError("model_context_policy.prepare() separated a tool call from its result.")
            if message.role == "tool":
                if message.tool_calls:
                    raise ConfigurationError("Prepared tool messages cannot request tool calls.")
                call_id = message.tool_call_id
                if not isinstance(call_id, str) or call_id not in pending_call_ids:
                    raise ConfigurationError(
                        "model_context_policy.prepare() returned an orphan or duplicate tool result."
                    )
                pending_call_ids.remove(call_id)
                continue

            if message.tool_call_id is not None:
                raise ConfigurationError("Only prepared tool messages may define tool_call_id.")
            if message.tool_calls and message.role != "assistant":
                raise ConfigurationError("Only prepared assistant messages may request tool calls.")
            if not message.tool_calls:
                continue

            try:
                cls._validate_tool_calls(message.tool_calls)
                for call in message.tool_calls:
                    validate_json_value(dict(call.arguments))
            except (ModelProtocolError, TypeError, ValueError) as exc:
                raise ConfigurationError("Prepared messages contain invalid tool calls.") from exc
            current_call_ids = {call.id for call in message.tool_calls}
            if seen_call_ids.intersection(current_call_ids):
                raise ConfigurationError("Prepared tool call IDs must be unique across the message sequence.")
            seen_call_ids.update(current_call_ids)
            pending_call_ids = current_call_ids

        if pending_call_ids:
            raise ConfigurationError("model_context_policy.prepare() removed one or more required tool results.")
