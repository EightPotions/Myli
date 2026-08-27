"""Custom-tool dispatch, middleware, and evidence retrieval."""

from __future__ import annotations

import asyncio
import copy
import inspect
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace

from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from .._json import canonical_json_dumps, first_json_difference, validate_json_value
from ..contracts import (
    Allow,
    Defer,
    EventHandler,
    FailureMode,
    Reject,
    TDesign,
    ToolCall,
    ToolContext,
)
from ..errors import (
    ToolExecutionError,
)
from ._helpers import (
    _concise,
    _required_decision_message,
    _resolve_json_pointer,
    _serialize_tool_result,
)
from ._types import (
    COMMIT_RENDER_TOOL_NAME,
    INSPECT_ASSET_TOOL_NAME,
    RENDER_TOOL_NAME,
    RETRIEVE_EVIDENCE_TOOL_NAME,
    _CallExecution,
    _RunState,
    _ToolBatchMetadata,
    _ToolConfig,
)


class ToolExecutionMixin:
    async def _execute_tool_batch(
        self,
        calls: Sequence[ToolCall],
        *,
        state: _RunState[TDesign],
        step: int,
        step_id: str,
        on_event: EventHandler | None,
    ) -> tuple[tuple[_CallExecution, ...], float]:
        parallel = self.parallel_tool_calls and len(calls) > 1 and all(self._parallel_safe(call) for call in calls)
        batch_id = f"{step_id}:tools"
        batch_metadata = tuple(
            _ToolBatchMetadata(
                model_step=step,
                model_step_id=step_id,
                tool_batch_id=batch_id,
                tool_batch_index=index,
                tool_batch_size=len(calls),
            )
            for index in range(len(calls))
        )
        if parallel:
            for call in calls:
                await self._tool_started(call, state, step, step_id, on_event)
            tasks = [
                asyncio.create_task(self._invoke_tool(call, state, metadata))
                for call, metadata in zip(calls, batch_metadata)
            ]
            try:
                executions = await asyncio.gather(*tasks)
            except BaseException:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise
            for execution in executions:
                self._record_tool_result(state, execution.outcome)
                state.outcomes.append(execution.outcome)
                await self._tool_finished(
                    execution.outcome,
                    step=step,
                    step_id=step_id,
                    on_event=on_event,
                )
            return tuple(executions), sum(execution.outcome.latency_seconds or 0.0 for execution in executions)

        executions: list[_CallExecution] = []
        total_latency = 0.0
        for call, metadata in zip(calls, batch_metadata):
            await self._tool_started(call, state, step, step_id, on_event)
            execution = await self._invoke_tool(call, state, metadata)
            executions.append(execution)
            self._record_tool_result(state, execution.outcome)
            state.outcomes.append(execution.outcome)
            total_latency += execution.outcome.latency_seconds or 0.0
            await self._tool_finished(
                execution.outcome,
                step=step,
                step_id=step_id,
                on_event=on_event,
            )
            if not execution.outcome.succeeded and execution.failure_mode is FailureMode.RAISE:
                break
        return tuple(executions), total_latency

    async def _invoke_tool(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        metadata: _ToolBatchMetadata,
    ) -> _CallExecution:
        started = time.perf_counter()
        failure_mode = self._failure_mode(call.name)
        context = self._tool_context(state, metadata=metadata)

        for middleware in self.middleware:
            try:
                decision = await middleware.before_call(call, context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ToolExecutionError(f"Tool middleware failed before {call.name}: {exc}") from exc
            if isinstance(decision, Reject):
                execution = _CallExecution(
                    outcome=self._outcome(
                        state,
                        call,
                        status="rejected",
                        message=_required_decision_message(decision.message),
                    ),
                    failure_mode=failure_mode,
                )
                return await self._finish_invocation(
                    call,
                    execution,
                    state,
                    started,
                    metadata,
                )
            if isinstance(decision, Defer):
                execution = _CallExecution(
                    outcome=self._outcome(
                        state,
                        call,
                        status="deferred",
                        message=_required_decision_message(decision.message),
                    ),
                    failure_mode=FailureMode.RETURN_ERROR,
                )
                return await self._finish_invocation(
                    call,
                    execution,
                    state,
                    started,
                    metadata,
                )
            if not isinstance(decision, Allow):
                raise ToolExecutionError("ToolMiddleware.before_call must return Allow, Reject, or Defer.")

        try:
            validate_json_value(call.arguments)
        except (TypeError, ValueError) as exc:
            execution = _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="rejected",
                    message=f"Tool arguments are not strict JSON: {exc}",
                ),
                failure_mode=failure_mode,
            )
        else:
            execution = await self._dispatch_tool(call, state, failure_mode, context)
        return await self._finish_invocation(call, execution, state, started, metadata)

    async def _finish_invocation(
        self,
        call: ToolCall,
        execution: _CallExecution,
        state: _RunState[TDesign],
        started: float,
        metadata: _ToolBatchMetadata,
    ) -> _CallExecution:
        outcome = replace(
            execution.outcome,
            latency_seconds=time.perf_counter() - started,
        )
        execution = replace(execution, outcome=outcome)
        context = self._tool_context(state, metadata=metadata, extra=(outcome,))
        for middleware in self.middleware:
            try:
                await middleware.after_call(call, outcome, context)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise ToolExecutionError(f"Tool middleware failed after {call.name}: {exc}") from exc
        return execution

    async def _dispatch_tool(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
        context: ToolContext,
    ) -> _CallExecution:
        if call.name == RENDER_TOOL_NAME and self.renderer is not None and self.vision_model is not None:
            return await self._render_design(call, state, failure_mode, context)
        if call.name == COMMIT_RENDER_TOOL_NAME and self.renderer is not None and self.vision_model is not None:
            return self._commit_render(call, state, failure_mode)
        if call.name == INSPECT_ASSET_TOOL_NAME and self._search_providers and self.vision_model is not None:
            return await self._inspect_asset(call, state, failure_mode, context)
        if call.name == RETRIEVE_EVIDENCE_TOOL_NAME and (
            self._compact_render_context or any(config.model_view is not None for config in self._tools.values())
        ):
            return self._retrieve_evidence(call, state, failure_mode)
        provider = self._search_providers.get(call.name)
        if provider is not None:
            return await self._search_assets(call, provider, state, failure_mode)
        config = self._tools.get(call.name)
        if config is not None:
            return await self._execute_custom_tool(
                call,
                config,
                state,
                context,
                failure_mode,
            )
        return _CallExecution(
            outcome=self._outcome(
                state,
                call,
                status="rejected",
                message=f"Unknown or unavailable tool: {call.name}.",
            ),
            failure_mode=FailureMode.RETURN_ERROR,
        )

    async def _execute_custom_tool(
        self,
        call: ToolCall,
        config: _ToolConfig,
        state: _RunState[TDesign],
        context: ToolContext,
        failure_mode: FailureMode,
    ) -> _CallExecution:
        missing = config.required_capabilities.difference(state.capabilities)
        if missing:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="rejected",
                    message=("Required capabilities are unavailable: " + ", ".join(sorted(missing))),
                ),
                failure_mode=failure_mode,
            )
        arguments = copy.deepcopy(dict(call.arguments))
        try:
            config.validator.validate(arguments)
        except JsonSchemaValidationError as exc:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="rejected",
                    message=f"Tool arguments do not match the schema: {exc.message}",
                ),
                failure_mode=failure_mode,
            )
        calls = state.custom_calls.get(call.name, 0)
        if calls >= config.max_calls_per_run:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="rejected",
                    message="The per-run tool call budget is exhausted.",
                ),
                failure_mode=failure_mode,
            )
        state.custom_calls[call.name] = calls + 1
        try:
            value = await asyncio.wait_for(
                config.tool.execute(arguments, context),
                timeout=config.timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="timed_out",
                    message=f"Tool timed out after {config.timeout_seconds:g} seconds.",
                ),
                failure_mode=failure_mode,
                cause=exc,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="failed",
                    message=f"Tool {call.name} failed.",
                ),
                failure_mode=failure_mode,
                cause=exc,
            )
        try:
            serialized, size = _serialize_tool_result(value)
        except (TypeError, ValueError) as exc:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="failed",
                    message=f"Tool result is not strict JSON: {exc}",
                ),
                failure_mode=failure_mode,
                cause=exc,
            )
        if size > config.max_result_bytes:
            return _CallExecution(
                outcome=self._outcome(
                    state,
                    call,
                    status="failed",
                    message=(f"Tool result exceeds the {config.max_result_bytes}-byte limit."),
                ),
                failure_mode=failure_mode,
            )
        outcome = self._outcome(
            state,
            call,
            status="succeeded",
            result=serialized,
            result_size_bytes=size,
        )
        if config.model_view is None:
            return _CallExecution(
                outcome=outcome,
                failure_mode=failure_mode,
            )
        try:
            model_value = config.model_view(copy.deepcopy(serialized), context)
            if inspect.isawaitable(model_value):
                close = getattr(model_value, "close", None)
                if callable(close):
                    close()
                raise TypeError("model_view must be synchronous.")
            model_value, _ = _serialize_tool_result(model_value)
            evidence_ref: str | None = None
            if first_json_difference(serialized, model_value) is not None:
                evidence_ref = f"{state.run_id}:evidence:{state.evidence_count + 1}"
                if isinstance(model_value, Mapping) and "evidence_ref" not in model_value:
                    model_value = {**model_value, "evidence_ref": evidence_ref}
                else:
                    model_value = {"result": model_value, "evidence_ref": evidence_ref}
            model_value, model_size = _serialize_tool_result(model_value)
            if model_size > config.max_result_bytes:
                raise ValueError(f"model_view result exceeds the {config.max_result_bytes}-byte limit.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return _CallExecution(
                outcome=replace(
                    outcome,
                    status="failed",
                    message=f"Tool {call.name} model_view failed: {_concise(str(exc))}",
                ),
                failure_mode=failure_mode,
                cause=exc,
            )
        if evidence_ref is not None:
            registered_reference = self._register_evidence(state, serialized)
            if registered_reference != evidence_ref:
                raise RuntimeError("Evidence references must be allocated in order.")
            outcome = replace(outcome, evidence_ref=evidence_ref)
        return _CallExecution(
            outcome=outcome,
            failure_mode=failure_mode,
            model_content=canonical_json_dumps(model_value),
        )

    def _retrieve_evidence(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
    ) -> _CallExecution:
        argument_names = set(call.arguments)
        if "evidence_ref" not in argument_names or not argument_names.issubset({"evidence_ref", "json_pointer"}):
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="retrieve_evidence requires evidence_ref and accepts optional json_pointer.",
            )
        reference = call.arguments["evidence_ref"]
        if not isinstance(reference, str) or not reference:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="retrieve_evidence evidence_ref must be non-empty text.",
            )
        if reference not in state.evidence:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The evidence_ref does not identify retained evidence from this run.",
            )
        source = state.evidence[reference]
        pointer = call.arguments.get("json_pointer", "")
        if not isinstance(pointer, str) or len(pointer) > self.limits.max_evidence_pointer_chars:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=(
                    "retrieve_evidence json_pointer must be text no longer than "
                    f"{self.limits.max_evidence_pointer_chars} characters."
                ),
            )
        if state.evidence_retrievals >= self.limits.max_evidence_retrievals:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The evidence retrieval budget is exhausted.",
            )
        state.evidence_retrievals += 1
        try:
            value = _resolve_json_pointer(
                source,
                pointer,
                max_depth=self.limits.max_pointer_depth,
            )
            result, size = _serialize_tool_result(
                {
                    "evidence_ref": reference,
                    "json_pointer": pointer,
                    "value": value,
                }
            )
        except (TypeError, ValueError) as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=f"Evidence retrieval failed: {exc}",
                cause=exc,
            )
        if size > self.limits.max_evidence_result_bytes:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=(
                    "Retrieved evidence exceeds the "
                    f"{self.limits.max_evidence_result_bytes}-byte limit; use json_pointer "
                    "to select a smaller value."
                ),
            )
        return _CallExecution(
            outcome=self._outcome(
                state,
                call,
                status="succeeded",
                result=result,
                result_size_bytes=size,
            ),
            failure_mode=failure_mode,
        )
