# Myli

Myli is a provider-neutral, pre-alpha Python harness for agents that propose
RFC 6902 changes to application-owned JSON design documents. It never persists
or applies the returned candidate to application state.

The core is independent of Pydantic, canvas formats, rendering stacks, model
providers, ORMs, progress transports, and persistence systems.

Read the [Myli documentation](https://myli.readthedocs.io/en/latest/) for the
complete guide and API reference.

## Run contract

~~~python
result = await myli.run(
    request=user_request,
    design=current_design,
    can_edit=True,
    capabilities={"media.transform"},
    history=history,
    on_event=handle_event,
    on_step=persist_step,
)
~~~

RunResult contains the user-facing message, optional validated candidate,
changed flag, proposed patch, approved run-scoped assets, all tool outcomes,
step traces, run ID, and final provider metadata. Successful and unsuccessful
tool outcomes are also available as filtered properties.

## Generic documents and injected models

~~~python
from typing import Any

from myli import DesignSpec, ModelRequest, ModelResponse, Myli


def load_untrusted_document(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("document must be an object")
    return value


design_spec = DesignSpec(
    name="design",
    schema={"type": "object"},
    validator=load_untrusted_document,
    serializer=lambda document: document,
    normalizer=normalize_trusted_stored_document,
    input_migrator=migrate_stored_document,
)


class ApplicationMainModel:
    async def complete(self, request: ModelRequest) -> ModelResponse:
        return await application_transport.complete(request)


myli = Myli(
    design_spec=design_spec,
    main_model=ApplicationMainModel(),
    vision_model=application_vision_model,
    renderer=application_renderer,
)
~~~

DesignSpec has deliberately separate methods for trusted stored input and
untrusted model candidates. Candidate JSON is validated at runtime against the
schema and must serialize back without coercion, default insertion, field
dropping, or any other silent rewrite. Stored input may opt into migration and
normalization before each run.

MainModel and VisionModel are public protocols. Applications may inject
different transports, endpoints, credentials, or fakes. LiteLLMMainModel and
LiteLLMVisionModel are the default implementations and are included with Myli:

~~~bash
python -m pip install myli
~~~

The LiteLLM main implementation supports structured, JSON, and text output
modes, rejects options that override client-owned request fields, and translates
provider failures into Myli's stable exception hierarchy.

Vision calls use `VisualReviewRequest`, which keeps labels and ordering explicit
for one or more images and can request output with an application-owned JSON
Schema. `LiteLLMVisionModel` handles the multi-image transport and validates
structured output.

Application attachments can be passed to `Myli.run(input_artifacts=...)` as eager
or lazy `InputArtifact` values. They are isolated to the run and available to tools
through `ToolContext.input_artifacts` and `load_input_artifact()`.

Pydantic remains optional:

~~~python
from myli.integrations.pydantic import PydanticDesignSpec
~~~

## Tools, policies, and middleware

AgentTool is domain-neutral. A tool declares a JSON input schema, capabilities,
per-run call budget, timeout, and result-size limit, then receives a ToolContext
with the current run ID, isolated evidence, zero-based model step, run-unique
step and tool-batch IDs, and its zero-based position and size within that batch:

~~~python
async def execute(arguments, context): ...

def model_view(result, context): ...  # optional, synchronous
~~~

Myli validates arguments and results, enforces capabilities and limits, and
records a ToolOutcome with one of succeeded, failed, rejected, timed_out, or
deferred. When a tool defines `model_view`, only that strict-JSON projection is
returned to the model. Candidate policies, middleware, traces, and `RunResult`
retain the complete `execute` result. Without `model_view`, the complete result
is also the model-facing result. When the projection differs from the complete
result, Myli adds a run-scoped `evidence_ref` and exposes the bounded built-in
`retrieve_evidence(evidence_ref, json_pointer?)` tool. The model can retrieve the
complete result or one RFC 6901 subtree without making all evidence visible by
default. Calls are sequential by default. Parallel
execution requires both the harness option and an explicit parallel_safe
declaration on every call in the batch. Each tool can select
FailureMode.RETURN_ERROR or FailureMode.RAISE.

ToolMiddleware can allow, reject, or defer work before execution and observe the
outcome afterward. Applications can use it for approval, audit, tenancy,
ordering, transactions, and rate limiting. The step and batch metadata lets
middleware apply a rule exactly once per model work phase, including when a
batch executes in parallel.

CandidatePolicy runs before rendering a proposal and before returning the final
candidate. CandidateContext includes capabilities, approved assets, and all
run-scoped outcomes; successful_tool_outcomes is the authorization-safe subset.

## Assets and visual review

Named asset providers return generic Asset values with application-defined
kinds and metadata. Myli assigns run-scoped references, detects conflicting
provider identities, applies search budgets and result limits, and tracks
provenance.

An asset can contain a RenderedArtifact preview or an async preview_loader.
Previews load only after discovery and explicit inspection, are cached per run,
and are bounded by timeout, byte-size, and media-type checks. Loading failures
are returned to the model and can be retried. Binary artifacts never appear in
step trace dictionaries.

When a renderer and vision model are configured, render_design validates the
candidate and application policies before rendering, validates the artifact,
and returns visual feedback as a tool result. The main model can attach a bounded
`questions` array to ask one or more open, image-grounded questions about a
rendered design or inspected asset preview. A successful render returns a
run-scoped `render_ref`. A later `render_design` call can pass that reference as
`base_render_ref` to apply its patch to the rendered candidate; Myli composes the
chain into a patch relative to the original document. `commit_render` selects a
render as the run's proposal without rendering again or persisting it. The final
response should then use `patch: null`; an equivalent patch is accepted, while a
conflicting patch enters normal validation retry.
Final policies run again with all completed tool outcomes before the committed
proposal is returned. Uncommitted renders remain previews and do not affect the
returned proposal fields. Committing requires editing to be enabled. A changed
candidate cannot be rendered while editing is disabled, but the unchanged
current design can be rendered diagnostically.

## Events, traces, and limits

Events cover the run, model, tool, and validation lifecycle. They contain run
and step IDs, status, safe messages, elapsed time, and an optional tool name,
but no arguments or results. Applications can map them to logs, metrics, SSE,
WebSockets, or ignore them.

StepTrace retains normalized model output, provider-exposed reasoning, tool
calls and outcomes, validation failures, response ID, model, finish reason,
usage, and latency. It provides to_dict() and redacted(). Async on_step and a
configurable trace redactor make no persistence assumptions.

HarnessLimits bounds model steps, retries, renders, searches, search results,
asset inspections, patches, documents, history, artifacts, operation timeouts,
and optionally the entire run. Every custom tool retains its own limits.

## Patch and concurrency safety

apply_json_patch implements add, remove, replace, move, copy, and test, including
root operations and escaped pointers, against a deep copy. JsonPatchLimits guard
operation count, patch bytes, pointer depth, value bytes, document bytes, invalid
indexes, non-JSON values, non-finite numbers, child moves, and candidate
expansion.

CancelledError is never wrapped. Cancellation propagates into model, renderer,
vision, search, preview, and custom-tool work. All counters, assets, outcomes,
and caches live in local run state, so one Myli instance can safely serve
concurrent requests.

## Development

~~~bash
uv sync
uv run ruff format --check .
uv run ruff check .
python -m unittest discover -s tests -v
~~~

Myli is distributed under the MIT License.
