# Integrating Myli

MainModel and VisionModel are public injectable protocols. LiteLLM is the
default provider, while applications can still inject custom transports and
fakes.

## Models

```console
$ python -m pip install myli
```

```python
import os

from myli import FailureMode, FunctionRenderer, Myli

myli = Myli(
    model="provider/your-main-model-api-name",
    vision_model="provider/your-vision-model-api-name",
    base_url="https://your-provider.example/v1",  # optional
    api_key=os.environ["MODEL_API_KEY"],  # or omit for provider env vars
    model_options={"temperature": 0.2, "timeout": 120},
    vision_options={"max_tokens": 1200},
    design_spec=design_spec,
    renderer=FunctionRenderer(render),
)
```

The vision model defaults to the main model and inherits its endpoint and key.
Supply `vision_model`, `vision_base_url`, or `vision_api_key` only when vision
uses a different model service.

`output_mode` supports three portability modes:

- `"structured"` (default) requests strict schema-constrained output;
- `"json"` requests a JSON object without provider-side schema enforcement;
- `"text"` relies on Myli's final parsing, validation, and retry.

Pass generation settings such as `timeout`, `temperature`, or token limits via
`model_options` and `vision_options`. Model names, connection values, messages,
tools, output format, and streaming are client-owned and cannot be overridden
through those option mappings.

### Model-context policy

Myli calls a synchronous {py:class}`~myli.ModelContextPolicy` immediately before
every `MainModel.complete` call. The default policy preserves the complete
message sequence. Applications can inject a policy to compact older exchanges,
apply a token budget, or rank context with application-specific relevance:

```python
class ApplicationContextPolicy:
    def prepare(self, messages, context):
        pinned = set(context.pinned_message_indexes)
        selected = select_relevant_indexes(messages, context.tool_outcomes)
        selected.update(pinned)
        return tuple(messages[index] for index in sorted(selected))


myli = Myli(..., model_context_policy=ApplicationContextPolicy())
```

{py:class}`~myli.ModelContext` exposes the run request, step identity,
capabilities, completed tool outcomes, and pinned system/current-run message
indexes. A policy receives an isolated copy and affects only the next provider
request; Myli retains its complete internal run history. Returned messages must
keep every pinned message and complete assistant tool-call/result group. Myli
rejects orphaned, partial, or duplicate tool exchanges before calling a
provider. This lets relevance policies retain errors and recent evidence without
creating provider-invalid request histories.

## Rendering

The renderer receives a document that already passed the `DesignSpec` validator
and every candidate policy. It must return non-empty bytes with an `image/*`
media type.

```python
from myli import FunctionRenderer, RenderedArtifact


async def render(design: dict) -> RenderedArtifact:
    png = await application_renderer.render_png(design)
    return RenderedArtifact(data=png, media_type="image/png")


renderer = FunctionRenderer(render)
```

Myli sends the rendered artifact to its vision client and returns factual visual
feedback to the main model. The vision client does not receive authority to
mutate or persist the document. Calls to `render_design` may include a `questions`
array, allowing the main model to ask one or more open questions about visible
details. `inspect_asset` supports the same optional array for asset previews.
Questions are bounded by {py:class}`~myli.HarnessLimits`, treated as untrusted
data, and must be answered only from visible evidence.

A successful `render_design` call returns a run-scoped `render_ref`. Pass that
reference back to `render_design` as `base_render_ref` to apply a revision patch
to that rendered candidate. Incremental renders can be chained, and Myli composes
their patches so the selected proposal remains relative to the original document.
Pass any successful reference to `commit_render` to select the already rendered
document as the run's proposal without invoking the renderer or vision model
again. Commit is selection, not persistence: the application still decides
whether to save {py:attr}`myli.RunResult.design`. After a commit, the model should
return `patch: null`. Myli also accepts a final patch that produces a canonically
identical document; a conflicting patch is a normal validation failure and retry.
Before returning, candidate policies run in the `final` phase with every completed
tool outcome. `RunResult.design`, `changed`, and `patch` describe the committed
render, while uncommitted renders remain previews only. Committing a render
requires `can_edit=True`.

## Asset search

Each asset search adapter becomes a separately named model tool such as
`search_assets_brand_library`. Names must match
`^[a-z][a-z0-9_]{0,31}$`.

```python
from myli import Asset, FunctionAssetSearch, RenderedArtifact


async def search_brand_library(query: str, limit: int) -> list[Asset]:
    matches = await brand_client.search(query=query, limit=limit)
    return [
        Asset(
            id=item.id,
            uri=item.application_uri,
            kind=item.kind,
            description=item.description,
            preview=RenderedArtifact(item.preview_bytes, item.preview_media_type),
            metadata={"license": item.license},
        )
        for item in matches
    ]


brand_search = FunctionAssetSearch(
    name="brand_library",
    description="Search approved brand assets, fonts, palettes, and templates.",
    function=search_brand_library,
)
```

Pass this adapter through `Myli(asset_search_providers=[brand_search], ...)`. Asset
`kind` values are application-defined and can represent photos, stickers,
icons, fonts, templates, palettes, video, or brand-library components.

Only an asset returned during the current run can be passed to `inspect_asset`.
A preview is optional and can be supplied eagerly or through `preview_loader`.
It is resolved only after discovery and explicit inspection. The preview itself
must be an image even when the underlying asset is another application kind.

## Candidate policies

`DesignSpec.validator` establishes the document's type and structural validity.
Candidate policies add application semantics such as bounds, immutable nodes,
design tokens, permissions, and searched-asset provenance.

```python
from myli import CandidateContext


def require_approved_images(
    candidate: dict,
    current: dict,
    context: CandidateContext,
) -> None:
    del current
    approved_uris = {asset.uri for asset in context.approved_assets}
    for element in candidate["elements"]:
        if element.get("type") == "image":
            if element.get("uri") not in approved_uris:
                raise ValueError("image did not come from an approved search")
```

The same policies run before a candidate render and before the final result.
Use {py:attr}`myli.CandidateContext.phase` when a rule legitimately differs
between those phases.

## Custom agent tools

Implement {py:class}`~myli.AgentTool` to expose an application operation to the
main model. Tool names must be unique, and results must be JSON-serializable.
The harness checks `required_capabilities`, `max_calls_per_run`,
`timeout_seconds`, and `max_result_bytes` for every run. It also validates each
model-produced argument object against `input_schema` before calling `execute`.

```python
class BrandLookup:
    name = "lookup_brand"
    description = "Look up one approved brand token."
    input_schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
        "required": ["name"],
        "additionalProperties": False,
    }
    required_capabilities = frozenset({"brand.read"})
    max_calls_per_run = 3
    timeout_seconds = 2.0
    max_result_bytes = 16_384

    failure_mode = FailureMode.RETURN_ERROR
    parallel_safe = False

    async def execute(self, arguments, context):
        return await brand_store.lookup(arguments["name"])

    def model_view(self, result, context):
        # Keep complete store metadata in ToolOutcome, but spend model context
        # only on the value needed for the next step.
        return {"value": result["value"]}


myli = Myli(..., tools=[BrandLookup()])
result = await myli.run(
    request="Use our primary brand color.",
    design=current_design,
    capabilities={"brand.read"},
)
```

`model_view` is optional and synchronous. Its return value must be strict JSON
and fit the tool's `max_result_bytes` limit. It affects only the tool message sent
to the model; candidate policies, middleware, traces, and `RunResult` retain the
complete validated result returned by `execute`.

If the model view differs from the complete result, its tool message also contains
a run-scoped `evidence_ref`. Myli then exposes
`retrieve_evidence(evidence_ref, json_pointer?)`: omitting `json_pointer` requests
the complete result, while an RFC 6901 pointer requests one subtree. Retrieval is
bounded by `HarnessLimits.max_evidence_retrievals`,
`max_evidence_result_bytes`, `max_evidence_pointer_chars`, and
`max_pointer_depth`. References from other runs are rejected.

Capability names are application-defined and independent of `can_edit`, which
continues to guard design patches.

## Budgets, events, traces, and history

Use {py:class}`~myli.HarnessLimits` to bound model steps, validation retries,
renders, searches, inspections, and returned search results.

```python
from myli import HarnessLimits

limits = HarnessLimits(
    max_model_steps=10,
    max_model_response_retries=1,
    max_validation_retries=1,
    max_renders=2,
    max_searches_per_provider=2,
    max_asset_inspections=4,
    max_vision_questions=6,
    max_vision_question_chars=1000,
    max_search_results=6,
    model_timeout_seconds=90,
    render_timeout_seconds=20,
    vision_timeout_seconds=60,
    search_timeout_seconds=10,
)
```

Timeouts are finite positive seconds and apply independently to each external
operation. Patch, document, artifact, history, and optional total-run limits are
available on the same object.

Pass an async `on_event` handler to {py:meth}`myli.Myli.run` for progress UI or
telemetry. Deterministic model, tool, and validation events carry run and step
IDs, phase, optional tool name, status, a safe message, and elapsed time. The
The `kind` field combines phase and status, such as
`tool.started`, `model.completed`, or `validation.failed`.

Use async `on_step` to persist each completed model/tool iteration immediately.
Step traces include assistant and provider-exposed reasoning content, tool calls
and outcomes, provider metadata, model/tool latency, validation errors, and the
{py:class}`~myli.ModelCallTrace` values associated with the step. The complete
ordered call sequence is available on {py:attr}`myli.RunResult.model_calls`, and
{py:attr}`myli.RunResult.usage` aggregates reported input, output, cached, and
reasoning tokens with request counts, retries, and latency. Model-call purposes
identify main, render-review, asset-inspection, and comparison requests.
Configure `trace_redactor` on {py:class}`~myli.Myli` to synchronously or
asynchronously replace each trace before `on_step` and the final result receive
it.

Conversation history may contain only plain-text user and assistant
{py:class}`~myli.Message` values. System messages, tool calls, and tool results
are owned by the current harness run and are rejected in supplied history.

## Application instructions

The `instructions` argument to {py:class}`~myli.Myli` can describe document
conventions or design-system guidance. It cannot override Myli's safety,
capability, or output rules. Do not place secrets in instructions because they
are sent to the main model on every step.

`main_prompt` and `vision_prompt` replace the default main-agent and visual-review
system prompts. Applications that override them are responsible for retaining
the required safety, trust-boundary, and structured-output instructions.
