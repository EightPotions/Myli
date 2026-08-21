# Integrating Myli

MainModel and VisionModel are public injectable protocols. Applications can use
custom transports and fakes, or the optional LiteLLM implementations.

## Models

```console
$ python -m pip install "myli[litellm]"
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
mutate or persist the document.

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


myli = Myli(..., tools=[BrandLookup()])
result = await myli.run(
    request="Use our primary brand color.",
    design=current_design,
    capabilities={"brand.read"},
)
```

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
and outcomes, provider metadata, token usage, model/tool latency, and validation
errors. Configure `trace_redactor` on {py:class}`~myli.Myli` to synchronously or
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
