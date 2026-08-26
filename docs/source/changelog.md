# Changelog

Myli is pre-alpha. This page records user-visible changes and migration notes as
the public contracts evolve.

## Unreleased

- `DesignSpec.prompt_schema` can now explicitly provide a compact, model-facing
  schema while `DesignSpec.schema` remains the full authoritative runtime schema.
  Both schemas are checked independently; Myli does not attempt to prove their
  semantic equivalence.
- Every main and internal vision request now produces a provider-neutral
  `ModelCallTrace` with its purpose, model/provider identity, latency, outcome,
  retry count, and reported input, output, cached, and reasoning tokens.
  `RunResult` exposes the ordered calls and aggregate `RunUsage`; each
  `StepTrace` includes the calls associated with that model step.
- A synchronous `ModelContextPolicy` hook now runs before every main-model
  completion. `ModelContext` exposes run and step metadata, complete tool
  outcomes, and pinned message indexes. The safe default retains all messages;
  injected policies may compact or relevance-rank context while Myli enforces
  pinned prompts and provider-valid tool-call/result groups.
- Custom `AgentTool` implementations may define a synchronous `model_view` to
  send a smaller strict-JSON result to the model while retaining the complete
  `ToolOutcome` for policies, middleware, traces, and application persistence.
  Reduced projections receive a run-scoped `evidence_ref`; the bounded built-in
  `retrieve_evidence` tool can retrieve the complete result or an RFC 6901
  subtree on demand.
- `render_design` accepts `base_render_ref` for incremental revisions of a
  successful rendered candidate. Patch chains are composed so `commit_render`
  still returns a proposal relative to the original document.

## 0.2.0

- `VisualReviewRequest` and `VisualReviewImage` now carry labeled one- or
  multi-image reviews, image-detail hints, per-request system guidance, and an
  optional strict JSON output schema across the provider-neutral vision boundary.
- `LiteLLMVisionModel` transports multi-image requests and validates structured
  results against the requested JSON Schema before returning them.
- `Myli.run(input_artifacts=...)` registers eager or lazy application inputs for
  one run. Tools see descriptors in `ToolContext.input_artifacts` and load bounded,
  cached bytes with `ToolContext.load_input_artifact()`.
- Input descriptors are included in the main-model run prompt while artifact bytes,
  loaders, and credentials remain excluded.

Migration: implementations of `VisionModel.review` now accept one
`VisualReviewRequest` instead of separate `RenderedArtifact` and `prompt`
arguments.

## 0.1.4

- Successful run-scoped render artifacts are available transiently to application
  tools through `ToolContext.rendered_artifacts`, keyed by the `render_ref` returned
  by `render_design`. Artifact bytes remain excluded from outcomes and traces.
- Vision models may return either text or a structured JSON object; structured
  reviews remain objects in render and asset-inspection tool results.

## 0.1.3

- A successful `render_design` call returns a run-scoped reference that
  `commit_render` can select without rendering again. Final responses may use
  `patch: null` or an equivalent patch; committed proposals are rechecked by
  final-phase policies with all outcomes.
- The main model can ask one or more bounded, open, image-grounded questions when
  rendering a design or inspecting an asset preview.
- LiteLLM is now the default model provider and is installed with Myli; custom
  MainModel and VisionModel implementations remain injectable. The previous
  `litellm` extra remains available as a compatibility alias.

## 0.1.1

- custom tools honor per-run `tool_failure_modes` overrides consistently;
- candidate round-trip failures identify the first differing RFC 6901 JSON
  Pointer, including escaped object keys;
- ToolContext exposes model-step and tool-batch identity, position, and size for
  reliable once-per-work-phase middleware;
- validation retry prompts retain successive failures from the current work
  phase;
- public injectable MainModel and VisionModel protocols with optional LiteLLM
  implementations and separate provider settings;
- strict DesignSpec trust boundaries for stored-input migration/normalization
  and untrusted candidate validation;
- run-scoped ToolContext and ToolOutcome evidence, configurable failure modes,
  safe optional parallelism, and generic tool middleware;
- generic candidate policies with successful prerequisite-tool evidence;
- run-scoped asset provenance, conflict detection, and lazy cached previews;
- full run lifecycle events, complete serializable/redactable step traces, and
  expanded document, history, artifact, patch, and total-run limits;
- bounded RFC 6902 pointer depth, values, patch bytes, and result expansion;
- strict JSON parsing and serialization that rejects non-finite numbers;
- isolated current-design snapshots for the complete asynchronous run;
- runtime JSON Schema validation for custom-tool arguments;
- default timeouts for model, renderer, vision, and asset-search operations;
- stable provider, model-protocol, tool-execution, design-validation, and run-limit
  exception taxonomy;
- public bounded `AgentTool` protocol with run capabilities;
- configurable main and vision system prompts;
- incremental `on_step` callbacks and trace redaction;
- provider reasoning, identifiers, usage, finish details, and latency in traces;
- structured progress events with run/step IDs, phase, status, safe messages,
  and elapsed time.

## 0.1.0

Initial provider-neutral harness release:

- structured design validation and candidate policies;
- minimal RFC 6902 model output with safe, deep-copy patch application;
- render and vision-review loop;
- multiple named asset-search adapters and asset-preview inspection;
- edit-capability enforcement and run budgets;
- progress events, step traces, and function adapters;
- a generic `Myli` constructor that owns main and vision client creation from
  model API names, optional base URLs, keys, and model options.

No earlier public version requires migration.
