# Changelog

Myli is pre-alpha. This page records user-visible changes and migration notes as
the public contracts evolve.

## Unreleased

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
