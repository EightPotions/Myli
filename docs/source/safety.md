# Safety model

Myli treats the main model, user text, current document, search metadata, and
rendered content as untrusted. The harness narrows what those inputs can cause;
the host application remains responsible for authorization and persistence.

## Trust boundaries

The application owns:

- injected model transports or model names, endpoint settings, and credentials;
- the design schema, parser, serializer, and semantic policies;
- rendering and asset-search implementations;
- user authorization and document persistence.

Myli owns:

- the model/tool loop and generic tool schemas;
- structural and application-defined validation before rendering and return;
- edit-capability enforcement;
- run-scoped asset references and bounded run budgets;
- in-memory result traces.

## Edit capability

`can_edit=True` grants the run the capability to return a changed document. It
does not prove that the user requested a change; the main model is instructed to
return `patch: null` for advice, critique, inspection, or questions.

With `can_edit=False`:

- the final patch must be `null`;
- diagnostic rendering remains available;
- only a patch producing the unchanged current document may be rendered.
- diagnostic renders cannot be committed as proposals.

The application should calculate `can_edit` from its authorization layer and
must still decide whether to save the returned candidate.

## Validation layers

Every candidate crosses these layers:

1. Parse the model response as strict JSON with exactly `message` and `patch`;
   reject non-finite numbers and other non-JSON values.
2. Validate and apply the RFC 6902 operations to a deep copy.
3. Strictly validate it through {py:class}`~myli.DesignSpec`.
4. Run every application candidate policy.
5. Require the validated candidate to serialize exactly to the patched document.
6. Enforce edit capability and compare a canonical serialized document.
7. Return the patch and candidate without writing either one anywhere.

`DesignSpec.schema` tells the model what document shape its patch must produce,
while `DesignSpec.validator` is the trusted runtime check on the patched result.
Validators must reject invalid values instead of silently normalizing them; a
normalizing candidate is rejected so the returned patch always reconstructs the
returned design.
Do not assume that a model provider's structured-output feature replaces runtime
validation.

Myli migrates and normalizes trusted stored input, then snapshots its canonical
serialization before a provider request. Mutating the caller's original object
during a run does not change the patch base.

## Asset provenance

Search results receive references scoped to one run. Myli exposes the approved
{py:class}`~myli.Asset` values to candidate policies, but it cannot infer
which fields in an application-specific document represent external assets.
Enforce URI provenance, licensing, tenant access, and other asset policies in a
candidate policy.

## Rendered and searched content

Visible text in a render or asset preview can contain prompt-injection content.
Myli tells the vision model to treat visible text as content rather than
instructions, then returns the review to the main model as tool output.
Provider integrations preserve tool-role boundaries when translating messages.

## Operational guidance

- Pass model credentials through Myli's connection settings or provider
  environment variables, never in document metadata.
- Avoid sensitive data in traces, provider metadata, search metadata, or events.
- Configure `trace_redactor` to remove sensitive assistant reasoning or tool
  outcomes before traces reach `on_step` or the final run result.
- Grant custom-tool `capabilities` from the host authorization layer and keep
  their call, timeout, and result-size limits narrow. Tool arguments are checked
  against each tool's JSON input schema before application code runs.
- Validate media size and decode limits in the renderer and model service.
- Set {py:class}`~myli.HarnessLimits` according to latency and cost budgets.
- Log final acceptance and persistence in the host application, separately from
  Myli's in-memory traces.
