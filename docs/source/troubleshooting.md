# Troubleshooting

## The final response is rejected

The main model must return valid JSON containing exactly `message` and `patch`.
The message must be non-empty. The patch must be `null` or a valid RFC 6902 array
that produces a document accepted by the supplied schema and runtime validator.

Myli sends a correction request when validation fails, up to
`HarnessLimits.max_validation_retries`. It then raises
{py:class}`~myli.ModelProtocolError`.
Each correction prompt repeats the validation failures from earlier attempts in
the same work phase, oldest first, so a later error does not hide an earlier
constraint that still applies.

## A tool call fails but the run continues

Model-correctable failures become structured error tool results so the main
model can recover. Inspect
{py:attr}`myli.StepTrace.tool_outcomes` and the `tool.failed` events to find the
underlying message.

Common causes include:

- invalid or unexpected tool arguments;
- exhausted render, search, or inspection budgets;
- an empty or non-image {py:class}`~myli.RenderedArtifact`;
- an asset result without an inspectable visual preview;
- an asset search adapter returning something other than `Asset` values;
- a candidate policy rejecting the proposed document.

Tools using FailureMode.RETURN_ERROR report recoverable outcomes. Tools using
FailureMode.RAISE abort with {py:class}`~myli.ToolExecutionError` or a stable
provider exception.

Malformed provider responses, including tool-call arguments that are not strict
JSON objects, are retried before any tool executes. Configure the number of
consecutive correction attempts with
`HarnessLimits.max_model_response_retries`. Exhausting that budget raises
{py:class}`~myli.ModelProtocolError`.

Model, renderer, vision, and asset-search waits also have finite defaults in
{py:class}`~myli.HarnessLimits`. Increase the corresponding timeout only after
checking the underlying integration and its own retry behavior.

## A model provider request fails

Provider SDK exceptions do not form part of Myli's public contract. Catch
{py:class}`~myli.ProviderConnectionError`,
{py:class}`~myli.ProviderTimeoutError`, or
{py:class}`~myli.ProviderRateLimitError` for those specific failure modes. Other
provider request failures raise the common {py:class}`~myli.ProviderError` base.

## Editing is disabled

Set `can_edit=True` only after the host application authorizes editing. When it
is false, a model-produced final patch is invalid, and rendering a changed
candidate returns a tool error. Advice and inspection can still return a message
with `patch: null`.

## An asset cannot be inspected

`inspect_asset` accepts the run-scoped reference returned by a configured asset
search tool, not the provider's raw asset ID or URI. The matching
{py:class}`~myli.Asset` must include a non-empty image
{py:class}`~myli.RenderedArtifact` in its `preview`, or an async
`preview_loader`.

## A run reaches its step limit

{py:class}`~myli.RunLimitExceeded` means the main model reached the step budget or
crossed an optional run-wide provider-token, tool-result, or stagnation limit.
The exception message identifies the exhausted control. Review the step traces,
tool descriptions, model connection settings, and provider tool-choice settings
before increasing it.

## Configuration fails at construction

{py:class}`~myli.ConfigurationError` commonly indicates an empty design name, a
non-mapping JSON Schema, or an invalid/duplicate asset-search name. Configuration
errors can also occur at run time if the current design cannot be serialized or
the supplied history contains anything other than plain user/assistant text.
