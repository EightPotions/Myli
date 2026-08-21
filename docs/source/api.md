# API reference

The package root exports the provider-neutral core. Optional integrations are
available below myli.integrations.

## Harness and results

~~~{currentmodule} myli
~~~

~~~{autoclass} Myli
:members:
~~~

~~~{autoclass} HarnessLimits
:members:
~~~

~~~{autoclass} RunResult
:members:
~~~

~~~{autoclass} RunEvent
:members:
~~~

~~~{autoclass} StepTrace
:members:
~~~

~~~{autoclass} ToolOutcome
:members:
~~~

## Documents and policies

~~~{autoclass} DesignSpec
:members:
~~~

~~~{autoclass} CandidatePolicy
:members:
~~~

~~~{autoclass} CandidateContext
:members:
~~~

~~~{autoclass} DesignRenderer
:members:
~~~

~~~{autoclass} RenderedArtifact
:members:
~~~

## Models and messages

~~~{autoclass} MainModel
:members:
~~~

~~~{autoclass} VisionModel
:members:
~~~

~~~{autoclass} LiteLLMMainModel
:members:
~~~

~~~{autoclass} LiteLLMVisionModel
:members:
~~~

~~~{autoclass} ModelRequest
:members:
~~~

~~~{autoclass} ModelResponse
:members:
~~~

~~~{autoclass} Message
:members:
~~~

## Tools, middleware, and assets

~~~{autoclass} AgentTool
:members:
~~~

~~~{autoclass} ToolContext
:members:
~~~

~~~{autoclass} ToolMiddleware
:members:
~~~

~~~{autoclass} Allow
~~~

~~~{autoclass} Reject
~~~

~~~{autoclass} Defer
~~~

~~~{autoclass} FailureMode
:members:
~~~

~~~{autoclass} AssetSearchProvider
:members:
~~~

~~~{autoclass} Asset
:members:
~~~

## Patch engine and exceptions

~~~{autofunction} apply_json_patch
~~~

~~~{autoclass} JsonPatchLimits
:members:
~~~

All public failures derive from MyliError. Provider,
model-protocol, tool-execution, design-validation, patch, run-limit, and
run-timeout subclasses let applications handle failures without depending on a
provider SDK.
