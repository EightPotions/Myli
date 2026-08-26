"""Provider- and document-neutral public contracts for Myli."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Generic, Literal, Protocol, TypeVar, runtime_checkable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from jsonschema.exceptions import ValidationError as JsonSchemaValidationError

from ._json import first_json_difference, validate_json_value


JsonObject = dict[str, Any]
TDesign = TypeVar("TDesign")
TDesignContra = TypeVar("TDesignContra", contravariant=True)
PreviewLoader = Callable[[], Awaitable["RenderedArtifact"]]
InputArtifactLoader = Callable[[], Awaitable["RenderedArtifact"]]


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A provider-neutral tool call requested by a model."""

    id: str
    name: str
    arguments: Mapping[str, Any]

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "arguments": {} if redact else _safe_trace_value(self.arguments),
        }


@dataclass(frozen=True, slots=True)
class Message:
    """A provider-neutral conversation message."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    tool_call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """JSON Schema definition of a tool visible to the main model."""

    name: str
    description: str
    input_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """Complete provider-neutral input for one main-model step."""

    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    output_schema: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Provider-neutral normalized model response."""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    reasoning_content: str | None = None

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        return {
            "content": self.content,
            "reasoning_content": None if redact else self.reasoning_content,
            "tool_calls": [call.to_dict(redact=redact) for call in self.tool_calls],
            "metadata": _safe_trace_value(self.metadata),
        }


@runtime_checkable
class MainModel(Protocol):
    """Injectable main-model client."""

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one model step and return a normalized response."""


@dataclass(frozen=True, slots=True)
class RenderedArtifact:
    """In-memory artifact suitable for a configured vision model."""

    data: bytes = field(repr=False)
    media_type: str = "image/png"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)


@dataclass(frozen=True, slots=True)
class VisualReviewImage:
    """One labeled image in a visual-review request."""

    artifact: RenderedArtifact
    label: str = "Image"
    detail: Literal["auto", "low", "high", "original"] = "auto"

    def __post_init__(self) -> None:
        if not isinstance(self.artifact, RenderedArtifact):
            raise TypeError("VisualReviewImage.artifact must use RenderedArtifact.")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("VisualReviewImage.label cannot be empty.")
        if self.detail not in {"auto", "low", "high", "original"}:
            raise ValueError("VisualReviewImage.detail is invalid.")
        object.__setattr__(self, "label", self.label.strip())


@dataclass(frozen=True, slots=True)
class VisualReviewRequest:
    """Provider-neutral request for reviewing one or more labeled images."""

    images: tuple[VisualReviewImage, ...]
    prompt: str
    system_prompt: str | None = None
    response_schema: Mapping[str, Any] | None = field(default=None, compare=False)
    response_schema_name: str = "visual_review"

    def __post_init__(self) -> None:
        images = tuple(self.images)
        if not images:
            raise ValueError("VisualReviewRequest.images cannot be empty.")
        if any(not isinstance(image, VisualReviewImage) for image in images):
            raise TypeError("VisualReviewRequest.images must contain VisualReviewImage values.")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("VisualReviewRequest.prompt cannot be empty.")
        if self.system_prompt is not None and (
            not isinstance(self.system_prompt, str) or not self.system_prompt.strip()
        ):
            raise ValueError("VisualReviewRequest.system_prompt cannot be empty.")
        if not isinstance(self.response_schema_name, str) or not self.response_schema_name.strip():
            raise ValueError("VisualReviewRequest.response_schema_name cannot be empty.")

        schema = self.response_schema
        if schema is not None:
            if not isinstance(schema, Mapping):
                raise TypeError("VisualReviewRequest.response_schema must be a mapping.")
            schema = copy.deepcopy(dict(schema))
            try:
                Draft202012Validator.check_schema(schema)
            except SchemaError as exc:
                raise ValueError(f"VisualReviewRequest.response_schema is invalid: {exc.message}") from exc

        object.__setattr__(self, "images", images)
        object.__setattr__(self, "prompt", self.prompt.strip())
        if self.system_prompt is not None:
            object.__setattr__(self, "system_prompt", self.system_prompt.strip())
        object.__setattr__(
            self,
            "response_schema_name",
            self.response_schema_name.strip(),
        )
        object.__setattr__(self, "response_schema", schema)


@runtime_checkable
class VisionModel(Protocol):
    """Injectable visual-review client."""

    async def review(
        self,
        request: VisualReviewRequest,
    ) -> str | Mapping[str, Any]:
        """Return textual or structured JSON feedback about a validated artifact."""


@runtime_checkable
class DesignRenderer(Protocol[TDesignContra]):
    """Application-owned rendering boundary."""

    async def render(self, design: TDesignContra) -> RenderedArtifact:
        """Render a validated design without mutating it."""


@dataclass(frozen=True, slots=True)
class Asset:
    """A generic asset discovered during a run.

    The reference, source, and run ID are assigned by Myli to copies of provider
    results. The application-provided ID and URI are preserved.
    """

    id: str
    uri: str
    kind: str = "asset"
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    preview: RenderedArtifact | None = field(default=None, repr=False, compare=False)
    preview_loader: PreviewLoader | None = field(default=None, repr=False, compare=False)
    provenance: Mapping[str, Any] = field(default_factory=dict, compare=False)
    reference: str | None = None
    source: str | None = None
    run_id: str | None = None

    def for_run(self, *, run_id: str, source: str, reference: str) -> Asset:
        """Return a run-scoped copy without changing the provider's object."""

        preview = self.preview
        if preview is not None:
            preview = replace(
                preview,
                data=bytes(preview.data),
                metadata=copy.deepcopy(dict(preview.metadata)),
            )
        return replace(
            self,
            metadata=copy.deepcopy(dict(self.metadata)),
            preview=preview,
            provenance=copy.deepcopy(dict(self.provenance)),
            run_id=run_id,
            source=source,
            reference=reference,
        )


@dataclass(frozen=True, slots=True)
class InputArtifact:
    """An application-provided artifact made available only for one run.

    Exactly one of ``artifact`` or ``loader`` supplies the bytes. Myli assigns the
    run ID to an isolated copy and exposes it through ``ToolContext`` under ``id``.
    """

    id: str
    kind: str = "input"
    description: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)
    artifact: RenderedArtifact | None = field(default=None, repr=False, compare=False)
    loader: InputArtifactLoader | None = field(default=None, repr=False, compare=False)
    run_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("InputArtifact.id cannot be empty.")
        if not isinstance(self.kind, str) or not self.kind.strip():
            raise ValueError("InputArtifact.kind cannot be empty.")
        if self.description is not None and not isinstance(self.description, str):
            raise TypeError("InputArtifact.description must be text or None.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("InputArtifact.metadata must be a mapping.")
        metadata = copy.deepcopy(dict(self.metadata))
        validate_json_value(metadata)
        if (self.artifact is None) == (self.loader is None):
            raise ValueError("InputArtifact requires exactly one of artifact or loader.")
        if self.artifact is not None and not isinstance(self.artifact, RenderedArtifact):
            raise TypeError("InputArtifact.artifact must use RenderedArtifact.")
        if self.loader is not None and not callable(self.loader):
            raise TypeError("InputArtifact.loader must be callable.")

        object.__setattr__(self, "id", self.id.strip())
        object.__setattr__(self, "kind", self.kind.strip())
        object.__setattr__(self, "metadata", metadata)

    def for_run(self, *, run_id: str) -> InputArtifact:
        """Return an isolated copy associated with one Myli run."""

        artifact = self.artifact
        if artifact is not None:
            artifact = replace(
                artifact,
                data=bytes(artifact.data),
                metadata=copy.deepcopy(dict(artifact.metadata)),
            )
        return replace(
            self,
            metadata=copy.deepcopy(dict(self.metadata)),
            artifact=artifact,
            run_id=run_id,
        )


@runtime_checkable
class AssetSearchProvider(Protocol):
    """A named application-owned asset discovery provider."""

    name: str
    description: str

    async def search(self, query: str, *, limit: int) -> Sequence[Asset]:
        """Return at most the requested limit of generic assets."""


DesignValidator = Callable[[Any], TDesign]
DesignSerializer = Callable[[TDesign], Mapping[str, Any]]
InputMigrator = Callable[[Any], Any]


@dataclass(frozen=True, slots=True)
class DesignSpec(Generic[TDesign]):
    """Schema and codecs for an application-owned JSON-object document.

    Stored input is trusted and may be migrated or normalized. Model candidates
    are untrusted: their raw JSON is schema checked, passed through the validator,
    and required to serialize back canonically without changing any value.
    """

    name: str
    schema: Mapping[str, Any]
    validator: DesignValidator[TDesign]
    serializer: DesignSerializer[TDesign]
    normalizer: DesignValidator[TDesign] | None = None
    input_migrator: InputMigrator | None = None
    _schema_validator: Draft202012Validator = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("DesignSpec.name cannot be empty.")
        if not isinstance(self.schema, Mapping):
            raise TypeError("DesignSpec.schema must be a mapping.")
        if not callable(self.validator) or not callable(self.serializer):
            raise TypeError("DesignSpec validator and serializer must be callable.")
        if self.normalizer is not None and not callable(self.normalizer):
            raise TypeError("DesignSpec.normalizer must be callable.")
        if self.input_migrator is not None and not callable(self.input_migrator):
            raise TypeError("DesignSpec.input_migrator must be callable.")
        try:
            schema = copy.deepcopy(dict(self.schema))
            Draft202012Validator.check_schema(schema)
        except SchemaError as exc:
            raise ValueError(f"DesignSpec.schema is invalid: {exc.message}") from exc
        object.__setattr__(self, "schema", schema)
        object.__setattr__(self, "_schema_validator", Draft202012Validator(schema))

    def normalize_input(self, value: Any) -> TDesign:
        """Migrate and normalize a trusted stored input before a run."""

        migrated = self.input_migrator(copy.deepcopy(value)) if self.input_migrator else copy.deepcopy(value)
        loader = self.normalizer or self.validator
        design = loader(migrated)
        document = self.serialize(design)
        self._validate_schema(document)
        return design

    def validate_candidate(self, value: Any) -> TDesign:
        """Strictly validate an untrusted model-produced JSON-object candidate."""

        if not isinstance(value, Mapping):
            raise TypeError("A design document must be a JSON object.")
        raw = copy.deepcopy(dict(value))
        validate_json_value(raw)
        self._validate_schema(raw)
        design = self.validator(copy.deepcopy(raw))
        serialized = self.serialize(design)
        difference = first_json_difference(raw, serialized)
        if difference is not None:
            raise ValueError(
                "The design validator or serializer altered untrusted model output; "
                "candidates must round-trip without normalization. The first difference "
                f"is at JSON Pointer {difference!r}."
            )
        return design

    def serialize(self, design: TDesign) -> JsonObject:
        """Return an isolated strict-JSON object in canonicalizable form."""

        value = self.serializer(design)
        if not isinstance(value, Mapping):
            raise TypeError("DesignSpec.serializer must return a mapping.")
        document = copy.deepcopy(dict(value))
        validate_json_value(document)
        return document

    def canonical_serialize(self, design: TDesign) -> str:
        """Serialize with stable object-key order and no non-JSON values."""

        return _canonical_json(self.serialize(design))

    def _validate_schema(self, document: Mapping[str, Any]) -> None:
        try:
            self._schema_validator.validate(document)
        except JsonSchemaValidationError as exc:
            raise ValueError(f"Design does not match its JSON Schema: {exc.message}") from exc


class FailureMode(str, Enum):
    """How a tool failure affects the containing run."""

    RETURN_ERROR = "return_error"
    RAISE = "raise"


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Run-scoped record of what actually happened for one tool call."""

    run_id: str
    call_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    status: Literal["succeeded", "failed", "rejected", "timed_out", "deferred"]
    result: Any | None = None
    message: str | None = None
    latency_seconds: float | None = None
    result_size_bytes: int | None = None
    evidence_ref: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == "succeeded"

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments": {} if redact else _safe_trace_value(self.arguments),
            "status": self.status,
            "result": None if redact else _safe_trace_value(self.result),
            "evidence_ref": self.evidence_ref,
            "message": self.message,
            "latency_seconds": self.latency_seconds,
            "result_size_bytes": self.result_size_bytes,
        }


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Immutable per-call view of isolated run state."""

    run_id: str
    phase: str
    can_edit: bool
    capabilities: frozenset[str]
    input_artifacts: Mapping[str, InputArtifact]
    approved_assets: tuple[Asset, ...]
    tool_outcomes: tuple[ToolOutcome, ...]
    rendered_artifacts: Mapping[str, RenderedArtifact]
    request: str
    _input_artifact_resolver: Callable[[str], Awaitable[RenderedArtifact]] = field(
        repr=False,
        compare=False,
    )
    model_step: int | None = None
    model_step_id: str | None = None
    tool_batch_id: str | None = None
    tool_batch_index: int | None = None
    tool_batch_size: int | None = None

    @property
    def successful_tool_outcomes(self) -> tuple[ToolOutcome, ...]:
        return tuple(outcome for outcome in self.tool_outcomes if outcome.succeeded)

    @property
    def is_first_in_tool_batch(self) -> bool:
        """Whether this is the first requested call in its model tool batch."""

        return self.tool_batch_index == 0

    async def load_input_artifact(self, artifact_id: str) -> RenderedArtifact:
        """Load and validate one registered run input, with run-scoped caching."""

        if artifact_id not in self.input_artifacts:
            raise KeyError(f"Unknown input artifact: {artifact_id}")
        return await self._input_artifact_resolver(artifact_id)

    @property
    def step(self) -> int | None:
        """Zero-based model step, as a concise alias for ``model_step``."""

        return self.model_step

    @property
    def step_id(self) -> str | None:
        """Run-unique model step ID, as an alias for ``model_step_id``."""

        return self.model_step_id

    @property
    def batch_id(self) -> str | None:
        """Run-unique tool batch ID, as an alias for ``tool_batch_id``."""

        return self.tool_batch_id

    @property
    def batch_index(self) -> int | None:
        """Zero-based call position, as an alias for ``tool_batch_index``."""

        return self.tool_batch_index

    @property
    def batch_size(self) -> int | None:
        """Requested call count, as an alias for ``tool_batch_size``."""

        return self.tool_batch_size


@runtime_checkable
class AgentTool(Protocol):
    """Domain-neutral executable tool contract.

    Implementations may also define a synchronous ``model_view(result, context)``
    method. Myli uses its strict-JSON return value only in the tool message sent
    to the model; complete results remain in :class:`ToolOutcome` evidence.
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    required_capabilities: frozenset[str]
    max_calls_per_run: int
    timeout_seconds: float
    max_result_bytes: int
    failure_mode: FailureMode
    parallel_safe: bool

    async def execute(self, arguments: Mapping[str, Any], context: ToolContext) -> Any:
        """Execute one validated call in its run context."""

    def model_view(self, result: Any, context: ToolContext) -> Any:
        """Optionally project a complete result into model-visible strict JSON.

        Myli detects this method dynamically, so tool implementations may omit it.
        """

        return result


@dataclass(frozen=True, slots=True)
class Allow:
    """Middleware decision permitting a call."""


@dataclass(frozen=True, slots=True)
class Reject:
    """Middleware decision rejecting a call without executing it."""

    message: str


@dataclass(frozen=True, slots=True)
class Defer:
    """Middleware decision deferring a call without executing it."""

    message: str


ToolDecision = Allow | Reject | Defer


@runtime_checkable
class ToolMiddleware(Protocol):
    """Application-defined policy around every built-in and custom tool call."""

    async def before_call(self, call: ToolCall, context: ToolContext) -> ToolDecision:
        """Allow, reject, or defer the call."""

    async def after_call(self, call: ToolCall, outcome: ToolOutcome, context: ToolContext) -> None:
        """Observe a completed, rejected, failed, or deferred call."""


@dataclass(frozen=True, slots=True)
class CandidateContext:
    """Run evidence available to application-owned candidate policies."""

    run_id: str
    phase: Literal["render", "final"]
    can_edit: bool
    capabilities: frozenset[str]
    approved_assets: tuple[Asset, ...]
    tool_outcomes: tuple[ToolOutcome, ...]

    @property
    def successful_tool_outcomes(self) -> tuple[ToolOutcome, ...]:
        """Only these outcomes may be treated as authorization evidence."""

        return tuple(outcome for outcome in self.tool_outcomes if outcome.succeeded)


@runtime_checkable
class CandidatePolicy(Protocol[TDesignContra]):
    """Application-owned semantic validation for candidate designs."""

    def validate(
        self,
        candidate: TDesignContra,
        current: TDesignContra,
        context: CandidateContext,
    ) -> None:
        """Raise an ordinary validation exception when the candidate is invalid."""


CandidateValidator = Callable[[TDesign, TDesign, CandidateContext], None]


@dataclass(frozen=True, slots=True)
class StepTrace:
    """Complete provider-neutral trace for one model step."""

    index: int
    run_id: str
    step_id: str
    response: ModelResponse
    tool_outcomes: tuple[ToolOutcome, ...] = ()
    validation_failures: tuple[str, ...] = ()
    model_latency_seconds: float | None = None
    tool_latency_seconds: float | None = None

    @property
    def model_response(self) -> ModelResponse:
        return self.response

    @property
    def assistant_content(self) -> str | None:
        return self.response.content

    @property
    def reasoning_content(self) -> str | None:
        return self.response.reasoning_content

    @property
    def tool_calls(self) -> tuple[ToolCall, ...]:
        return self.response.tool_calls

    @property
    def provider_response_id(self) -> str | None:
        value = self.response.metadata.get("provider_response_id")
        return value if isinstance(value, str) else None

    @property
    def model_name(self) -> str | None:
        value = self.response.metadata.get("model")
        return value if isinstance(value, str) else None

    @property
    def finish_reason(self) -> str | None:
        value = self.response.metadata.get("finish_reason")
        return value if isinstance(value, str) else None

    @property
    def usage(self) -> Mapping[str, Any] | None:
        value = self.response.metadata.get("usage")
        return value if isinstance(value, Mapping) else None

    @property
    def token_usage(self) -> Mapping[str, Any] | None:
        return self.usage

    def redacted(self) -> StepTrace:
        """Return a trace with reasoning, tool arguments, and results removed."""

        return replace(
            self,
            response=replace(
                self.response,
                reasoning_content=None,
                tool_calls=tuple(replace(call, arguments={}) for call in self.response.tool_calls),
            ),
            tool_outcomes=tuple(replace(outcome, arguments={}, result=None) for outcome in self.tool_outcomes),
        )

    def to_dict(self, *, redact: bool = False) -> dict[str, Any]:
        trace = self.redacted() if redact else self
        return {
            "index": trace.index,
            "run_id": trace.run_id,
            "step_id": trace.step_id,
            "model_response": trace.response.to_dict(redact=False),
            "assistant_content": trace.assistant_content,
            "reasoning_content": trace.reasoning_content,
            "tool_calls": [call.to_dict() for call in trace.tool_calls],
            "tool_outcomes": [outcome.to_dict() for outcome in trace.tool_outcomes],
            "validation_failures": list(trace.validation_failures),
            "provider_response_id": trace.provider_response_id,
            "model_name": trace.model_name,
            "finish_reason": trace.finish_reason,
            "usage": _safe_trace_value(trace.usage),
            "model_latency_seconds": trace.model_latency_seconds,
            "tool_latency_seconds": trace.tool_latency_seconds,
        }


StepHandler = Callable[[StepTrace], Awaitable[None]]
TraceRedactor = Callable[[StepTrace], StepTrace | Awaitable[StepTrace]]

EVENT_KINDS = frozenset(
    {
        "run.started",
        "model.started",
        "model.completed",
        "model.failed",
        "tool.started",
        "tool.completed",
        "tool.failed",
        "tool.deferred",
        "validation.started",
        "validation.failed",
        "validation.completed",
        "run.completed",
        "run.failed",
        "run.cancelled",
    }
)


@dataclass(frozen=True, slots=True)
class RunEvent:
    """A deterministic event containing no tool arguments or results."""

    kind: str
    run_id: str
    step_id: str = ""
    step: int | None = None
    tool_name: str | None = None
    message: str = ""
    elapsed_seconds: float = 0.0
    phase: str = field(init=False)
    status: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"Unsupported event kind: {self.kind}.")
        phase, status = self.kind.split(".", 1)
        object.__setattr__(self, "phase", phase)
        object.__setattr__(self, "status", status)

    @property
    def safe_message(self) -> str:
        return self.message


EventHandler = Callable[[RunEvent], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class RunResult(Generic[TDesign]):
    """Validated proposal and evidence returned without persistence or mutation."""

    message: str
    design: TDesign | None
    changed: bool
    patch: tuple[Mapping[str, Any], ...] | None
    approved_assets: tuple[Asset, ...]
    tool_outcomes: tuple[ToolOutcome, ...]
    traces: tuple[StepTrace, ...]
    run_id: str
    model_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def candidate(self) -> TDesign | None:
        return self.design

    @property
    def successful_tool_outcomes(self) -> tuple[ToolOutcome, ...]:
        return tuple(outcome for outcome in self.tool_outcomes if outcome.succeeded)

    @property
    def failed_tool_outcomes(self) -> tuple[ToolOutcome, ...]:
        return tuple(outcome for outcome in self.tool_outcomes if not outcome.succeeded)

    @property
    def provider_metadata(self) -> Mapping[str, Any]:
        return self.model_metadata


def _canonical_json(value: Mapping[str, Any]) -> str:
    validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
        allow_nan=False,
    )


def _safe_trace_value(value: Any) -> Any:
    """Convert trace values without ever embedding raw binary artifacts."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"redacted_binary_bytes": len(value)}
    if isinstance(value, RenderedArtifact):
        return {
            "media_type": value.media_type,
            "size_bytes": len(value.data),
            "metadata": _safe_trace_value(value.metadata),
        }
    if isinstance(value, Mapping):
        return {str(key): _safe_trace_value(item) for key, item in value.items()}
    if isinstance(value, Collection) and not isinstance(value, (str, bytes, bytearray)):
        return [_safe_trace_value(item) for item in value]
    return {"unsupported_type": type(value).__name__}
