"""Public exception hierarchy for Myli."""


class MyliError(RuntimeError):
    """Base class for errors raised by the Myli harness."""


class ConfigurationError(MyliError):
    """Raised when incompatible harness components are configured."""


class ProviderError(MyliError):
    """Base class for failures reported while calling a model provider."""


class ProviderConnectionError(ProviderError):
    """Raised when Myli cannot connect to a configured model provider."""


class ProviderTimeoutError(ProviderConnectionError):
    """Raised when a model provider request exceeds its timeout."""


class ProviderRateLimitError(ProviderError):
    """Raised when a model provider rejects a request because of a rate limit."""


class ModelProtocolError(MyliError):
    """Raised when a model response violates the expected response contract."""


class ToolExecutionError(MyliError):
    """Raised when application-owned tool or rendering code fails."""


class DesignValidationError(MyliError):
    """Raised when a proposed design fails schema or application policy checks."""


class JsonPatchError(MyliError):
    """Raised when an RFC 6902 JSON Patch cannot be applied."""


class RunLimitExceeded(MyliError):
    """Raised when the main agent does not finish within the configured step budget."""


class RunTimeoutError(RunLimitExceeded):
    """Raised when the optional total run timeout is exhausted."""
