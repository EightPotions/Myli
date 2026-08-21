"""Tests for the public Myli package interface."""

import inspect
import unittest

import myli


def test_version_is_exposed() -> None:
    """The package exposes its current release version."""
    assert myli.__version__ == "0.1.0"


def test_core_harness_is_exposed() -> None:
    """Applications can import the harness from the package root."""
    assert myli.Myli.__name__ == "Myli"
    assert myli.FunctionRenderer.__name__ == "FunctionRenderer"
    assert myli.FunctionAssetSearch.__name__ == "FunctionAssetSearch"
    assert myli.AssetSearchProvider.__name__ == "AssetSearchProvider"
    assert myli.AgentTool.__name__ == "AgentTool"
    assert not hasattr(myli, "ImageAsset")
    assert not hasattr(myli, "ImageSearchTool")
    assert not hasattr(myli, "FunctionImageSearch")


def test_model_backend_is_not_part_of_the_public_api() -> None:
    parameters = inspect.signature(myli.Myli).parameters

    assert "model" in parameters
    assert "base_url" in parameters
    assert "main_agent" not in parameters
    assert "vision_agent" not in parameters
    assert not hasattr(myli, "LiteLLMMainAgent")
    assert not hasattr(myli, "LiteLLMVisionAgent")


def test_stable_exception_taxonomy_is_public() -> None:
    assert issubclass(myli.ProviderError, myli.MyliError)
    assert issubclass(myli.ProviderConnectionError, myli.ProviderError)
    assert issubclass(myli.ProviderTimeoutError, myli.ProviderConnectionError)
    assert issubclass(myli.ProviderRateLimitError, myli.ProviderError)
    assert issubclass(myli.ModelProtocolError, myli.MyliError)
    assert issubclass(myli.ToolExecutionError, myli.MyliError)
    assert issubclass(myli.DesignValidationError, myli.MyliError)
    assert issubclass(myli.RunLimitExceeded, myli.MyliError)


def load_tests(loader, standard_tests, pattern):
    """Expose the dependency-free function tests to ``unittest`` discovery."""

    del loader, pattern
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            standard_tests.addTest(unittest.FunctionTestCase(value, description=name))
    return standard_tests
