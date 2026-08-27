# Contributing

## Set up the project

Create the project environment, then install the documentation dependencies:

```console
$ uv sync
$ uv pip install -r docs/requirements.txt
```

## Format and lint

Run Ruff before submitting a change:

```console
$ uv run ruff format --check .
$ uv run ruff check .
```

To apply formatting and safe lint fixes, run `uv run ruff format .` followed by
`uv run ruff check --fix .`.

## Run the tests

The package test suite uses offline model doubles and does not call external
services:

```console
$ uv run pytest
```

## Build the documentation

Treat warnings as errors so local behavior matches Read the Docs:

```console
$ python -m sphinx -W --keep-going -b dirhtml docs/source docs/_build/dirhtml
```

Open `docs/_build/dirhtml/index.html` in a browser to inspect the Furo site.

When changing a public class, protocol, exception, or behavior, update its
docstring, the [API reference](api.md), relevant user guides, and the
[changelog](changelog.md) in the same change.
