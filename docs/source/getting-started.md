# Getting started

Myli does not prescribe a design-document format or model transport. LiteLLM is
the default provider: supply model names and connection settings, or inject
custom MainModel and VisionModel implementations.

## Installation

Myli requires Python 3.10 or newer. Once a release is available on PyPI, install
it with:

```console
$ python -m pip install myli
```

For development from a checkout, install the repository itself:

```console
$ python -m pip install -e .
```

## A complete first run

This example defines a small poster document, configures its models and renderer,
and starts an editable run.

```python
import asyncio
import os
from typing import Any

from myli import (
    DesignSpec,
    FunctionRenderer,
    Myli,
    RenderedArtifact,
)


def validate_poster(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("a poster must be an object")
    if set(value) != {"background", "elements"}:
        raise ValueError("a poster requires background and elements")
    if not isinstance(value["background"], str):
        raise ValueError("background must be a string")
    if not isinstance(value["elements"], list):
        raise ValueError("elements must be a list")
    return {
        "background": value["background"],
        "elements": [dict(element) for element in value["elements"]],
    }


poster_spec = DesignSpec[dict[str, Any]](
    name="poster",
    schema={
        "type": "object",
        "properties": {
            "background": {"type": "string"},
            "elements": {"type": "array", "items": {"type": "object"}},
        },
        "required": ["background", "elements"],
        "additionalProperties": False,
    },
    validator=validate_poster,
    serializer=lambda poster: poster,
)


async def render(design: dict[str, Any]) -> RenderedArtifact:
    png = await application_renderer.render_png(design)
    return RenderedArtifact(data=png, media_type="image/png")


async def main() -> None:
    myli = Myli(
        model="provider/main-model-api-name",
        vision_model="provider/vision-model-api-name",
        base_url=os.environ.get("MODEL_BASE_URL"),
        api_key=os.environ.get("MODEL_API_KEY"),
        design_spec=poster_spec,
        renderer=FunctionRenderer(render),
    )
    result = await myli.run(
        request="Make the background warmer.",
        design={"background": "#ffffff", "elements": []},
        can_edit=True,
    )
    print(result.message)
    print(result.patch)
    print(result.design)


asyncio.run(main())
```

The model produces only the changed paths as an RFC 6902 patch. Myli applies the
patch to a deep copy and returns both the patch and the validated candidate, but
does not save either. Check {py:attr}`myli.RunResult.changed` and persist
{py:attr}`myli.RunResult.design` through your application's normal authorization
path. The patch is also available as {py:attr}`myli.RunResult.patch` for audit or
patch-native persistence.

## Application-owned pieces

Your application supplies:

- injected main and optional vision models, or LiteLLM connection settings;
- a {py:class}`~myli.DesignRenderer` that returns real image bytes;
- the {py:class}`~myli.DesignSpec` schema, serializer, and validator.

See [integrating Myli](integrations.md) for those boundaries and optional asset
search, validation policies, events, history, and budgets.
