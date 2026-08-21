"""Small adapters for applications that prefer async functions to classes."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Generic

from .contracts import Asset, RenderedArtifact, TDesign


RendererFunction = Callable[[TDesign], Awaitable[RenderedArtifact]]
AssetSearchFunction = Callable[[str, int], Awaitable[Sequence[Asset]]]


@dataclass(frozen=True, slots=True)
class FunctionRenderer(Generic[TDesign]):
    """Adapt an async render function to the ``DesignRenderer`` protocol."""

    function: RendererFunction[TDesign]

    async def render(self, design: TDesign) -> RenderedArtifact:
        return await self.function(design)


@dataclass(frozen=True, slots=True)
class FunctionAssetSearch:
    """Adapt a named async search function to AssetSearchProvider."""

    name: str
    description: str
    function: AssetSearchFunction

    async def search(self, query: str, *, limit: int) -> Sequence[Asset]:
        return await self.function(query, limit)
