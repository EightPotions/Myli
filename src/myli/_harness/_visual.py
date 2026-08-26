"""Rendering, asset search, and visual inspection tools."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Mapping
from itertools import islice
from typing import Any

from .._json import validate_json_value
from ..contracts import (
    Asset,
    AssetSearchProvider,
    FailureMode,
    TDesign,
    ToolCall,
    ToolContext,
    VisualReviewImage,
    VisualReviewRequest,
)
from ..errors import (
    DesignValidationError,
)
from ._helpers import (
    _canonical_json,
    _same_asset,
)
from ._types import (
    _CallExecution,
    _RenderProposal,
    _RunState,
)


class VisualToolsMixin:
    async def _render_design(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
        context: ToolContext,
    ) -> _CallExecution:
        argument_names = set(call.arguments)
        if "patch" not in argument_names or not argument_names.issubset({"base_render_ref", "patch", "questions"}):
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="render_design requires patch and accepts optional base_render_ref and questions.",
            )
        try:
            questions = self._vision_questions(call.arguments["questions"]) if "questions" in argument_names else ()
        except (TypeError, ValueError) as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=f"Invalid vision questions: {exc}",
                cause=exc,
            )
        base_document: Mapping[str, Any] = state.current_document
        base_patch: tuple[Mapping[str, Any], ...] = ()
        if "base_render_ref" in argument_names:
            base_render_ref = call.arguments["base_render_ref"]
            if not isinstance(base_render_ref, str) or not base_render_ref:
                return self._call_error(
                    state,
                    call,
                    failure_mode,
                    status="rejected",
                    message="render_design base_render_ref must be non-empty text.",
                )
            base_proposal = state.render_proposals.get(base_render_ref)
            if base_proposal is None:
                return self._call_error(
                    state,
                    call,
                    failure_mode,
                    status="rejected",
                    message="The base_render_ref does not identify a successful render from this run.",
                )
            base_document = base_proposal.document
            base_patch = base_proposal.patch
        if state.render_calls >= self.limits.max_renders:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The render budget is exhausted.",
            )
        state.render_calls += 1
        try:
            revision_patch = call.arguments["patch"]
            patched = self._apply_patch(base_document, revision_patch)
            composed_patch = base_patch + tuple(copy.deepcopy(operation) for operation in revision_patch)
            if base_patch:
                composed_document = self._apply_patch(
                    state.current_document,
                    list(composed_patch),
                )
                if _canonical_json(composed_document) != _canonical_json(patched):
                    raise DesignValidationError(
                        "The incremental render patch could not be composed from the current document."
                    )
            candidate = self.design_spec.validate_candidate(patched)
            self._validate_candidate(candidate, state, phase="render")
            candidate_document = self.design_spec.serialize(candidate)
            self._require_exact_candidate(patched, candidate_document)
            changed = _canonical_json(candidate_document) != _canonical_json(state.current_document)
            if changed and not state.can_edit:
                raise DesignValidationError("Editing is disabled; a changed candidate cannot be rendered.")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=f"Render candidate validation failed: {exc}",
                cause=exc,
            )
        try:
            artifact = await asyncio.wait_for(
                self.renderer.render(candidate),  # type: ignore[union-attr]
                timeout=self.limits.render_timeout_seconds,
            )
            self._validate_artifact(artifact)
            review = await self._review_with_trace(
                VisualReviewRequest(
                    images=(
                        VisualReviewImage(
                            artifact=artifact,
                            label="Rendered design",
                        ),
                    ),
                    prompt=self._render_prompt(state, candidate, questions),
                    purpose="render_review",
                ),
                state=state,
                context=context,
            )
        except asyncio.TimeoutError as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="timed_out",
                message="Rendering or visual review timed out.",
                cause=exc,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="failed",
                message="Rendering or visual review failed.",
                cause=exc,
            )
        render_ref = f"render:{state.render_calls}"
        state.render_proposals[render_ref] = _RenderProposal(
            document=copy.deepcopy(candidate_document),
            patch=composed_patch,
            artifact=artifact,
        )
        return self._call_success(
            state,
            call,
            failure_mode,
            {"visual_review": review, "render_ref": render_ref},
        )

    def _commit_render(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
    ) -> _CallExecution:
        if set(call.arguments) != {"render_ref"}:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="commit_render requires exactly render_ref.",
            )
        if not state.can_edit:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="Editing is disabled; a render cannot be committed.",
            )
        render_ref = call.arguments["render_ref"]
        if not isinstance(render_ref, str) or not render_ref:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="commit_render render_ref must be non-empty text.",
            )
        proposal = state.render_proposals.get(render_ref)
        if proposal is None:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The render_ref does not identify a successful render from this run.",
            )
        state.committed_render = _RenderProposal(
            document=copy.deepcopy(proposal.document),
            patch=tuple(copy.deepcopy(operation) for operation in proposal.patch),
            artifact=proposal.artifact,
        )
        return self._call_success(
            state,
            call,
            failure_mode,
            {"render_ref": render_ref, "committed": True},
        )

    async def _search_assets(
        self,
        call: ToolCall,
        provider: AssetSearchProvider,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
    ) -> _CallExecution:
        if "query" not in call.arguments or not set(call.arguments).issubset({"query", "limit"}):
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="Asset search requires query and optional limit.",
            )
        query = call.arguments["query"]
        limit = call.arguments.get(
            "limit",
            min(6, self.limits.max_search_results),
        )
        if not isinstance(query, str) or not query.strip():
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="Asset search query must be non-empty text.",
            )
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.limits.max_search_results:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=(f"Asset search limit must be between 1 and {self.limits.max_search_results}."),
            )
        calls = state.search_calls.get(call.name, 0)
        if calls >= self.limits.max_searches_per_provider:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The search budget for this provider is exhausted.",
            )
        state.search_calls[call.name] = calls + 1
        try:
            found = await asyncio.wait_for(
                provider.search(query.strip(), limit=limit),
                timeout=self.limits.search_timeout_seconds,
            )
        except asyncio.TimeoutError as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="timed_out",
                message="Asset search timed out.",
                cause=exc,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="failed",
                message="Asset search failed.",
                cause=exc,
            )

        try:
            pending: dict[str, Asset] = {}
            payload: list[dict[str, Any]] = []
            for asset in islice(found, limit):
                self._validate_asset(asset)
                reference = f"{state.run_id}:{provider.name}:{asset.id}"
                scoped = asset.for_run(
                    run_id=state.run_id,
                    source=provider.name,
                    reference=reference,
                )
                existing = pending.get(reference) or state.assets.get(reference)
                if existing is not None:
                    if not _same_asset(existing, scoped):
                        raise ValueError(f"Conflicting asset identity from provider {provider.name}: {asset.id}.")
                    continue
                pending[reference] = scoped
                payload.append(
                    {
                        "asset_ref": reference,
                        "id": scoped.id,
                        "uri": scoped.uri,
                        "kind": scoped.kind,
                        "description": scoped.description,
                        "metadata": dict(scoped.metadata),
                        "provenance": dict(scoped.provenance),
                        "preview_available": (scoped.preview is not None or scoped.preview_loader is not None),
                    }
                )
            state.assets.update(pending)
            result = {"source": provider.name, "assets": payload}
            validate_json_value(result)
        except Exception as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="failed",
                message=f"Asset search returned invalid or conflicting assets: {exc}",
                cause=exc,
            )
        return self._call_success(state, call, failure_mode, result)

    async def _inspect_asset(
        self,
        call: ToolCall,
        state: _RunState[TDesign],
        failure_mode: FailureMode,
        context: ToolContext,
    ) -> _CallExecution:
        argument_names = set(call.arguments)
        if "asset_ref" not in argument_names or not argument_names.issubset({"asset_ref", "questions"}):
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="inspect_asset requires asset_ref and accepts optional questions.",
            )
        try:
            questions = self._vision_questions(call.arguments["questions"]) if "questions" in argument_names else ()
        except (TypeError, ValueError) as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message=f"Invalid vision questions: {exc}",
                cause=exc,
            )
        reference = call.arguments["asset_ref"]
        if not isinstance(reference, str) or reference not in state.assets:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The asset reference must come from this run.",
            )
        if state.inspection_calls >= self.limits.max_asset_inspections:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="rejected",
                message="The asset inspection budget is exhausted.",
            )
        state.inspection_calls += 1
        asset = state.assets[reference]
        artifact = state.preview_cache.get(reference)
        if artifact is None:
            if asset.preview is not None:
                artifact = asset.preview
            elif asset.preview_loader is None:
                return self._call_error(
                    state,
                    call,
                    failure_mode,
                    status="rejected",
                    message="This asset has no inspectable preview.",
                )
            else:
                try:
                    artifact = await asyncio.wait_for(
                        asset.preview_loader(),
                        timeout=self.limits.asset_load_timeout_seconds,
                    )
                except asyncio.TimeoutError as exc:
                    return self._call_error(
                        state,
                        call,
                        failure_mode,
                        status="timed_out",
                        message="Asset preview loading timed out.",
                        cause=exc,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    return self._call_error(
                        state,
                        call,
                        failure_mode,
                        status="failed",
                        message="Asset preview loading failed.",
                        cause=exc,
                    )
            try:
                self._validate_artifact(artifact)
            except Exception as exc:
                return self._call_error(
                    state,
                    call,
                    failure_mode,
                    status="failed",
                    message=f"The loaded asset preview is invalid: {exc}",
                    cause=exc,
                )
            state.preview_cache[reference] = artifact
        try:
            review = await self._review_with_trace(
                VisualReviewRequest(
                    images=(
                        VisualReviewImage(
                            artifact=artifact,
                            label="Asset preview",
                        ),
                    ),
                    prompt=self._asset_prompt(asset, questions),
                    purpose="asset_inspection",
                ),
                state=state,
                context=context,
            )
        except asyncio.TimeoutError as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="timed_out",
                message="Asset visual review timed out.",
                cause=exc,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return self._call_error(
                state,
                call,
                failure_mode,
                status="failed",
                message="Asset visual review failed.",
                cause=exc,
            )
        return self._call_success(
            state,
            call,
            failure_mode,
            {"asset_ref": reference, "visual_review": review},
        )
