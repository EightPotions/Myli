# Architecture

Myli is an orchestration boundary, not a provider client, document model,
rendering stack, or persistence layer.

~~~text
request + trusted stored input
             |
             v
       isolated run state <------ application policies
        /       |       \
 main model   tools    asset providers
                  \       /
                   renderer -> vision model
                         |
                         v
             bounded RFC 6902 proposal
~~~

## Trust boundaries

DesignSpec separates trusted-input migration and normalization from strict
untrusted-candidate validation. The latter always performs runtime JSON Schema
validation and rejects a validator or serializer that silently changes model
output.

MainModel and VisionModel are public protocols. Applications can inject fakes,
custom transports, or different providers. LiteLLM is the default integration
and supports separate main and vision configuration.

## Evidence and policy

Every tool call produces a run-scoped ToolOutcome. CandidateContext exposes all
outcomes with explicit status and provides successful_tool_outcomes as the
authorization-safe subset. CandidatePolicy runs before proposed rendering and
before final return, allowing the host to enforce prerequisites, permissions,
provenance, licensing, tenant access, geometry, tokens, and invariants.

ToolMiddleware sits around every built-in and application tool. It can allow,
reject, or defer work and can observe the resulting outcome. This supports
approval, audit, ordering, transactions, rate limits, and tenant boundaries
without embedding an application progress protocol in Myli.

## Assets and rendering

Each named provider returns generic Asset values. Myli creates run references,
tracks provider provenance, detects conflicts, and bounds discovery. Preview
artifacts can be eager or lazy. Lazy resolution is cached only within the run,
is available only after discovery, and is guarded by timeout, size, and media
validation.

Rendering receives a strictly validated candidate. The returned artifact is
validated before the configured vision model receives it. The model's text
review is a normal tool result; raw artifact bytes are excluded from traces. A
successful render receives a run-scoped reference and can later be committed as
the run proposal without rendering again. The selection remains isolated run
state and is revalidated with final-phase policy evidence; Myli never applies or
persists it.

## Isolation and ownership

The Myli instance stores immutable configuration only. Tool counters, search
budgets, assets, outcomes, preview caches, traces, and identifiers are created
inside each run. Cancellation propagates through awaited work and is never
wrapped. The application remains solely responsible for accepting, applying,
and persisting a returned candidate.
