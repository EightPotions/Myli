# Myli

Myli is a provider-neutral Python harness for agents that edit structured visual
designs. It coordinates injectable main and vision models with application
tools, policies, rendering, and optional asset discovery.

Myli validates every proposed document before rendering or returning it. It also
enforces edit capabilities, asset provenance policies, and per-run budgets.

```{warning}
Myli is pre-alpha software. Expect breaking API changes before version 1.0.
```

## Start here

- Follow the [getting started guide](getting-started.md) for a complete runnable
  example.
- Read [integrating Myli](integrations.md) when configuring models, renderers,
  or asset search.
- Review the [safety model](safety.md) before accepting agent-produced designs.
- Use the [API reference](api.md) for public classes and protocols.

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

getting-started
integrations
safety
architecture
troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Reference
:hidden:

api
changelog
contributing
```
