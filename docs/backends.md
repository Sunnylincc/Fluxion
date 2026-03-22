# Backend Abstraction

Fluxion separates control-plane scheduling from backend execution modeling.

## Core interface

`Backend` provides:

- `estimate_prefill(prompt_tokens)`
- `estimate_decode(tokens)`
- `can_host(required_blocks, placement_tags)`
- `load_score()`

## Capability model

`BackendCapabilities` captures backend-specific constraints:

- placement tags (e.g. `latency-optimized`, `high-throughput`)
- max batch token budget
- optional features (e.g. KV offload)

## Why this matters

The abstraction is intentionally designed to support:

- simulator mode (today)
- real kernel integration and backend SDK bindings (future)
- planner decisions that include memory headroom, transfer penalties, and backend-specific constraints
