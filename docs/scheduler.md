# Scheduler Design

Fluxion uses a two-stage scheduler:

1. **Prefill admission stage** with token budget per step.
2. **Decode stage** with token-level continuous batching.

## Goals

- avoid static-batch behavior under mixed prompt lengths
- bound starvation through aging-aware policy scoring
- maintain explicit knobs for prefill throughput vs decode fairness

## Config knobs

- `max_batch_size`
- `max_prefill_tokens_per_step`
- `decode_token_budget`
- `starvation_slo_ms`
- policy: `fcfs`, `latency_priority`, `token_budget`

## Policy behavior

- `fcfs`: baseline arrival-order scheduler.
- `latency_priority`: prioritizes older/high-priority requests using weighted aging.
- `token_budget`: shortest-remaining-decode-first with aging compensation.

## Continuous batch rebuild

Each step:

- build prefill batch from queue under token budget
- process decode batch by one token/request
- requeue unfinished decode requests with fresh priority score

This allows dynamic adaptation when arrivals are heterogeneous and bursty.
