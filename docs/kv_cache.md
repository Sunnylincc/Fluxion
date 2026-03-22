# KV-Cache Subsystem

Fluxion KV-cache is block-based and request-scoped.

## Implementation

- C++ allocator (`cpp/kv_cache/block_allocator.*`) with pybind11 binding (`cpp/bindings/pybind_module.cpp`)
- Python compatibility allocator with matching semantics for environments where extension build is unavailable

## Operations

- `allocate(request_id, num_blocks)`
- `free(request_id)`
- `can_allocate(num_blocks)`

## Reported metrics

- `total_blocks`, `free_blocks`, `used_blocks`
- `utilization`
- `largest_contiguous_free_run`
- `external_fragmentation` (1 - largest_free_run / free_blocks)
- `pressure`

These metrics are intended for admission control and benchmark analysis.
