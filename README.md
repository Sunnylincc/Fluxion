# Fluxion: Hardware-Aware Inference Runtime

Fluxion is an inference systems project focused on serving-time runtime concerns:

- token-level scheduling and continuous batching
- block-based KV-cache allocation and pressure control
- multi-device placement across heterogeneous backends
- policy benchmarking under mixed and stress workloads

Fluxion is **not** an app-layer chatbot demo. The repository centers on runtime internals and interfaces that can evolve toward real executors.

## MVP Scope (v1)

- Python control plane for API, admission, scheduling, planner, and benchmark harness
- C++17 KV-cache allocator exposed through pybind11
- prefill/decode scheduling split with policy comparison
- mock CPU/GPU/accelerator backends with placement constraints and transfer/latency modeling
- OpenAI-compatible minimal endpoints for integration testing (`/v1/completions`, `/v1/chat/completions`)

## Architecture

```mermaid
flowchart TB
    Client --> API[FastAPI API Layer]
    API --> RM[Request Manager]
    RM --> SCHED["Token Scheduler<br/>prefill queue + decode queue"]
    SCHED --> PLANNER["Multi-device Planner<br/>headroom + load + constraints"]
    PLANNER --> CPU[CPU Backend]
    PLANNER --> GPU[Mock GPU Backend]
    PLANNER --> ACC[Mock Accelerator Backend]
    SCHED --> KVC[KVCacheManager]
    KVC --> CPP[BlockAllocator (C++/pybind11)]
    SCHED --> MET[Metrics Collector]
    MET --> BENCH[Benchmark Matrix]
```

## Repository Layout

- `fluxion/runtime/` – core runtime primitives (scheduler, engine, planner, KV-cache, metrics)
- `fluxion/backends/` – backend interface + backend profiles
- `fluxion/api/` – OpenAI-compatible integration endpoints
- `fluxion/benchmarks/` – reproducible benchmark matrix + stress workload
- `cpp/kv_cache/` – C++ block allocator
- `cpp/bindings/` – pybind11 module bindings
- `docs/` – design docs for scheduler/KV-cache/backends
- `tests/` – runtime and subsystem tests

## Build

```bash
pip install -r requirements.txt
cmake -S . -B build
cmake --build build -j
```

If the extension is not built, Fluxion uses a Python-compatible fallback allocator so runtime tests can still execute.

## Run

### API server

```bash
uvicorn fluxion.api.server:app --host 0.0.0.0 --port 8000
```

### Benchmark matrix

```bash
python -m fluxion.benchmarks.mixed_workload --workload mixed --requests 80 --trials 3
python -m fluxion.benchmarks.mixed_workload --workload stress_burst --requests 80 --trials 3
```

## Reported metrics

- queue delay p50/p95
- TTFT p50/p95
- TPOT p50
- request latency p50/p95
- tokens/sec
- KV pressure/utilization/fragmentation
- per-device request share and utilization

## Documentation

- Scheduler design: `docs/scheduler.md`
- KV-cache subsystem: `docs/kv_cache.md`
- Backend abstraction: `docs/backends.md`

## Next Steps

1. add asynchronous runtime loop + streaming decode path
2. integrate real model executors (PyTorch/TensorRT/accelerator SDK)
3. add prefix cache + speculative decode controls
4. add Prometheus/OpenTelemetry exporter and trace replay benchmarks
