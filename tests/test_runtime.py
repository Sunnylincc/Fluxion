from fluxion.backends.mock_backends import CPUBackend, MockAcceleratorBackend, MockGPUBackend
from fluxion.runtime.engine import FluxionRuntime
from fluxion.runtime.scheduler import SchedulerConfig


def test_runtime_end_to_end() -> None:
    runtime = FluxionRuntime(
        scheduler_config=SchedulerConfig(policy="token_budget", max_batch_size=8, max_prefill_tokens_per_step=1024, decode_token_budget=16),
        backends=[CPUBackend(), MockGPUBackend(), MockAcceleratorBackend()],
        kv_total_blocks=512,
    )
    runtime.submit(prompt="hello", max_new_tokens=8)
    runtime.run_until_complete()

    summary = runtime.metrics.summary()
    assert summary["requests"] >= 1
    assert summary["ttft_p50_ms"] >= 0


def test_runtime_respects_placement_constraints() -> None:
    runtime = FluxionRuntime(
        scheduler_config=SchedulerConfig(policy="latency_priority", max_batch_size=8, max_prefill_tokens_per_step=1024, decode_token_budget=16),
        backends=[CPUBackend(), MockGPUBackend(), MockAcceleratorBackend()],
        kv_total_blocks=512,
    )

    rid = runtime.submit(prompt="token " * 20, max_new_tokens=16, placement_tags={"general", "latency-optimized"})
    runtime.run_until_complete()
    req = runtime.requests.get(rid)
    assert req.assigned_device == "accel-0"
