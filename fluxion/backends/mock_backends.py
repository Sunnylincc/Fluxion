from __future__ import annotations

from fluxion.backends.base import Backend, BackendCapabilities, DeviceProfile


class CPUBackend(Backend):
    def __init__(self, name: str = "cpu-0") -> None:
        super().__init__(
            profile=DeviceProfile(
                name=name,
                memory_capacity_blocks=2048,
                prefill_ms_per_token=0.7,
                decode_ms_per_token=0.5,
                fixed_kernel_launch_ms=0.01,
            ),
            capabilities=BackendCapabilities(backend_type="cpu", placement_tags=frozenset({"general", "cpu"})),
        )


class MockGPUBackend(Backend):
    def __init__(self, name: str = "gpu-0") -> None:
        super().__init__(
            profile=DeviceProfile(
                name=name,
                memory_capacity_blocks=8192,
                prefill_ms_per_token=0.22,
                decode_ms_per_token=0.12,
                host_to_device_ms_per_kb=0.01,
                device_to_host_ms_per_kb=0.005,
                fixed_kernel_launch_ms=0.04,
            ),
            capabilities=BackendCapabilities(
                backend_type="mock_gpu",
                supports_kv_offload=True,
                max_batch_tokens=8192,
                placement_tags=frozenset({"general", "gpu", "high-throughput"}),
            ),
        )


class MockAcceleratorBackend(Backend):
    """Mock accelerator with stricter placement constraints and transfer overhead."""

    def __init__(self, name: str = "accel-0") -> None:
        super().__init__(
            profile=DeviceProfile(
                name=name,
                memory_capacity_blocks=3072,
                prefill_ms_per_token=0.35,
                decode_ms_per_token=0.09,
                host_to_device_ms_per_kb=0.04,
                device_to_host_ms_per_kb=0.03,
                fixed_kernel_launch_ms=0.06,
            ),
            capabilities=BackendCapabilities(
                backend_type="mock_accelerator",
                supports_kv_offload=False,
                max_batch_tokens=6144,
                placement_tags=frozenset({"general", "accelerator", "latency-optimized"}),
            ),
            metadata={"compiler": "mock_xla", "runtime": "mock_rt_v1"},
        )
