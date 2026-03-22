from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True)
class BackendCapabilities:
    backend_type: str
    supports_prefill: bool = True
    supports_decode: bool = True
    supports_kv_offload: bool = False
    max_batch_tokens: int = 4096
    placement_tags: frozenset[str] = frozenset()


@dataclass(slots=True)
class DeviceProfile:
    name: str
    memory_capacity_blocks: int
    prefill_ms_per_token: float
    decode_ms_per_token: float
    host_to_device_ms_per_kb: float = 0.0
    device_to_host_ms_per_kb: float = 0.0
    fixed_kernel_launch_ms: float = 0.05


@dataclass(slots=True)
class ExecutionEstimate:
    prefill_ms: float = 0.0
    decode_ms: float = 0.0
    transfer_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.prefill_ms + self.decode_ms + self.transfer_ms


class Backend:
    """Execution backend contract used by Fluxion's planner and runtime.

    The backend only provides latency/memory estimates in v1; real kernel invocations can be
    integrated behind the same interface without changing scheduler/control-plane components.
    """

    def __init__(
        self,
        profile: DeviceProfile,
        capabilities: BackendCapabilities,
        metadata: Mapping[str, str] | None = None,
    ) -> None:
        self.profile = profile
        self.capabilities = capabilities
        self.metadata = dict(metadata or {})
        self.inflight_requests = 0

    def estimate_prefill(self, prompt_tokens: int) -> ExecutionEstimate:
        prefill_ms = self.profile.fixed_kernel_launch_ms + (prompt_tokens * self.profile.prefill_ms_per_token)
        transfer_ms = self.profile.host_to_device_ms_per_kb * (prompt_tokens * 4 / 1024)
        return ExecutionEstimate(prefill_ms=prefill_ms, transfer_ms=transfer_ms)

    def estimate_decode(self, tokens: int) -> ExecutionEstimate:
        decode_ms = self.profile.fixed_kernel_launch_ms + (tokens * self.profile.decode_ms_per_token)
        transfer_ms = self.profile.device_to_host_ms_per_kb * (tokens * 4 / 1024)
        return ExecutionEstimate(decode_ms=decode_ms, transfer_ms=transfer_ms)

    def can_host(self, required_blocks: int, placement_tags: set[str] | None = None) -> bool:
        if required_blocks > self.profile.memory_capacity_blocks:
            return False
        if placement_tags and not placement_tags.issubset(self.capabilities.placement_tags):
            return False
        return True

    def load_score(self) -> float:
        return float(self.inflight_requests)
