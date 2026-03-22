from __future__ import annotations

from dataclasses import dataclass

from fluxion.backends.base import Backend


@dataclass(slots=True)
class PlacementDecision:
    device: str
    reason: str
    score: float


class MultiDevicePlanner:
    """Planner with placement tags and anti-overcommit headroom control."""

    def __init__(self, backends: list[Backend], max_memory_utilization: float = 0.92) -> None:
        self.backends = {b.profile.name: b for b in backends}
        self.memory_used: dict[str, int] = {b.profile.name: 0 for b in backends}
        self.max_memory_utilization = max_memory_utilization

    def assign(self, request_id: str, required_blocks: int, placement_tags: set[str] | None = None) -> PlacementDecision:
        candidates: list[tuple[float, Backend]] = []
        for backend in self.backends.values():
            if not backend.can_host(required_blocks, placement_tags):
                continue

            cap = backend.profile.memory_capacity_blocks
            used = self.memory_used[backend.profile.name]
            projected = (used + required_blocks) / cap
            if projected > self.max_memory_utilization:
                continue

            memory_pressure = projected
            load_pressure = backend.load_score() / 8.0
            transfer_penalty = backend.profile.host_to_device_ms_per_kb
            score = memory_pressure * 0.55 + load_pressure * 0.35 + transfer_penalty * 0.1
            candidates.append((score, backend))

        if not candidates:
            raise RuntimeError(f"no device can host request {request_id} with required_blocks={required_blocks}")

        score, picked = min(candidates, key=lambda x: x[0])
        self.memory_used[picked.profile.name] += required_blocks
        picked.inflight_requests += 1
        return PlacementDecision(
            device=picked.profile.name,
            reason="min(score=memory+load+transfer) with headroom guard",
            score=score,
        )

    def release(self, device_name: str, used_blocks: int) -> None:
        self.memory_used[device_name] = max(0, self.memory_used[device_name] - used_blocks)
        self.backends[device_name].inflight_requests = max(0, self.backends[device_name].inflight_requests - 1)

    def utilization(self) -> dict[str, float]:
        return {
            name: self.memory_used[name] / backend.profile.memory_capacity_blocks
            for name, backend in self.backends.items()
        }
