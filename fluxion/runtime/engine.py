from __future__ import annotations

import time

from fluxion.backends.base import Backend
from fluxion.models.types import GenerationRequest
from fluxion.runtime.kv_cache import KVCacheManager
from fluxion.runtime.metrics import MetricsCollector, RuntimeSample
from fluxion.runtime.planner import MultiDevicePlanner
from fluxion.runtime.request_manager import RequestManager
from fluxion.runtime.scheduler import SchedulerConfig, TokenScheduler


class FluxionRuntime:
    def __init__(self, scheduler_config: SchedulerConfig, backends: list[Backend], kv_total_blocks: int = 4096) -> None:
        self.requests = RequestManager()
        self.scheduler = TokenScheduler(scheduler_config)
        self.kv = KVCacheManager(total_blocks=kv_total_blocks)
        self.backends = {b.profile.name: b for b in backends}
        self.planner = MultiDevicePlanner(backends)
        self.metrics = MetricsCollector()
        self.sim_time_ms = 0.0

    @staticmethod
    def _estimate_blocks(req: GenerationRequest) -> int:
        # Prefill footprint + rolling decode reservation.
        return max(1, req.prompt_tokens // 8 + req.max_new_tokens // 12)

    def submit(self, prompt: str, max_new_tokens: int, priority: int = 0, placement_tags: set[str] | None = None) -> str:
        req = self.requests.create(prompt, max_new_tokens, priority)
        req.placement_tags = placement_tags or {"general"}

        blocks_needed = self._estimate_blocks(req)
        if not self.kv.can_allocate(blocks_needed):
            raise RuntimeError("KV cache pressure too high; cannot admit request")

        decision = self.planner.assign(req.request_id, blocks_needed, req.placement_tags)
        req.assigned_device = decision.device
        req.kv_blocks_reserved = blocks_needed
        self.kv.allocate(req.request_id, blocks_needed)
        self.scheduler.enqueue(req)
        return req.request_id

    def step(self) -> None:
        step_cost_ms = 0.0
        now_ts = time.time()

        for req in self.scheduler.pop_prefill_batch(now_ts=now_ts):
            backend = self.backends[req.assigned_device or ""]
            est = backend.estimate_prefill(req.prompt_tokens)
            step_cost_ms = max(step_cost_ms, est.total_ms)
            req.first_token_ts = req.arrival_ts + (self.sim_time_ms + est.total_ms) / 1000.0
            self.scheduler.to_decode(req, now_ts=now_ts)

        decode_batch = self.scheduler.pop_decode_batch(now_ts=now_ts)
        if decode_batch:
            decode_step_max = 0.0
            for req in decode_batch:
                backend = self.backends[req.assigned_device or ""]
                est = backend.estimate_decode(tokens=1)
                decode_step_max = max(decode_step_max, est.total_ms)
                req.generated_tokens += 1
                if req.done:
                    req.completed_ts = req.arrival_ts + (self.sim_time_ms + step_cost_ms + est.total_ms) / 1000.0
                    self._finalize(req)
                else:
                    self.scheduler.requeue_decode(req, now_ts=now_ts)
            step_cost_ms += decode_step_max

        self.sim_time_ms += step_cost_ms

    def run_until_complete(self, max_steps: int = 100000) -> None:
        steps = 0
        while self.scheduler.prefill_q or self.scheduler.decode_q:
            self.step()
            steps += 1
            if steps > max_steps:
                raise RuntimeError("runtime exceeded max_steps; possible scheduler livelock")

    def _finalize(self, req: GenerationRequest) -> None:
        assert req.first_token_ts is not None and req.completed_ts is not None and req.assigned_device is not None
        queue_delay_ms = max(0.0, (req.first_token_ts - req.arrival_ts) * 1000.0)
        ttft_ms = queue_delay_ms
        decode_steps = max(1, req.generated_tokens - 1)
        tpot_ms = ((req.completed_ts - req.first_token_ts) * 1000.0) / decode_steps
        total_latency_ms = (req.completed_ts - req.arrival_ts) * 1000.0
        self.metrics.add(
            RuntimeSample(
                request_id=req.request_id,
                queue_delay_ms=queue_delay_ms,
                ttft_ms=ttft_ms,
                tpot_ms=tpot_ms,
                total_latency_ms=total_latency_ms,
                tokens_generated=req.generated_tokens,
                assigned_device=req.assigned_device,
            )
        )
        self.kv.free(req.request_id)
        self.planner.release(req.assigned_device, req.kv_blocks_reserved)
