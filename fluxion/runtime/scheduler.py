from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import time
from typing import Literal

from fluxion.models.types import GenerationRequest, RequestState

PolicyName = Literal["fcfs", "latency_priority", "token_budget"]


@dataclass(slots=True)
class SchedulerConfig:
    policy: PolicyName = "fcfs"
    max_batch_size: int = 16
    max_prefill_tokens_per_step: int = 4096
    decode_token_budget: int = 64
    starvation_slo_ms: float = 500.0


class TokenScheduler:
    """Token-level scheduler with prefill/decode separation and continuous batch rebuild."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self.prefill_q: deque[GenerationRequest] = deque()
        self.decode_q: list[tuple[float, int, GenerationRequest]] = []
        self._seq = 0

    def enqueue(self, request: GenerationRequest) -> None:
        request.state = RequestState.QUEUED
        self.prefill_q.append(request)

    def pop_prefill_batch(self, now_ts: float | None = None) -> list[GenerationRequest]:
        now = now_ts or time.time()
        batch: list[GenerationRequest] = []
        tokens_budget = self.config.max_prefill_tokens_per_step

        while self.prefill_q and len(batch) < self.config.max_batch_size:
            candidate = self.prefill_q[0]
            ptoks = candidate.prompt_tokens
            if batch and ptoks > tokens_budget:
                break
            self.prefill_q.popleft()
            candidate.state = RequestState.PREFILL
            batch.append(candidate)
            tokens_budget -= ptoks
            if (now - candidate.arrival_ts) * 1000 >= self.config.starvation_slo_ms:
                # drain one starved request immediately even if budget is exhausted
                continue
            if tokens_budget <= 0:
                break
        return batch

    def to_decode(self, request: GenerationRequest, now_ts: float | None = None) -> None:
        request.state = RequestState.DECODE
        score = self._priority_score(request, now_ts=now_ts)
        heapq.heappush(self.decode_q, (score, self._next_seq(), request))

    def pop_decode_batch(self, now_ts: float | None = None) -> list[GenerationRequest]:
        if not self.decode_q:
            return []

        now = now_ts or time.time()
        batch: list[GenerationRequest] = []
        budget = self.config.decode_token_budget

        while self.decode_q and len(batch) < self.config.max_batch_size and budget > 0:
            _, _, req = heapq.heappop(self.decode_q)
            if req.done:
                continue
            batch.append(req)
            budget -= 1

            # in token_budget mode, allow additional admission for short-tail requests
            if self.config.policy == "token_budget" and req.remaining_decode_tokens <= 4 and self.decode_q and budget > 0:
                continue

            waited_ms = (now - req.arrival_ts) * 1000.0
            if waited_ms > self.config.starvation_slo_ms and self.decode_q and budget > 0:
                continue

        return batch

    def requeue_decode(self, request: GenerationRequest, now_ts: float | None = None) -> None:
        score = self._priority_score(request, now_ts=now_ts)
        heapq.heappush(self.decode_q, (score, self._next_seq(), request))

    def _priority_score(self, request: GenerationRequest, now_ts: float | None = None) -> float:
        now = now_ts or time.time()
        if self.config.policy == "fcfs":
            return request.arrival_ts
        if self.config.policy == "latency_priority":
            waited_ms = (now - request.arrival_ts) * 1000
            # smaller score is higher priority in heap; older + higher user priority first
            return -(0.7 * waited_ms + 100.0 * request.priority)
        if self.config.policy == "token_budget":
            # shortest remaining decode first with aging to avoid starvation
            waited_ms = (now - request.arrival_ts) * 1000
            return float(request.remaining_decode_tokens) - (0.002 * waited_ms)
        raise ValueError(f"unknown policy {self.config.policy}")

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq
