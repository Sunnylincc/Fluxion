from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import time


class RequestState(str, Enum):
    QUEUED = "queued"
    PREFILL = "prefill"
    DECODE = "decode"
    COMPLETE = "complete"


@dataclass(slots=True)
class GenerationRequest:
    request_id: str
    prompt: str
    max_new_tokens: int
    priority: int = 0
    placement_tags: set[str] = field(default_factory=set)
    arrival_ts: float = field(default_factory=time.time)
    generated_tokens: int = 0
    state: RequestState = RequestState.QUEUED
    first_token_ts: float | None = None
    completed_ts: float | None = None
    assigned_device: str | None = None
    kv_blocks_reserved: int = 0

    @property
    def prompt_tokens(self) -> int:
        return max(1, len(self.prompt.split()))

    @property
    def done(self) -> bool:
        return self.generated_tokens >= self.max_new_tokens

    @property
    def remaining_decode_tokens(self) -> int:
        return max(0, self.max_new_tokens - self.generated_tokens)
