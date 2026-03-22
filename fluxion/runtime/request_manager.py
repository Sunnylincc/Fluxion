from __future__ import annotations

import time
import uuid

from fluxion.models.types import GenerationRequest


class RequestManager:
    def __init__(self) -> None:
        self.requests: dict[str, GenerationRequest] = {}

    def create(self, prompt: str, max_new_tokens: int, priority: int = 0) -> GenerationRequest:
        req = GenerationRequest(
            request_id=str(uuid.uuid4()),
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            priority=priority,
            arrival_ts=time.time(),
        )
        self.requests[req.request_id] = req
        return req

    def get(self, request_id: str) -> GenerationRequest:
        return self.requests[request_id]
