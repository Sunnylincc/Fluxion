from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field

from fluxion.backends.mock_backends import CPUBackend, MockAcceleratorBackend, MockGPUBackend
from fluxion.runtime.engine import FluxionRuntime
from fluxion.runtime.scheduler import SchedulerConfig

app = FastAPI(title="Fluxion Inference Runtime API", version="0.2.0")
runtime = FluxionRuntime(
    scheduler_config=SchedulerConfig(policy="latency_priority", max_batch_size=16, max_prefill_tokens_per_step=4096, decode_token_budget=64),
    backends=[CPUBackend(), MockGPUBackend(), MockAcceleratorBackend()],
)


class CompletionRequest(BaseModel):
    model: str = "fluxion-mock-1"
    prompt: str
    max_tokens: int = Field(default=32, ge=1, le=1024)
    priority: int = Field(default=0, ge=0, le=10)


class ChatCompletionRequest(BaseModel):
    model: str = "fluxion-mock-1"
    messages: list[dict[str, str]]
    max_tokens: int = Field(default=32, ge=1, le=1024)
    priority: int = Field(default=0, ge=0, le=10)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/completions")
def completions(req: CompletionRequest) -> dict:
    rid = runtime.submit(req.prompt, req.max_tokens, req.priority)
    runtime.run_until_complete()
    result = runtime.requests.get(rid)

    return {
        "id": rid,
        "object": "text_completion",
        "choices": [{"index": 0, "text": f"<generated:{result.generated_tokens}_tokens>", "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.generated_tokens,
            "total_tokens": result.prompt_tokens + result.generated_tokens,
        },
        "runtime_metrics": runtime.metrics.raw()[-1],
        "kv_metrics": runtime.kv.metrics_dict(),
    }


@app.post("/v1/chat/completions")
def chat_completions(req: ChatCompletionRequest) -> dict:
    prompt = "\n".join(m.get("content", "") for m in req.messages)
    rid = runtime.submit(prompt, req.max_tokens, req.priority)
    runtime.run_until_complete()
    result = runtime.requests.get(rid)

    return {
        "id": rid,
        "object": "chat.completion",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": f"<generated:{result.generated_tokens}_tokens>"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.generated_tokens,
            "total_tokens": result.prompt_tokens + result.generated_tokens,
        },
        "runtime_metrics": runtime.metrics.raw()[-1],
        "kv_metrics": runtime.kv.metrics_dict(),
    }
