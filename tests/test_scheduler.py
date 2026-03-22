from fluxion.models.types import GenerationRequest
from fluxion.runtime.scheduler import SchedulerConfig, TokenScheduler


def _req(rid: str, prompt_words: int, max_new: int, priority: int = 0) -> GenerationRequest:
    return GenerationRequest(request_id=rid, prompt=("x " * prompt_words).strip(), max_new_tokens=max_new, priority=priority)


def test_scheduler_prefill_token_budget() -> None:
    scheduler = TokenScheduler(SchedulerConfig(policy="fcfs", max_batch_size=10, max_prefill_tokens_per_step=20, decode_token_budget=4))
    scheduler.enqueue(_req("a", prompt_words=18, max_new=8))
    scheduler.enqueue(_req("b", prompt_words=10, max_new=8))

    prefill = scheduler.pop_prefill_batch()
    assert [r.request_id for r in prefill] == ["a"]


def test_scheduler_token_budget_prefers_short_remaining() -> None:
    scheduler = TokenScheduler(SchedulerConfig(policy="token_budget", max_batch_size=8, decode_token_budget=4))
    long_req = _req("long", 8, 50)
    short_req = _req("short", 8, 8)
    short_req.generated_tokens = 7

    scheduler.to_decode(long_req)
    scheduler.to_decode(short_req)
    decode = scheduler.pop_decode_batch()
    assert decode[0].request_id == "short"
