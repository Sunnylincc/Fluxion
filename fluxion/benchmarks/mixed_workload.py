from __future__ import annotations

import argparse
import json
import random
from typing import Callable

from fluxion.backends.mock_backends import CPUBackend, MockAcceleratorBackend, MockGPUBackend
from fluxion.runtime.engine import FluxionRuntime
from fluxion.runtime.scheduler import SchedulerConfig


def _new_runtime(policy: str) -> FluxionRuntime:
    return FluxionRuntime(
        scheduler_config=SchedulerConfig(
            policy=policy,
            max_batch_size=16,
            max_prefill_tokens_per_step=4096,
            decode_token_budget=64,
            starvation_slo_ms=500,
        ),
        backends=[CPUBackend(), MockGPUBackend(), MockAcceleratorBackend()],
    )


def generate_mixed_requests(n: int, rng: random.Random) -> list[tuple[str, int, int, set[str]]]:
    out: list[tuple[str, int, int, set[str]]] = []
    for _ in range(n):
        prompt_words = rng.randint(8, 256)
        max_new_tokens = rng.randint(16, 128)
        priority = rng.randint(0, 5)
        placement = {"general"}
        if rng.random() < 0.2:
            placement.add("latency-optimized")
        out.append(("tok " * prompt_words, max_new_tokens, priority, placement))
    return out


def generate_stress_burst(n: int, rng: random.Random) -> list[tuple[str, int, int, set[str]]]:
    """Tail-heavy workload with bursts of long prompts and decode-heavy requests."""
    out: list[tuple[str, int, int, set[str]]] = []
    for i in range(n):
        if i % 5 == 0:
            prompt_words = rng.randint(512, 1500)
            max_new_tokens = rng.randint(32, 96)
            priority = 5
            placement = {"general", "latency-optimized"}
        else:
            prompt_words = rng.randint(12, 96)
            max_new_tokens = rng.randint(160, 320)
            priority = rng.randint(0, 3)
            placement = {"general", "high-throughput"} if rng.random() < 0.3 else {"general"}
        out.append(("tok " * prompt_words, max_new_tokens, priority, placement))
    return out


def run_trial(policy: str, workload_fn: Callable[[int, random.Random], list[tuple[str, int, int, set[str]]]], requests: int, seed: int) -> dict[str, float]:
    rng = random.Random(seed)
    runtime = _new_runtime(policy)
    reqs = workload_fn(requests, rng)

    # warmup path for scheduler/planner state machinery
    for _ in range(5):
        runtime.submit(prompt="warmup token " * 16, max_new_tokens=16, priority=0)
    runtime.run_until_complete()

    for prompt, max_new_tokens, priority, tags in reqs:
        runtime.submit(prompt=prompt, max_new_tokens=max_new_tokens, priority=priority, placement_tags=tags)

    runtime.run_until_complete()

    summary = runtime.metrics.summary()
    kv = runtime.kv.metrics_dict()
    summary["kv_utilization"] = float(kv["utilization"])
    summary["kv_external_fragmentation"] = float(kv["external_fragmentation"])
    summary["kv_pressure"] = float(kv["pressure"])
    utils = runtime.planner.utilization()
    summary["device_utilization_avg"] = sum(utils.values()) / max(1, len(utils))
    return summary


def aggregate(trials: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({k for t in trials for k in t})
    return {k: sum(t.get(k, 0.0) for t in trials) / len(trials) for k in keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Fluxion scheduler benchmark matrix")
    parser.add_argument("--requests", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--workload", choices=["mixed", "stress_burst"], default="mixed")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    workload_map = {
        "mixed": generate_mixed_requests,
        "stress_burst": generate_stress_burst,
    }

    results: dict[str, dict[str, float]] = {}
    for policy in ["fcfs", "latency_priority", "token_budget"]:
        trials = [
            run_trial(policy, workload_map[args.workload], args.requests, args.seed + i)
            for i in range(args.trials)
        ]
        results[policy] = aggregate(trials)

    if args.json:
        print(json.dumps(results, indent=2, sort_keys=True))
        return

    print(f"Workload={args.workload} requests={args.requests} trials={args.trials}")
    for policy, summary in results.items():
        print(f"\nPolicy={policy}")
        for key, value in sorted(summary.items()):
            print(f"  {key}: {value:.3f}")


if __name__ == "__main__":
    main()
