from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(slots=True)
class RuntimeSample:
    request_id: str
    queue_delay_ms: float
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    tokens_generated: int
    assigned_device: str


class MetricsCollector:
    def __init__(self) -> None:
        self.samples: list[RuntimeSample] = []

    def add(self, sample: RuntimeSample) -> None:
        self.samples.append(sample)

    @staticmethod
    def _percentile(vals: list[float], p: float) -> float:
        if not vals:
            return 0.0
        ordered = sorted(vals)
        if len(ordered) == 1:
            return ordered[0]
        idx = int(round((p / 100.0) * (len(ordered) - 1)))
        return ordered[max(0, min(len(ordered) - 1, idx))]

    def summary(self) -> dict[str, float]:
        if not self.samples:
            return {}

        queue = [s.queue_delay_ms for s in self.samples]
        ttfts = [s.ttft_ms for s in self.samples]
        tpots = [s.tpot_ms for s in self.samples]
        lat = [s.total_latency_ms for s in self.samples]
        toks = [s.tokens_generated for s in self.samples]
        total_latency_s = max(1e-9, sum(lat) / 1000.0)

        by_device: dict[str, int] = {}
        for s in self.samples:
            by_device[s.assigned_device] = by_device.get(s.assigned_device, 0) + 1

        summary: dict[str, float] = {
            "requests": float(len(self.samples)),
            "queue_p50_ms": self._percentile(queue, 50),
            "queue_p95_ms": self._percentile(queue, 95),
            "ttft_p50_ms": self._percentile(ttfts, 50),
            "ttft_p95_ms": self._percentile(ttfts, 95),
            "tpot_p50_ms": self._percentile(tpots, 50),
            "latency_p50_ms": self._percentile(lat, 50),
            "latency_p95_ms": self._percentile(lat, 95),
            "tokens_per_sec": sum(toks) / total_latency_s,
        }

        for dev, count in by_device.items():
            summary[f"device_share_{dev}"] = count / len(self.samples)
        return summary

    def raw(self) -> list[dict[str, float | str | int]]:
        return [asdict(s) for s in self.samples]
