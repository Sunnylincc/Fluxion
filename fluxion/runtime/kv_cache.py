from __future__ import annotations

from dataclasses import asdict, dataclass

try:
    import fluxion_cpp  # type: ignore
except ImportError:  # pragma: no cover
    fluxion_cpp = None


@dataclass(slots=True)
class KVCacheMetrics:
    total_blocks: int
    free_blocks: int
    used_blocks: int
    largest_contiguous_free_run: int
    utilization: float
    external_fragmentation: float
    pressure: float


class _PyBlockAllocator:
    def __init__(self, total_blocks: int, block_size_bytes: int) -> None:
        self.total_blocks = total_blocks
        self.block_size_bytes = block_size_bytes
        self.free = list(range(total_blocks - 1, -1, -1))
        self.used_bitmap = [False] * total_blocks
        self.allocations: dict[str, list[int]] = {}

    def can_allocate(self, n: int) -> bool:
        return len(self.free) >= n

    def allocate(self, request_id: str, n: int) -> list[int]:
        if not self.can_allocate(n):
            raise RuntimeError("insufficient KV cache blocks")
        blocks = self.allocations.setdefault(request_id, [])
        for _ in range(n):
            b = self.free.pop()
            blocks.append(b)
            self.used_bitmap[b] = True
        return blocks

    def free_req(self, request_id: str) -> None:
        blocks = self.allocations.pop(request_id, [])
        for b in blocks:
            self.used_bitmap[b] = False
            self.free.append(b)

    def _largest_run(self) -> int:
        cur = 0
        best = 0
        for used in self.used_bitmap:
            if not used:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        return best

    def stats(self):
        used = self.total_blocks - len(self.free)
        util = used / self.total_blocks if self.total_blocks else 0.0
        largest_run = self._largest_run()
        external_frag = 0.0
        if self.free:
            external_frag = 1.0 - (largest_run / len(self.free))
        return type(
            "Stats",
            (),
            dict(
                total_blocks=self.total_blocks,
                free_blocks=len(self.free),
                used_blocks=used,
                largest_contiguous_free_run=largest_run,
                utilization=util,
                external_fragmentation=external_frag,
            ),
        )


class KVCacheManager:
    """Block-based KV cache allocator with consistent metrics across Python/C++ paths."""

    def __init__(self, total_blocks: int = 1024, block_size_bytes: int = 4096) -> None:
        if fluxion_cpp is None:
            self._alloc = _PyBlockAllocator(total_blocks, block_size_bytes)
            self._cpp = False
        else:
            self._alloc = fluxion_cpp.BlockAllocator(total_blocks, block_size_bytes)
            self._cpp = True

    def allocate(self, request_id: str, num_blocks: int) -> list[int]:
        return list(self._alloc.allocate(request_id, num_blocks))

    def free(self, request_id: str) -> None:
        if self._cpp:
            self._alloc.free(request_id)
        else:
            self._alloc.free_req(request_id)

    def can_allocate(self, num_blocks: int) -> bool:
        return bool(self._alloc.can_allocate(num_blocks))

    def metrics(self) -> KVCacheMetrics:
        s = self._alloc.stats()
        pressure = 1.0 - (s.free_blocks / s.total_blocks if s.total_blocks else 1.0)
        return KVCacheMetrics(
            total_blocks=s.total_blocks,
            free_blocks=s.free_blocks,
            used_blocks=s.used_blocks,
            largest_contiguous_free_run=s.largest_contiguous_free_run,
            utilization=s.utilization,
            external_fragmentation=s.external_fragmentation,
            pressure=pressure,
        )

    def metrics_dict(self) -> dict[str, float | int]:
        return asdict(self.metrics())
