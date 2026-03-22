from fluxion.runtime.kv_cache import KVCacheManager


def test_kv_cache_allocate_free() -> None:
    kv = KVCacheManager(total_blocks=10, block_size_bytes=1024)
    blocks = kv.allocate("r1", 4)
    assert len(blocks) == 4
    assert kv.metrics().used_blocks == 4

    kv.free("r1")
    m = kv.metrics()
    assert m.used_blocks == 0
    assert m.free_blocks == 10


def test_kv_cache_fragmentation_signal_changes() -> None:
    kv = KVCacheManager(total_blocks=12, block_size_bytes=1024)
    kv.allocate("r1", 4)
    kv.allocate("r2", 4)
    kv.free("r1")

    m = kv.metrics()
    assert m.largest_contiguous_free_run >= 4
    assert 0.0 <= m.external_fragmentation <= 1.0
