"""Regression coverage for large, strip-heavy OME time-series cache eviction."""
from collections import OrderedDict

import numpy as np

from omeify.io.read_cache import ReadCache


class _NoEntryScan(OrderedDict):
    """Fail if an eviction performs a linear search through cached blocks."""

    def items(self):
        raise AssertionError("Eviction must not scan all native cache entries")


def test_demand_eviction_does_not_scan_all_cached_strips():
    # A 1-row-strip TIFF has thousands of small blocks, not one big page.
    # Evictions must stay O(1), independent of the number of retained blocks.
    cache = ReadCache(128 * 1024)
    strip = np.zeros(128, dtype=np.uint8)
    for i in range(1024):
        cache.load(i, lambda: strip.copy())
    cache._entries = _NoEntryScan(cache._entries)
    for i in range(1024, 1536):
        cache.load(i, lambda: strip.copy())
    info = cache.info()
    assert info["loads"] == 1536
    assert info["evictions"] == 512
    assert info["resident_bytes"] == 128 * 1024
    assert 0 not in cache and 512 in cache and 1535 in cache
    assert info["speculative_bytes"] == 0


def test_speculative_lru_hits_promotion_and_demand_priority():
    cache = ReadCache(64)
    def value():
        return np.arange(8, dtype=np.uint8)

    def unexpected():
        raise AssertionError("Cached data must not invoke a loader")
    cache.load("d0", value)
    cache.load("s1", value, speculative=True)
    cache.load("s2", value, speculative=True)  # Reaches 16 B speculative cap.
    cache.load("s1", unexpected, speculative=True)  # Touch speculative LRU.
    cache.load("s3", value, speculative=True)
    assert "s2" not in cache
    assert "s1" in cache and "s3" in cache and "d0" in cache
    assert cache.info()["prefetched_wasted_bytes"] == 8
    assert cache.info()["speculative_bytes"] == 16

    # Promotion removes only s1 from speculative eviction candidates.
    np.testing.assert_array_equal(cache.load("s1", unexpected), value())
    assert cache.info()["prefetched_hits"] == 1
    assert cache.info()["speculative_bytes"] == 8
    cache.load("s4", value, speculative=True)
    cache.load("s5", value, speculative=True)  # Evicts s3, not demanded s1.
    assert "s3" not in cache and "s1" in cache
    assert "d0" in cache and "s4" in cache and "s5" in cache

    # A sufficiently large demand evicts remaining guesses first, then the
    # oldest demanded entries, with no stale speculative keys left behind.
    cache.load("big", lambda: np.zeros(64, dtype=np.uint8))
    assert cache.info()["speculative_bytes"] == 0
    assert cache.info()["resident_bytes"] == 64
    assert "big" in cache and "s1" not in cache
    assert cache.info()["prefetched_hits"] == 1


def test_speculative_admissions_never_evict_demand_and_clear_resets_index():
    cache = ReadCache(64)
    cache.load("demand", lambda: np.zeros(64, dtype=np.uint8))
    cache._entries = _NoEntryScan(cache._entries)
    cache.load("guess", lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    assert "demand" in cache and "guess" not in cache
    assert cache.info()["evictions"] == 0
    cache.clear()
    assert cache.info()["entries"] == 0
    assert cache.info()["speculative_bytes"] == 0
    cache.load("guess", lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    assert "guess" in cache
    cache.clear()
    assert cache.info()["prefetched_wasted_bytes"] == 8
    assert cache.info()["speculative_bytes"] == 0


def test_discard_namespace_updates_speculative_eviction_index():
    cache = ReadCache(64)
    cache.load(("a", 1), lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    cache.load(("b", 2), lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    cache.load(("a", 3), lambda: np.zeros(16, dtype=np.uint8))
    cache.discard_namespace("a")
    assert ("a", 1) not in cache and ("a", 3) not in cache
    assert ("b", 2) in cache
    assert cache.info()["speculative_bytes"] == 8
    assert cache.info()["prefetched_wasted_bytes"] == 8
    cache.load(("b", 4), lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    cache.load(("b", 5), lambda: np.zeros(8, dtype=np.uint8), speculative=True)
    assert ("b", 2) not in cache
    assert cache.info()["speculative_bytes"] == 16


def test_real_one_row_strip_time_series_remains_correct_after_eviction(tmp_path):
    """Reader integration: churn cannot mislabel planes or corrupt source pixels."""
    import tifffile

    from omeify import open_dataset

    values = np.arange(12 * 32 * 48, dtype=np.uint16).reshape(12, 32, 48)
    path = tmp_path / "many-strips.ome.tif"
    tifffile.imwrite(
        path, values, ome=True, rowsperstrip=1, photometric="minisblack",
        metadata={"axes": "TYX", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5},
    )
    with open_dataset(path, input_type="ome_tiff", cache_bytes=1024) as dataset:
        for t in range(12):
            with dataset.plane(t=t) as image:
                np.testing.assert_array_equal(image.read_region(0, 32, 0, 48), values[t])
        assert dataset.cache_info()["evictions"] > 0
        with dataset.plane(t=0) as first:
            np.testing.assert_array_equal(first.read_region(7, 27, 2, 39), values[0, 7:27, 2:39])
