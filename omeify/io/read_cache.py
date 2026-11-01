"""Session-local, byte-bounded demand/speculative caching.

No process-global cache, pixel conversion, or background worker is introduced here.
The owner supplies immutable storage-block identities and loaders. Clearing the
cache invalidates in-flight admission, not the value promised to existing callers.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Hashable, TypeVar

import numpy as np

from .image_metadata import integer

T = TypeVar("T")


def _size(value) -> int:
    if isinstance(value, np.ndarray):
        return value.nbytes
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value)
    if isinstance(value, tuple):
        return sum(_size(item) for item in value)
    return 0


def _freeze(value) -> None:
    if isinstance(value, np.ndarray):
        value.flags.writeable = False
    elif isinstance(value, tuple):
        for item in value:
            _freeze(item)


@dataclass
class _Entry:
    value: object
    size: int
    speculative: bool


@dataclass
class _Flight:
    result: Future
    epoch: int
    speculative: bool


class ReadCache:
    """One aggregate native-block allowance shared by a source's plane views.

    Speculative entries use at most one quarter of the allowance and never evict
    demanded entries. Oversized results are returned but not retained. A cache
    budget bounds retained payloads, not return arrays or upstream codec scratch.
    """

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self.max_bytes = integer(max_bytes, "cache_bytes")
        self._entries: OrderedDict[Hashable, _Entry] = OrderedDict()
        # OrderedDict keeps the LRU for all entries. An independent index of
        # speculative keys makes "evict speculation before demand" O(1), even
        # after tens of thousands of demanded TIFF strips fill the cache.
        self._speculative: OrderedDict[Hashable, None] = OrderedDict()
        self._flights: dict[Hashable, _Flight] = {}
        self._lock = RLock()
        self._epoch = 0
        self._bytes = 0
        self._speculative_bytes = 0
        self._counts = dict(hits=0, misses=0, loads=0, waits=0, evictions=0,
                            prefetched_hits=0, prefetched_wasted_bytes=0,
                            load_errors=0, high_water_bytes=0)

    def __contains__(self, key) -> bool:
        with self._lock:
            return key in self._entries

    def _drop(self, key) -> None:
        entry = self._entries.pop(key)
        self._bytes -= entry.size
        if entry.speculative:
            del self._speculative[key]
            self._speculative_bytes -= entry.size
            self._counts["prefetched_wasted_bytes"] += entry.size
        self._counts["evictions"] += 1

    def _admit(self, key, value, *, speculative: bool) -> None:
        size = _size(value)
        if not 0 < size <= self.max_bytes:
            return
        speculative_limit = self.max_bytes // 4
        if speculative and size > speculative_limit:
            return
        while (self._bytes + size > self.max_bytes or
               (speculative and self._speculative_bytes + size > speculative_limit)):
            if self._speculative:
                victim = next(iter(self._speculative))
            elif speculative:
                # Never evict a demanded block to admit a guess.
                return
            else:
                victim = next(iter(self._entries))
            self._drop(victim)
        _freeze(value)
        self._entries[key] = _Entry(value, size, speculative)
        self._bytes += size
        if speculative:
            self._speculative[key] = None
            self._speculative_bytes += size
        self._counts["high_water_bytes"] = max(self._counts["high_water_bytes"], self._bytes)

    def load(self, key: Hashable, loader: Callable[[], T], *, speculative=False) -> T:
        """Reuse cached/in-progress work; exceptions never become missing pixels."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._counts["hits"] += 1
                self._entries.move_to_end(key)
                if entry.speculative:
                    if speculative:
                        self._speculative.move_to_end(key)
                    else:
                        # The demanded hit takes this entry out of the
                        # speculative eviction pool, preserving the LRU order.
                        entry.speculative = False
                        del self._speculative[key]
                        self._speculative_bytes -= entry.size
                        self._counts["prefetched_hits"] += 1
                return entry.value
            flight = self._flights.get(key)
            if flight is not None:
                self._counts["waits"] += 1
                if not speculative:
                    flight.speculative = False
                leader = False
            else:
                self._counts["misses"] += 1
                flight = _Flight(Future(), self._epoch, bool(speculative))
                self._flights[key] = flight
                leader = True
        if not leader:
            return flight.result.result()
        try:
            value = loader()
        except BaseException as error:
            with self._lock:
                self._counts["load_errors"] += 1
                if self._flights.get(key) is flight:
                    del self._flights[key]
                flight.result.set_exception(error)
            raise
        with self._lock:
            self._counts["loads"] += 1
            if flight.epoch == self._epoch:
                self._admit(key, value, speculative=flight.speculative)
            if self._flights.get(key) is flight:
                del self._flights[key]
            flight.result.set_result(value)
        return value

    def clear(self) -> None:
        with self._lock:
            self._epoch += 1
            self._counts["prefetched_wasted_bytes"] += self._speculative_bytes
            self._entries.clear()
            self._speculative.clear()
            self._bytes = self._speculative_bytes = 0
            # Existing waiters retain their Future. New requests after a clear must
            # not join a load from a previous cache generation.
            self._flights.clear()

    def discard_namespace(self, namespace) -> None:
        with self._lock:
            for key in tuple(self._entries):
                if isinstance(key, tuple) and key[0] == namespace:
                    self._drop(key)

    def info(self) -> dict[str, int]:
        with self._lock:
            return {**self._counts, "max_bytes": self.max_bytes,
                    "resident_bytes": self._bytes, "speculative_bytes": self._speculative_bytes,
                    "entries": len(self._entries), "in_flight": len(self._flights)}


class CacheNamespace:
    """Read-only membership view used by TIFF segment diagnostics."""

    def __init__(self, cache: ReadCache, namespace):
        self.cache, self.namespace = cache, namespace

    def __contains__(self, key):
        return (self.namespace, key) in self.cache

    def clear(self):
        self.cache.discard_namespace(self.namespace)
