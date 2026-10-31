"""Bounded lazy plane-reader handles for streamed stack construction."""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

from .image_metadata import integer


class ReaderSequence(Sequence):
    """Keep at most a few open decoder/view handles, not one per stack plane."""
    def __init__(self, count, factory, *, capacity=4):
        self._count = integer(count, "plane count")
        self._capacity = integer(capacity, "reader capacity", minimum=1)
        self._factory = factory
        self._readers = OrderedDict()

    def __len__(self):
        return self._count

    def __getitem__(self, index):
        if isinstance(index, slice):
            indices = range(*index.indices(self._count))
            return ReaderSequence(len(indices), lambda i: self[indices[i]], capacity=self._capacity)
        index = integer(index, "plane index")
        if index >= self._count:
            raise IndexError("Plane index outside reader sequence")
        if index in self._readers:
            value = self._readers.pop(index)
        else:
            value = self._factory(index)
        self._readers[index] = value
        while len(self._readers) > self._capacity:
            self._readers.popitem(last=False)[1].clear_cache()
        return value

    def clear_cache(self):
        for reader in self._readers.values():
            reader.clear_cache()
        self._readers.clear()


class SequenceWindow(Sequence):
    """A metadata sequence range without eagerly copying all TIFF frames."""
    def __init__(self, parent, start, stop):
        self.parent, self.start, self.stop = parent, start, stop

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, index):
        if isinstance(index, slice):
            lo, hi, step = index.indices(len(self))
            if step != 1:
                raise ValueError("Sequence windows require unit stride")
            return SequenceWindow(self.parent, self.start+lo, self.start+hi)
        if not 0 <= index < len(self):
            raise IndexError(index)
        return self.parent[self.start+index]
