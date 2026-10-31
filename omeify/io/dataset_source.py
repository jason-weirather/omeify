"""Multidimensional source contract independent of the 2D ImageSource interface.

File backends can implement both contracts: their existing 2D reader selects one
plane, while ImageDataset uses the native Z/T interface. Array-backed datasets
and borrowed dataset views need only this contract, not a pretend 2D reader.
"""
from __future__ import annotations

from pathlib import Path
from inspect import getattr_static
from typing import Protocol, Self

import numpy as np

from .dataset_metadata import DatasetLevel
from .image_metadata import ImageMetadata


class DatasetSource(Protocol):
    """Operations required by ImageDataset; no ``read_region`` or 2D inheritance.

    This is a static typing protocol, not an ``isinstance`` guard. Runtime
    checking of property-backed protocols can invoke unopened descriptors on
    Python 3.11; use ``check_dataset_source`` without opening source metadata.
    Optional cache/prefetch capabilities are queried separately by the session.
    """

    @property
    def is_open(self) -> bool: ...

    @property
    def generation(self) -> int: ...

    @property
    def backing_paths(self) -> tuple[Path, ...]: ...

    @property
    def dataset_levels(self) -> tuple[DatasetLevel, ...]: ...

    def open(self) -> Self: ...

    def close(self) -> None: ...

    def plane_metadata(self, *, z: int | None, t: int | None,
                       level: int) -> ImageMetadata: ...

    def read_plane_region(self, y0: int, y1: int, x0: int, x1: int, *,
                          z: int | None, t: int | None, level: int,
                          channels: tuple[int, ...], speculative: bool = False) -> np.ndarray: ...

    def clear_cache(self) -> None: ...

    def cache_info(self) -> dict: ...


def check_dataset_source(source) -> None:
    """Validate structural requirements without touching lazy source properties.

    ``isinstance(source, @runtime_checkable Protocol)`` could evaluate an
    unopened ``dataset_levels`` property on Python 3.11. Static lookup avoids
    discovery, I/O and that interpreter-dependent behavior.
    """
    required = ("is_open", "generation", "backing_paths", "dataset_levels",
                "open", "close", "plane_metadata", "read_plane_region",
                "clear_cache", "cache_info")
    missing = [name for name in required
               if getattr_static(source, name, None) is None]
    if missing:
        raise TypeError("source must implement the DatasetSource contract: " +
                        ", ".join(missing))


class DatasetSourceBase:
    """Resource lifecycle for sources that expose only native dataset planes.

    The existing ImageSource lifecycle is unchanged. File readers continue to
    implement ImageSource plus the DatasetSource protocol structurally.
    Dataset-specific array/borrowed sources inherit this lightweight base.
    """

    supports_prefetch = False

    def __init__(self, metadata: ImageMetadata | None = None):
        if metadata is not None and not isinstance(metadata, ImageMetadata):
            raise TypeError("metadata must be ImageMetadata or None")
        self._metadata = metadata
        self._is_open = False
        self._generation = 0
        self._context_active = False

    @property
    def metadata(self) -> ImageMetadata:
        if self._metadata is None:
            raise RuntimeError("Open the source before discovering its metadata")
        return self._metadata

    @property
    def is_open(self) -> bool:
        return self._is_open

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return ()

    def _open(self) -> None:
        """Acquire metadata/resources without decoding pixels."""

    def _close(self) -> None:
        """Release owned resources only."""

    def open(self) -> Self:
        if not self._is_open:
            try:
                self._open()
                if not isinstance(self._metadata, ImageMetadata):
                    raise TypeError("A dataset source must publish ImageMetadata")
            except BaseException:
                self._close()
                raise
            self._is_open = True
            self._generation += 1
        return self

    def close(self) -> None:
        if self._is_open:
            try:
                self._close()
            finally:
                self._is_open = False
                self._generation += 1

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("Dataset source is closed; use 'with' or open()")

    def clear_cache(self) -> None:
        """No-op for sources without retained pixel caches."""

    def cache_info(self) -> dict:
        return {"max_bytes": 0, "resident_bytes": 0, "entries": 0, "in_flight": 0}

    def __enter__(self) -> Self:
        if self._context_active:
            raise RuntimeError("Nested contexts on the same dataset source are not supported")
        self.open()
        self._context_active = True
        return self

    def __exit__(self, *args) -> None:
        self._context_active = False
        self.close()
