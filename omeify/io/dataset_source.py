"""Multidimensional source contract independent of the 2D ImageSource interface.

File backends can implement both contracts: their existing 2D reader selects one
plane, while ImageDataset uses the native Z/T interface. Array-backed datasets
and borrowed dataset views need only this contract, not a pretend 2D reader.
"""
from __future__ import annotations

from inspect import getattr_static
from pathlib import Path
from typing import Protocol, Self

import numpy as np

from ._source_session import _SourceSession
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


def validate_dataset_levels(levels) -> tuple[DatasetLevel, ...]:
    """Validate native metadata for both dataset-only and dual-interface sources."""
    levels = tuple(levels)
    if not levels or any(not isinstance(item, DatasetLevel) or item.index != i
                         for i, item in enumerate(levels)):
        raise TypeError("A dataset source must publish ordered DatasetLevel descriptors")
    return levels


class DatasetSourceBase(_SourceSession):
    """Resource lifecycle for sources that expose only native dataset planes.

    ImageSource and DatasetSourceBase inherit the same private session lifecycle.
    File readers implement ImageSource plus this protocol structurally. Dataset
    sources with no fixed 2D observation inherit this lightweight base.
    """

    supports_prefetch = False
    _closed_message = "Dataset source is closed; use 'with' or open()"
    _context_label = "dataset source"

    def _validate_open(self) -> None:
        # DatasetLevel is authoritative: a dataset source need not construct or
        # expose a stand-alone ImageMetadata just to enter a session.
        validate_dataset_levels(self.dataset_levels)

    def cache_info(self) -> dict:
        return {"max_bytes": 0, "resident_bytes": 0, "entries": 0, "in_flight": 0}
