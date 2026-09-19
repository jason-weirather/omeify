"""Backend contract and defensive regional-read boundary, without a file assumption."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from ._selection import normalize_channel_indices
from .image_metadata import ImageMetadata, integer

ChannelSelection = int | Sequence[int] | None


class ImageSource(ABC):
    """Subclass to provide stable metadata and on-demand, repeatable pixel regions.

    Implement ``_read_region``; optionally implement ``_open`` and ``_close`` for
    handles/workers. A file backend may discover its descriptor in ``_open``.
    The base read validates bounds, shape, and dtype and returns owned, C-contiguous
    native-endian pixels without intensity conversion. Empty spatial reads do not
    call the backend. Reads must be independent of request size/order/subsets.

    This interface requires random access, not bounded implementation cost: an
    array-backed source is file-free but is not a lazy renderer. Thread safety is
    not assumed. Metadata, cached source pixels, and caller arrays must not change
    during an open observation. ``close`` is idempotent; re-open starts a new session.
    """

    def __init__(self, metadata: ImageMetadata | None = None) -> None:
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
        """Session token, not a persistent identity or a content checksum."""
        return self._generation

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        """Optional local dependencies protected from overwrite by image writers.

        File backends must report local pixel dependencies, including a borrowed
        memory-mapped array's file. Empty for non-file sources. Not OME metadata.
        """
        return ()

    def open(self) -> ImageSource:
        if not self._is_open:
            try:
                self._open()
                if not isinstance(self._metadata, ImageMetadata):
                    raise TypeError("A source must publish ImageMetadata when opened")
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

    def _open(self) -> None:
        """Acquire resources and publish metadata, without rendering pixels."""

    def _close(self) -> None:
        """Release owned resources only."""

    def clear_cache(self) -> None:
        """Optional cache eviction; never closes the source or changes its pixels."""

    def _ensure_open(self) -> None:
        if not self.is_open:
            raise RuntimeError("Image source is closed; use 'with' or open()")

    def read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int = 0, channels: ChannelSelection = None,
    ) -> np.ndarray:
        """Read half-open integer bounds in the selected level's pixel coordinates.

        CYX channel selection preserves the C axis and requested order/duplicates.
        YX and YXS permit only logical channel zero; RGB always returns all three
        samples. No implicit crop clipping, padding, or resampling.
        """
        self._ensure_open()
        metadata = self.metadata
        descriptor = metadata.level(level)
        bounds = tuple(integer(v, n) for v, n in zip((y0, y1, x0, x1), ("y0", "y1", "x0", "x1")))
        y0, y1, x0, x1 = bounds
        height, width = descriptor.spatial_shape
        if not (y0 <= y1 <= height and x0 <= x1 <= width):
            raise ValueError(
                f"Region {bounds} is outside level {descriptor.index} shape {descriptor.shape}"
            )
        selected = tuple(normalize_channel_indices(channels, metadata.logical_channel_count))
        if metadata.axes != "CYX" and selected != (0,):
            raise ValueError("A YX or RGB image permits exactly logical channel zero")
        shape = (y1 - y0, x1 - x0)
        if metadata.axes == "CYX":
            shape = (len(selected), *shape)
        elif metadata.axes == "YXS":
            shape = (*shape, 3)
        if y0 == y1 or x0 == x1:
            return np.empty(shape, dtype=metadata.dtype)
        values = self._read_region(y0, y1, x0, x1, level=descriptor.index, channels=selected)
        if not isinstance(values, np.ndarray) or np.ma.isMaskedArray(values):
            raise TypeError("ImageSource._read_region must return an unmasked NumPy ndarray")
        if values.shape != shape:
            raise ValueError(f"Source returned shape {values.shape}; expected {shape}")
        if values.dtype.newbyteorder("=") != metadata.dtype:
            raise TypeError(f"Source returned dtype {values.dtype}; expected {metadata.dtype}")
        if not values.dtype.isnative:
            # Byte swap, not numeric casting: preserve NaN payloads and signed zero.
            values = values.byteswap().view(metadata.dtype)
        return np.array(values, copy=True, order="C", subok=False)

    @abstractmethod
    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        """Return precisely the requested samples in the descriptor's canonical axes."""

    def __repr__(self) -> str:
        if self._metadata is None:
            return f"{type(self).__name__}(is_open={self.is_open})"
        return (
            f"{type(self).__name__}(is_open={self.is_open}, "
            f"axes={self._metadata.axes!r}, shape={self._metadata.shape}, "
            f"dtype={self._metadata.dtype})"
        )

    def __enter__(self) -> ImageSource:
        if self._context_active:
            raise RuntimeError("Nested contexts on the same source are not supported")
        self.open()
        self._context_active = True
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._context_active = False
        self.close()
