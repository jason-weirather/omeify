"""NumPy and memory-map backend; no disk serialization and no generated pyramid."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ImageSource
from .pixel_size import PixelSize
from .spec import ImageType


class ArraySource(ImageSource):
    """Borrow or copy one array, optionally with explicitly described reduced levels.

    ``copy=False`` borrows the arrays: the caller must not mutate them while read.
    ``copy=True`` takes a complete snapshot (and allocates it). Regional results
    never alias the inputs. Closing never closes a caller's NumPy memory map.
    No shape-based channel/RGB guessing, implicit resampling, or dtype conversion.
    """

    def __init__(
        self, array: np.ndarray, *, axes: str, image_type: ImageType = "multichannel",
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
        channel_ids: Sequence[str | None] | None = None, icc_profile: bytes | None = None,
        background_label: int = 0, copy: bool = False,
        level_arrays: Sequence[np.ndarray] = (), levels: Sequence[ImageLevel] = (),
    ) -> None:
        if not isinstance(copy, bool):
            raise TypeError("copy must be a boolean")
        if isinstance(channel_names, (str, bytes)) or isinstance(channel_ids, (str, bytes)):
            raise TypeError("channel_names/channel_ids must be sequences, not strings")
        arrays = (array, *tuple(level_arrays))
        if not all(isinstance(a, np.ndarray) and not np.ma.isMaskedArray(a) for a in arrays):
            raise TypeError(
                "ArraySource accepts unmasked NumPy arrays, not implicit array-like materialization"
            )
        descriptors = tuple(levels)
        if len(arrays) != (len(descriptors) if descriptors else 1):
            raise ValueError("Supply one ImageLevel per array, including level zero")
        metadata = ImageMetadata(
            axes=axes, shape=array.shape, dtype=array.dtype, image_type=image_type,
            channel_names=None if channel_names is None else tuple(channel_names),
            pixel_size=pixel_size, levels=descriptors,
            channel_ids=None if channel_ids is None else tuple(channel_ids),
            icc_profile=icc_profile, background_label=background_label,
        )
        for item, level in zip(arrays, metadata.levels):
            if item.shape != level.shape or item.dtype.newbyteorder("=") != metadata.dtype:
                raise ValueError("Each level array must match its declared shape and image dtype")
        super().__init__(metadata)
        self._paths = () if copy else _array_paths(arrays)
        self._arrays = (
            tuple(np.array(a, copy=True, subok=False) for a in arrays) if copy else arrays
        )

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self._paths

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        array = self._arrays[level]
        if array.shape != self.metadata.level(level).shape:
            raise ValueError("Caller resized a borrowed array during the observation")
        if self.metadata.axes == "CYX":
            return array[:, y0:y1, x0:x1][list(channels)]
        if self.metadata.axes == "YXS":
            return array[y0:y1, x0:x1, :][..., list(channels)]
        return array[y0:y1, x0:x1]


def _array_paths(arrays: Sequence[np.ndarray]) -> tuple[Path, ...]:
    paths = set()
    for array in arrays:
        seen = set()
        current = array
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, np.memmap) and current.filename is not None:
                paths.add(Path(current.filename).resolve())
            current = getattr(current, "base", None)
    return tuple(sorted(paths))
