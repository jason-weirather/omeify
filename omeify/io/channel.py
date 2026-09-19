"""Lazy logical channels; all reads go through the owning image contract."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .image_metadata import integer

if TYPE_CHECKING:
    from .base import Image


class Channel:
    """A read-only logical-channel view of one image descriptor and open session.

    Obtain channels from the parent image. Metadata is a projection of the same
    immutable descriptor, not an independently editable copy. asarray() explicitly
    materializes a whole channel. Scalar channels return YX; RGB returns YXS.
    """

    def __init__(self, image: Image, index: int) -> None:
        image._ensure_open()
        index = integer(index, "channel index")
        if index >= image.channel_count:
            raise IndexError("Channel index is outside the image")
        self._image = image
        self._generation = image._generation
        self._description = image._metadata
        self._index = index

    @property
    def index(self) -> int:
        return self._index

    @property
    def id(self) -> str:
        declared = self._description.channel_ids[self.index]
        return declared if declared is not None else f"Channel:0:{self.index}"

    @property
    def name(self) -> str:
        return self._description.channel_names[self.index]

    @property
    def dtype(self) -> np.dtype:
        return self._description.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        spatial = self._description.levels[0].spatial_shape
        return (*spatial, 3) if self.samples_per_pixel == 3 else spatial

    @property
    def samples_per_pixel(self) -> int:
        return self._description.samples_per_pixel

    @property
    def source_id(self) -> str | None:
        return self._description.channel_source_ids[self.index]

    @property
    def id_is_generated(self) -> bool:
        return self._description.channel_ids[self.index] is None

    @property
    def source_metadata(self) -> Mapping[str, Any]:
        return self._description.channel_metadata[self.index]

    @property
    def height(self) -> int:
        return self.shape[0]

    @property
    def width(self) -> int:
        return self.shape[1]

    def _ensure(self) -> None:
        self._image._ensure_session(self._generation)

    def read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int = 0,
    ) -> np.ndarray:
        self._ensure()
        values = self._image.read_region(y0, y1, x0, x1, level=level, channels=(self.index,))
        return values[0] if self._description.axes == "CYX" else values

    def asarray(self, *, level: int = 0) -> np.ndarray:
        self._ensure()
        height, width = self._description.level(level).spatial_shape
        return self.read_region(0, height, 0, width, level=level)

    def clear_cache(self) -> None:
        """Release the owning source's cache; no independent channel cache exists."""
        self._ensure()
        self._image.clear_cache()

    def __repr__(self) -> str:
        return (
            f"Channel(index={self.index}, id={self.id!r}, name={self.name!r}, "
            f"shape={self.shape}, dtype={self.dtype})"
        )
