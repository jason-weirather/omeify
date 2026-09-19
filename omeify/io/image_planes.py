"""Bridge semantic images into the existing shared TIFF writer, without asarray."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .image_metadata import integer
from .pixel_size import PixelSize
from .spec import ImageType, OMEImageSpec

if TYPE_CHECKING:
    from .base import Image


class ImagePlaneReader:
    """One logical channel (or an entire RGB plane), borrowing the image lifetime."""

    def __init__(self, image: Image, channel: int, *, level: int = 0) -> None:
        from .base import Image
        if not isinstance(image, Image):
            raise TypeError("image must be an omeify Image")
        if not image.is_open:
            raise RuntimeError("Image must be open for regional plane access")
        self._image = image
        self._metadata = image.metadata
        descriptor = self._metadata.level(level)
        self._level = descriptor.index
        self._channel = integer(channel, "channel")
        if self._channel >= self._metadata.logical_channel_count:
            raise IndexError("Logical channel does not exist")
        self.height, self.width = descriptor.spatial_shape
        self.dtype = self._metadata.dtype
        self.samples_per_pixel = self._metadata.samples_per_pixel
        self._epoch = getattr(image, "_image_epoch", None)
        self._reader_epoch = getattr(image, "_image_generation", None)

    def clear_cache(self) -> None:
        """This adapter owns no cache; it never closes/evicts the borrowed source."""

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        image = self._image
        if not image.is_open or self._epoch != getattr(image, "_image_epoch", None) or (
            self._reader_epoch != getattr(image, "_image_generation", None)
        ):
            raise RuntimeError("Plane reader belongs to a closed or earlier image session")
        bounds = tuple(integer(v, n) for v, n in zip((y0, y1, x0, x1), ("y0", "y1", "x0", "x1")))
        y0, y1, x0, x1 = bounds
        if not (y0 <= y1 <= self.height and x0 <= x1 <= self.width):
            raise ValueError("Region is outside the selected image plane")
        if self._metadata.image_type == "label":
            values = image.read_region(y0, y1, x0, x1, level=self._level)
        elif hasattr(image, "_wrapped_source"):
            selected = None if self.samples_per_pixel == 3 else (self._channel,)
            values = image.read_region(y0, y1, x0, x1, level=self._level, channels=selected)
            if self._metadata.axes == "CYX":
                values = values[0]
        else:
            # Legacy readers retain native-axis and RGB-selection conveniences.
            # Logical-channel regional access is canonical for every file reader.
            values = image.channels[self._channel].read_region(y0, y1, x0, x1, level=self._level)
        expected = (y1 - y0, x1 - x0)
        if self.samples_per_pixel == 3:
            expected = (*expected, 3)
        if not isinstance(values, np.ndarray) or values.shape != expected:
            raise ValueError(f"Image plane must return exactly shape {expected}")
        if values.dtype.newbyteorder("=") != self.dtype:
            raise TypeError(f"Image plane returned {values.dtype}; expected {self.dtype}")
        if not values.dtype.isnative:
            values = values.byteswap().view(self.dtype)
        return np.ascontiguousarray(values)


class ImagePlaneSource:
    """A bounded-read, borrowed adapter; both public TIFF writers share this path."""

    def __init__(self, image: Image, *, level: int = 0) -> None:
        from .base import Image
        if not isinstance(image, Image):
            raise TypeError("image must be an omeify Image")
        if not image.is_open:
            raise RuntimeError("Open the image with 'with' or open() before writing")
        self.image = image
        self.metadata = image.metadata
        self.level = self.metadata.level(level).index
        self._epoch = getattr(image, "_image_epoch", None)
        self._reader_epoch = getattr(image, "_image_generation", None)

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self.image.backing_paths

    def plane_readers(self, *, cache_mib: int = 64) -> list[ImagePlaneReader]:
        integer(cache_mib, "cache_mib")
        if not self.image.is_open or self._epoch != getattr(self.image, "_image_epoch", None) or (
            self._reader_epoch != getattr(self.image, "_image_generation", None)
        ):
            raise RuntimeError("Writer source belongs to a closed or earlier image session")
        return [ImagePlaneReader(self.image, i, level=self.level)
                for i in range(self.metadata.logical_channel_count)]

    def output_spec(
        self, *, image_type: ImageType | None = None,
        channel_names: Sequence[str] | None = None, pixel_size: PixelSize | None = None,
        icc_profile: bytes | None = None,
    ) -> OMEImageSpec:
        metadata = self.metadata
        descriptor = metadata.level(self.level)
        if image_type is not None and image_type != metadata.image_type:
            raise ValueError("Writer image_type conflicts with the image's declared semantics")
        size = pixel_size if pixel_size is not None else descriptor.pixel_size
        if size is None:
            raise ValueError("Writing requires calibrated pixels; supply pixel_size=PixelSize(...)")
        return OMEImageSpec.from_shape(
            image_type=metadata.image_type, axes=descriptor.axes, shape=descriptor.shape,
            dtype=metadata.dtype,
            channel_names=metadata.channel_names if channel_names is None else channel_names,
            pixel_size=size,
            icc_profile=metadata.icc_profile if icc_profile is None else icc_profile,
        )


def protect_source_paths(source: object, destination: Path) -> None:
    """Do not replace a declared pixel dependency, including symlinks/hard links."""
    for value in getattr(source, "backing_paths", ()):
        path = Path(value)
        if destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError("Output must differ from every local file backing the source image")
