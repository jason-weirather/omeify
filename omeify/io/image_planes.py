"""Private bridge from the one Image contract to physical TIFF planes."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import Image
from .image_metadata import integer
from .spec import OMEImageSpec


class ImagePlaneReader:
    """One logical channel bound to an image session, with no independent cache."""

    def __init__(self, image: Image, channel: int, *, level: int = 0) -> None:
        image._ensure_open()
        descriptor = image._metadata.level(level)
        self._image = image
        self._channel = image[channel]
        self._level = descriptor.index
        self.height, self.width = descriptor.spatial_shape
        self.dtype = image.dtype
        self.samples_per_pixel = image.samples_per_pixel

    def clear_cache(self) -> None:
        # Evict between writer passes/planes without closing the caller's source.
        if self._image.is_open and self._image._generation == self._channel._generation:
            self._channel.clear_cache()

    def read_region(self, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
        return self._channel.read_region(y0, y1, x0, x1, level=self._level)


class ImagePlaneSource:
    """The only image-to-writer adapter; no file/array/procedural dispatch."""

    def __init__(self, image: Image, *, level: int = 0) -> None:
        if not isinstance(image, Image):
            raise TypeError("Writing requires an Image; construct an image before writing")
        image._ensure_open()
        self._image = image
        self._metadata = image._metadata
        self._level = self._metadata.level(level).index
        self._generation = image._generation

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self._image.backing_paths

    def plane_readers(self, *, cache_mib: int = 64) -> list[ImagePlaneReader]:
        integer(cache_mib, "cache_mib")
        self._image._ensure_session(self._generation)
        return [ImagePlaneReader(self._image, i, level=self._level)
                for i in range(self._metadata.logical_channel_count)]

    def output_spec(self) -> OMEImageSpec:
        self._image._ensure_session(self._generation)
        metadata = self._metadata
        descriptor = metadata.level(self._level)
        if descriptor.pixel_size is None:
            raise ValueError(
                "Writing requires calibrated pixels; use image.with_metadata(pixel_size=...)",
            )
        return OMEImageSpec.from_shape(
            image_type=metadata.image_type, axes=descriptor.axes, shape=descriptor.shape,
            dtype=metadata.dtype, channel_names=metadata.channel_names,
            pixel_size=descriptor.pixel_size, icc_profile=metadata.icc_profile,
        )


def protect_source_paths(source: ImagePlaneSource, destination: Path) -> None:
    """Do not replace a declared pixel dependency, including symlinks/hard links."""
    for path in source.backing_paths:
        if destination.resolve() == path.resolve() or (
            destination.exists() and path.exists() and destination.samefile(path)
        ):
            raise ValueError("Output must differ from every local file backing the source image")
