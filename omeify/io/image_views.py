"""Borrowed metadata and channel views for the current library workflows."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from .base import Image
from .channel import Channel
from .image_metadata import ImageLevel, ImageMetadata, integer
from .image_source import ImageSource
from .pixel_size import PixelSize


class BorrowedSource(ImageSource):
    """Own no parent resources; pin every dependency to its original open session."""

    def __init__(self, metadata: ImageMetadata, parents: Sequence[Image]) -> None:
        super().__init__(metadata)
        self._parents = tuple(dict.fromkeys(parents))
        for image in self._parents:
            image._ensure_open()
        self._sessions = tuple(image._generation for image in self._parents)

    @property
    def is_open(self) -> bool:
        return self._is_open and all(
            image.is_open and image._generation == generation
            for image, generation in zip(self._parents, self._sessions)
        )

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return tuple(dict.fromkeys(path for image in self._parents for path in image.backing_paths))

    def _open(self) -> None:
        for image, generation in zip(self._parents, self._sessions):
            image._ensure_session(generation)

    def clear_cache(self) -> None:
        self._ensure_open()
        for image in self._parents:
            image.clear_cache()


class MetadataSource(BorrowedSource):
    """Reinterpret declared metadata without altering any pixel values."""

    def __init__(
        self, image: Image, *, channel_names: Sequence[str] | None,
        pixel_size: PixelSize | None, icc_profile: bytes | None,
    ) -> None:
        metadata = image._metadata
        levels = metadata.levels
        if pixel_size is not None:
            if not isinstance(pixel_size, PixelSize):
                raise TypeError("pixel_size must be a PixelSize")
            levels = tuple(replace(
                item, pixel_size=None if item.downsample_yx is None else PixelSize(
                    pixel_size.x * item.downsample_yx[1],
                    pixel_size.y * item.downsample_yx[0], pixel_size.unit,
                ),
            ) for item in levels)
        metadata = replace(
            metadata,
            channel_names=metadata.channel_names if channel_names is None else channel_names,
            pixel_size=metadata.pixel_size if pixel_size is None else pixel_size,
            icc_profile=metadata.icc_profile if icc_profile is None else icc_profile,
            levels=levels,
        )
        super().__init__(metadata, (image,))
        self._image = image

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        return self._image.read_region(y0, y1, x0, x1, level=level, channels=channels)


class ChannelSource(BorrowedSource):
    """Assemble a single CYX resolution from scalar channel views."""

    def __init__(
        self, channels: Sequence[Channel], *, level: int,
        channel_names: Sequence[str] | None, pixel_size: PixelSize | None,
    ) -> None:
        selected = tuple(channels)
        if not selected or any(not isinstance(c, Channel) for c in selected):
            raise TypeError("channels must be a nonempty sequence of Channel objects")
        level = integer(level, "level")
        for channel in selected:
            channel._ensure()
            if channel.samples_per_pixel != 1:
                raise ValueError("Channel assembly requires scalar channels, not interleaved RGB")
        descriptions = [c._image._metadata.level(level) for c in selected]
        shape = descriptions[0].spatial_shape
        dtype = selected[0].dtype
        if any(d.spatial_shape != shape for d in descriptions):
            raise ValueError("Channel images must have matching spatial shapes")
        if any(c.dtype != dtype for c in selected):
            raise TypeError("Channel images must have the same dtype; no implicit casting")
        if pixel_size is None:
            pixel_size = descriptions[0].pixel_size
            sizes = [None if d.pixel_size is None else d.pixel_size.converted_to("µm")
                     for d in descriptions]
            if any(s != sizes[0] for s in sizes):
                raise ValueError("Channel calibrations differ; supply an explicit pixel_size")
        metadata = ImageMetadata(
            axes="CYX", shape=(len(selected), *shape), dtype=dtype,
            channel_names=(
                tuple(c.name for c in selected) if channel_names is None else channel_names
            ),
            pixel_size=pixel_size,
            # New assembled channels get unique IDs; origins remain separate provenance.
            channel_source_ids=tuple(c.id for c in selected),
            channel_metadata=tuple(c.source_metadata for c in selected),
        )
        super().__init__(metadata, tuple(c._image for c in selected))
        self._channels = selected
        self._level = level

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        return np.stack([
            self._channels[i].read_region(y0, y1, x0, x1, level=self._level) for i in channels
        ])


class CropSource(BorrowedSource):
    """Offset level-zero regional reads without taking ownership of the parent."""

    def __init__(self, image: Image, y0: int, y1: int, x0: int, x1: int) -> None:
        y0, y1, x0, x1 = (integer(v, "crop bound") for v in (y0, y1, x0, x1))
        height, width = image.levels[0].spatial_shape
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError("Crop must be a nonempty, in-bounds level-zero rectangle")
        shape = list(image.shape)
        shape[image.axes.index("Y")] = y1 - y0
        shape[image.axes.index("X")] = x1 - x0
        level = ImageLevel(0, image.axes, tuple(shape), image.pixel_size, (1., 1.))
        metadata = replace(image._metadata, shape=tuple(shape), levels=(level,))
        super().__init__(metadata, (image,))
        self._image, self._y0, self._x0 = image, y0, x0

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        return self._image.read_region(
            y0 + self._y0, y1 + self._y0, x0 + self._x0, x1 + self._x0, channels=channels,
        )
