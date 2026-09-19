"""File-reader adapters. Format interpretation remains in the existing TIFF readers."""
from __future__ import annotations

from numbers import Integral
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .image_metadata import ImageLevel, ImageMetadata, integer
from .image_planes import ImagePlaneReader
from .image_source import ImageSource
from .spec import ImageType

if TYPE_CHECKING:
    from .base import Image


def metadata_from_reader(reader: Image) -> ImageMetadata:
    """Describe canonical reads, preserving unknown reduced-level calibration.

    Legacy ``reader.axes/shape/levels`` retain their native TIFF meaning. New
    descriptors describe the supported 2D regional-access layout, not singleton
    Z/T storage dimensions. This function performs no pixel reads.
    """
    if not reader.is_open:
        raise RuntimeError("Open the reader before requesting its image descriptor")
    kind = reader.image_type
    if kind == "label":
        axes, names, ids = "YX", ("Labels",), (None,)
    else:
        axes = reader.output_axes
        channels = reader.channels
        names = tuple(c.name for c in channels)
        ids = tuple(None if c.id_is_generated else c.id for c in channels)
    base_size = reader.pixel_size
    descriptors = []
    for i, item in enumerate(reader.levels):
        if np.dtype(item.dtype).newbyteorder("=") != np.dtype(reader.dtype).newbyteorder("="):
            raise ValueError("Pyramid levels must preserve the source dtype")
        layout = dict(zip(str(item.axes), item.shape))
        if "Y" not in layout or "X" not in layout or any(
            size != 1 for axis, size in layout.items() if axis not in "CSYX"
        ):
            raise NotImplementedError(
                "Image sources currently support 2D images, not non-singleton Z/T"
            )
        height, width = int(layout["Y"]), int(layout["X"])
        if axes == "CYX":
            if int(layout.get("C", 1)) != len(names):
                raise ValueError("Pyramid level channel layout differs from its base image")
            shape = (len(names), height, width)
        elif axes == "YXS":
            if int(layout.get("S", 0)) != 3:
                raise ValueError("RGB source must have three stored samples at every level")
            shape = (height, width, 3)
        else:
            if int(layout.get("C", 1)) != 1 or int(layout.get("S", 1)) != 1:
                raise ValueError("YX source must have one scalar plane at every level")
            shape = (height, width)
        size = base_size if i == 0 else reader.pixel_size_at_level(i)
        scale = (1.0, 1.0) if i == 0 else None
        if i and size is not None and base_size is not None:
            a, b = base_size.converted_to("µm"), size.converted_to("µm")
            scale = (b.y / a.y, b.x / a.x)
        descriptors.append(ImageLevel(i, axes, shape, size, scale))
    return ImageMetadata(
        axes=axes, shape=descriptors[0].shape, dtype=reader.dtype, image_type=kind,
        channel_names=names, channel_ids=ids, pixel_size=base_size, levels=tuple(descriptors),
        icc_profile=reader.icc_profile if kind == "rgb" else None,
        background_label=reader.background_label if kind == "label" else 0,
    )


def _epoch(reader: Image) -> tuple[object, object]:
    return getattr(reader, "_image_epoch", None), getattr(reader, "_image_generation", None)


class ReaderImageSource(ImageSource):
    """Adapt an existing Omeify reader/image without taking ownership by default.

    This is also the canonical bridge for vendor readers' native-layout APIs.
    Borrowed readers must be open and must outlive every image using this adapter.
    """

    def __init__(self, reader: Image, *, owns_reader: bool = False) -> None:
        from .base import Image
        if not isinstance(reader, Image):
            raise TypeError("reader must be an omeify Image")
        if not isinstance(owns_reader, bool):
            raise TypeError("owns_reader must be a boolean")
        super().__init__()
        self._reader = reader
        self._owns_reader = owns_reader
        self._reader_epoch = None
        self._planes: dict[tuple[int, int], ImagePlaneReader] = {}

    @property
    def is_open(self) -> bool:
        return (self._is_open and self._reader.is_open
                and self._reader_epoch == _epoch(self._reader))

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self._reader.backing_paths

    def open(self) -> ReaderImageSource:
        if self._is_open and not self.is_open:
            raise RuntimeError(
                "Borrowed reader session changed; close this source before reopening"
            )
        super().open()
        return self

    def _open(self) -> None:
        if self._owns_reader:
            self._reader.open()
        elif not self._reader.is_open:
            raise RuntimeError("A borrowed image/reader must already be open")
        self._reader_epoch = _epoch(self._reader)
        self._metadata = self._reader.metadata

    def _close(self) -> None:
        self._planes.clear()
        if self._owns_reader and self._reader_epoch == _epoch(self._reader):
            self._reader.close()

    def clear_cache(self) -> None:
        self._ensure_open()
        self._reader.clear_cache()

    def _plane(self, channel: int, level: int) -> ImagePlaneReader:
        key = (channel, level)
        if key not in self._planes:
            self._planes[key] = ImagePlaneReader(self._reader, channel, level=level)
        return self._planes[key]

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        if self.metadata.axes == "CYX":
            return np.stack([
                self._plane(i, level).read_region(y0, y1, x0, x1)
                for i in channels
            ])
        values = self._plane(0, level).read_region(y0, y1, x0, x1)
        return values[..., list(channels)] if self.metadata.axes == "YXS" else values


class OMETiffSource(ReaderImageSource):
    """OME-TIFF backend for any semantic image wrapper, reusing the proven decoder.

    A filename does not identify labels. Pass ``image_type='label'`` explicitly
    for a categorical raster; the default recognizes scalar intensity or RGB.
    """

    def __init__(
        self, path: str | Path, *, series: int = 0,
        image_type: ImageType | None = None, background_label: int = 0,
    ) -> None:
        from .ome_tiff_reader import OMETiffLabelReader, OMETiffReader
        series = integer(series, "series")
        if isinstance(background_label, (bool, np.bool_)) or not isinstance(
            background_label, Integral
        ):
            raise TypeError("background_label must be an integer")
        if image_type not in {None, "multichannel", "rgb", "label"}:
            raise ValueError("Unsupported image_type")
        if image_type != "label" and background_label != 0:
            raise ValueError("background_label requires image_type='label'")
        reader = (
            OMETiffLabelReader(path, series=series, background_label=background_label)
            if image_type == "label" else OMETiffReader(path, series=series)
        )
        super().__init__(reader, owns_reader=True)
        self._expected_type = image_type

    @property
    def path(self) -> Path:
        return self._reader.path

    def _open(self) -> None:
        super()._open()
        if self._expected_type is not None and self.metadata.image_type != self._expected_type:
            raise TypeError("File image semantics differ from the requested image_type")
