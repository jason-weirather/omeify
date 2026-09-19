"""Shared TIFF lifecycle, level geometry, and bounded regional decoding."""
from __future__ import annotations

import logging
import threading
from abc import abstractmethod
from pathlib import Path

import numpy as np
import tifffile

from omeify.inspection import TiffInspector

from ..image_metadata import ImageLevel, ImageMetadata, integer
from ..image_source import ImageSource
from ..pixel_size import PixelSize, consistent_tiff_resolution_pixel_size
from ..tiff import TiffPlaneReader


LOGGER = logging.getLogger(__name__)


class TiffSource(ImageSource):
    def __init__(self, path: str | Path, *, series: int = 0) -> None:
        super().__init__()
        self.path = Path(path)
        self.series_index = integer(series, "series")
        self._tiff: tifffile.TiffFile | None = None
        self._series: tifffile.TiffPageSeries | None = None
        self._inspection: TiffInspector | None = None
        self._planes: dict[tuple[int, int], TiffPlaneReader] = {}
        self._read_lock = threading.RLock()

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return (self.path,)

    @property
    def native_levels(self) -> tuple[tifffile.TiffPageSeries, ...]:
        if self._series is None:
            raise RuntimeError("TIFF source is not open")
        return tuple(self._series.levels) or (self._series,)

    def inspect(self) -> TiffInspector:
        if self._tiff is None:
            raise RuntimeError("TIFF source is not open")
        if self._inspection is None:
            self._inspection = TiffInspector.from_tiff(self._tiff, file_path=self.path, detail=1)
        return self._inspection

    def _open(self) -> None:
        self._tiff = tifffile.TiffFile(self.path, _multifile=False)
        self._load_file(self._tiff)
        self._metadata = self._describe()

    def _close(self) -> None:
        for plane in self._planes.values():
            plane.clear_cache()
        self._planes.clear()
        if self._tiff is not None:
            self._tiff.close()
        self._tiff = None
        self._series = None
        self._inspection = None
        self._metadata = None

    def clear_cache(self) -> None:
        self._ensure_open()
        for plane in self._planes.values():
            plane.clear_cache()
        self._planes.clear()

    @abstractmethod
    def _load_file(self, tiff: tifffile.TiffFile) -> None:
        """Interpret the container and select its physical series."""

    @abstractmethod
    def _describe(self) -> ImageMetadata:
        """Normalize file metadata once, without decoding pixels."""

    @abstractmethod
    def series_names(self) -> tuple[str | None, ...]:
        """Names in TIFF series order, not OME image order."""

    def series_name(self) -> str | None:
        return self.series_names()[self.series_index]

    def _level_pages(self, level: int) -> list[tifffile.TiffPage]:
        return [page.aspage() for page in self.native_levels[level].pages]

    def _levels(
        self, axes: str, channels: int, pixel_size: PixelSize | None,
    ) -> tuple[ImageLevel, ...]:
        descriptions = []
        base_dtype = np.dtype(self.native_levels[0].dtype).newbyteorder("=")
        for index, item in enumerate(self.native_levels):
            pages = self._level_pages(index)
            if len(pages) != channels:
                raise ValueError("Pyramid level plane count differs from its logical channels")
            if np.dtype(item.dtype).newbyteorder("=") != base_dtype:
                raise TypeError("Pyramid levels must retain the base dtype")
            layout = dict(zip(str(item.axes), item.shape))
            if "Y" not in layout or "X" not in layout:
                raise ValueError("TIFF level has no Y/X geometry")
            # Channel axis letters may be vendor-defined (I/Q), so check physical planes.
            height, width = int(layout["Y"]), int(layout["X"])
            for page in pages:
                if (int(page.imagelength), int(page.imagewidth)) != (height, width):
                    raise ValueError("Pyramid channel geometry differs from the advertised raster")
                if np.dtype(page.dtype).newbyteorder("=") != base_dtype:
                    raise TypeError("Pyramid channel dtype differs from the base image")
                if int(page.samplesperpixel) != (3 if axes == "YXS" else 1):
                    raise ValueError("Pyramid sample organization differs from the base image")
            shape = ((channels, height, width) if axes == "CYX" else
                     (height, width, 3) if axes == "YXS" else (height, width))
            size = pixel_size if index == 0 else consistent_tiff_resolution_pixel_size(pages)
            if index and size is None:
                LOGGER.warning(
                    "Pyramid level %s has no physical calibration; its scale will not be inferred "
                    "from rounded dimensions", index,
                )
            scale = (1., 1.) if index == 0 else None
            if index and pixel_size is not None and size is not None:
                a, b = pixel_size.converted_to("µm"), size.converted_to("µm")
                scale = (b.y / a.y, b.x / a.x)
            descriptions.append(ImageLevel(index, axes, shape, size, scale))
        return tuple(descriptions)

    def _plane(self, channel: int, level: int) -> TiffPlaneReader:
        key = (channel, level)
        if key not in self._planes:
            self._planes[key] = TiffPlaneReader(
                self.native_levels[level].pages[channel].aspage(), lock=self._read_lock,
            )
        return self._planes[key]

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *, level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        if self.metadata.axes == "CYX":
            return np.stack([self._plane(c, level).read_region(y0, y1, x0, x1) for c in channels])
        return self._plane(0, level).read_region(y0, y1, x0, x1)
