from __future__ import annotations

import threading
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from omeify.inspection import TiffInspector
from omeify.io.base import ChannelSelection, MultichannelImage
from omeify.utils.tiff_image_features import TiffPlaneReader


class OMETiffReader(MultichannelImage):
    """Context-managed reader for one OME-TIFF image series.

    The TIFF file remains open for the lifetime of the context. ``print(reader)``
    uses the same default renderer as ``omeify inspect PATH``.
    """

    def __init__(self, path: str | Path, *, series: int = 0) -> None:
        self._path = Path(path)
        self.series_index = int(series)
        if self.series_index < 0:
            raise ValueError("series must be zero or greater")
        self._tiff: tifffile.TiffFile | None = None
        self._inspection: TiffInspector | None = None
        self._read_lock = threading.RLock()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._tiff is not None

    def open(self) -> "OMETiffReader":
        if self._tiff is not None:
            return self
        tiff = tifffile.TiffFile(self.path)
        try:
            if not tiff.is_ome:
                raise ValueError(f"TIFF does not contain recognized OME metadata: {self.path}")
            if self.series_index >= len(tiff.series):
                raise IndexError(
                    f"Series {self.series_index} does not exist; OME-TIFF has "
                    f"{len(tiff.series)} series"
                )
        except Exception:
            tiff.close()
            raise
        self._tiff = tiff
        self._inspection = None
        return self

    def close(self) -> None:
        if self._tiff is not None:
            self._tiff.close()
        self._tiff = None
        self._inspection = None

    def __enter__(self) -> "OMETiffReader":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _require_open(self) -> tifffile.TiffFile:
        if self._tiff is None:
            raise RuntimeError("OMETiffReader must be opened with 'with' or open()")
        return self._tiff

    @property
    def tiff(self) -> tifffile.TiffFile:
        return self._require_open()

    @property
    def series(self) -> tifffile.TiffPageSeries:
        return self.tiff.series[self.series_index]

    @property
    def levels(self) -> tuple[tifffile.TiffPageSeries, ...]:
        return tuple(self.series.levels)

    @property
    def axes(self) -> str:
        return str(self.series.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.series.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.series.dtype).newbyteorder("=")

    @property
    def size_c(self) -> int:
        image = self._ome_image_summary()
        value = image["size"].get("c") if image is not None else None
        if value is not None:
            return int(value)
        if "C" in self.axes:
            return int(self.shape[self.axes.index("C")])
        if "S" in self.axes:
            return int(self.shape[self.axes.index("S")])
        return 1

    @property
    def channel_names(self) -> tuple[str, ...]:
        image = self._ome_image_summary()
        if image is not None:
            labels = [
                channel.get("name")
                or channel.get("id")
                or f"Channel {int(channel['index']) + 1}"
                for channel in image["channels"]
            ]
            if labels:
                return tuple(str(item) for item in labels)
        return tuple(f"Channel {index + 1}" for index in range(self.size_c))

    @property
    def level_count(self) -> int:
        return len(self.levels)

    @property
    def is_pyramidal(self) -> bool:
        return self.level_count > 1

    @property
    def inspection(self) -> TiffInspector:
        tiff = self._require_open()
        if self._inspection is None:
            self._inspection = TiffInspector.from_tiff(
                tiff,
                file_path=self.path,
                detail=1,
            )
        return self._inspection

    @property
    def inspection_report(self) -> dict[str, Any]:
        return self.inspection.report

    def _ome_image_summary(self) -> dict[str, Any] | None:
        ome = self.inspection_report.get("ome")
        if ome is None:
            return None
        images = ome.get("images", [])
        if self.series_index >= len(images):
            return None
        return images[self.series_index]

    def _selected_level(self, level: int) -> tifffile.TiffPageSeries:
        level_index = int(level)
        if level_index < 0 or level_index >= len(self.levels):
            raise IndexError(
                f"Pyramid level {level_index} does not exist; series has {len(self.levels)} levels"
            )
        return self.levels[level_index]

    def asarray(self, *, level: int = 0) -> np.ndarray:
        """Materialize one pyramid level.

        Whole-slide arrays can be extremely large. Prefer :meth:`read_region`
        when only a spatial window is needed.
        """

        return np.asarray(self._selected_level(level).asarray())

    @staticmethod
    def _normalize_channels(channels: ChannelSelection, size: int) -> list[int]:
        if channels is None:
            return list(range(size))
        selected = (
            [int(channels)]
            if isinstance(channels, Integral)
            else [int(item) for item in channels]
        )
        if not selected:
            raise ValueError("channels must contain at least one index")
        for index in selected:
            if index < 0 or index >= size:
                raise IndexError(f"Channel {index} is outside the available range 0..{size - 1}")
        return selected

    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        level: int = 0,
        channels: ChannelSelection = None,
    ) -> np.ndarray:
        """Read a Y/X window from common planar or interleaved OME-TIFF layouts.

        ``CYX``, ``YX``, and ``YXS`` layouts are read tile-by-tile or strip-by-strip.
        Additional Z/T axes are accepted only when they are singleton.
        """

        selected_level = self._selected_level(level)
        axes = str(selected_level.axes)
        shape = tuple(int(value) for value in selected_level.shape)
        axis_sizes = dict(zip(axes, shape))
        unsupported = {
            axis: size
            for axis, size in axis_sizes.items()
            if axis not in {"C", "S", "Y", "X"} and size != 1
        }
        if unsupported:
            raise NotImplementedError(
                "read_region currently requires singleton non-spatial Z/T axes; "
                f"found {unsupported} in axes {axes!r}"
            )
        if "Y" not in axes or "X" not in axes:
            raise NotImplementedError(f"read_region requires Y and X axes, found {axes!r}")

        height = axis_sizes["Y"]
        width = axis_sizes["X"]
        if not (0 <= y0 <= y1 <= height and 0 <= x0 <= x1 <= width):
            raise ValueError(
                f"Requested region {(y0, y1, x0, x1)} is outside level shape {shape} "
                f"with axes {axes!r}"
            )

        pages = [item.aspage() for item in selected_level.pages]
        if "C" in axes:
            size_c = axis_sizes["C"]
            selected_channels = self._normalize_channels(channels, size_c)
            if len(pages) != size_c:
                raise NotImplementedError(
                    f"CYX region reading expects one TIFF page per channel; found "
                    f"{len(pages)} pages for C={size_c}"
                )
            readers = [
                TiffPlaneReader(pages[index], lock=self._read_lock)
                for index in selected_channels
            ]
            return np.stack(
                [reader.read_region(y0, y1, x0, x1) for reader in readers],
                axis=0,
            )

        if len(pages) != 1:
            raise NotImplementedError(
                f"Region reading for axes {axes!r} expects one TIFF page; found {len(pages)}"
            )
        reader = TiffPlaneReader(pages[0], lock=self._read_lock)
        result = reader.read_region(y0, y1, x0, x1)
        if "S" in axes:
            selected_samples = self._normalize_channels(channels, axis_sizes["S"])
            return np.ascontiguousarray(result[..., selected_samples])
        if channels is not None:
            selected = self._normalize_channels(channels, 1)
            if selected != [0]:
                raise IndexError("Single-channel YX images only accept channel 0")
        return result

    def __str__(self) -> str:
        return self.inspection.render_text()

    def __repr__(self) -> str:
        if self.is_open:
            return self.inspection.render_text()
        return f"OMETiffReader(path={str(self.path)!r}, series={self.series_index}, closed=True)"
