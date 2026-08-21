from __future__ import annotations

import threading
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from omeify.inspection import TiffInspector
from omeify.io.base import ChannelSelection, LabelImage, MultichannelImage
from omeify.io.tiff import TiffPlaneReader


class _OMETiffReaderCore:
    """Shared context management and TIFF access for concrete OME readers."""

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

    def open(self):
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

    def __enter__(self):
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _require_open(self) -> tifffile.TiffFile:
        if self._tiff is None:
            raise RuntimeError(f"{type(self).__name__} must be opened with 'with' or open()")
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

    def _read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        level: int = 0,
        channels: ChannelSelection = None,
    ) -> np.ndarray:
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
        return (
            f"{type(self).__name__}(path={str(self.path)!r}, "
            f"series={self.series_index}, closed=True)"
        )


class OMETiffReader(_OMETiffReaderCore, MultichannelImage):
    """Context-managed reader for one multichannel or RGB OME-TIFF series.

    ``print(reader)`` uses the same default renderer as ``omeify inspect PATH``.
    ``channel_names`` are logical OME channels. For RGB this is normally
    ``('RGB',)`` while ``sample_names`` are red, green, and blue.
    """

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
        count = self.shape[self.axes.index("C")] if "C" in self.axes else 1
        return tuple(f"Channel {index + 1}" for index in range(int(count)))

    @property
    def samples_per_pixel(self) -> int:
        image = self._ome_image_summary()
        if image is not None:
            samples = [
                int(item["samples_per_pixel"])
                for item in image.get("channels", [])
                if item.get("samples_per_pixel") is not None
            ]
            if samples and len(set(samples)) == 1:
                return samples[0]
        if "S" in self.axes:
            return int(self.shape[self.axes.index("S")])
        return 1

    @property
    def sample_count(self) -> int:
        return self.size_c

    @property
    def is_rgb(self) -> bool:
        return self.samples_per_pixel == 3 and self.size_c == 3 and len(self.channel_names) == 1

    @property
    def sample_names(self) -> tuple[str, ...]:
        if self.is_rgb:
            return ("Red", "Green", "Blue")
        if self.samples_per_pixel == 1:
            return self.channel_names
        return tuple(f"Sample {index + 1}" for index in range(self.sample_count))

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
        """Read a Y/X window from common planar or interleaved layouts.

        For planar ``CYX`` images, ``channels`` selects logical channels. For
        interleaved ``YXS`` RGB images it selects stored samples.
        """

        return self._read_region(
            y0,
            y1,
            x0,
            x1,
            level=level,
            channels=channels,
        )


class OMETiffLabelReader(_OMETiffReaderCore, LabelImage):
    """Virtual-access reader for one integer OME-TIFF label raster.

    The caller explicitly chooses label semantics by using this class. OME-TIFF
    does not intrinsically distinguish intensity rasters from label rasters.
    No region measurements or label counting are performed.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        background_label: int = 0,
    ) -> None:
        super().__init__(path, series=series)
        self.background_label = int(background_label)

    def open(self) -> "OMETiffLabelReader":
        super().open()
        try:
            self.validate_label_dtype()
            axis_sizes = dict(zip(self.axes, self.shape))
            unsupported = {
                axis: size
                for axis, size in axis_sizes.items()
                if axis not in {"C", "Y", "X"} and size != 1
            }
            if unsupported:
                raise ValueError(
                    "Label OME-TIFF requires singleton non-spatial axes; "
                    f"found {unsupported} in axes {self.axes!r}"
                )
            if "S" in axis_sizes or int(axis_sizes.get("C", 1)) != 1:
                raise ValueError(
                    "Label OME-TIFF reader requires one grayscale label plane; "
                    f"found axes={self.axes!r}, shape={self.shape}."
                )
        except Exception:
            self.close()
            raise
        return self

    def asarray(self, *, level: int = 0) -> np.ndarray:
        value = super().asarray(level=level)
        axes = list(str(self._selected_level(level).axes))
        for axis_index in range(len(axes) - 1, -1, -1):
            if axes[axis_index] in {"Y", "X"}:
                continue
            if value.shape[axis_index] != 1:
                raise ValueError(
                    "Label image contains a non-singleton non-spatial axis: "
                    f"axes={''.join(axes)!r}, shape={value.shape}"
                )
            value = np.take(value, 0, axis=axis_index)
            axes.pop(axis_index)
        if axes != ["Y", "X"] or value.ndim != 2:
            raise ValueError(
                f"Label image did not resolve to one YX raster: axes={axes}, shape={value.shape}"
            )
        return np.ascontiguousarray(value)

    def read_region(
        self,
        y0: int,
        y1: int,
        x0: int,
        x1: int,
        *,
        level: int = 0,
    ) -> np.ndarray:
        value = self._read_region(
            y0,
            y1,
            x0,
            x1,
            level=level,
            channels=0,
        )
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 2:
            raise ValueError(f"Label region did not resolve to one YX raster: {value.shape}")
        return np.ascontiguousarray(value)
