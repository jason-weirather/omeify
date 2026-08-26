from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from omeify.inspection import TiffInspector

from .base import ChannelSelection, LabelImage, MultichannelImage
from .channel import Channel, normalize_channel_indices
from .pixel_size import (
    PixelSize,
    consistent_tiff_resolution_pixel_size,
    pixel_size_from_xy_units,
)
from .tiff import TiffPlaneReader

LOGGER = logging.getLogger(__name__)


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
        self._region_readers: dict[tuple[int, int], TiffPlaneReader] = {}

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
        for reader in self._region_readers.values():
            reader.clear_cache()
        self._region_readers.clear()
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
        values = tuple(getattr(self.series, "levels", ()) or ())
        return values or (self.series,)

    @property
    def axes(self) -> str:
        return str(self.series.axes)

    @property
    def source_axes(self) -> str:
        return self.axes

    @property
    def source_byte_order(self) -> str:
        return "big" if self.tiff.byteorder == ">" else "little"

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

    @property
    def pixel_size(self) -> PixelSize | None:
        image = self._ome_image_summary()
        physical = (image.get("physical_size") or {}) if image is not None else {}
        x = physical.get("x")
        y = physical.get("y")
        x_complete = bool(x and x.get("value") is not None and x.get("unit"))
        y_complete = bool(y and y.get("value") is not None and y.get("unit"))
        if x_complete and y_complete:
            assert x is not None and y is not None
            return pixel_size_from_xy_units(
                float(x["value"]),
                str(x["unit"]),
                float(y["value"]),
                str(y["unit"]),
            )

        if x is not None or y is not None:
            LOGGER.warning(
                "OME-TIFF contains incomplete PhysicalSizeX/PhysicalSizeY metadata; "
                "TIFF resolution fallback will not be used. Supply an explicit pixel-size "
                "override for conversion."
            )
            return None

        fallback = consistent_tiff_resolution_pixel_size(self._level_pages(0))
        if fallback is not None:
            LOGGER.warning(
                "OME-TIFF does not provide PhysicalSizeX/PhysicalSizeY metadata; "
                "using TIFF XResolution/YResolution/ResolutionUnit tags (%s).",
                fallback.to_tuple(),
            )
        return fallback

    def pixel_size_at_level(self, level: int) -> PixelSize | None:
        selected = self._selected_level(level)
        base_size = self.pixel_size
        if base_size is None:
            return None
        base_shape = dict(zip(self.axes, self.shape))
        level_shape = dict(zip(str(selected.axes), tuple(int(item) for item in selected.shape)))
        if not {"X", "Y"}.issubset(base_shape) or not {"X", "Y"}.issubset(level_shape):
            raise ValueError("Cannot calculate level pixel size without X and Y axes")
        return base_size.scaled(
            base_shape["X"] / level_shape["X"],
            base_shape["Y"] / level_shape["Y"],
        )

    def _selected_level(self, level: int) -> tifffile.TiffPageSeries:
        level_index = int(level)
        if level_index < 0 or level_index >= len(self.levels):
            raise IndexError(
                f"Pyramid level {level_index} does not exist; series has {len(self.levels)} levels"
            )
        return self.levels[level_index]

    def _level_pages(self, level: int) -> list[tifffile.TiffPage]:
        return [item.aspage() for item in self._selected_level(level).pages]

    def _region_reader(
        self,
        page: tifffile.TiffPage,
        *,
        level: int,
        page_index: int,
    ) -> TiffPlaneReader:
        """Reuse decoded TIFF segments across repeated regional reads."""

        self._require_open()
        key = (int(level), int(page_index))
        reader = self._region_readers.get(key)
        if reader is None:
            reader = TiffPlaneReader(page, lock=self._read_lock)
            self._region_readers[key] = reader
        return reader

    def asarray(self, *, level: int = 0) -> np.ndarray:
        """Materialize one pyramid level.

        Whole-slide arrays can be extremely large. Prefer :meth:`read_region`
        when only a spatial window is needed.
        """

        return np.asarray(self._selected_level(level).asarray())

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

        pages = self._level_pages(level)
        if "C" in axes:
            size_c = axis_sizes["C"]
            selected_channels = normalize_channel_indices(channels, size_c)
            if len(pages) != size_c:
                raise NotImplementedError(
                    f"CYX region reading expects one TIFF page per channel; found "
                    f"{len(pages)} pages for C={size_c}"
                )
            readers = [
                self._region_reader(
                    pages[index],
                    level=level,
                    page_index=index,
                )
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
        reader = self._region_reader(
            pages[0],
            level=level,
            page_index=0,
        )
        result = reader.read_region(y0, y1, x0, x1)
        if "S" in axes:
            selected_samples = normalize_channel_indices(channels, axis_sizes["S"])
            return np.ascontiguousarray(result[..., selected_samples])
        if channels is not None:
            selected = normalize_channel_indices(channels, 1)
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
    """Context-managed reader for one multichannel or RGB OME-TIFF series."""

    input_type_description = "OME-TIFF"

    def __init__(self, path: str | Path, *, series: int = 0) -> None:
        super().__init__(path, series=series)
        self._channels: tuple[Channel, ...] | None = None

    def open(self) -> "OMETiffReader":
        super().open()
        self._channels = None
        return self

    def close(self) -> None:
        if self._channels is not None:
            for channel in self._channels:
                channel.clear_cache()
        self._channels = None
        super().close()

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

    def _logical_channel_count_from_layout(self) -> int:
        if "C" in self.axes:
            return int(self.shape[self.axes.index("C")])
        return 1

    def _channel_page(self, channel_index: int, level: int) -> tifffile.TiffPage:
        selected = self._selected_level(level)
        axes = str(selected.axes)
        pages = self._level_pages(level)
        if "C" in axes:
            size_c = int(selected.shape[axes.index("C")])
            if len(pages) != size_c:
                raise NotImplementedError(
                    f"OME planar level {level} has {len(pages)} pages for C={size_c}"
                )
            return pages[channel_index]
        if channel_index != 0 or len(pages) != 1:
            raise IndexError(f"Logical channel {channel_index} is unavailable at level {level}")
        return pages[0]

    def _channel_plane_reader(self, channel_index: int, level: int) -> TiffPlaneReader:
        selected = self._selected_level(level)
        page_index = channel_index if "C" in str(selected.axes) else 0
        return self._region_reader(
            self._channel_page(channel_index, level),
            level=level,
            page_index=page_index,
        )

    def _channel_array(self, channel_index: int, level: int) -> np.ndarray:
        page = self._channel_page(channel_index, level)
        value = np.asarray(page.asarray())
        if int(page.samplesperpixel) == 1:
            if value.ndim == 3 and value.shape[0] == 1:
                value = value[0]
            if value.ndim == 3 and value.shape[-1] == 1:
                value = value[..., 0]
            if value.ndim != 2:
                raise ValueError(
                    f"OME channel {channel_index} did not materialize as YX: {value.shape}"
                )
        elif value.ndim != 3 or value.shape[-1] != int(page.samplesperpixel):
            raise ValueError(
                f"OME channel {channel_index} did not materialize as YXS: {value.shape}"
            )
        return np.ascontiguousarray(value)

    @property
    def channels(self) -> tuple[Channel, ...]:
        self._require_open()
        if self._channels is None:
            image = self._ome_image_summary()
            summaries = list(image.get("channels", [])) if image is not None else []
            logical_count = len(summaries) or self._logical_channel_count_from_layout()
            pages = self._level_pages(0)
            if "C" in self.axes and logical_count != len(pages):
                raise ValueError(
                    f"OME metadata declares {logical_count} logical channels, but the "
                    f"selected series has {len(pages)} full-resolution pages"
                )
            if "C" not in self.axes and logical_count != 1:
                raise ValueError(
                    f"OME series axes {self.axes!r} can expose one logical channel, but "
                    f"metadata contains {logical_count}"
                )

            axis_sizes = dict(zip(self.axes, self.shape))
            channel_shape: tuple[int, ...]
            if "S" in self.axes:
                channel_shape = (
                    int(axis_sizes["Y"]),
                    int(axis_sizes["X"]),
                    int(axis_sizes["S"]),
                )
            else:
                channel_shape = (int(axis_sizes["Y"]), int(axis_sizes["X"]))

            result: list[Channel] = []
            for index in range(logical_count):
                summary = summaries[index] if index < len(summaries) else {}
                source_id = summary.get("id")
                channel_id = str(source_id) if source_id else f"Channel:0:{index}"
                page = pages[index] if "C" in self.axes else pages[0]
                samples_per_pixel = int(
                    summary.get("samples_per_pixel")
                    or page.samplesperpixel
                    or 1
                )
                name = summary.get("name")
                if not name and logical_count == 1 and samples_per_pixel == 3:
                    name = "RGB"
                if not name:
                    # Keep a source Channel ID as provenance, not as a synthesized
                    # display name. IDs can contain arbitrary source vocabulary and
                    # should not leak into a minimized output merely because Name is absent.
                    name = f"Channel {index + 1}"
                result.append(
                    Channel(
                        index=index,
                        channel_id=channel_id,
                        name=str(name),
                        dtype=page.dtype,
                        shape=channel_shape,
                        samples_per_pixel=samples_per_pixel,
                        plane_reader_factory=(
                            lambda level, channel_index=index: self._channel_plane_reader(
                                channel_index,
                                level,
                            )
                        ),
                        array_reader=(
                            lambda level, channel_index=index: self._channel_array(
                                channel_index,
                                level,
                            )
                        ),
                        ensure_available=lambda: self._require_open(),
                        source_id=None if source_id is None else str(source_id),
                        id_is_generated=source_id is None,
                        source_metadata=summary,
                    )
                )
            self._channels = tuple(result)
        return self._channels

    @property
    def samples_per_pixel(self) -> int:
        values = {channel.samples_per_pixel for channel in self.channels}
        return values.pop() if len(values) == 1 else 1

    @property
    def sample_count(self) -> int:
        return self.size_c

    @property
    def is_rgb(self) -> bool:
        return self.samples_per_pixel == 3 and self.size_c == 3 and len(self.channels) == 1

    @property
    def sample_names(self) -> tuple[str, ...]:
        if self.is_rgb:
            return ("Red", "Green", "Blue")
        if self.samples_per_pixel == 1:
            return self.channel_names
        return tuple(f"Sample {index + 1}" for index in range(self.sample_count))

    @property
    def image_type(self) -> str:
        return "rgb" if self.is_rgb else "multichannel"

    def _normalized_output_layout(self) -> tuple[str, tuple[int, ...]]:
        axis_sizes = dict(zip(self.axes, self.shape))
        if "Y" not in axis_sizes or "X" not in axis_sizes:
            raise NotImplementedError(
                f"OME-TIFF conversion requires spatial Y/X axes, found {self.axes!r}"
            )
        unsupported = {
            axis: size
            for axis, size in axis_sizes.items()
            if axis not in {"C", "S", "Y", "X"} and size != 1
        }
        if unsupported:
            raise NotImplementedError(
                "OME-TIFF normalization currently supports singleton Z/T and other "
                f"non-spatial axes only; found {unsupported} in axes {self.axes!r}"
            )
        height = int(axis_sizes["Y"])
        width = int(axis_sizes["X"])
        if self.is_rgb:
            if int(axis_sizes.get("S", 0)) != 3:
                raise NotImplementedError(
                    f"RGB OME-TIFF normalization requires YXS storage, found {self.axes!r}"
                )
            return "YXS", (height, width, 3)
        if "S" in axis_sizes and int(axis_sizes["S"]) != 1:
            raise NotImplementedError(
                "Non-RGB OME-TIFFs with multiple stored samples per pixel are unsupported"
            )
        if "C" in axis_sizes:
            return "CYX", (len(self.channels), height, width)
        return "YX", (height, width)

    @property
    def output_axes(self) -> str:
        return self._normalized_output_layout()[0]

    @property
    def output_shape(self) -> tuple[int, ...]:
        return self._normalized_output_layout()[1]

    @property
    def icc_profile(self) -> bytes | None:
        self._require_open()
        if not self.is_rgb:
            return None
        pages = self._level_pages(0)
        if len(pages) != 1:
            return None
        profile = pages[0].iccprofile
        return None if profile is None else bytes(profile)

    @property
    def source_channel_metadata(self) -> tuple[dict[str, Any], ...]:
        return tuple(dict(channel.source_metadata) for channel in self.channels)

    def plane_readers(self, *, cache_mib: int = 64) -> list[TiffPlaneReader]:
        self._require_open()
        if self.output_axes == "YXS":
            return [
                TiffPlaneReader(
                    self._channel_page(0, 0),
                    lock=self._read_lock,
                    cache_mib=cache_mib,
                )
            ]
        if self.output_axes == "YX":
            return [
                TiffPlaneReader(
                    self._channel_page(0, 0),
                    lock=self._read_lock,
                    cache_mib=cache_mib,
                )
            ]
        return [
            TiffPlaneReader(
                self._channel_page(index, 0),
                lock=self._read_lock,
                cache_mib=cache_mib,
            )
            for index in range(len(self.channels))
        ]

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
        interleaved ``YXS`` RGB images it retains the historical sample-selection
        behavior; logical RGB channel access is available through ``reader[0]``.
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
    """Virtual-access reader for one integer OME-TIFF label raster."""

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
