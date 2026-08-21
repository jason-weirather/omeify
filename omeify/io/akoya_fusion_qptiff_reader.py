from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import tifffile

from omeify.inspection import TiffInspector

from .akoya_qptiff import (
    AkoyaQPIChannelMetadata,
    ChannelNameField,
    consistent_akoya_pixel_size,
    parse_akoya_qpi_channel_metadata,
    select_akoya_channel_name,
)
from .base import ChannelSelection, MultichannelImage
from .channel import Channel, normalize_channel_indices
from .pixel_size import PixelSize
from .tiff import TiffPlaneReader


class AkoyaFusionQPTiffReader(MultichannelImage):
    """Context-managed, lazy reader for planar Akoya Fusion multiplex QPTIFF.

    Both vendor ``Name`` and ``Biomarker`` values are retained in each
    :class:`Channel` object's ``source_metadata``. ``channel_name_field`` only
    selects which field becomes the normalized logical channel name.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        channel_name_field: ChannelNameField = "auto",
    ) -> None:
        if channel_name_field not in {"name", "biomarker", "auto"}:
            raise ValueError("channel_name_field must be 'name', 'biomarker', or 'auto'")
        self._path = Path(path)
        self.series_index = int(series)
        if self.series_index < 0:
            raise ValueError("series must be zero or greater")
        self.channel_name_field: ChannelNameField = channel_name_field
        self._tiff: tifffile.TiffFile | None = None
        self._inspection: TiffInspector | None = None
        self._read_lock = threading.RLock()
        self._channels: tuple[Channel, ...] | None = None
        self._channel_metadata: tuple[AkoyaQPIChannelMetadata, ...] = ()
        self._pixel_size: PixelSize | None = None

    @property
    def path(self) -> Path:
        return self._path

    @property
    def is_open(self) -> bool:
        return self._tiff is not None

    def open(self) -> "AkoyaFusionQPTiffReader":
        if self._tiff is not None:
            return self
        tiff = tifffile.TiffFile(self.path)
        try:
            if not tiff.is_qpi:
                raise ValueError(f"TIFF is not recognized as Akoya/PerkinElmer QPI: {self.path}")
            if self.series_index >= len(tiff.series):
                raise IndexError(
                    f"Series {self.series_index} does not exist; QPTIFF has "
                    f"{len(tiff.series)} series"
                )
            self._tiff = tiff
            pages = self._level_pages(0)
            if not pages:
                raise ValueError("Selected Fusion QPTIFF series contains no channel pages")
            height = int(pages[0].imagelength)
            width = int(pages[0].imagewidth)
            dtype = np.dtype(pages[0].dtype).newbyteorder("=")
            for index, page in enumerate(pages):
                if int(page.samplesperpixel) != 1:
                    raise ValueError(
                        "Fusion multiplex QPTIFF requires one grayscale sample per channel page; "
                        f"channel {index} has SamplesPerPixel={int(page.samplesperpixel)}"
                    )
                if (int(page.imagelength), int(page.imagewidth)) != (height, width):
                    raise ValueError(
                        "Fusion channel pages must have matching full-resolution shapes; "
                        f"channel 0={(height, width)}, channel {index}="
                        f"{(int(page.imagelength), int(page.imagewidth))}"
                    )
                if np.dtype(page.dtype).newbyteorder("=") != dtype:
                    raise TypeError(
                        "Fusion channel pages must have matching dtypes; "
                        f"channel 0={dtype}, channel {index}={page.dtype}"
                    )
            metadata = tuple(
                parse_akoya_qpi_channel_metadata(
                    page.description,
                    channel_index=index,
                )
                for index, page in enumerate(pages)
            )
            # Validate the explicitly selected naming policy while opening the
            # file. Metadata access should fail early when, for example,
            # ``channel_name_field="biomarker"`` is requested but a page does
            # not contain a Biomarker value.
            for index, item in enumerate(metadata):
                select_akoya_channel_name(
                    item,
                    self.channel_name_field,
                    channel_index=index,
                )
            self._channel_metadata = metadata
            self._pixel_size = consistent_akoya_pixel_size(metadata, pages)
            self._channels = None
            self._inspection = None
        except Exception:
            tiff.close()
            self._tiff = None
            self._channel_metadata = ()
            self._pixel_size = None
            raise
        return self

    def close(self) -> None:
        if self._channels is not None:
            for channel in self._channels:
                channel.clear_cache()
        if self._tiff is not None:
            self._tiff.close()
        self._tiff = None
        self._inspection = None
        self._channels = None
        self._channel_metadata = ()
        self._pixel_size = None

    def __enter__(self) -> "AkoyaFusionQPTiffReader":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def _require_open(self) -> tifffile.TiffFile:
        if self._tiff is None:
            raise RuntimeError(
                "AkoyaFusionQPTiffReader must be opened with 'with' or open()"
            )
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

    def _selected_level(self, level: int) -> tifffile.TiffPageSeries:
        level_index = int(level)
        if level_index < 0 or level_index >= len(self.levels):
            raise IndexError(
                f"Pyramid level {level_index} does not exist; series has {len(self.levels)} levels"
            )
        return self.levels[level_index]

    def _level_pages(self, level: int) -> list[tifffile.TiffPage]:
        selected = self._selected_level(level)
        return [item.aspage() for item in selected.pages]

    @property
    def axes(self) -> str:
        return str(self.series.axes)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.series.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.series.dtype).newbyteorder("=")

    @property
    def pixel_size(self) -> PixelSize | None:
        self._require_open()
        return self._pixel_size

    @property
    def source_channel_metadata(self) -> tuple[dict[str, str | float | None], ...]:
        self._require_open()
        return tuple(item.as_dict() for item in self._channel_metadata)

    def _plane_reader(self, channel_index: int, level: int) -> TiffPlaneReader:
        pages = self._level_pages(level)
        if len(pages) != len(self._channel_metadata):
            raise NotImplementedError(
                f"Fusion pyramid level {level} has {len(pages)} pages, but the base level "
                f"has {len(self._channel_metadata)} channels"
            )
        return TiffPlaneReader(pages[channel_index], lock=self._read_lock)

    def _channel_array(self, channel_index: int, level: int) -> np.ndarray:
        pages = self._level_pages(level)
        if channel_index >= len(pages):
            raise IndexError(f"Channel {channel_index} is unavailable at pyramid level {level}")
        value = np.asarray(pages[channel_index].asarray())
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        if value.ndim == 3 and value.shape[-1] == 1:
            value = value[..., 0]
        if value.ndim != 2:
            raise ValueError(
                f"Fusion channel {channel_index} did not materialize as YX: {value.shape}"
            )
        return np.ascontiguousarray(value)

    @property
    def channels(self) -> tuple[Channel, ...]:
        self._require_open()
        if self._channels is None:
            pages = self._level_pages(0)
            result: list[Channel] = []
            for index, (page, metadata) in enumerate(zip(pages, self._channel_metadata)):
                normalized_name, selected_field = select_akoya_channel_name(
                    metadata,
                    self.channel_name_field,
                    channel_index=index,
                )
                source_metadata = metadata.as_dict()
                source_metadata["selected_name_field"] = selected_field
                result.append(
                    Channel(
                        index=index,
                        channel_id=f"Channel:0:{index}",
                        name=normalized_name,
                        dtype=page.dtype,
                        shape=(int(page.imagelength), int(page.imagewidth)),
                        samples_per_pixel=1,
                        plane_reader_factory=(
                            lambda level, channel_index=index: self._plane_reader(
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
                        source_id=None,
                        id_is_generated=True,
                        source_metadata=source_metadata,
                    )
                )
            self._channels = tuple(result)
        return self._channels

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
        selected = normalize_channel_indices(channels, len(self.channels))
        return np.stack(
            [self.channels[index].read_region(y0, y1, x0, x1, level=level) for index in selected],
            axis=0,
        )

    def asarray(self, *, level: int = 0) -> np.ndarray:
        return np.stack([channel.asarray(level=level) for channel in self.channels], axis=0)

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

    def __str__(self) -> str:
        return self.inspection.render_text()

    def __repr__(self) -> str:
        if self.is_open:
            return self.inspection.render_text()
        return (
            f"AkoyaFusionQPTiffReader(path={str(self.path)!r}, "
            f"series={self.series_index}, channel_name_field={self.channel_name_field!r}, "
            "closed=True)"
        )
