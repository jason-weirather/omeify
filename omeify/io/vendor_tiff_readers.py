from __future__ import annotations

import logging
import math
import re
import threading
from pathlib import Path
from typing import Any, Literal, Mapping
from xml.etree import ElementTree

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
from .indica_mif import (
    IndicaMIFMetadata,
    build_indica_mif_series,
    parse_indica_mif_metadata,
)
from .pixel_size import (
    PixelSize,
    consistent_tiff_resolution_pixel_size,
)
from .tiff import TiffPlaneReader

LOGGER = logging.getLogger(__name__)

SourceProfile = Literal[
    "akoya_mif_qptiff",
    "akoya_fusion_qptiff",
    "akoya_he_qptiff",
    "svs",
    "component",
    "indica_mif",
]

_SUPPORTED_PLANAR_DTYPES = {
    "int8",
    "int16",
    "int32",
    "uint8",
    "uint16",
    "uint32",
    "float32",
    "float64",
}
_SUPPORTED_RGB_DTYPES = {"uint8"}
_APERIO_MPP_PATTERN = re.compile(
    r"(?:^|\|)\s*MPP\s*=\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)",
    flags=re.IGNORECASE,
)
_XML_ENCODING_DECLARATION = re.compile(
    r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2",
    flags=re.IGNORECASE,
)


def _local_name(tag: str) -> str:
    return str(tag).rsplit("}", 1)[-1]


def _parse_xml(xml: str | bytes | None) -> ElementTree.Element | None:
    if not xml:
        return None
    payload: str | bytes
    if isinstance(xml, bytes):
        payload = xml
    else:
        # tifffile may already have decoded the tag while leaving a stale
        # encoding declaration such as encoding="utf-16" in the text.
        payload = _XML_ENCODING_DECLARATION.sub(r"\1", str(xml), count=1)
    try:
        return ElementTree.fromstring(payload)
    except (ElementTree.ParseError, ValueError, TypeError):
        return None


def _first_child_text(root: ElementTree.Element, local_name: str) -> str | None:
    for child in root:
        if _local_name(child.tag) == local_name and child.text:
            value = child.text.strip()
            if value:
                return value
    return None


def _first_descendant_text(root: ElementTree.Element, local_name: str) -> str | None:
    for element in root.iter():
        if _local_name(element.tag) == local_name and element.text:
            value = element.text.strip()
            if value:
                return value
    return None


class _VendorTiffReader(MultichannelImage):
    """Shared lazy reader for the TIFF-family source profiles supported by omeify.

    Concrete subclasses define source-format interpretation only. Pixel access is
    always delegated to :class:`TiffPlaneReader`, so the same region-based I/O
    contract is consumed by :class:`OMETiffWriter` during conversion.
    """

    profile: SourceProfile
    input_type_description: str
    image_type: Literal["multichannel", "rgb"] = "multichannel"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
        channel_name_field: ChannelNameField = "name",
    ) -> None:
        if channel_name_field not in {"name", "biomarker", "auto"}:
            raise ValueError("channel_name_field must be 'name', 'biomarker', or 'auto'")
        self._path = Path(path)
        self.series_index = int(series)
        if self.series_index < 0:
            raise ValueError("series must be zero or greater")
        self.pixel_size_override = pixel_size
        self.channel_name_field: ChannelNameField = channel_name_field
        self._tiff: tifffile.TiffFile | None = None
        self._series: tifffile.TiffPageSeries | None = None
        self._level0: tifffile.TiffPageSeries | None = None
        self._pages: list[tifffile.TiffPage] = []
        self._channels: tuple[Channel, ...] | None = None
        self._pixel_size: PixelSize | None = None
        self._icc_profile: bytes | None = None
        self._source_channel_metadata: tuple[Mapping[str, Any], ...] = ()
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
            self._validate_container(tiff)
            series, level0, pages = self._select_series(tiff)

            self._tiff = tiff
            self._series = series
            self._level0 = level0
            self._pages = pages
            if self.image_type == "rgb":
                self._inspect_rgb()
            else:
                self._inspect_planar()
            self._channels = None
            self._inspection = None
            return self
        except Exception:
            tiff.close()
            self._reset()
            raise

    def close(self) -> None:
        if self._channels is not None:
            for channel in self._channels:
                channel.clear_cache()
        if self._tiff is not None:
            self._tiff.close()
        self._reset()

    def _reset(self) -> None:
        self._tiff = None
        self._series = None
        self._level0 = None
        self._pages = []
        self._channels = None
        self._pixel_size = None
        self._icc_profile = None
        self._source_channel_metadata = ()
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
        self._require_open()
        assert self._series is not None
        return self._series

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
    def shape(self) -> tuple[int, ...]:
        return tuple(int(item) for item in self.series.shape)

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.series.dtype).newbyteorder("=")

    @property
    def source_byte_order(self) -> str:
        return "big" if self.tiff.byteorder == ">" else "little"

    @property
    def pixel_size(self) -> PixelSize | None:
        self._require_open()
        return self._pixel_size

    @property
    def icc_profile(self) -> bytes | None:
        self._require_open()
        return self._icc_profile

    @property
    def source_channel_metadata(self) -> tuple[Mapping[str, Any], ...]:
        self._require_open()
        return self._source_channel_metadata

    @property
    def output_axes(self) -> str:
        return "YXS" if self.image_type == "rgb" else "CYX"

    @property
    def output_shape(self) -> tuple[int, ...]:
        if self.image_type == "rgb":
            page = self._pages[0]
            return (int(page.imagelength), int(page.imagewidth), 3)
        page = self._pages[0]
        return (len(self._pages), int(page.imagelength), int(page.imagewidth))

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
            self._inspection = TiffInspector.from_tiff(tiff, file_path=self.path, detail=1)
        return self._inspection

    def _validate_container(self, tiff: tifffile.TiffFile) -> None:
        if self.profile in {"akoya_mif_qptiff", "akoya_fusion_qptiff", "akoya_he_qptiff"}:
            if not tiff.is_qpi:
                raise ValueError(
                    f"{type(self).__name__} requires an Akoya/PerkinElmer QPI TIFF source"
                )
        elif self.profile == "svs" and not tiff.is_svs:
            raise ValueError("AperioSVSReader requires an Aperio SVS source")

    def _select_series(
        self,
        tiff: tifffile.TiffFile,
    ) -> tuple[tifffile.TiffPageSeries, tifffile.TiffPageSeries, list[tifffile.TiffPage]]:
        if self.series_index >= len(tiff.series):
            raise IndexError(
                f"Series {self.series_index} does not exist; file has {len(tiff.series)} series"
            )
        series = tiff.series[self.series_index]
        levels = tuple(getattr(series, "levels", ()) or ())
        level0 = levels[0] if levels else series
        pages = [item.aspage() for item in level0.pages]
        if not pages:
            raise ValueError("Selected TIFF series contains no readable full-resolution pages")
        return series, level0, pages

    def _inspect_planar(self) -> None:
        page0 = self._pages[0]
        height = int(page0.imagelength)
        width = int(page0.imagewidth)
        dtype = np.dtype(page0.dtype).newbyteorder("=")
        if dtype.name not in _SUPPORTED_PLANAR_DTYPES:
            supported = ", ".join(sorted(_SUPPORTED_PLANAR_DTYPES))
            raise TypeError(
                f"Unsupported planar source dtype {dtype}. Supported dtypes are: {supported}. "
                "omeify does not cast unsupported pixel data."
            )
        for index, page in enumerate(self._pages):
            if (int(page.imagelength), int(page.imagewidth)) != (height, width):
                raise ValueError(
                    "All planar channel pages must have the same full-resolution shape; "
                    f"channel 0={(height, width)}, channel {index}="
                    f"{(int(page.imagelength), int(page.imagewidth))}."
                )
            if np.dtype(page.dtype).newbyteorder("=") != dtype:
                raise TypeError(
                    "All planar channel pages must have the same dtype; "
                    f"channel 0={dtype}, channel {index}={page.dtype}."
                )
            if int(page.samplesperpixel) != 1:
                raise ValueError(
                    "Planar multiplex input requires SamplesPerPixel=1 for every channel page"
                )

        axis_sizes = dict(zip(str(self._level0.axes), tuple(int(v) for v in self._level0.shape)))
        for axis, size in axis_sizes.items():
            if axis not in {"X", "Y", "C", "I", "Q"} and size != 1:
                raise ValueError(
                    f"Unsupported non-singleton axis {axis}={size} in planar input "
                    f"with axes {self._level0.axes!r}"
                )

        metadata: list[dict[str, Any]] = []
        names: list[str] = []
        if self.profile == "akoya_fusion_qptiff":
            parsed = tuple(
                parse_akoya_qpi_channel_metadata(page.description, channel_index=index)
                for index, page in enumerate(self._pages)
            )
            for index, item in enumerate(parsed):
                normalized_name, selected_field = select_akoya_channel_name(
                    item,
                    self.channel_name_field,
                    channel_index=index,
                )
                source = item.as_dict()
                source["selected_name_field"] = selected_field
                names.append(normalized_name)
                metadata.append(source)
            pixel_size = consistent_akoya_pixel_size(parsed, self._pages)
            if pixel_size is not None and any(
                item.pixel_size_microns is None for item in parsed
            ):
                self._warn_tiff_resolution_fallback(
                    "Akoya PixelSizeMicrons",
                    pixel_size=pixel_size,
                )
            self._pixel_size = self.pixel_size_override or pixel_size
        elif self.profile == "indica_mif":
            metadata_value = getattr(self, "_indica_metadata", None)
            if not isinstance(metadata_value, IndicaMIFMetadata):
                raise RuntimeError("Indica metadata was not initialized")
            if len(metadata_value.channels) != len(self._pages):
                raise ValueError(
                    "Indica channel metadata count does not match the full-resolution IFD map"
                )
            for channel in metadata_value.channels:
                names.append(channel.name)
                metadata.append(channel.as_dict())
            pixel_size = consistent_tiff_resolution_pixel_size(self._pages)
            self._pixel_size = self.pixel_size_override or pixel_size
        elif self.profile == "component":
            pixel_size = consistent_tiff_resolution_pixel_size(self._pages)
            self._pixel_size = self.pixel_size_override or pixel_size
            for index, page in enumerate(self._pages):
                name = self._tolerant_akoya_name(page) or f"Channel {index + 1}"
                names.append(name)
                metadata.append({"name": name})
        else:
            for index, page in enumerate(self._pages):
                name = self._tolerant_akoya_name(page) or f"Channel {index + 1}"
                names.append(name)
                metadata.append({"name": name})
            source_pixel_size = self._akoya_description_pixel_size(page0)
            if source_pixel_size is None:
                source_pixel_size = self._warn_tiff_resolution_fallback(
                    "Akoya PixelSizeMicrons"
                )
            self._pixel_size = self.pixel_size_override or source_pixel_size

        self._source_channel_metadata = tuple(metadata)
        self._normalized_names = tuple(names)

    def _inspect_rgb(self) -> None:
        if len(self._pages) != 1:
            raise ValueError(
                "RGB input expects exactly one full-resolution TIFF page in the selected "
                f"series; found {len(self._pages)}"
            )
        page = self._pages[0]
        dtype = np.dtype(page.dtype).newbyteorder("=")
        if dtype.name not in _SUPPORTED_RGB_DTYPES:
            raise TypeError(f"RGB input requires uint8 pixels, found {dtype}")
        if int(page.samplesperpixel) != 3:
            raise ValueError(
                f"RGB input requires SamplesPerPixel=3, found {int(page.samplesperpixel)}"
            )
        if int(page.planarconfig) != 1:
            raise ValueError("RGB input must use contiguous samples (PlanarConfiguration=1)")
        if int(page.photometric) not in {2, 6}:
            raise ValueError(
                "RGB input requires RGB or YCbCr photometric interpretation; "
                f"found {int(page.photometric)}"
            )
        axes = str(self._level0.axes)
        shape = tuple(int(item) for item in self._level0.shape)
        expected = (int(page.imagelength), int(page.imagewidth), 3)
        if axes != "YXS" or shape != expected:
            raise ValueError(
                "RGB input must resolve to one YXS series with shape (Y, X, 3); "
                f"found axes={axes!r}, shape={shape}"
            )

        if self.profile == "svs":
            source_pixel_size = self._svs_description_pixel_size(page)
            if source_pixel_size is None:
                source_pixel_size = self._warn_tiff_resolution_fallback("Aperio MPP")
        else:
            source_pixel_size = self._akoya_description_pixel_size(page)
            if source_pixel_size is None:
                source_pixel_size = self._warn_tiff_resolution_fallback(
                    "Akoya PixelSizeMicrons"
                )
        self._pixel_size = self.pixel_size_override or source_pixel_size
        self._icc_profile = bytes(page.iccprofile) if page.iccprofile is not None else None
        self._normalized_names = ("RGB",)
        self._source_channel_metadata = ({"name": "RGB"},)

    @staticmethod
    def _tolerant_akoya_name(page: tifffile.TiffPage) -> str | None:
        root = _parse_xml(page.description)
        if root is None:
            return None
        return _first_child_text(root, "Name") or _first_descendant_text(root, "Name")

    @staticmethod
    def _akoya_description_pixel_size(page: tifffile.TiffPage) -> PixelSize | None:
        try:
            parsed = parse_akoya_qpi_channel_metadata(page.description, channel_index=0)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.pixel_size_microns is not None:
            value = parsed.pixel_size_microns
            return PixelSize(value, value, "µm")
        root = _parse_xml(page.description)
        raw = _first_descendant_text(root, "PixelSizeMicrons") if root is not None else None
        if raw is not None:
            value = float(raw)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid Akoya PixelSizeMicrons={raw!r}")
            return PixelSize(value, value, "µm")
        return None

    @staticmethod
    def _svs_description_pixel_size(page: tifffile.TiffPage) -> PixelSize | None:
        match = _APERIO_MPP_PATTERN.search(page.description or "")
        if match is not None:
            value = float(match.group("value"))
            return PixelSize(value, value, "µm")
        return None

    def _warn_tiff_resolution_fallback(
        self,
        expected_metadata: str,
        *,
        pixel_size: PixelSize | None = None,
    ) -> PixelSize | None:
        if pixel_size is None:
            pixel_size = consistent_tiff_resolution_pixel_size(self._pages)
        if pixel_size is not None:
            LOGGER.warning(
                "%s does not provide expected %s calibration; using TIFF "
                "XResolution/YResolution/ResolutionUnit tags (%s).",
                self.input_type_description,
                expected_metadata,
                pixel_size.to_tuple(),
            )
        return pixel_size

    def _selected_level(self, level: int) -> tifffile.TiffPageSeries:
        level_index = int(level)
        if level_index < 0 or level_index >= len(self.levels):
            raise IndexError(
                f"Pyramid level {level_index} does not exist; series has {len(self.levels)} levels"
            )
        return self.levels[level_index]

    def _level_pages(self, level: int) -> list[tifffile.TiffPage]:
        return [item.aspage() for item in self._selected_level(level).pages]

    def _channel_page(self, channel_index: int, level: int) -> tifffile.TiffPage:
        pages = self._level_pages(level)
        if self.image_type == "rgb":
            if channel_index != 0 or len(pages) != 1:
                raise IndexError(
                    f"RGB logical channel {channel_index} is unavailable at level {level}"
                )
            return pages[0]
        if len(pages) != len(self._pages):
            raise NotImplementedError(
                f"Pyramid level {level} has {len(pages)} pages, but the base level has "
                f"{len(self._pages)} logical channels"
            )
        return pages[channel_index]

    def _channel_plane_reader(
        self,
        channel_index: int,
        level: int,
        *,
        cache_mib: int = 64,
    ) -> TiffPlaneReader:
        return TiffPlaneReader(
            self._channel_page(channel_index, level),
            lock=self._read_lock,
            cache_mib=cache_mib,
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
                    f"Channel {channel_index} did not materialize as YX: {value.shape}"
                )
        elif value.ndim != 3 or value.shape[-1] != int(page.samplesperpixel):
            raise ValueError(
                f"RGB channel {channel_index} did not materialize as YXS: {value.shape}"
            )
        return np.ascontiguousarray(value)

    @property
    def channels(self) -> tuple[Channel, ...]:
        self._require_open()
        if self._channels is None:
            result: list[Channel] = []
            for index, (name, source_metadata) in enumerate(
                zip(self._normalized_names, self._source_channel_metadata)
            ):
                page = self._pages[index] if self.image_type != "rgb" else self._pages[0]
                shape: tuple[int, ...]
                if self.image_type == "rgb":
                    shape = (int(page.imagelength), int(page.imagewidth), 3)
                else:
                    shape = (int(page.imagelength), int(page.imagewidth))
                source_id_value = source_metadata.get("id")
                source_id = None if source_id_value is None else str(source_id_value)
                result.append(
                    Channel(
                        index=index,
                        channel_id=f"Channel:0:{index}",
                        name=name,
                        dtype=page.dtype,
                        shape=shape,
                        samples_per_pixel=3 if self.image_type == "rgb" else 1,
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
                        source_id=source_id,
                        id_is_generated=True,
                        source_metadata=source_metadata,
                    )
                )
            self._channels = tuple(result)
        return self._channels

    @property
    def is_rgb(self) -> bool:
        return self.image_type == "rgb"

    @property
    def sample_names(self) -> tuple[str, ...]:
        return ("Red", "Green", "Blue") if self.is_rgb else self.channel_names

    def plane_readers(self, *, cache_mib: int = 64) -> list[TiffPlaneReader]:
        self._require_open()
        if self.image_type == "rgb":
            return [self._channel_plane_reader(0, 0, cache_mib=cache_mib)]
        return [
            self._channel_plane_reader(index, 0, cache_mib=cache_mib)
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
        if self.image_type == "rgb":
            if channels is not None:
                selected = normalize_channel_indices(channels, 1)
                if selected != [0]:
                    raise IndexError("RGB logical-channel selection only accepts channel 0")
            return self.channels[0].read_region(y0, y1, x0, x1, level=level)
        selected = normalize_channel_indices(channels, len(self.channels))
        return np.stack(
            [self.channels[index].read_region(y0, y1, x0, x1, level=level) for index in selected],
            axis=0,
        )

    def asarray(self, *, level: int = 0) -> np.ndarray:
        if self.image_type == "rgb":
            return self.channels[0].asarray(level=level)
        return np.stack([channel.asarray(level=level) for channel in self.channels], axis=0)

    def __str__(self) -> str:
        return self.inspection.render_text()

    def __repr__(self) -> str:
        if self.is_open:
            return self.inspection.render_text()
        return (
            f"{type(self).__name__}(path={str(self.path)!r}, "
            f"series={self.series_index}, closed=True)"
        )


class AkoyaMIFQPTiffReader(_VendorTiffReader):
    profile = "akoya_mif_qptiff"
    input_type_description = "Akoya mIF QPTIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(path, series=series, pixel_size=pixel_size)


class AkoyaFusionQPTiffReader(_VendorTiffReader):
    profile = "akoya_fusion_qptiff"
    input_type_description = "Akoya Fusion multiplex QPTIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        channel_name_field: ChannelNameField = "auto",
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(
            path,
            series=series,
            pixel_size=pixel_size,
            channel_name_field=channel_name_field,
        )


class IndicaMIFTiffReader(_VendorTiffReader):
    profile = "indica_mif"
    input_type_description = "Indica Labs/HALO mIF TIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(path, series=series, pixel_size=pixel_size)
        self._indica_metadata: IndicaMIFMetadata | None = None

    def _reset(self) -> None:
        super()._reset()
        self._indica_metadata = None

    def _validate_container(self, tiff: tifffile.TiffFile) -> None:
        self._indica_metadata = parse_indica_mif_metadata(tiff.pages.first.description)

    def _select_series(
        self,
        tiff: tifffile.TiffFile,
    ) -> tuple[tifffile.TiffPageSeries, tifffile.TiffPageSeries, list[tifffile.TiffPage]]:
        if self.series_index != 0:
            raise IndexError("Indica mIF TIFF contains one XML-defined image; series must be 0")
        if self._indica_metadata is None:
            raise RuntimeError("Indica metadata was not initialized")
        series = build_indica_mif_series(tiff, self._indica_metadata)
        pages = [item.aspage() for item in series.pages]
        return series, series, pages


class AkoyaHEQPTiffReader(_VendorTiffReader):
    profile = "akoya_he_qptiff"
    input_type_description = "Akoya H&E QPTIFF"
    image_type = "rgb"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(path, series=series, pixel_size=pixel_size)


class AperioSVSReader(_VendorTiffReader):
    profile = "svs"
    input_type_description = "Aperio SVS"
    image_type = "rgb"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(path, series=series, pixel_size=pixel_size)


class AkoyaComponentTiffReader(_VendorTiffReader):
    profile = "component"
    input_type_description = "Akoya Component TIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(path, series=series, pixel_size=pixel_size)
