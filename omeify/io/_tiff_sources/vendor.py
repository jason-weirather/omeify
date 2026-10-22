from __future__ import annotations

import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile


from .base import TiffSource
from ..akoya_qptiff import (
    ChannelNameField,
    consistent_akoya_pixel_size,
    parse_akoya_qpi_channel_metadata,
    select_akoya_channel_name,
)
from ..image_metadata import ImageMetadata
from ..leica_scn import (
    SCNImage, SCNMetadata, build_leica_scn_he_series, parse_leica_scn, scn_he_pixel_size,
)
from ..indica_mif import (
    IndicaMIFMetadata,
    build_indica_mif_series,
    parse_indica_mif_metadata,
)
from ..pixel_size import (
    PixelSize,
    consistent_tiff_resolution_pixel_size,
)

LOGGER = logging.getLogger(__name__)

SourceProfile = Literal[
    "akoya_mif_qptiff",
    "akoya_fusion_qptiff",
    "akoya_he_qptiff",
    "svs",
    "scn_he",
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


class VendorSource(TiffSource):
    def __init__(
        self, path: str | Path, *, profile: SourceProfile,
        description: str, series: int = 0, pixel_size: PixelSize | None = None,
        channel_name_field: ChannelNameField = "name",
    ) -> None:
        super().__init__(path, series=series)
        if channel_name_field not in {"name", "biomarker", "auto"}:
            raise ValueError("channel_name_field must be 'name', 'biomarker', or 'auto'")
        if pixel_size is not None and not isinstance(pixel_size, PixelSize):
            raise TypeError("pixel_size must be PixelSize")
        if profile not in {"akoya_mif_qptiff", "akoya_fusion_qptiff", "akoya_he_qptiff",
                           "svs", "scn_he", "indica_mif"}:
            raise ValueError("Unsupported vendor source profile")
        self.profile = profile
        self.input_type_description = description
        self.pixel_size_override = pixel_size
        self.channel_name_field = channel_name_field
        self.image_type = ("rgb" if profile in {"svs", "scn_he", "akoya_he_qptiff"}
                           else "multichannel")
        self._pages: list[tifffile.TiffPage] = []
        self._level0 = None
        self._pixel_size = None
        self._icc_profile = None
        self._source_channel_metadata = ()
        self._normalized_names = ()
        self._indica_metadata = None
        self._scn_metadata: SCNMetadata | None = None
        self._scn_image: SCNImage | None = None
        self._origin_yx: tuple[float, float] | None = None

    def _load_file(self, tiff: tifffile.TiffFile) -> None:
        if self.profile == "indica_mif":
            if self.series_index != 0:
                raise IndexError("Indica mIF TIFF contains one XML-defined image; series must be 0")
            self._indica_metadata = parse_indica_mif_metadata(tiff.pages.first.description)
            self._series = build_indica_mif_series(tiff, self._indica_metadata)
            self._level0 = self._series
            self._pages = [item.aspage() for item in self._series.pages]
        elif self.profile == "scn_he":
            self._scn_metadata = parse_leica_scn(tiff)
            if self.series_index >= len(self._scn_metadata.images):
                raise IndexError(
                    f"Series {self.series_index} does not exist; SCN has "
                    f"{len(self._scn_metadata.images)} images"
                )
            self._scn_image = self._scn_metadata.images[self.series_index]
            self._series = build_leica_scn_he_series(tiff, self._scn_image)
            self._level0 = self._series
            self._pages = [self._series.pages[0].aspage()]
        else:
            self._validate_container(tiff)
            self._series, self._level0, self._pages = self._select_series(tiff)
        if self.image_type == "rgb":
            self._inspect_rgb()
        else:
            self._inspect_planar()

    def _close(self) -> None:
        super()._close()
        self._pages = []
        self._level0 = None
        self._indica_metadata = None
        self._scn_metadata = None
        self._scn_image = None
        self._origin_yx = None
        self._source_channel_metadata = ()
        self._normalized_names = ()
        self._pixel_size = None
        self._icc_profile = None

    def _describe(self) -> ImageMetadata:
        axes = "YXS" if self.image_type == "rgb" else "CYX"
        count = len(self._normalized_names)
        levels = self._levels(axes, count, self._pixel_size)
        if self._origin_yx is not None:
            levels = (replace(levels[0], origin_yx=self._origin_yx), *levels[1:])
        return ImageMetadata(
            axes=axes, shape=levels[0].shape, dtype=self._pages[0].dtype,
            image_type=self.image_type, channel_names=self._normalized_names,
            pixel_size=self._pixel_size, levels=levels, icc_profile=self._icc_profile,
            channel_metadata=self._source_channel_metadata,
            channel_source_ids=tuple(None if r.get("id") is None else str(r["id"])
                                     for r in self._source_channel_metadata),
        )

    def series_names(self) -> tuple[str | None, ...]:
        if self._tiff is None:
            raise RuntimeError("TIFF source is not open")
        if self.profile == "scn_he":
            assert self._scn_metadata is not None
            return tuple(image.name for image in self._scn_metadata.images)
        return (None,) if self.profile == "indica_mif" else (None,) * len(self._tiff.series)

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
        else:
            # Name is only the direct QPI channel field. Never promote an unrelated
            # descendant Name (patient, slide, scanner, etc.) to channel identity.
            parsed = tuple(
                parse_akoya_qpi_channel_metadata(page.description, channel_index=index)
                for index, page in enumerate(self._pages)
            )
            for index, item in enumerate(parsed):
                name = item.name or f"Channel {index + 1}"
                names.append(name)
                metadata.append({"name": name})
            source_pixel_size = consistent_akoya_pixel_size(parsed, self._pages)
            if source_pixel_size is not None and any(
                item.pixel_size_microns is None for item in parsed
            ):
                self._warn_tiff_resolution_fallback(
                    "Akoya PixelSizeMicrons", pixel_size=source_pixel_size,
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
        elif self.profile == "scn_he":
            assert self._scn_image is not None
            source_pixel_size = scn_he_pixel_size(self._scn_image, page)
        else:
            source_pixel_size = self._akoya_description_pixel_size(page)
            if source_pixel_size is None:
                source_pixel_size = self._warn_tiff_resolution_fallback(
                    "Akoya PixelSizeMicrons"
                )
        self._pixel_size = self.pixel_size_override or source_pixel_size
        if self._scn_image is not None:
            self._origin_yx = self._scn_image.center_origin(self._pixel_size)
        self._icc_profile = bytes(page.iccprofile) if page.iccprofile is not None else None
        self._normalized_names = ("RGB",)
        self._source_channel_metadata = ({"name": "RGB"},)

    @staticmethod
    def _akoya_description_pixel_size(page: tifffile.TiffPage) -> PixelSize | None:
        parsed = parse_akoya_qpi_channel_metadata(page.description, channel_index=0)
        if parsed.pixel_size_microns is None:
            return None
        return PixelSize(parsed.pixel_size_microns, parsed.pixel_size_microns, "µm")

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
