"""OME metadata interpretation over the shared TIFF pixel backend."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import tifffile

from ..image_metadata import ImageMetadata
from ..pixel_size import PixelSize, consistent_tiff_resolution_pixel_size, pixel_size_from_xy_units
from .base import TiffSource

LOGGER = logging.getLogger(__name__)


class OMESource(TiffSource):
    def __init__(
        self, path: str | Path, *, series: int = 0, labels: bool = False,
        background_label: int = 0,
    ) -> None:
        super().__init__(path, series=series)
        self._labels = labels
        self._background_label = background_label

    def _load_file(self, tiff: tifffile.TiffFile) -> None:
        if not tiff.is_ome:
            raise ValueError(f"TIFF does not contain recognized OME metadata: {self.path}")
        if self.series_index >= len(tiff.series):
            raise IndexError(
                f"Series {self.series_index} does not exist ({len(tiff.series)} series)",
            )
        self._series = tiff.series[self.series_index]
        self._ome_image_summary()

    def _mapped_ome_image(self, series_index: int) -> dict[str, Any]:
        """Resolve metadata through inspection's authoritative local TiffData map.

        Ordinals, names and equal dimensions are not evidence of identity.
        Incomplete/ambiguous/scan-limited mappings must not become generated
        channel names or TIFF-calibration fallbacks during conversion.
        """

        report = self.inspect().report
        ome = report.get("ome")
        mappings = report["calibration"]["series"]
        mapping = mappings[series_index]
        image_index = mapping["ome_image_index"]
        images = [] if ome is None else ome.get("images", [])
        if image_index is None or not 0 <= image_index < len(images):
            raise ValueError(
                f"Cannot associate TIFF series {series_index} with an OME Image: "
                f"{mapping['mapping_reason']}. Refusing to guess channel identity or calibration."
            )
        return images[image_index]

    def _ome_image_summary(self) -> dict[str, Any]:
        return self._mapped_ome_image(self.series_index)

    def _pixel_size(self) -> PixelSize | None:
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

    def _describe(self) -> ImageMetadata:
        native = self.native_levels[0]
        layout = dict(zip(str(native.axes), native.shape))
        if "Y" not in layout or "X" not in layout or any(
            n != 1 for axis, n in layout.items() if axis not in "CSYX"
        ):
            raise NotImplementedError("OME images require Y/X and singleton non-spatial axes")
        pages = self._level_pages(0)
        summaries = list(self._ome_image_summary().get("channels", []))
        count = len(summaries) or int(layout.get("C", 1))
        if count != len(pages) or count != int(layout.get("C", 1)):
            raise ValueError("OME logical channels do not match the selected TIFF planes")
        is_rgb = count == 1 and int(pages[0].samplesperpixel) == 3
        if self._labels:
            if count != 1 or "S" in layout or int(pages[0].samplesperpixel) != 1:
                raise ValueError("Label OME-TIFF requires one grayscale label plane")
            axes, kind = "YX", "label"
        elif is_rgb:
            if int(layout.get("S", 0)) != 3:
                raise ValueError("RGB OME-TIFF must store three contiguous samples")
            axes, kind = "YXS", "rgb"
        else:
            if int(layout.get("S", 1)) != 1:
                raise NotImplementedError("Non-RGB multi-sample OME-TIFF is unsupported")
            axes, kind = ("CYX" if "C" in layout else "YX"), "multichannel"
        names, ids, records = [], [], []
        for index in range(count):
            summary = summaries[index] if index < len(summaries) else {}
            samples = int(summary.get("samples_per_pixel") or pages[index].samplesperpixel or 1)
            if samples != (3 if is_rgb else 1):
                raise ValueError("OME SamplesPerPixel disagrees with the TIFF sample layout")
            source_id = summary.get("id")
            ids.append(None if source_id is None else str(source_id))
            names.append(str(summary.get("name") or ("RGB" if is_rgb else f"Channel {index + 1}")))
            records.append(summary)
        pixel_size = self._pixel_size()
        levels = self._levels(axes, count, pixel_size)
        profile = pages[0].iccprofile if kind == "rgb" else None
        return ImageMetadata(
            axes=axes, shape=levels[0].shape, dtype=native.dtype, image_type=kind,
            channel_names=names, channel_ids=ids, channel_metadata=records,
            pixel_size=pixel_size, levels=levels,
            icc_profile=None if profile is None else bytes(profile),
            background_label=self._background_label,
        )

    def series_name(self) -> str | None:
        name = self._ome_image_summary().get("name")
        return None if name in (None, "") else str(name)

    def series_names(self) -> tuple[str | None, ...]:
        if self._tiff is None:
            raise RuntimeError("TIFF source is not open")
        images = (self._mapped_ome_image(i) for i in range(len(self._tiff.series)))
        return tuple(
            (None if item.get("name") in (None, "") else str(item["name"]) for item in images),
        )
