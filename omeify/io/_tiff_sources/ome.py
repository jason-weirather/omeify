"""OME metadata interpretation over the shared TIFF pixel backend."""
from __future__ import annotations

import logging
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import tifffile
from lxml import etree
from omeify.utils.ome_geometry import calibrated_levels, read_geometry

from ..image_metadata import ImageLevel, ImageMetadata
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

    def _infer_subifd_sampling(self, levels: tuple[ImageLevel, ...]) -> tuple[ImageLevel, ...]:
        """Recover exact integer OME SubIFD scales, but never guess spatial offsets.

        TIFF resolutions take precedence. Only an actual per-plane SubIFD hierarchy
        with one constant, isotropic integer step and *exact* division in both axes
        supplies enough structural evidence. Rounded shapes, extra/missing levels,
        and conflicting TIFF calibration remain unresolved.
        """
        if len(levels) < 2 or all(level.downsample_yx is not None for level in levels):
            return levels

        def unresolved(reason: str) -> tuple[ImageLevel, ...]:
            LOGGER.debug("OME SubIFD sampling inference skipped: %s", reason)
            return levels

        pages_by_level = [self._level_pages(index) for index in range(len(levels))]
        for channel, parent in enumerate(pages_by_level[0]):
            offsets = parent.subifds
            if parent.is_reduced or offsets is None or len(offsets) != len(levels) - 1:
                return unresolved("incomplete direct SubIFD linkage")
            for index, child_pages in enumerate(pages_by_level[1:]):
                child = child_pages[channel]
                if not child.is_reduced or int(child.offset) != int(offsets[index]):
                    return unresolved("reduced plane does not match its parent SubIFD")

        base_height, base_width = levels[0].spatial_shape
        first_height, first_width = levels[1].spatial_shape
        if base_height % first_height or base_width % first_width:
            return unresolved("first reduction is not an exact integer division")
        step_y, step_x = base_height // first_height, base_width // first_width
        if step_y != step_x or step_y < 2:
            return unresolved("first reduction is not isotropic and greater than one")

        base_size = levels[0].pixel_size
        for index, level in enumerate(levels[1:], start=1):
            factor = step_y ** index
            height, width = level.spatial_shape
            if height * factor != base_height or width * factor != base_width:
                return unresolved("levels do not share an exact constant integer reduction")
            if level.downsample_yx is not None and not all(
                math.isclose(scale, factor, rel_tol=1e-5, abs_tol=0)
                for scale in level.downsample_yx
            ):
                return unresolved("declared level scale conflicts with SubIFD dimensions")
            if level.pixel_size is not None:
                if base_size is None:
                    return unresolved("reduced calibration cannot be related to an unknown base")
                actual = level.pixel_size.converted_to("µm")
                expected = base_size.scaled(factor).converted_to("µm")
                if not all(
                    math.isclose(a, b, rel_tol=1e-5, abs_tol=0)
                    for a, b in ((actual.y, expected.y), (actual.x, expected.x))
                ):
                    return unresolved("declared TIFF pixel sizes conflict with SubIFD dimensions")
            if level.offset_yx not in (None, (0.0, 0.0)):
                return unresolved("declared spatial offset cannot be extended")

        resolved = tuple(
            replace(
                level,
                downsample_yx=(float(step_y ** index),) * 2,
                pixel_size=(None if base_size is None else base_size.scaled(step_y ** index)),
                offset_yx=None,  # SubIFDs specify scale, not the sampling phase.
            ) if index and level.downsample_yx is None else level
            for index, level in enumerate(levels)
        )
        LOGGER.info(
            "Recovered %d OME SubIFD sampling scales using an exact %dx hierarchy; "
            "missing spatial offsets remain unknown",
            sum(a.downsample_yx is None and b.downsample_yx is not None
                for a, b in zip(levels, resolved, strict=True)), step_y,
        )
        return resolved

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
        # Match via the established local TiffData mapping, never series ordinal.
        summary = self._ome_image_summary()
        parser = etree.XMLParser(resolve_entities=False, no_network=True, load_dtd=False)
        root = etree.fromstring(self._tiff.ome_metadata.encode("utf-8") if isinstance(self._tiff.ome_metadata, str) else self._tiff.ome_metadata, parser)
        images = root.findall(f"{{{etree.QName(root).namespace}}}Image")
        geometry = read_geometry(root, images[summary["index"]])
        # A linked Omeify geometry record is authoritative. Do not use a structural
        # fallback to paper over broken calibration in a self-described image.
        if geometry is None:
            levels = self._infer_subifd_sampling(levels)
        levels = calibrated_levels(levels, geometry)
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
