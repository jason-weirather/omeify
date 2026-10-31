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
from ..dataset_metadata import AxisInfo, DatasetLevel, selected_index
from ..tiff import TiffPlaneReader
from .ome_index import OMEPlaneIndex
from ..pixel_size import PixelSize, consistent_tiff_resolution_pixel_size, pixel_size_from_xy_units
from .base import TiffSource

LOGGER = logging.getLogger(__name__)


class OMESource(TiffSource):
    supports_prefetch = True

    def __init__(
        self, path: str | Path, *, series: int = 0, labels: bool = False,
        background_label: int = 0, z=None, t=None,
        cache_bytes=64 * 1024 * 1024, _allow_default_selection=False,
    ) -> None:
        super().__init__(path, series=series, cache_bytes=cache_bytes)
        self._labels = labels
        self._background_label = background_label
        self.z, self.t = z, t
        self._allow_default_selection = _allow_default_selection
        self._plane_index = None
        self._plane_map = None

    def _load_file(self, tiff: tifffile.TiffFile) -> None:
        if not tiff.is_ome:
            raise ValueError(f"TIFF does not contain recognized OME metadata: {self.path}")
        if self.series_index >= len(tiff.series):
            raise IndexError(
                f"Series {self.series_index} does not exist ({len(tiff.series)} series)",
            )
        self._series = tiff.series[self.series_index]
        self._plane_index = OMEPlaneIndex(tiff)
        self._plane_map = self._plane_index.map_series(self.series_index)
        from omeify.utils.stack_geometry import read_stack_geometry
        pixels = self._plane_map.pixels
        self._stack_geometry = read_stack_geometry(pixels.getroottree().getroot(), pixels.getparent())
        self.z = selected_index(self._axis_info("Z"), self.z, name="Z",
                                default=self._allow_default_selection)
        self.t = selected_index(self._axis_info("T"), self.t, name="T",
                                default=self._allow_default_selection)

    def _mapped_ome_image(self, series_index: int) -> dict[str, Any]:
        if self._plane_index is None:
            raise RuntimeError("TIFF source is not open")
        return self._plane_index.summary(series_index)

    def _ome_image_summary(self) -> dict[str, Any]:
        return self._mapped_ome_image(self.series_index)

    def _pixel_size(self) -> PixelSize | None:
        image = self._ome_image_summary()
        physical = (image.get("physical_size") or {}) if image is not None else {}
        x = physical.get("x")
        y = physical.get("y")
        # OME-XML 2016-06 defaults omitted PhysicalSize*Unit attributes to µm.
        # The inspection summary deliberately retains the *raw* absent units;
        # use its schema-aware calibration interpreter for effective read units.
        # This does not invent missing numeric PhysicalSizeX/Y measurements.
        from omeify._calibration import ome_calibration
        effective = ome_calibration(self._plane_map.pixels)
        x_unit = effective["x"]["effective_unit"]
        y_unit = effective["y"]["effective_unit"]
        x_complete = bool(x and x.get("value") is not None and x_unit)
        y_complete = bool(y and y.get("value") is not None and y_unit)
        if x_complete and y_complete:
            assert x is not None and y is not None
            return pixel_size_from_xy_units(
                float(x["value"]), str(x_unit),
                float(y["value"]), str(y_unit),
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
            n != 1 for axis, n in layout.items() if axis not in "TCZSYX"
        ):
            raise NotImplementedError("OME images require Y/X and supported T/C/Z/sample axes")
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

    def _axis_info(self, name, *, size=None):
        pixels = self._plane_map.pixels
        if name in "ZT":
            count = self._plane_map.sizes[name] if size is None else size
            value = pixels.get("PhysicalSizeZ" if name == "Z" else "TimeIncrement")
            unit = pixels.get("PhysicalSizeZUnit", "µm") if name == "Z" else pixels.get("TimeIncrementUnit", "s")
            scale = None if value is None else float(value)
            if name == "Z" and scale is not None:
                scale = PixelSize(scale, scale, unit).converted_to("µm").x
                unit = "µm"
            elif name == "T" and scale is not None:
                from ..dataset_metadata import time_unit_seconds
                scale *= time_unit_seconds(unit)
                unit = "s"
            origin = (getattr(self, "_stack_geometry", None) or {}).get(
                "z_origin_um" if name == "Z" else "time_origin_s")
            if origin is not None and scale is None:
                raise ValueError("Intrinsic stack origin requires axis calibration")
            return AxisInfo(name, count, "space" if name == "Z" else "time",
                            unit if scale is not None else None, scale, origin)
        raise ValueError("Expected Z or T")

    @property
    def dataset_levels(self):
        self._ensure_open()
        result = []
        for level in self.metadata.levels:
            size = None if level.pixel_size is None else level.pixel_size.converted_to("µm")
            origin = level.origin_yx
            axes = [self._axis_info("T")]
            if self.metadata.axes == "CYX":
                axes.append(AxisInfo("C", self.metadata.logical_channel_count, "channel"))
            axes.append(self._axis_info("Z"))
            for j, name in enumerate("YX"):
                axes.append(AxisInfo(name, level.spatial_shape[j], "space",
                    "µm" if size else None, (size.y, size.x)[j] if size else None,
                    origin[j] if origin else None))
            if self.metadata.axes == "YXS":
                axes.append(AxisInfo("S", 3, "channel"))
            base = replace(level, index=0, downsample_yx=(1., 1.), offset_yx=(0., 0.))
            meta = replace(self.metadata, shape=base.shape, pixel_size=base.pixel_size,
                           levels=(base,))
            result.append(DatasetLevel(level.index, tuple(axes), meta))
        return tuple(result)

    def _pages_for(self, z, t, level):
        pages = []
        for c in range(self._plane_map.sizes["C"]):
            index = self._plane_map.ifd(z=z, t=t, c=c)
            base = self._tiff.pages[index].aspage()
            if level:
                if base.subifds is None or level > len(base.subifds):
                    raise ValueError("Selected OME plane has no matching SubIFD level")
                page = base.pages[level - 1].aspage()
                if page.offset != base.subifds[level - 1]:
                    raise ValueError("Invalid selected-plane SubIFD linkage")
            else:
                page = base
                if page.is_reduced:
                    raise ValueError("OME base plane cannot be a reduced IFD")
            pages.append(page)
        return pages

    def _level_pages(self, level):
        return self._pages_for(self.z, self.t, level)

    def plane_metadata(self, *, z=None, t=None, level=0):
        self._ensure_open()
        z = selected_index(self._axis_info("Z"), z, name="Z")
        t = selected_index(self._axis_info("T"), t, name="T")
        descriptor = self.metadata.level(level)
        # Validate metadata for the selected planes, not just a keyframe inherited
        # by tifffile's lightweight frames. No pixel decode is performed here.
        for i in range(level, len(self.metadata.levels) if level == 0 else level + 1):
            expected = self.metadata.level(i)
            pages = self._pages_for(z, t, i)
            for page in pages:
                if ((int(page.imagelength), int(page.imagewidth)) != expected.spatial_shape or
                        np.dtype(page.dtype).newbyteorder("=") != self.metadata.dtype or
                        int(page.samplesperpixel) != self.metadata.samples_per_pixel):
                    raise ValueError("Selected OME plane geometry/dtype/samples disagree with source")
                profile = None if page.iccprofile is None else bytes(page.iccprofile)
                if self.metadata.image_type == "rgb" and profile != self.metadata.icc_profile:
                    raise ValueError("Selected RGB plane ICC profile differs from the source image")
            size = consistent_tiff_resolution_pixel_size(pages)
            if size is not None and expected.pixel_size is not None:
                a, b = size.converted_to("µm"), expected.pixel_size.converted_to("µm")
                if not all(math.isclose(x, y, rel_tol=1e-5) for x, y in ((a.x,b.x),(a.y,b.y))):
                    raise ValueError("Selected OME plane calibration differs from source")
        if level == 0:
            return self.metadata
        base = replace(descriptor, index=0, downsample_yx=(1., 1.), offset_yx=(0., 0.))
        return replace(self.metadata, shape=base.shape, pixel_size=base.pixel_size, levels=(base,))

    def read_plane_region(self, y0, y1, x0, x1, *, z, t, level, channels, speculative=False):
        pages = self._pages_for(z, t, level)
        values = []
        for c in channels:
            page = pages[c]
            key = (int(page.offset),)
            with self._read_lock:
                if key not in self._planes:
                    self._planes[key] = TiffPlaneReader(page, lock=self._read_lock,
                                                        cache=self._block_cache)
                reader = self._planes[key]
                self._planes.move_to_end(key)
                while len(self._planes) > 32:
                    self._planes.popitem(last=False)
            values.append(reader.read_region(y0, y1, x0, x1, speculative=speculative))
        return np.stack(values) if self.metadata.axes == "CYX" else values[0]

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        return self.read_plane_region(y0, y1, x0, x1, z=self.z, t=self.t,
                                      level=level, channels=channels)

    def cache_info(self):
        return self._block_cache.info()

    def acquisition_plane(self, *, z, t, channel=0):
        """Declared per-plane measurements, separate from intrinsic grid coordinates."""
        from omeify._calibration import _children
        from types import MappingProxyType
        self._ensure_open()
        self._plane_map.linear(z=z, t=t, c=channel)
        records = getattr(self, "_acquisition_records", None)
        if records is None:
            records = {}
            for node in _children(self._plane_map.pixels, "Plane"):
                key = tuple(int(node.get("The" + a, "-1")) for a in "ZTC")
                self._plane_map.linear(z=key[0], t=key[1], c=key[2])
                if key in records:
                    raise ValueError("Duplicate OME acquisition plane declarations")
                fields = {}
                for name in ("DeltaT", "ExposureTime", "PositionX", "PositionY", "PositionZ"):
                    raw = node.get(name)
                    if raw is not None:
                        value = float(raw)
                        if not math.isfinite(value):
                            raise ValueError("Non-finite acquisition plane coordinate")
                        unit = node.get(name + "Unit", "s" if name in {"DeltaT", "ExposureTime"}
                                        else "reference frame")
                        fields[name] = (value, unit)
                records[key] = MappingProxyType(fields)
            self._acquisition_records = records
        return records.get((z,t,channel), MappingProxyType({}))

    def _close(self):
        super()._close()
        self._plane_map = self._plane_index = None
        self._acquisition_records = None

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
