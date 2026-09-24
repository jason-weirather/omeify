"""Named, tile-streamed crop exports through the existing OME-TIFF writer."""
from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path
from typing import Any

from .io._writer.configuration import DEFAULT_JPEG_QUALITY, DEFAULT_JPEG_SUBSAMPLING
from .io.image_metadata import integer
from .io.ome_multi_series_writer import OMEImageSeries, OMEMultiSeriesWriter
from .io.ome_tiff_reader import OMETiffLabelReader
from .io.ome_tiff_writer import JPEGSubsampling
from .io.pixel_size import PixelSize
from .io.source_reader import InputType, source_reader
from .regions import (
    MAX_GEOJSON_CHARS,
    ShatterMode,
    group_outputs,
    load_geojson,
    parse_regions,
    pixel_bounds,
    rectangle_feature,
)

LOGGER = logging.getLogger(__name__)


def _same_path(a: Path, b: Path) -> bool:
    return a.resolve() == b.resolve() or (a.exists() and b.exists() and a.samefile(b))


def crop(
    input_path: str | Path, output_path: str | Path, *,
    bounds: Sequence[int] | None = None,
    geojson: Mapping[str, Any] | list[Any] | str | Path | None = None,
    input_type: InputType = "ome_tiff", series: int = 0,
    shatter: ShatterMode | None = None, clip: bool = False, labels: bool = False,
    pixel_size: PixelSize | None = None, channel_name_field: str | None = None,
    compression: str | None = None, tile_size: int | None = None,
    pyramid_levels: int | None = None, max_workers: int | None = None,
    overwrite: bool = True, cache_directory: str | Path | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    jpeg_subsampling: JPEGSubsampling = DEFAULT_JPEG_SUBSAMPLING,
) -> dict[str, Any]:
    """Export full-resolution rectangles into one named, ordered OME-TIFF collection.

    Supply exactly one of integer ``bounds=(x0,y0,x1,y1)`` or pixel-coordinate
    ``geojson`` (a dict/list or a JSON file path). Float geometry edges are rounded
    outward. No polygon masking, resampling, stitching, or label renumbering occurs.
    Input coordinates are never interpreted as physical units or longitude/latitude.

    ``output_path`` is the exact destination, with one series per ROI in input
    order, even for a single ROI. Series names have a padded one-based index plus
    the supplied name (or ROI). Model-supplied names never choose paths by default.
    ``shatter='by_index'`` writes one file per ROI; ``shatter='by_name'`` groups
    equal names into files, with unnamed ROIs falling back to their input indices.
    Shattered filenames insert the name/index before a TIFF extension, or append
    .ome.tiff when output_path is a bare prefix. No file is written at the prefix.
    All geometry and destinations are checked before writing any pixels. Each file
    installs atomically; a multiple-file export is NOT an all-or-nothing transaction.
    Writer defaults remain authoritative, including lossy RGB JPEG 90 / 422 / 512.
    """
    if (bounds is None) == (geojson is None):
        raise ValueError("Choose exactly one of bounds or geojson")
    series = integer(series, "series")
    if any(not isinstance(v, bool) for v in (clip, labels, overwrite)):
        raise TypeError("clip, labels, and overwrite must be booleans")
    if labels and input_type != "ome_tiff":
        raise ValueError("labels=True requires input_type='ome_tiff'")
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize")
    source_path = Path(input_path)
    protected = [source_path]
    if bounds is not None:
        if len(bounds) != 4:
            raise ValueError("bounds must contain X0 Y0 X1 Y1")
        # Permit negative integer bounds only for explicit clipping.
        if any(isinstance(v, bool) or not isinstance(v, Integral) for v in bounds):
            raise TypeError("Explicit bounds must be integers")
        bounds = tuple(int(v) for v in bounds)
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError("Explicit bounds must have X0 < X1 and Y0 < Y1")
        document = rectangle_feature(bounds, name="")
    elif isinstance(geojson, (str, Path)):
        path = Path(geojson)
        with path.open(encoding="utf-8") as handle:
            document = load_geojson(handle.read(MAX_GEOJSON_CHARS + 1))
        protected.append(path)
    else:
        document = geojson
    regions = parse_regions(document)
    outputs = group_outputs(regions, output_path, shatter=shatter)
    index_width = max(2, len(str(len(regions))))
    reader = (OMETiffLabelReader(source_path, series=series) if labels else source_reader(
        source_path, input_type=input_type, series=series, channel_name_field=channel_name_field,
    ))
    if labels and channel_name_field is not None:
        raise ValueError("channel_name_field is only valid for qptiff_fusion input")
    with reader:
        height, width = reader.levels[0].spatial_shape
        # Omeify's generated GeoJSON carries a coordinate guard, not a content identity.
        context = document.get("omeify") if isinstance(document, Mapping) else None
        if context is not None:
            if not isinstance(context, Mapping):
                raise ValueError("GeoJSON omeify context must be an object")
            if context.get("coordinate_system") != "level0_pixels":
                raise ValueError("GeoJSON must use level0_pixels coordinates")
            if context.get("image_size") != [width, height] or context.get("series") != series:
                raise ValueError(
                    "GeoJSON image dimensions or series do not match the selected image"
                )
        effective_size = pixel_size or reader.pixel_size
        if effective_size is None:
            raise ValueError("Crop export requires pixel calibration; supply pixel_size explicitly")
        rectangles = {
            r.index: pixel_bounds(r.bounds, width, height, clip=clip) for r in regions
        }
        protected.extend(reader.backing_paths)
        writers = []
        planned_paths: list[Path] = []
        # Preflight the WHOLE batch, including input aliases and dangling symlinks.
        for path, members in outputs:
            if any(_same_path(path, p) for p in protected):
                raise ValueError("A crop destination would replace the image or GeoJSON input")
            if any(_same_path(path, p) for p in planned_paths):
                raise ValueError("Crop destinations alias each other")
            if path.is_dir():
                raise IsADirectoryError(path)
            if not overwrite and os.path.lexists(path):
                raise FileExistsError(path)
            planned_paths.append(path)
            writers.append((OMEMultiSeriesWriter(
                path, compression=compression, tile_size=tile_size, pyramid_levels=pyramid_levels,
                jpeg_quality=jpeg_quality, jpeg_subsampling=jpeg_subsampling,
                max_workers=max_workers, overwrite=overwrite, cache_directory=cache_directory,
            ), members))
        results = []
        with reader.with_metadata(pixel_size=effective_size) as image:
            for writer, members in writers:
                LOGGER.info("Cropping %s region(s) into %s", len(members), writer.path)
                roi_records = []
                with ExitStack() as stack:
                    entries = []
                    for output_series, region in enumerate(members):
                        x0, y0, x1, y1 = rectangles[region.index]
                        view = stack.enter_context(image.crop(y0, y1, x0, x1))
                        name = f"{region.index:0{index_width}d} - {region.name or 'ROI'}"
                        entries.append(OMEImageSeries(name, view))
                        roi_records.append({
                            "input_index": region.index, "name": region.name,
                            "output_series": output_series, "output_name": name,
                            "bounds": [x0, y0, x1, y1],
                            "requested_bounds": list(region.bounds),
                            "clipped": (region.bounds[0] < 0 or region.bounds[1] < 0
                                        or region.bounds[2] > width or region.bounds[3] > height),
                            "source_offset_xy": [x0, y0],
                        })
                    write_report = writer.write(entries, provenance={
                        "schema": "omeify.crop/2", "coordinate_system": "level0_pixels",
                        "shatter": shatter,
                        "source_series": series, "source_size": [width, height],
                        "crop_mode": "bounding_rectangle", "regions": roi_records,
                    })
                results.append({"path": str(writer.path), "regions": roi_records,
                                "write_report": write_report})
    return {"schema": "omeify.crop/2", "input_path": str(source_path), "source_series": series,
            "output_path": str(output_path), "shatter": shatter, "clip": clip, "outputs": results}
