"""Named, regional crop exports through the selected OME output writer."""
from __future__ import annotations

import logging
import math
import os
from collections.abc import Mapping, Sequence
from contextlib import ExitStack
from numbers import Integral
from pathlib import Path
from typing import Any

from .io._writer.configuration import DEFAULT_JPEG_QUALITY, DEFAULT_JPEG_SUBSAMPLING
from .io.base import MultichannelImage
from .io.image_metadata import integer
from .io.ome_multi_series_writer import OMEImageSeries
from .io.ome_tiff_writer import JPEGSubsampling
from .io.ome_zarr.store import is_remote, location
from .io.output import OutputFormat, image_writer
from .io.pixel_size import PixelSize
from .io.source_reader import InputType, source_reader
from .path_safety import paths_overlap
from .regions import (
    MAX_GEOJSON_CHARS,
    ShatterMode,
    group_outputs,
    load_geojson,
    parse_regions,
    pixel_bounds,
    rectangle_feature,
)
from .reports import complete_report
from .roi_raster import MaskBackground, geometry_crop, mask_fill_value
from .workflow import (
    ChannelRenameMapping,
    RenameChannelsBy,
    apply_channel_renames,
    validate_channel_rename_mapping,
)

LOGGER = logging.getLogger(__name__)


def load_crop_document(
    *, bounds: Sequence[int] | None = None,
    geojson: Mapping[str, Any] | list[Any] | str | Path | None = None,
) -> tuple[Any, tuple[Path, ...]]:
    """Load one bounded crop input and report any auxiliary file it depends on."""
    if (bounds is None) == (geojson is None):
        raise ValueError("Choose exactly one of bounds or geojson")
    protected: list[Path] = []
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
    return document, tuple(protected)


# A plain selector retains its source name; (selector, new_name) renames just
# that occurrence. Python strings are always exact names; CLI parses numeric indices.
CropChannel = int | str | tuple[int | str, str]


def _channel_requests(
    channels: Sequence[CropChannel] | None,
) -> tuple[tuple[int | str, str | None], ...] | None:
    if channels is None:
        return None
    if isinstance(channels, (str, bytes)) or not isinstance(channels, Sequence):
        raise TypeError("channels must be a sequence of names, indices or (selector, name) pairs")
    if not channels:
        raise ValueError("channels must not be empty; omit channels to keep all source channels")
    result = []
    for item in channels:
        selector, name = item if isinstance(item, tuple) and len(item) == 2 else (item, None)
        if isinstance(selector, bool) or not isinstance(selector, (Integral, str)):
            raise TypeError("Channel selectors must be integer indices or exact source names")
        if isinstance(selector, Integral):
            selector = integer(selector, "channel index")
        elif not selector:
            raise ValueError("Channel names must not be empty")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Selected output channel names must be non-empty strings")
            name = name.strip()
        result.append((selector, name))
    return tuple(result)


def _resolved_channels(
    source_names: Sequence[str], requests: tuple[tuple[int | str, str | None], ...] | None,
    renames: ChannelRenameMapping, rename_by: RenameChannelsBy | None,
) -> tuple[tuple[int, ...], tuple[str, ...], list[dict[str, Any]]]:
    source_names = tuple(source_names)
    renamed_names = apply_channel_renames(source_names, renames, rename_by)
    if requests is None:
        requests = tuple((index, None) for index in range(len(source_names)))
    resolved = []
    for selector, new_name in requests:
        if isinstance(selector, str):
            matches = [i for i, name in enumerate(source_names) if name == selector]
            if not matches:
                raise ValueError(f"Unknown channel name {selector!r}")
            if len(matches) != 1:
                raise ValueError(
                    f"Channel name {selector!r} is ambiguous; select its zero-based index"
                )
            index = matches[0]
        else:
            index = selector
            if index >= len(source_names):
                raise IndexError(
                    f"Channel index {index} is outside 0..{len(source_names) - 1}"
                )
        resolved.append((index, renamed_names[index] if new_name is None else new_name))
    mapping = [
        {"source_index": index, "source_name": source_names[index],
         "output_index": output_index, "output_name": name}
        for output_index, (index, name) in enumerate(resolved)
    ]
    return (tuple(index for index, _ in resolved),
            tuple(name for _, name in resolved), mapping)


def crop(
    input_path: str | Path, output_path: str | Path, *,
    bounds: Sequence[int] | None = None,
    geojson: Mapping[str, Any] | list[Any] | str | Path | None = None,
    input_type: InputType = "ome_tiff", series: int = 0,
    output_format: OutputFormat = "ome_tiff", group: str = "",
    z: int | None = None, t: int | None = None, image_type: str | None = None,
    intrinsic_only: bool = False, storage_options: dict | None = None,
    shatter: ShatterMode | None = None, clip: bool = False, labels: bool = False,
    mask_background: MaskBackground = "auto", padding: int = 0,
    pad_outside: bool = False,
    pixel_size: PixelSize | None = None, channel_name_field: str | None = None,
    channels: Sequence[CropChannel] | None = None,
    rename_channels: ChannelRenameMapping | None = None,
    rename_channels_by: RenameChannelsBy | None = None,
    compression: str | None = None, tile_size: int | None = None,
    pyramid_levels: int | None = None, max_workers: int | None = None,
    overwrite: bool = True, cache_directory: str | Path | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    jpeg_subsampling: JPEGSubsampling = DEFAULT_JPEG_SUBSAMPLING,
) -> dict[str, Any]:
    """Export full-resolution rectangles into one named, ordered OME image collection.

    Supply exactly one of integer ``bounds=(x0,y0,x1,y1)`` or pixel-coordinate
    ``geojson`` (a dict/list or a JSON file path). Float geometry edges are rounded
    outward. GeoJSON polygons are masked at pixel centers, retaining holes and
    multipart unions; use mask_background='keep' for the legacy bounding box.
    No resampling, stitching or label renumbering occurs.
    Input coordinates are never interpreted as physical units or longitude/latitude.

    ``output_path`` is the exact destination, with one series per ROI in input
    order, even for a single ROI. Series names have a padded one-based index plus
    the supplied name (or ROI). Model-supplied names never choose paths by default.
    ``shatter='by_index'`` writes one file per ROI; ``shatter='by_name'`` groups
    equal names into files, with unnamed ROIs falling back to their input indices.
    Shattered filenames insert the name/index before the selected format's suffix.
    Bare prefixes receive .ome.tiff or .ome.zarr; no product is written at the prefix.
    All geometry and destinations are checked before writing any pixels. Products
    publish individually; a batch is NOT an all-or-nothing transaction. TIFF RGB
    defaults remain JPEG 90 / 422 / 512; Zarr defaults to lossless Zstd / 512 and
    requires a new local directory for every destination.

    ``channels`` selects logical channels by zero-based integer index or exact
    source name, in output order. ``(selector, output_name)`` renames one selected
    channel. Omitting channels exports all source channels, optionally renamed
    with the same explicit name/index mapping as ``convert``. RGB and labels each
    have one indivisible logical channel. Selection and renaming remain lazy.

    ``padding`` adds source-resolution pixels on all sides of the polygon envelope.
    Out-of-image requests fail unless ``clip=True`` (trim the canvas) or
    ``pad_outside=True`` (fill areas beyond the source). The latter two are
    mutually exclusive. ``mask_background='auto'`` uses white for RGB, zero for
    intensity images and the declared background ID for labels. White on float
    channels is literal 1.0, not a normalized display value.
    """
    series = integer(series, "series")
    if any(not isinstance(v, bool) for v in (clip, labels, overwrite, pad_outside)):
        raise TypeError("clip, labels, overwrite, and pad_outside must be booleans")
    padding = integer(padding, "padding")
    if clip and pad_outside:
        raise ValueError("clip and pad_outside are mutually exclusive")
    if mask_background not in ("auto", "black", "white", "keep"):
        raise ValueError("mask_background must be auto, black, white, or keep")
    if labels:
        if image_type not in {None, "label"}:
            raise ValueError("labels=True conflicts with image_type")
        image_type = "label"
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize")
    requests = _channel_requests(channels)
    renames, rename_by = validate_channel_rename_mapping(rename_channels, rename_channels_by)
    if renames and requests is not None and any(name is not None for _, name in requests):
        raise ValueError("Choose either named channel selections or a rename mapping, not both")
    source_path = location(input_path)
    document, auxiliary = load_crop_document(bounds=bounds, geojson=geojson)
    protected = [source_path, *auxiliary]
    regions = parse_regions(document)
    outputs = group_outputs(regions, output_path, shatter=shatter, output_format=output_format)
    index_width = max(2, len(str(len(regions))))
    reader = source_reader(
        source_path, input_type=input_type, series=series, channel_name_field=channel_name_field,
        group=group, z=z, t=t, image_type=image_type, intrinsic_only=intrinsic_only,
        storage_options=storage_options,
    )
    if labels and channel_name_field is not None:
        raise ValueError("channel_name_field is only valid for qptiff_fusion input")
    with reader:
        indices, output_names, channel_mapping = _resolved_channels(
            reader.channel_names, requests, renames, rename_by,
        )
        if reader.image_type != "multichannel" and (
            len(indices) != 1 or indices[0] != 0
        ):
            raise ValueError(
                "RGB and label images have one indivisible logical channel (index 0)"
            )
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
        # Preflight every padded extent before opening ANY output writer.
        rectangles = {}
        for region in regions:
            x0, y0, x1, y1 = region.bounds
            padded = (x0 - padding, y0 - padding, x1 + padding, y1 + padding)
            if pad_outside:
                # Require at least some source overlap; never write a completely
                # invented region even when the user permits off-slide padding.
                pixel_bounds(padded, width, height, clip=True)
                rectangle = (math.floor(padded[0]),
                             math.floor(padded[1]),
                             math.ceil(padded[2]),
                             math.ceil(padded[3]))
            else:
                rectangle = pixel_bounds(padded, width, height, clip=clip)
            # Prevent an untrusted GeoJSON extent from declaring an impractically
            # huge raster dimension when off-slide padding is enabled.
            if (rectangle[2] - rectangle[0] > 2_147_483_647
                    or rectangle[3] - rectangle[1] > 2_147_483_647):
                raise ValueError("Crop output dimensions exceed the supported pixel-edge range")
            rectangles[region.index] = rectangle
        fill_value = mask_fill_value(reader, mask_background)
        # Integer-edged axis-aligned rectangles require no raster mask. Retain
        # the original zero-copy crop path (and RGB JPEG default) for them.
        mask_required = {
            region.index: (mask_background != "keep"
                           and (padding != 0 or not region.is_integer_rectangle))
            for region in regions
        }
        protected.extend(reader.backing_paths)
        # Masked RGB crops default to lossless TIFF storage so JPEG cannot bleed
        # boundary colors back into the black/white background. An explicit
        # compression choice is always authoritative (including lossy JPEG).
        compression_for_crop = compression
        if (compression is None and output_format == "ome_tiff"
                and reader.image_type == "rgb"
                and (any(mask_required.values()) or pad_outside)):
            compression_for_crop = "Deflate"
        writers = []
        planned_paths: list[Path] = []
        # Preflight the WHOLE batch, including input aliases and dangling symlinks.
        for path, members in outputs:
            if any(not is_remote(p) and paths_overlap(path, p) for p in protected):
                raise ValueError("A crop destination would replace the image or GeoJSON input")
            if any(paths_overlap(path, p) for p in planned_paths):
                raise ValueError("Crop destinations alias each other")
            if path.is_dir():
                raise IsADirectoryError(path)
            if not overwrite and os.path.lexists(path):
                raise FileExistsError(path)
            planned_paths.append(path)
            writers.append((image_writer(
                path, output_format=output_format, collection=True, compression=compression_for_crop, tile_size=tile_size, pyramid_levels=pyramid_levels,
                jpeg_quality=jpeg_quality, jpeg_subsampling=jpeg_subsampling,
                max_workers=max_workers, overwrite=overwrite, cache_directory=cache_directory,
            ), members))
        results = []
        with reader.with_metadata(pixel_size=effective_size) as image:
            with ExitStack() as channel_stack:
                selected = image
                if indices != tuple(range(image.channel_count)):
                    selected = channel_stack.enter_context(MultichannelImage.from_channels(
                        [image[index] for index in indices], channel_names=output_names,
                    ))
                elif output_names != image.channel_names:
                    selected = channel_stack.enter_context(image.with_metadata(
                        channel_names=output_names,
                    ))
                for writer, members in writers:
                    LOGGER.info("Cropping %s region(s) into %s", len(members), writer.path)
                    roi_records = []
                    with ExitStack() as stack:
                        entries = []
                        for output_series, region in enumerate(members):
                            x0, y0, x1, y1 = rectangles[region.index]
                            # The polygon source reads only the overlapping pixels
                            # and masks a single writer-requested tile at a time.
                            # keep preserves bbox pixels but still supports optional
                            # padding beyond image edges.
                            if not mask_required[region.index] and not pad_outside:
                                view = stack.enter_context(selected.crop(y0, y1, x0, x1))
                            else:
                                view = stack.enter_context(geometry_crop(
                                    selected, (x0, y0, x1, y1), polygons=region.polygons,
                                    mask=mask_required[region.index], fill_value=fill_value,
                                ))
                            name = f"{region.index:0{index_width}d} - {region.name or 'ROI'}"
                            entries.append(OMEImageSeries(name, view))
                            roi_records.append({
                                "input_index": region.index, "name": region.name,
                                "output_series": output_series, "output_name": name,
                                "bounds": [x0, y0, x1, y1],
                                "requested_bounds": list(region.bounds),
                                "clipped": (clip and (
                                    region.bounds[0] - padding < 0 or
                                    region.bounds[1] - padding < 0 or
                                    region.bounds[2] + padding > width or
                                    region.bounds[3] + padding > height)),
                                "source_offset_xy": [x0, y0],
                                "geometry_type": region.geometry_type,
                                "geometry_sha256": region.geometry_sha256,
                                "mask_applied": mask_required[region.index],
                            })
                        provenance = complete_report({
                            "coordinate_system": "level0_pixels",
                            "shatter": shatter,
                            "source_series": series, "source_size": [width, height],
                            "crop_mode": ("bounding_rectangle" if mask_background == "keep"
                                          else "polygon_mask"),
                            "mask_background": mask_background, "fill_value": fill_value,
                            "padding": padding, "pad_outside": pad_outside,
                            "regions": roi_records, "channel_mapping": channel_mapping,
                        }, "crop_provenance.schema.json")
                        write_report = writer.write(entries, provenance=provenance)
                    results.append({"path": str(writer.path), "regions": roi_records,
                                    "write_report": write_report})
    report = {"input_path": str(source_path), "source_series": series,
            "output_path": str(output_path), "shatter": shatter, "clip": clip,
            "crop_mode": ("bounding_rectangle" if mask_background == "keep"
                          else "polygon_mask"),
            "mask_background": mask_background, "fill_value": fill_value,
            "padding": padding, "pad_outside": pad_outside,
            "channel_mapping": channel_mapping, "outputs": results}

    return complete_report(report, "crop_report.schema.json")
