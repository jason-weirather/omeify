from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from ._version import get_version_info
from .io._writer.configuration import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_JPEG_SUBSAMPLING,
    resolve_compression_name,
    resolve_tile_size,
)
from .io.ome_tiff_writer import JPEGSubsampling, OMETiffWriter
from .io.pixel_size import PixelSize
from .io.source_reader import INPUT_TYPES, InputType, source_reader
from .provenance import readable_runtime
from .workflow import (
    ChannelRenameMapping,
    RenameChannelsBy,
    apply_channel_renames as _apply_channel_renames,
    build_input_report,
    log_channel_mapping,
    log_source_summary,
    validate_channel_rename_mapping as _validate_channel_rename_mapping,
    validate_image_paths,
)

LOGGER = logging.getLogger(__name__)

DownsampleMethod = Literal["mean", "nearest"]


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_type: InputType,
    series: int = 0,
    channel_name_field: Literal["name", "biomarker", "auto"] | None = None,
    rename_channels: ChannelRenameMapping | None = None,
    rename_channels_by: RenameChannelsBy | None = None,
    pixel_size: PixelSize | None = None,
    compression: str | None = None,
    jpeg_quality: int = DEFAULT_JPEG_QUALITY,
    jpeg_subsampling: JPEGSubsampling = DEFAULT_JPEG_SUBSAMPLING,
    tile_size: int | None = None,
    pyramid_levels: int | None = None,
    downsample: DownsampleMethod = "mean",
    max_workers: int | None = None,
    display_uuid: bool = True,
    software: str | None = None,
    overwrite: bool = True,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    """Normalize one supported image into omeify's canonical OME-TIFF contract.

    This is the public Python counterpart to ``omeify convert``. Source-format
    readers live in :mod:`omeify.io`; this function only selects the reader,
    applies explicit channel-renaming policy, and delegates all OME-TIFF
    construction and verification to :class:`OMETiffWriter`.

    Omitted compression and tile size follow the selected image's type, not its
    file profile: RGB uses lossy JPEG quality 90 / 4:2:2 and 256-pixel tiles;
    non-RGB uses lossless LZW and 1024-pixel tiles. This includes RGB OME-TIFF
    inputs. Supply a lossless codec explicitly to preserve RGB sample values.

    ``rename_channels`` uses native Python key types: strings in ``name`` mode
    and zero-based integers in ``index`` mode. The CLI converts JSON string keys
    into those same semantic values before calling this function.
    """

    input_file, output_file = validate_image_paths(
        input_path,
        output_path,
        overwrite=overwrite,
    )
    if input_type not in INPUT_TYPES:
        raise ValueError(f"Unsupported input_type {input_type!r}")
    normalized_renames, normalized_rename_mode = _validate_channel_rename_mapping(
        rename_channels,
        rename_channels_by,
    )
    if downsample not in {"mean", "nearest"}:
        raise ValueError("downsample must be 'mean' or 'nearest'")
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize instance")

    start_epoch = time.time()
    LOGGER.info(
        "Convert workflow: opening %s as input type %s, series %s",
        input_file,
        input_type,
        int(series),
    )
    reader = source_reader(
        input_file,
        input_type=input_type,
        series=int(series),
        channel_name_field=channel_name_field,
    )

    with reader:
        source_pixel_size = reader.pixel_size
        effective_pixel_size = pixel_size or source_pixel_size
        if effective_pixel_size is None:
            raise ValueError(
                "Input does not provide a usable physical pixel size. Supply pixel_size="
                "PixelSize(x, y, unit) explicitly so omeify can satisfy its OME-TIFF contract."
            )

        log_source_summary(
            LOGGER,
            reader,
            source_pixel_size=source_pixel_size,
            effective_pixel_size=effective_pixel_size,
            pixel_size_overridden=pixel_size is not None,
        )
        source_channels = tuple(reader.channels)
        source_channel_names = tuple(channel.name for channel in source_channels)
        output_channel_names = _apply_channel_renames(
            source_channel_names,
            normalized_renames,
            normalized_rename_mode,
        )
        log_channel_mapping(
            LOGGER,
            source_channels,
            output_channel_names,
            rename_mode=normalized_rename_mode,
        )
        effective_compression = resolve_compression_name(reader.image_type, compression)
        effective_tile_size = resolve_tile_size(reader.image_type, tile_size)
        LOGGER.info(
            "Convert fidelity boundary: source samples enter the writer with dtype %s and "
            "without an intensity rescale or explicit numeric cast",
            reader.dtype,
        )
        if effective_compression.strip().lower() in {"jpeg", "jpg"}:
            LOGGER.info(
                "Output storage: JPEG quality %s, subsampling %s; RGB pixels are re-encoded",
                jpeg_quality,
                jpeg_subsampling,
            )
        else:
            LOGGER.info("Output storage: %s lossless compression", effective_compression)
        LOGGER.info(
            "Using the shared OME-TIFF writer directly on the source reader "
            "(tile=%s, downsample=%s)",
            effective_tile_size,
            downsample,
        )
        writer = OMETiffWriter(
            output_file,
            compression=effective_compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=effective_tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,
            max_workers=max_workers,
            display_uuid=display_uuid,
            software=software,
            overwrite=overwrite,
            cache_directory=cache_directory,
        )
        writer_started = time.monotonic()
        with reader.with_metadata(
            channel_names=output_channel_names, pixel_size=effective_pixel_size,
        ) as output_image:
            write_report = writer.write(output_image)
        LOGGER.info(
            "Shared writer path completed in %s",
            readable_runtime(time.monotonic() - writer_started),
        )

        input_report = build_input_report(reader, input_file, source_pixel_size)

    output_size = output_file.stat().st_size
    output_report = dict(write_report["output_file"])
    image_report = dict(write_report["image"])
    pyramid_report = dict(write_report["pyramid"])
    if write_report["output_file"]["axes"] == "CYX":  # type: ignore[index]
        pyramid_report["level_shapes_cyx"] = pyramid_report["level_shapes"]
    elif write_report["output_file"]["axes"] == "YXS":  # type: ignore[index]
        pyramid_report["level_shapes_yxs"] = pyramid_report["level_shapes"]

    options = dict(write_report["options"])
    options.update(
        {
            "deidentify_ome": True,
            "input_type": input_type,
            "series": int(series),
            "rename_channels": dict(normalized_renames),
            "rename_channels_by": normalized_rename_mode,
            "channel_name_field": channel_name_field,
            "pixel_size_override": None if pixel_size is None else list(pixel_size.to_tuple()),
        }
    )


    stop_epoch = time.time()
    report: dict[str, object] = {
        "ome": write_report["ome"],
        "miti_header": write_report["miti_header"],
        "input_file": input_report,
        "output_file": output_report,
        "image": image_report,
        "pyramid": pyramid_report,
        "verification": write_report["verification"],
        "options": options,
        "conversion_stats": {
            "start_time": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "stop_time": datetime.fromtimestamp(stop_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "run_time": readable_runtime(stop_epoch - start_epoch),
            "output_to_input_size_ratio": (
                output_size / int(input_report["size_bytes"])
                if int(input_report["size_bytes"])
                else None
            ),
            "compression_ratio": (
                output_size / int(input_report["size_bytes"])
                if int(input_report["size_bytes"])
                else None
            ),
        },
        "versions": get_version_info(),
    }
    LOGGER.info(
        "Convert complete: %s (%s bytes) in %s",
        output_file,
        f"{output_size:,}",
        readable_runtime(stop_epoch - start_epoch),
    )
    return report
