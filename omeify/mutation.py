from __future__ import annotations

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np

from ._version import get_version_info
from .dtype_mutation import (
    DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    DTYPE_MUTATION_PROTOCOL_VERSION,
    DTypeMutationSource,
    RANGE_MODES,
    RangeMode,
    TARGET_DTYPES,
    TargetDType,
    analyze_dtype_mutation,
)
from .io.ome_tiff_writer import DownsampleMethod, OMETiffWriter
from .io.pixel_size import PixelSize
from .io.source_reader import PLANAR_INPUT_TYPES, PlanarInputType, source_reader
from .provenance import readable_runtime
from .workflow import (
    ChannelRenameMapping,
    RenameChannelsBy,
    apply_channel_renames,
    build_input_report,
    log_channel_mapping,
    log_source_summary,
    validate_channel_rename_mapping,
    validate_image_paths,
)

LOGGER = logging.getLogger(__name__)


def mutate(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_type: PlanarInputType,
    dtype: TargetDType,
    range_mode: RangeMode = "auto",
    series: int = 0,
    channel_name_field: Literal["name", "biomarker", "auto"] | None = None,
    rename_channels: ChannelRenameMapping | None = None,
    rename_channels_by: RenameChannelsBy | None = None,
    pixel_size: PixelSize | None = None,
    sample_pixels_per_channel: int = DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    auto_max_normalized_rmse: float = DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    compression: str = "LZW",
    tile_size: int = 1024,
    pyramid_levels: int | None = None,
    downsample: DownsampleMethod = "mean",
    max_workers: int | None = None,
    display_uuid: bool = True,
    overwrite: bool = True,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    """Write a dtype-mutated OME-TIFF from one planar floating-point source.

    The current mutation operation is deliberately narrow: it converts a
    planar float32 or float64 source to uint8 or uint16. The source is scanned
    channel by channel to select and report one fixed mapping for each channel,
    then the transformed planes stream through the normal :class:`OMETiffWriter`.
    """

    input_file, output_file = validate_image_paths(
        input_path,
        output_path,
        overwrite=overwrite,
    )
    if input_type not in PLANAR_INPUT_TYPES:
        raise ValueError(f"Unsupported input_type {input_type!r}")
    normalized_renames, normalized_rename_mode = validate_channel_rename_mapping(
        rename_channels,
        rename_channels_by,
    )
    if dtype not in TARGET_DTYPES:
        raise ValueError("dtype must be 'uint8' or 'uint16'")
    if range_mode not in RANGE_MODES:
        raise ValueError("range_mode must be 'auto', 'preserve', or 'full'")
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize instance")
    if compression.strip().lower() in {"jpeg", "jpg"}:
        raise ValueError("dtype-mutated quantitative channels require lossless compression")

    start_epoch = time.time()
    LOGGER.info(
        "Mutate workflow: opening %s as input type %s, series %s",
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
        if reader.is_rgb:
            raise ValueError("dtype mutation currently supports planar grayscale channels only")
        if reader.output_axes not in {"YX", "CYX"}:
            raise ValueError(
                f"dtype mutation requires planar YX or CYX output, found {reader.output_axes!r}"
            )
        if not np.issubdtype(reader.dtype, np.floating):
            raise TypeError(
                f"dtype mutation requires float32 or float64 input, found {reader.dtype}"
            )
        source_pixel_size = reader.pixel_size
        effective_pixel_size = pixel_size or source_pixel_size
        if effective_pixel_size is None:
            raise ValueError(
                "Input does not provide a usable physical pixel size; mutation will not "
                "invent one"
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
        output_channel_names = apply_channel_renames(
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
        LOGGER.info(
            "Mutation boundary: %s -> %s using range mode %s; source metadata and geometry "
            "remain on the shared conversion path",
            reader.dtype,
            dtype,
            range_mode,
        )
        LOGGER.info(
            "Scanning every full-resolution channel before writing so each channel gets one "
            "fixed, auditable mapping"
        )
        analysis_started = time.monotonic()
        plans = analyze_dtype_mutation(
            reader,
            channel_names=source_channel_names,
            source_dtype=reader.dtype,
            dtype=dtype,
            range_mode=range_mode,
            sample_pixels_per_channel=sample_pixels_per_channel,
            auto_max_normalized_rmse=auto_max_normalized_rmse,
        )
        LOGGER.info(
            "Mutation planning completed in %s",
            readable_runtime(time.monotonic() - analysis_started),
        )
        transformed = DTypeMutationSource(reader, plans)
        LOGGER.info(
            "Using the same OME-TIFF writer as convert through a streaming dtype-transform "
            "adapter (compression=%s, tile=%s, downsample=%s)",
            compression,
            tile_size,
            downsample,
        )
        writer = OMETiffWriter(
            output_file,
            image_type="multichannel",
            channel_names=output_channel_names,
            pixel_size=effective_pixel_size,
            compression=compression,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,
            max_workers=max_workers,
            display_uuid=display_uuid,
            overwrite=overwrite,
            cache_directory=cache_directory,
        )
        writer_started = time.monotonic()
        write_report = writer.write_source(
            transformed,
            axes=reader.output_axes,
            shape=reader.output_shape,
            dtype=np.dtype(dtype),
        )
        LOGGER.info(
            "Shared writer path completed in %s",
            readable_runtime(time.monotonic() - writer_started),
        )

        input_report = build_input_report(
            reader,
            input_file,
            source_pixel_size,
            always_include_source_miti_header=True,
        )

    output_size = output_file.stat().st_size
    output_report = dict(write_report["output_file"])
    options = dict(write_report["options"])
    options.update(
        {
            "operation": "dtype",
            "input_type": input_type,
            "series": int(series),
            "channel_name_field": channel_name_field,
            "rename_channels": dict(normalized_renames),
            "rename_channels_by": normalized_rename_mode,
            "pixel_size_override": (
                None if pixel_size is None else list(pixel_size.to_tuple())
            ),
            "dtype": dtype,
            "range_mode": range_mode,
            "sample_pixels_per_channel": int(sample_pixels_per_channel),
            "auto_max_normalized_rmse": float(auto_max_normalized_rmse),
        }
    )


    stop_epoch = time.time()
    report: dict[str, object] = {
        "operation": "dtype",
        "ome": write_report["ome"],
        "miti_header": write_report["miti_header"],
        "input_file": input_report,
        "output_file": output_report,
        "image": write_report["image"],
        "pyramid": write_report["pyramid"],
        "verification": write_report["verification"],
        "dtype_mutation": {
            "protocol_version": DTYPE_MUTATION_PROTOCOL_VERSION,
            "source_dtype": input_report["dtype"],
            "target_dtype": dtype,
            "range_mode": range_mode,
            "rounding": "nearest, ties to even",
            "clipping_policy": "none for finite source values",
            "automatic_range_policy": {
                "near_integer_tolerance": 0.01,
                "moderate_near_integer_fraction": 0.10,
                "moderate_enrichment_over_uniform_fractional_parts": 5.0,
                "strong_near_integer_fraction": 0.50,
                "strong_enrichment_over_uniform_fractional_parts": 10.0,
                "maximum_unit_rounding_normalized_rmse": float(
                    auto_max_normalized_rmse
                ),
                "decision": (
                    "Auto preserves unit scale when the nearest-integer range fits the target "
                    "dtype and either full-resolution nonzero values provide strong "
                    "integer-lattice evidence or unit-rounding RMSE is within the configured "
                    "fraction of the sampled nonzero robust intensity span; otherwise it uses "
                    "a zero-anchored linear mapping."
                ),
            },
            "channels": [plan.report for plan in plans],
        },
        "options": options,
        "mutation_stats": {
            "start_time": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "stop_time": datetime.fromtimestamp(stop_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "run_time": readable_runtime(stop_epoch - start_epoch),
            "output_to_input_size_ratio": (
                output_size / int(input_report["size_bytes"])
                if int(input_report["size_bytes"])
                else None
            ),
        },
        "versions": get_version_info(),
    }
    LOGGER.info(
        "Mutate complete: %s (%s bytes) in %s",
        output_file,
        f"{output_size:,}",
        readable_runtime(stop_epoch - start_epoch),
    )
    return report
