from __future__ import annotations

import logging
import time
from datetime import datetime
from contextlib import ExitStack
from .io.base import MultichannelImage
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
from .io._writer.configuration import normalize_float32_mantissa_bits
from .io._writer.precision import float32_significant_bits
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

FLOAT32_PRECISION_MUTATION_PROTOCOL_VERSION = "1.0"


def mutate(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_type: PlanarInputType,
    dtype: TargetDType | None = None,
    float32_mantissa_bits: int | None = None,
    range_mode: RangeMode | None = None,
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
    software: str | None = None,
    overwrite: bool = True,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    """Write an explicitly pixel-mutated OME-TIFF from one planar float source.

    Exactly one mutation is selected:

    * ``dtype=`` converts float32/float64 pixels to uint8 or uint16 using the
      existing measured and reported dtype-mutation policy.
    * ``float32_mantissa_bits=`` keeps float32 storage and exponent range while
      rounding the stored fraction to the requested number of bits.

    Both operations stream through the same :class:`OMETiffWriter` used by
    conversion and rebuild all pyramid levels from the mutated representation.
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

    dtype_requested = dtype is not None
    precision_requested = float32_mantissa_bits is not None
    if dtype_requested == precision_requested:
        raise ValueError(
            "mutate requires exactly one pixel mutation: set dtype= or "
            "float32_mantissa_bits=, but not both"
        )

    normalized_mantissa_bits: int | None = None
    effective_range_mode: RangeMode | None = None
    if dtype_requested:
        if dtype not in TARGET_DTYPES:
            raise ValueError("dtype must be 'uint8' or 'uint16'")
        effective_range_mode = "auto" if range_mode is None else range_mode
        if effective_range_mode not in RANGE_MODES:
            raise ValueError("range_mode must be 'auto', 'preserve', or 'full'")
    else:
        if range_mode is not None:
            raise ValueError("range_mode is only valid with dtype mutation")
        normalized_mantissa_bits = normalize_float32_mantissa_bits(
            float32_mantissa_bits
        )
        assert normalized_mantissa_bits is not None
        if normalized_mantissa_bits == 23:
            raise ValueError(
                "float32_mantissa_bits=23 retains full float32 precision; use convert "
                "when no pixel-value mutation is intended"
            )

    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize instance")
    if compression.strip().lower() in {"jpeg", "jpg"}:
        raise ValueError("quantitative mutation output requires lossless compression")

    operation = "dtype" if dtype_requested else "float32_precision"
    start_epoch = time.time()
    LOGGER.info(
        "Mutate workflow: opening %s as input type %s, series %s; operation=%s",
        input_file,
        input_type,
        int(series),
        operation,
    )
    reader = source_reader(
        input_file,
        input_type=input_type,
        series=int(series),
        channel_name_field=channel_name_field,
    )
    with reader, ExitStack() as stack:
        if reader.image_type == "rgb":
            raise ValueError("mutation currently supports planar grayscale channels only")
        if reader.axes not in {"YX", "CYX"}:
            raise ValueError(
                f"mutation requires planar YX or CYX output, found {reader.axes!r}"
            )
        if not np.issubdtype(reader.dtype, np.floating):
            raise TypeError(
                f"mutation requires float32 or float64 input, found {reader.dtype}"
            )
        if precision_requested and reader.dtype != np.dtype("float32"):
            raise TypeError(
                "float32 mantissa mutation requires float32 input; "
                f"found {reader.dtype}"
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

        dtype_mutation_report: dict[str, object] | None = None
        float_precision_report: dict[str, object] | None = None

        if dtype_requested:
            assert dtype is not None
            assert effective_range_mode is not None
            LOGGER.info(
                "Mutation boundary: %s -> %s using range mode %s; source metadata and "
                "geometry remain on the shared conversion path",
                reader.dtype,
                dtype,
                effective_range_mode,
            )
            LOGGER.info(
                "Scanning every full-resolution channel before writing so each channel gets "
                "one fixed, auditable mapping"
            )
            analysis_started = time.monotonic()
            plans = analyze_dtype_mutation(
                reader,
                dtype=dtype,
                range_mode=effective_range_mode,
                sample_pixels_per_channel=sample_pixels_per_channel,
                auto_max_normalized_rmse=auto_max_normalized_rmse,
            )
            LOGGER.info(
                "Mutation planning completed in %s",
                readable_runtime(time.monotonic() - analysis_started),
            )
            transformed = stack.enter_context(MultichannelImage(DTypeMutationSource(reader, plans)))
            writer_mantissa_bits = None
            dtype_mutation_report = {
                "protocol_version": DTYPE_MUTATION_PROTOCOL_VERSION,
                "source_dtype": reader.dtype.name,
                "target_dtype": dtype,
                "range_mode": effective_range_mode,
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
                        "Auto preserves unit scale when the nearest-integer range fits the "
                        "target dtype and either full-resolution nonzero values provide "
                        "strong integer-lattice evidence or unit-rounding RMSE is within the "
                        "configured fraction of the sampled nonzero robust intensity span; "
                        "otherwise it uses a zero-anchored linear mapping."
                    ),
                },
                "channels": [plan.report for plan in plans],
            }
        else:
            assert normalized_mantissa_bits is not None
            transformed = reader
            writer_mantissa_bits = normalized_mantissa_bits
            precision_bits = normalized_mantissa_bits + 1
            significant_bits = float32_significant_bits(normalized_mantissa_bits)
            LOGGER.info(
                "Mutation boundary: float32 storage retained with %s fraction bits "
                "(%s-bit significand precision, OME SignificantBits=%s)",
                normalized_mantissa_bits,
                precision_bits,
                significant_bits,
            )
            float_precision_report = {
                "protocol_version": FLOAT32_PRECISION_MUTATION_PROTOCOL_VERSION,
                "source_dtype": "float32",
                "target_dtype": "float32",
                "float32_mantissa_bits": normalized_mantissa_bits,
                "float_significand_precision_bits": precision_bits,
                "ome_significant_bits": significant_bits,
                "rounding": "nearest, ties to even",
                "lossy": True,
                "storage_dtype_preserved": True,
                "float32_exponent_range_preserved": True,
                "description": (
                    "The float32 storage type and exponent range are preserved while lower "
                    "fraction bits are rounded away before writing. Pyramid levels are rebuilt "
                    "from the precision-trimmed representation and rounded to the same precision."
                ),
            }

        LOGGER.info(
            "Using the shared OME-TIFF writer through a streaming mutation adapter "
            "(compression=%s, tile=%s, downsample=%s)",
            compression,
            tile_size,
            downsample,
        )
        writer = OMETiffWriter(
            output_file,
            compression=compression,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,
            max_workers=max_workers,
            display_uuid=display_uuid,
            software=software,
            float32_mantissa_bits=writer_mantissa_bits,
            overwrite=overwrite,
            cache_directory=cache_directory,
        )
        writer_started = time.monotonic()
        with transformed.with_metadata(
            channel_names=output_channel_names, pixel_size=effective_pixel_size,
        ) as output_image:
            write_report = writer.write(output_image)
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
            "operation": operation,
            "input_type": input_type,
            "series": int(series),
            "channel_name_field": channel_name_field,
            "rename_channels": dict(normalized_renames),
            "rename_channels_by": normalized_rename_mode,
            "pixel_size_override": (
                None if pixel_size is None else list(pixel_size.to_tuple())
            ),
        }
    )
    if dtype_requested:
        options.update(
            {
                "dtype": dtype,
                "range_mode": effective_range_mode,
                "sample_pixels_per_channel": int(sample_pixels_per_channel),
                "auto_max_normalized_rmse": float(auto_max_normalized_rmse),
            }
        )
    else:
        options["float32_mantissa_bits"] = normalized_mantissa_bits

    stop_epoch = time.time()
    report: dict[str, object] = {
        "operation": operation,
        "ome": write_report["ome"],
        "miti_header": write_report["miti_header"],
        "input_file": input_report,
        "output_file": output_report,
        "image": write_report["image"],
        "pyramid": write_report["pyramid"],
        "verification": write_report["verification"],
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
    if dtype_mutation_report is not None:
        report["dtype_mutation"] = dtype_mutation_report
    if float_precision_report is not None:
        report["float_precision_mutation"] = float_precision_report

    LOGGER.info(
        "Mutate complete: %s (%s bytes) in %s",
        output_file,
        f"{output_size:,}",
        readable_runtime(stop_epoch - start_epoch),
    )
    return report
