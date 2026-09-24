from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import PlaneReader
from omeify.utils.miti_header_validator import (
    MITIHeaderValidation,
    validate_miti_ome_tiff_header,
)
from omeify.utils.ome_schema_validator import OMESchemaValidator

from .configuration import (
    DownsampleMethod,
    LossyCompressionPolicy,
    WriterSettings,
    compression_settings,
    resolve_compression_name,
    resolve_downsample,
    resolve_tile_size,
    validate_jpeg_alignment,
    validate_lossy_compression,
)
from .model import PlaneReaderSource, PreparedImage
from .pyramid import auto_level_shapes
from .precision import prepare_float_precision

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MetadataValidation:
    """Successful OME schema and Omeify MITI-header validation."""

    validator: OMESchemaValidator
    xml_is_valid: bool
    miti_header: MITIHeaderValidation


def prepare_image(
    source: PlaneReaderSource,
    spec: OMEImageSpec,
    *,
    name: str | None,
    downsample: DownsampleMethod | None,
    compression_name: str | None,
    settings: WriterSettings,
    lossy_policy: LossyCompressionPolicy,
) -> PreparedImage:
    """Validate one source and resolve its storage and pyramid policy."""

    if not isinstance(source, PlaneReaderSource):
        raise TypeError("source must implement PlaneReaderSource")
    if not isinstance(spec, OMEImageSpec):
        raise TypeError("spec must be an OMEImageSpec")
    source, spec, float32_mantissa_bits = prepare_float_precision(
        source,
        spec,
        settings.float32_mantissa_bits,
    )
    effective_downsample = resolve_downsample(spec.image_type, downsample)
    tile_size = resolve_tile_size(spec.image_type, settings.tile_size)
    display_name = "image" if name is None else f"series {name!r}"

    LOGGER.info(
        "Writer preflight: validating %s source plane(s) for %s %s %s %s",
        spec.plane_count,
        spec.image_type,
        spec.output_axes,
        spec.output_shape,
        display_name,
    )
    readers = source.plane_readers(cache_mib=64)
    try:
        validate_readers(readers, spec)
    finally:
        for reader in readers:
            reader.clear_cache()

    compression = compression_settings(
        resolve_compression_name(spec.image_type, compression_name),
        spec.dtype,
        is_rgb=spec.is_rgb,
        jpeg_quality=settings.jpeg_quality,
        jpeg_subsampling=settings.jpeg_subsampling,
    )
    validate_lossy_compression(
        image_type=spec.image_type,
        lossless=compression.lossless,
        policy=lossy_policy,
    )
    validate_jpeg_alignment(
        tile_size=tile_size,
        subsampling=compression.subsampling,
        jpeg_subsampling=settings.jpeg_subsampling,
    )
    level_shapes = auto_level_shapes(
        spec.output_shape,
        axes=spec.output_axes,
        tile_size=tile_size,
        pyramid_levels=settings.pyramid_levels,
    )
    LOGGER.info(
        "Writer plan for %s: compression=%s (%s), tile=%sx%s, workers=%s, "
        "downsample=%s, subresolution-levels=%s, float-mantissa-bits=%s",
        display_name,
        compression.name,
        "lossless" if compression.lossless else "lossy",
        tile_size,
        tile_size,
        settings.max_workers,
        effective_downsample,
        len(level_shapes) - 1,
        float32_mantissa_bits,
    )
    LOGGER.info(
        "%s pyramid shapes: %s",
        display_name.capitalize(),
        ", ".join(
            f"L{index}={shape}"
            for index, shape in enumerate(level_shapes)
        ),
    )
    return PreparedImage(
        source=source,
        spec=spec,
        downsample=effective_downsample,
        compression=compression,
        level_shapes=level_shapes,
        tile_size=tile_size,
        float32_mantissa_bits=float32_mantissa_bits,
        name=name,
    )


def validate_readers(
    readers: Sequence[PlaneReader],
    spec: OMEImageSpec,
) -> None:
    """Validate one reader per physical TIFF plane against an image spec."""

    if len(readers) != spec.plane_count:
        raise ValueError(
            f"Source supplied {len(readers)} planes; expected {spec.plane_count} "
            f"for {spec.output_axes} shape {spec.output_shape}."
        )
    for index, reader in enumerate(readers):
        if reader.height != spec.size_y or reader.width != spec.size_x:
            raise ValueError(
                f"Source plane {index} has shape {(reader.height, reader.width)}; "
                f"expected {(spec.size_y, spec.size_x)}."
            )
        if np.dtype(reader.dtype).newbyteorder("=") != spec.dtype:
            raise TypeError(
                f"Source plane {index} has dtype {reader.dtype}; expected {spec.dtype}."
            )
        if int(reader.samples_per_pixel) != spec.samples_per_pixel:
            raise ValueError(
                f"Source plane {index} has SamplesPerPixel={reader.samples_per_pixel}; "
                f"expected {spec.samples_per_pixel}."
            )


def validate_ome_xml(
    omexml: str,
    *,
    context: str,
) -> MetadataValidation:
    """Validate generated OME-XML against both supported metadata contracts."""

    LOGGER.info("Generating and validating %s OME-XML", context)
    validator = OMESchemaValidator()
    xml_is_valid = validator.validate(omexml)
    if xml_is_valid is None:
        raise RuntimeError(
            "OME-XML schema validation could not be performed because no local "
            "OME 2016-06 schema was available"
        )
    if not xml_is_valid:
        raise ValueError("Generated OME-XML failed OME 2016-06 schema validation")
    miti_header = validate_miti_ome_tiff_header(omexml)
    if not miti_header.is_valid:
        details = "; ".join(miti_header.errors)
        raise ValueError(
            f"Generated {context} OME header failed omeify MITI validation: {details}"
        )
    return MetadataValidation(
        validator=validator,
        xml_is_valid=xml_is_valid,
        miti_header=miti_header,
    )
