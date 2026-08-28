from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import tifffile

from omeify._version import __version__
from omeify.io.ome_tiff_writer import (
    JPEGSubsampling,
    OMETiffWriter,
    _auto_level_shapes,
    _compression_settings,
    _iter_downsampled_tiles,
    _iter_tiles,
    _OUTPUT_BYTEORDER,
    _page_readers,
    _resolution,
)
from omeify.io.tiff import PlaneReader

from .image_series import OMEImageSeries
from .model import PreparedSeries

LOGGER = logging.getLogger(__name__)


def prepare_series(
    image: OMEImageSeries,
    *,
    compression_name: str,
    jpeg_quality: int,
    jpeg_subsampling: JPEGSubsampling,
    tile_size: int,
    pyramid_levels: int | None,
) -> PreparedSeries:
    """Validate one source and resolve its storage and pyramid policy."""

    readers = image.source.plane_readers(cache_mib=64)
    OMETiffWriter._validate_readers(readers, image.spec)
    for reader in readers:
        reader.clear_cache()

    compression = _compression_settings(
        compression_name,
        image.spec.dtype,
        is_rgb=image.spec.is_rgb,
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
    )
    if not image.spec.is_rgb and not compression.lossless:
        raise ValueError(
            "Lossy compression is restricted to RGB OME-TIFF series; "
            "multichannel and label series require lossless compression"
        )
    if compression.subsampling is not None:
        jpeg_alignment = max(compression.subsampling) * 8
        if tile_size % jpeg_alignment != 0:
            raise ValueError(
                f"TIFF tile size {tile_size} is incompatible with JPEG "
                f"{jpeg_subsampling} subsampling; use a tile size divisible by "
                f"{jpeg_alignment}."
            )
    level_shapes = tuple(
        _auto_level_shapes(
            image.spec.output_shape,
            axes=image.spec.output_axes,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
        )
    )
    return PreparedSeries(image, level_shapes, compression)


def build_pyramid(
    prepared: PreparedSeries,
    directory: Path,
    *,
    tile_size: int,
) -> tuple[Path, ...]:
    """Build one temporary pyramid without materializing the base raster."""

    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    spec = prepared.image.spec
    source = prepared.image.source
    for level_index, output_shape in enumerate(
        prepared.level_shapes[1:],
        start=1,
    ):
        output_path = directory / f"level-{level_index}.tf8"
        LOGGER.info(
            "Building series %r pyramid level %s with shape %s",
            prepared.image.name,
            level_index,
            output_shape,
        )
        if level_index == 1:
            readers = source.plane_readers()
            _write_temporary_level(
                readers,
                output_shape,
                prepared,
                output_path,
                tile_size=tile_size,
            )
        else:
            with tifffile.TiffFile(paths[-1]) as previous:
                readers = _page_readers(previous, spec.plane_count)
                _write_temporary_level(
                    readers,
                    output_shape,
                    prepared,
                    output_path,
                    tile_size=tile_size,
                )
        paths.append(output_path)
    return tuple(paths)


def _write_temporary_level(
    readers: Sequence[PlaneReader],
    output_shape: tuple[int, ...],
    prepared: PreparedSeries,
    output_path: Path,
    *,
    tile_size: int,
) -> None:
    spec = prepared.image.spec
    options: dict[str, object] = {
        "shape": output_shape,
        "dtype": spec.dtype,
        "photometric": spec.photometric,
        "tile": (tile_size, tile_size),
        "compression": None,
        "metadata": None,
        "software": False,
        "maxworkers": 1,
    }
    if spec.is_rgb:
        options["planarconfig"] = "contig"
    with tifffile.TiffWriter(
        output_path,
        bigtiff=True,
        byteorder=_OUTPUT_BYTEORDER,
        ome=False,
    ) as writer:
        writer.write(
            _iter_downsampled_tiles(
                readers,
                output_shape,
                axes=spec.output_axes,
                plane_count=spec.plane_count,
                tile_size=tile_size,
                method=prepared.image.downsample,
            ),
            **options,
        )


def write_output(
    prepared: Sequence[PreparedSeries],
    level_paths: Sequence[Sequence[Path]],
    output_path: Path,
    omexml: str,
    *,
    tile_size: int,
    max_workers: int,
) -> None:
    """Write all base Images and their SubIFD pyramids into one BigTIFF."""

    software = f"omeify {__version__}"
    with tifffile.TiffWriter(
        output_path,
        bigtiff=True,
        byteorder=_OUTPUT_BYTEORDER,
        ome=False,
    ) as writer:
        for series_index, (item, paths) in enumerate(zip(prepared, level_paths)):
            spec = item.image.spec
            common_options = _common_write_options(
                item,
                tile_size=tile_size,
                max_workers=max_workers,
            )
            base_options: dict[str, object] = {
                "shape": item.level_shapes[0],
                "description": omexml.encode("utf-8") if series_index == 0 else None,
                "software": software if series_index == 0 else False,
                "subifds": len(paths),
                "resolution": _resolution(spec.pixel_size, 1),
                "resolutionunit": "CENTIMETER",
                **common_options,
            }
            writer.write(
                _iter_tiles(
                    item.image.source.plane_readers(),
                    item.level_shapes[0],
                    axes=spec.output_axes,
                    plane_count=spec.plane_count,
                    tile_size=tile_size,
                ),
                **base_options,
            )

            for level_index, (level_path, level_shape) in enumerate(
                zip(paths, item.level_shapes[1:]),
                start=1,
            ):
                with tifffile.TiffFile(level_path) as level_tiff:
                    readers = _page_readers(level_tiff, spec.plane_count)
                    writer.write(
                        _iter_tiles(
                            readers,
                            level_shape,
                            axes=spec.output_axes,
                            plane_count=spec.plane_count,
                            tile_size=tile_size,
                        ),
                        shape=level_shape,
                        software=False,
                        subfiletype=1,
                        resolution=_resolution(spec.pixel_size, 2**level_index),
                        resolutionunit="CENTIMETER",
                        **common_options,
                    )


def _common_write_options(
    prepared: PreparedSeries,
    *,
    tile_size: int,
    max_workers: int,
) -> dict[str, object]:
    spec = prepared.image.spec
    compression = prepared.compression
    options: dict[str, object] = {
        "dtype": spec.dtype,
        "photometric": spec.photometric,
        "tile": (tile_size, tile_size),
        "compression": compression.tifffile_value,
        "compressionargs": compression.compression_args,
        "predictor": compression.predictor,
        "metadata": None,
        "maxworkers": max_workers,
    }
    if spec.is_rgb:
        options["planarconfig"] = "contig"
        if compression.subsampling is not None:
            options["subsampling"] = compression.subsampling
        if spec.icc_profile is not None:
            options["iccprofile"] = spec.icc_profile
    return options
