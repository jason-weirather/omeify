from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

import tifffile

from .configuration import OUTPUT_BYTEORDER, WriterSettings
from .model import PreparedImage
from .pyramid import iter_tiles, page_readers, resolution

LOGGER = logging.getLogger(__name__)


def write_output(
    prepared: Sequence[PreparedImage],
    level_paths: Sequence[Sequence[Path]],
    output_path: Path,
    omexml: str,
    *,
    settings: WriterSettings,
) -> None:
    """Write every prepared base image and its SubIFD pyramid into one BigTIFF."""

    if not prepared:
        raise ValueError("at least one prepared image is required")
    if len(prepared) != len(level_paths):
        raise ValueError("prepared images and pyramid path groups must align")

    series_count = len(prepared)
    with tifffile.TiffWriter(
        output_path,
        bigtiff=True,
        byteorder=OUTPUT_BYTEORDER,
        ome=False,
    ) as writer:
        for series_index, (item, paths) in enumerate(
            zip(prepared, level_paths, strict=True)
        ):
            if len(paths) != len(item.level_shapes) - 1:
                raise ValueError(
                    f"{item.display_name} has {len(paths)} staged pyramid levels; "
                    f"expected {len(item.level_shapes) - 1}"
                )
            common_options = common_write_options(
                item,
                tile_size=settings.tile_size,
                max_workers=settings.max_workers,
            )
            base_progress = _base_progress_label(
                item,
                series_index=series_index,
                series_count=series_count,
            )
            base_started = time.monotonic()
            writer.write(
                iter_tiles(
                    item.source.plane_readers(),
                    item.level_shapes[0],
                    axes=item.spec.output_axes,
                    plane_count=item.spec.plane_count,
                    tile_size=settings.tile_size,
                    progress_label=base_progress,
                ),
                shape=item.level_shapes[0],
                description=(
                    omexml.encode("utf-8") if series_index == 0 else None
                ),
                software=settings.software if series_index == 0 else False,
                subifds=len(paths),
                resolution=resolution(item.spec.pixel_size, 1),
                resolutionunit="CENTIMETER",
                **common_options,
            )
            LOGGER.info(
                "%s full-resolution base encoded in %.2f seconds",
                item.display_name.capitalize(),
                time.monotonic() - base_started,
            )

            for level_index, (level_path, level_shape) in enumerate(
                zip(paths, item.level_shapes[1:], strict=True),
                start=1,
            ):
                level_started = time.monotonic()
                with tifffile.TiffFile(level_path) as level_tiff:
                    readers = page_readers(level_tiff, item.spec.plane_count)
                    writer.write(
                        iter_tiles(
                            readers,
                            level_shape,
                            axes=item.spec.output_axes,
                            plane_count=item.spec.plane_count,
                            tile_size=settings.tile_size,
                            progress_label=_level_progress_label(
                                item,
                                level_index=level_index,
                                level_count=len(paths),
                            ),
                        ),
                        shape=level_shape,
                        software=False,
                        subfiletype=1,
                        resolution=resolution(
                            item.spec.pixel_size,
                            2**level_index,
                        ),
                        resolutionunit="CENTIMETER",
                        **common_options,
                    )
                LOGGER.info(
                    "%s pyramid level %s/%s encoded in %.2f seconds",
                    item.display_name.capitalize(),
                    level_index,
                    len(paths),
                    time.monotonic() - level_started,
                )


def common_write_options(
    prepared: PreparedImage,
    *,
    tile_size: int,
    max_workers: int,
) -> dict[str, object]:
    """Return final tifffile options shared by base and pyramid levels."""

    spec = prepared.spec
    compression = prepared.compression
    options: dict[str, object] = {
        "dtype": spec.dtype,
        "photometric": spec.photometric,
        "tile": (tile_size, tile_size),
        "compression": compression.tifffile_value,
        # tifffile may add encoder-specific keys to this dictionary. Keep the
        # prepared policy immutable across base levels, SubIFDs, and series.
        "compressionargs": (
            None if compression.compression_args is None
            else dict(compression.compression_args)
        ),
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


def _base_progress_label(
    prepared: PreparedImage,
    *,
    series_index: int,
    series_count: int,
) -> str:
    if series_count == 1:
        return "Writing final full-resolution base"
    return (
        f"Writing {prepared.display_name} base "
        f"({series_index + 1}/{series_count})"
    )


def _level_progress_label(
    prepared: PreparedImage,
    *,
    level_index: int,
    level_count: int,
) -> str:
    if prepared.name is None:
        return f"Writing final pyramid level {level_index}/{level_count}"
    return (
        f"Writing {prepared.display_name} pyramid level "
        f"{level_index}/{level_count}"
    )
