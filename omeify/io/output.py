"""Explicit storage selection for the shared image workflows."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from .ome_multi_series_writer import OMEMultiSeriesWriter
from .ome_tiff_writer import OMETiffWriter

OutputFormat = Literal["ome_tiff", "ome_zarr"]
OUTPUT_FORMATS = ("ome_tiff", "ome_zarr")


def image_writer(path: str | Path, *, output_format: OutputFormat = "ome_tiff",
                 collection: bool = False, compression=None, tile_size=None,
                 pyramid_levels=None, downsample="mean", max_workers=None,
                 float32_mantissa_bits=None, overwrite=True, cache_directory=None,
                 display_uuid=True, software=None, jpeg_quality=90, jpeg_subsampling="422"):
    """Choose storage without duplicating image transforms or changing caller pixels.

    General workflow overwrite permission never authorizes replacing a Zarr tree.
    New Zarr stores always use atomic no-replace publication.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"Unsupported output_format {output_format!r}")
    settings = dict(compression=compression, tile_size=tile_size,
                    pyramid_levels=pyramid_levels, max_workers=max_workers,
                    float32_mantissa_bits=float32_mantissa_bits)
    if not collection:
        settings["downsample"] = downsample
    if output_format == "ome_zarr":
        from .ome_zarr import OMEZarrCollectionWriter, OMEZarrWriter
        if (cache_directory is not None or not display_uuid or software is not None
                or jpeg_quality != 90 or jpeg_subsampling != "422"):
            raise ValueError("TIFF scratch, UUID, software and JPEG settings do not apply to OME-Zarr")
        writer = OMEZarrCollectionWriter if collection else OMEZarrWriter
        return writer(path, **settings)
    settings.update(overwrite=overwrite, cache_directory=cache_directory,
                    display_uuid=display_uuid, software=software,
                    jpeg_quality=jpeg_quality, jpeg_subsampling=jpeg_subsampling)
    return (OMEMultiSeriesWriter if collection else OMETiffWriter)(path, **settings)
