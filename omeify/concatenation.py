"""Assemble selected OME-TIFF images into a verified OME-TIFF collection.

Concatenation is an ordered image-collection operation, not planar channel
concatenation, registration, or binary TIFF IFD splicing. Each source stays open
only while the existing verified, regional multi-series writer consumes it.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io.image_metadata import integer
from .io.ome_multi_series_writer import OMEImageSeries, OMEMultiSeriesWriter
from .io.ome_tiff_reader import OMETiffReader
from .path_safety import check_output_path


@dataclass(frozen=True, slots=True)
class ConcatSeries:
    """One zero-based source TIFF series, with an optional output name."""

    path: str | Path
    series: int
    name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)):
            raise TypeError("Source path must be a string or Path")
        if not str(self.path).strip():
            raise ValueError("Source path must not be empty")
        path = Path(self.path)
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "series", integer(self.series, "series"))
        if self.name is not None:
            if not isinstance(self.name, str) or not self.name.strip():
                raise ValueError("Explicit series names must be non-empty strings")
            object.__setattr__(self, "name", self.name.strip())


def _source_significant_bits(reader: OMETiffReader) -> int | None:
    """Preserve OME SignificantBits without re-quantizing existing source samples."""
    report = reader.inspect().report
    image_index = report["calibration"]["series"][reader.series_index]["ome_image_index"]
    if image_index is None:
        # The reader itself already rejects an ambiguous OME TiffData association.
        raise ValueError("Selected series has no unambiguous local OME Image mapping")
    images = report["ome"]["images"]
    bits = images[image_index]["significant_bits"]
    return None if bits is None else int(bits)


def _source_downsample(reader: OMETiffReader) -> str:
    """Honor known pyramid geometry; otherwise conservatively handle scalar IDs.

    A normal OME image does not say whether an integer grayscale plane contains
    labels. In the absence of an Omeify geometry record, nearest-neighbor avoids
    averaging unknown integer IDs. RGB/other intensity images use ordinary mean.
    """
    levels = reader.levels
    if reader.origin_yx is not None and len(levels) > 1:
        level = levels[1]
        if level.downsample_yx is not None and level.offset_yx is not None:
            offsets = tuple(float(v) for v in level.offset_yx)
            nearest = tuple((1.0 - float(v)) / 2.0 for v in level.downsample_yx)
            if all(math.isclose(a, b, rel_tol=0, abs_tol=1e-9)
                   for a, b in zip(offsets, nearest, strict=True)):
                return "nearest"
            if all(math.isclose(v, 0.0, rel_tol=0, abs_tol=1e-9) for v in offsets):
                return "mean"
    if (reader.channel_count == 1 and reader.image_type != "rgb"
            and np.issubdtype(reader.dtype, np.integer)):
        return "nearest"
    return "mean"


def concat(
    inputs: Sequence[ConcatSeries],
    output_path: str | Path,
    *,
    compression: str = "LZW",
    tile_size: int | None = None,
    pyramid_levels: int | None = None,
    max_workers: int | None = None,
    cache_directory: str | Path | None = None,
    overwrite: bool = True,
) -> dict[str, object]:
    """Write an ordered collection using lossless TIFF compression.

    ``ConcatSeries(path, index)`` retains the source OME Image name when present;
    ``ConcatSeries(path, index, name)`` overrides it. Source files may repeat and
    have different shapes, dtypes, channel counts, resolutions, or origins. Pixels
    are never registered, spatially resampled, cast, or quantized at level zero.
    Noncanonical RGB ICC profiles are mapped to sRGB before lossless encoding. Pyramids
    are reconstructed under the existing writer's bounded I/O/verification policy.

    Only lossless codecs are accepted; even RGB is lossless by default. An unnamed
    source gets a generic index-only fallback without copying source filenames.
    Duplicate output names are errors, with no silent disambiguation.
    """
    if isinstance(inputs, (str, bytes)) or not isinstance(inputs, Sequence):
        raise TypeError("inputs must be a sequence of ConcatSeries entries")
    entries = tuple(inputs)
    if not entries:
        raise ValueError("Provide at least one source series")
    if not all(isinstance(item, ConcatSeries) for item in entries):
        raise TypeError("inputs must contain only ConcatSeries entries")
    if not isinstance(compression, str) or compression.lower() not in {
        "lzw", "deflate", "zstd", "uncompressed",
    }:
        raise ValueError(
            "concat supports only lossless compression: LZW, Deflate, ZSTD, Uncompressed"
        )
    destination = Path(output_path)
    protected = tuple(item.path for item in entries)
    check_output_path(destination, protected=protected, overwrite=overwrite)

    with ExitStack() as resources:
        selected = []
        for item in entries:
            reader = resources.enter_context(OMETiffReader(item.path, series=item.series))
            name = item.name or reader.series_name or f"Series {item.series}"
            selected.append(OMEImageSeries(
                name, reader, downsample=_source_downsample(reader),
                significant_bits=_source_significant_bits(reader),
            ))
        # Preflight names before the expensive pyramid work; no name is silently
        # rewritten, including a name supplied explicitly by --named-series.
        names = [item.name for item in selected]
        if len(set(names)) != len(names):
            raise ValueError(
                "Output series names must be unique; use --named-series to rename duplicates"
            )
        return OMEMultiSeriesWriter(
            destination, compression=compression, tile_size=tile_size,
            pyramid_levels=pyramid_levels, max_workers=max_workers,
            cache_directory=cache_directory, overwrite=overwrite,
        ).write(selected)
