from __future__ import annotations

from dataclasses import dataclass

from omeify.io.ome_tiff_writer import CompressionSettings

from .image_series import OMEImageSeries


@dataclass(frozen=True, slots=True)
class PreparedSeries:
    image: OMEImageSeries
    level_shapes: tuple[tuple[int, ...], ...]
    compression: CompressionSettings
