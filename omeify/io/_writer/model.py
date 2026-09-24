from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import PlaneReader

from .configuration import CompressionSettings, DownsampleMethod


@runtime_checkable
class PlaneReaderSource(Protocol):
    """Source contract consumed by Omeify's shared writer engine."""

    def plane_readers(self) -> list[PlaneReader]:
        """Return one random-access reader for each physical TIFF plane."""


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """One validated image ready for shared pyramid and TIFF construction."""

    source: PlaneReaderSource
    spec: OMEImageSpec
    downsample: DownsampleMethod
    compression: CompressionSettings
    level_shapes: tuple[tuple[int, ...], ...]
    tile_size: int
    float32_mantissa_bits: int | None = None
    name: str | None = None

    @property
    def display_name(self) -> str:
        return "image" if self.name is None else f"series {self.name!r}"


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Result returned by the shared atomic writer engine."""

    verification: dict[str, object]
    output_size: int
