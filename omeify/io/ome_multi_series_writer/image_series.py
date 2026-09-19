"""One named heterogeneous output entry, borrowing an open Image."""
from __future__ import annotations

from dataclasses import dataclass, field

from omeify.io._writer import DownsampleMethod, resolve_downsample
from omeify.io.base import Image
from omeify.io.image_planes import ImagePlaneSource


@dataclass(frozen=True, slots=True)
class OMEImageSeries:
    """Name an Image for multi-series output and optionally choose storage policy.

    Keep the image open until the write completes. No pixels are read here; this
    entry pins the current image session, not a future reopening of its source.
    """

    name: str
    image: Image
    level: int = 0
    compression: str | None = None
    downsample: DownsampleMethod | None = None
    _source: ImagePlaneSource = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("OME series name must be a non-empty string")
        object.__setattr__(self, "name", self.name.strip())
        source = ImagePlaneSource(self.image, level=self.level)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "downsample", resolve_downsample(
            self.image.image_type,
            self.downsample,
        ))
        if self.compression is not None:
            if not isinstance(self.compression, str) or not self.compression.strip():
                raise ValueError("series compression must be a non-empty string or None")
            object.__setattr__(self, "compression", self.compression.strip())
