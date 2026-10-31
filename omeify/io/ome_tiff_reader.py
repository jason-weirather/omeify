"""OME-TIFF origins for the shared semantic image interface."""
from __future__ import annotations

from .dataset import PlaneReaderMixin

from pathlib import Path

from ._tiff_sources.ome import OMESource
from .base import LabelImage, MultichannelImage
from .file_image import FileImage


class OMETiffReader(PlaneReaderMixin, FileImage, MultichannelImage):
    """Open one OME-TIFF intensity or RGB series; common Image methods do all reads."""

    input_type_description = "OME-TIFF"

    def __init__(self, path: str | Path, *, series: int = 0, z=None, t=None,
                 cache_bytes=64 * 1024 * 1024) -> None:
        super().__init__(OMESource(path, series=series, z=z, t=t, cache_bytes=cache_bytes))


class OMETiffLabelReader(PlaneReaderMixin, FileImage, LabelImage):
    """Explicit categorical interpretation of one integer OME-TIFF series."""

    input_type_description = "OME-TIFF labels"

    def __init__(
        self, path: str | Path, *, series: int = 0, background_label: int = 0, z=None, t=None,
        cache_bytes=64 * 1024 * 1024,
    ) -> None:
        super().__init__(OMESource(
            path, series=series, labels=True, background_label=background_label,
            z=z, t=t, cache_bytes=cache_bytes,
        ))
