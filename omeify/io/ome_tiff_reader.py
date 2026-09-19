"""OME-TIFF origins for the shared semantic image interface."""
from __future__ import annotations

from pathlib import Path

from ._tiff_sources.ome import OMESource
from .base import LabelImage, MultichannelImage
from .file_image import FileImage


class OMETiffReader(FileImage, MultichannelImage):
    """Open one OME-TIFF intensity or RGB series; common Image methods do all reads."""

    input_type_description = "OME-TIFF"

    def __init__(self, path: str | Path, *, series: int = 0) -> None:
        super().__init__(OMESource(path, series=series))


class OMETiffLabelReader(FileImage, LabelImage):
    """Explicit categorical interpretation of one integer OME-TIFF series."""

    input_type_description = "OME-TIFF labels"

    def __init__(
        self, path: str | Path, *, series: int = 0, background_label: int = 0,
    ) -> None:
        super().__init__(OMESource(
            path, series=series, labels=True, background_label=background_label,
        ))
