from __future__ import annotations

from typing import Any

from .base import Image, LabelImage, MultichannelImage, RGBImage
from .spec import ImageType, OMEImageSpec
from .tiff import ArrayPlaneReader, PlaneReader, TiffPlaneReader

__all__ = [
    "ArrayPlaneReader",
    "Image",
    "ImageType",
    "LabelImage",
    "MultichannelImage",
    "OMEImageSpec",
    "OMETiffLabelReader",
    "OMETiffReader",
    "OMETiffWriter",
    "PlaneReader",
    "PlaneReaderSource",
    "RGBImage",
    "TiffPlaneReader",
]


def __getattr__(name: str) -> Any:
    if name in {"OMETiffReader", "OMETiffLabelReader"}:
        from .ome_tiff_reader import OMETiffLabelReader, OMETiffReader

        return {"OMETiffReader": OMETiffReader, "OMETiffLabelReader": OMETiffLabelReader}[name]
    if name in {"OMETiffWriter", "PlaneReaderSource"}:
        from .ome_tiff_writer import OMETiffWriter, PlaneReaderSource

        return {
            "OMETiffWriter": OMETiffWriter,
            "PlaneReaderSource": PlaneReaderSource,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
