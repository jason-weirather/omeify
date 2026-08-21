from __future__ import annotations

from typing import Any

from .base import Image, LabelImage, MultichannelImage, RGBImage
from .channel import Channel, RenameChannelsBy
from .pixel_size import PixelSize
from .spec import ImageType, OMEImageSpec
from .tiff import ArrayPlaneReader, PlaneReader, TiffPlaneReader

__all__ = [
    "AkoyaFusionQPTiffReader",
    "ArrayPlaneReader",
    "Channel",
    "Image",
    "ImageType",
    "LabelImage",
    "MultichannelImage",
    "OMEImageSpec",
    "OMETiffLabelReader",
    "OMETiffReader",
    "OMETiffWriter",
    "PixelSize",
    "PlaneReader",
    "PlaneReaderSource",
    "RGBImage",
    "RenameChannelsBy",
    "TemporaryOMETiffWriter",
    "TiffPlaneReader",
    "write_ometiff",
]


def __getattr__(name: str) -> Any:
    if name == "AkoyaFusionQPTiffReader":
        from .akoya_fusion_qptiff_reader import AkoyaFusionQPTiffReader

        return AkoyaFusionQPTiffReader
    if name in {"OMETiffReader", "OMETiffLabelReader"}:
        from .ome_tiff_reader import OMETiffLabelReader, OMETiffReader

        return {"OMETiffReader": OMETiffReader, "OMETiffLabelReader": OMETiffLabelReader}[name]
    if name in {"OMETiffWriter", "PlaneReaderSource", "TemporaryOMETiffWriter"}:
        from .ome_tiff_writer import (
            OMETiffWriter,
            PlaneReaderSource,
            TemporaryOMETiffWriter,
        )

        return {
            "OMETiffWriter": OMETiffWriter,
            "PlaneReaderSource": PlaneReaderSource,
            "TemporaryOMETiffWriter": TemporaryOMETiffWriter,
        }[name]
    if name == "write_ometiff":
        from .convenience import write_ometiff

        return write_ometiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
