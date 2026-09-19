from __future__ import annotations

from typing import Any

from .array_source import ArraySource
from .base import Image, LabelImage, MultichannelImage, RGBImage
from .channel import Channel
from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ImageSource
from .pixel_size import PixelSize
from .reader_source import OMETiffSource, ReaderImageSource
from .spec import ImageType, OMEImageSpec
from .tiff import ArrayPlaneReader, PlaneReader, TiffPlaneReader
from .vendor_tiff_readers import (
    AkoyaComponentTiffReader,
    AkoyaFusionQPTiffReader,
    AkoyaHEQPTiffReader,
    AkoyaMIFQPTiffReader,
    AperioSVSReader,
    IndicaMIFTiffReader,
)

__all__ = [
    "AkoyaComponentTiffReader",
    "AkoyaFusionQPTiffReader",
    "AkoyaHEQPTiffReader",
    "AkoyaMIFQPTiffReader",
    "AperioSVSReader",
    "ArrayPlaneReader",
    "ArraySource",
    "Channel",
    "Image",
    "ImageLevel",
    "ImageMetadata",
    "ImageSource",
    "ImageType",
    "IndicaMIFTiffReader",
    "LabelImage",
    "MultichannelImage",
    "OMEImageSeries",
    "OMEImageSpec",
    "OMEMultiSeriesWriter",
    "OMETiffLabelReader",
    "OMETiffReader",
    "OMETiffSource",
    "OMETiffWriter",
    "PixelSize",
    "PlaneReader",
    "PlaneReaderSource",
    "RGBImage",
    "ReaderImageSource",
    "TemporaryOMETiffWriter",
    "TiffPlaneReader",
    "write_ometiff",
]


def __getattr__(name: str) -> Any:
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
    if name in {"OMEImageSeries", "OMEMultiSeriesWriter"}:
        from .ome_multi_series_writer import (
            OMEImageSeries,
            OMEMultiSeriesWriter,
        )

        return {
            "OMEImageSeries": OMEImageSeries,
            "OMEMultiSeriesWriter": OMEMultiSeriesWriter,
        }[name]
    if name == "write_ometiff":
        from .convenience import write_ometiff

        return write_ometiff
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
