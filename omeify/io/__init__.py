from __future__ import annotations

from typing import Any

from .base import Image, LabelImage, MultichannelImage, RGBImage
from .channel import Channel
from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ImageSource
from .pixel_size import PixelSize
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
    "Channel",
    "Image",
    "ImageLevel",
    "ImageMetadata",
    "ImageSource",
    "IndicaMIFTiffReader",
    "LabelImage",
    "MultichannelImage",
    "OMEImageSeries",
    "OMEMultiSeriesWriter",
    "OMETiffLabelReader",
    "OMETiffReader",
    "OMETiffWriter",
    "PixelSize",
    "RGBImage",
    "TemporaryOMETiffWriter",
]


def __getattr__(name: str) -> Any:
    if name in {"OMETiffReader", "OMETiffLabelReader"}:
        from .ome_tiff_reader import OMETiffLabelReader, OMETiffReader

        return {"OMETiffReader": OMETiffReader, "OMETiffLabelReader": OMETiffLabelReader}[name]
    if name in {"OMETiffWriter", "TemporaryOMETiffWriter"}:
        from .ome_tiff_writer import (
            OMETiffWriter,
            TemporaryOMETiffWriter,
        )

        return {
            "OMETiffWriter": OMETiffWriter,
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
