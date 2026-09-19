"""Explicit file origins; all readers share Image's lifecycle and regional access."""
from __future__ import annotations

from pathlib import Path

from ._tiff_sources.vendor import VendorSource
from .akoya_qptiff import ChannelNameField
from .base import MultichannelImage, RGBImage
from .file_image import FileImage
from .pixel_size import PixelSize


class AkoyaMIFQPTiffReader(FileImage, MultichannelImage):
    input_type_description = "Akoya mIF QPTIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="akoya_mif_qptiff", description=self.input_type_description,
            series=series, pixel_size=pixel_size,
        ))


class AkoyaFusionQPTiffReader(FileImage, MultichannelImage):
    input_type_description = "Akoya Fusion multiplex QPTIFF"

    def __init__(
        self, path: str | Path, *, series: int = 0,
        channel_name_field: ChannelNameField = "auto", pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="akoya_fusion_qptiff", description=self.input_type_description,
            series=series, pixel_size=pixel_size, channel_name_field=channel_name_field,
        ))


class IndicaMIFTiffReader(FileImage, MultichannelImage):
    input_type_description = "Indica Labs/HALO mIF TIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="indica_mif", description=self.input_type_description,
            series=series, pixel_size=pixel_size,
        ))


class AkoyaHEQPTiffReader(FileImage, RGBImage):
    input_type_description = "Akoya H&E QPTIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="akoya_he_qptiff", description=self.input_type_description,
            series=series, pixel_size=pixel_size,
        ))


class AperioSVSReader(FileImage, RGBImage):
    input_type_description = "Aperio SVS"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="svs", description=self.input_type_description,
            series=series, pixel_size=pixel_size,
        ))


class AkoyaComponentTiffReader(FileImage, MultichannelImage):
    input_type_description = "Akoya Component TIFF"

    def __init__(
        self,
        path: str | Path,
        *,
        series: int = 0,
        pixel_size: PixelSize | None = None,
    ) -> None:
        super().__init__(VendorSource(
            path, profile="component", description=self.input_type_description,
            series=series, pixel_size=pixel_size,
        ))
