from __future__ import annotations

from pathlib import Path
from typing import Literal

from .ome_tiff_reader import OMETiffReader
from .vendor_tiff_readers import (
    AkoyaComponentTiffReader,
    AkoyaFusionQPTiffReader,
    AkoyaHEQPTiffReader,
    AkoyaMIFQPTiffReader,
    AperioSVSReader,
    IndicaMIFTiffReader,
)

InputType = Literal[
    "qptiff_mif",
    "qptiff_fusion",
    "qptiff_he",
    "svs",
    "ome_tiff",
    "component",
    "indica_mif",
]
PlanarInputType = Literal[
    "qptiff_mif",
    "qptiff_fusion",
    "ome_tiff",
    "component",
    "indica_mif",
]

INPUT_TYPES: tuple[str, ...] = (
    "qptiff_mif",
    "qptiff_fusion",
    "qptiff_he",
    "svs",
    "ome_tiff",
    "component",
    "indica_mif",
)
PLANAR_INPUT_TYPES: tuple[str, ...] = (
    "qptiff_mif",
    "qptiff_fusion",
    "ome_tiff",
    "component",
    "indica_mif",
)


def source_reader(
    input_path: str | Path,
    *,
    input_type: InputType,
    series: int,
    channel_name_field: str | None,
):
    """Construct the explicitly selected source-format reader.

    The caller supplies ``input_type``. Reader selection never guesses from a
    filename extension.
    """

    if input_type == "qptiff_mif":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return AkoyaMIFQPTiffReader(input_path, series=series)
    if input_type == "qptiff_fusion":
        return AkoyaFusionQPTiffReader(
            input_path,
            series=series,
            channel_name_field=channel_name_field or "auto",  # type: ignore[arg-type]
        )
    if input_type == "qptiff_he":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return AkoyaHEQPTiffReader(input_path, series=series)
    if input_type == "svs":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return AperioSVSReader(input_path, series=series)
    if input_type == "ome_tiff":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return OMETiffReader(input_path, series=series)
    if input_type == "indica_mif":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return IndicaMIFTiffReader(input_path, series=series)
    if input_type == "component":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return AkoyaComponentTiffReader(input_path, series=series)
    raise ValueError(f"Unsupported input_type {input_type!r}")
