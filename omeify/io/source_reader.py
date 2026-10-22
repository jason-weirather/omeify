from __future__ import annotations

from pathlib import Path
from typing import Literal

from .ome_tiff_reader import OMETiffReader, OMETiffLabelReader
from .vendor_tiff_readers import (
    AkoyaFusionQPTiffReader,
    AkoyaHEQPTiffReader,
    AkoyaMIFQPTiffReader,
    AperioSVSReader,
    IndicaMIFTiffReader,
    LeicaSCNHEReader,
)

InputType = Literal[
    "qptiff_mif",
    "qptiff_fusion",
    "qptiff_he",
    "svs",
    "scn_he",
    "ome_tiff",
    "ome_zarr",
    "indica_mif",
]
PlanarInputType = Literal[
    "qptiff_mif",
    "qptiff_fusion",
    "ome_tiff",
    "ome_zarr",
    "indica_mif",
]

INPUT_TYPES: tuple[str, ...] = (
    "qptiff_mif",
    "qptiff_fusion",
    "qptiff_he",
    "svs",
    "scn_he",
    "ome_tiff",
    "ome_zarr",
    "indica_mif",
)
PLANAR_INPUT_TYPES: tuple[str, ...] = (
    "qptiff_mif",
    "qptiff_fusion",
    "ome_tiff",
    "ome_zarr",
    "indica_mif",
)


def source_reader(
    input_path: str | Path,
    *,
    input_type: InputType,
    series: int = 0,
    channel_name_field: str | None = None,
    group: str = "", z: int | None = None, t: int | None = None,
    image_type: str | None = None, intrinsic_only: bool = False,
    storage_options: dict | None = None,
):
    """Construct the explicitly selected source-format reader.

    The caller supplies ``input_type``. Reader selection never guesses from a
    filename extension.
    """

    if input_type == "ome_zarr":
        from .ome_zarr import OMEZarrReader
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return OMEZarrReader(input_path, series=series, group=group, z=z, t=t,
                             image_type=image_type, intrinsic_only=intrinsic_only,
                             storage_options=storage_options)
    if group or z is not None or t is not None or intrinsic_only or storage_options is not None:
        raise ValueError("group, Z/T selection, intrinsic_only and remote options require OME-Zarr input")
    if "://" in str(input_path):
        raise ValueError("Remote input currently requires input_type='ome_zarr'")
    if image_type not in {None, "label"} or (image_type == "label" and input_type != "ome_tiff"):
        raise ValueError("For TIFF input only OME-TIFF image_type='label' is an explicit override")
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
    if input_type == "scn_he":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return LeicaSCNHEReader(input_path, series=series)
    if input_type == "ome_tiff":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return (OMETiffLabelReader if image_type == "label" else OMETiffReader)(input_path, series=series)
    if input_type == "indica_mif":
        if channel_name_field is not None:
            raise ValueError("channel_name_field is only valid for qptiff_fusion input")
        return IndicaMIFTiffReader(input_path, series=series)
    raise ValueError(f"Unsupported input_type {input_type!r}")
