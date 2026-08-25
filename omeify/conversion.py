from __future__ import annotations

import hashlib
import time
from datetime import datetime
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Literal

from ._version import get_version_info
from .io.ome_tiff_reader import OMETiffReader
from .io.ome_tiff_writer import JPEGSubsampling, OMETiffWriter
from .io.pixel_size import PixelSize
from .io.vendor_tiff_readers import (
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
DownsampleMethod = Literal["mean", "nearest"]

RenameChannelsBy = Literal["name", "index"]
ChannelRenameMapping = Mapping[str, str] | Mapping[int, str]


def _validate_channel_rename_mapping(
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[dict[str, str] | dict[int, str], RenameChannelsBy | None]:
    """Validate a channel rename mapping at the conversion boundary."""

    if not mapping:
        if by is not None and by not in {"name", "index"}:
            raise ValueError("rename_channels_by must be 'name' or 'index'")
        return {}, by
    if by not in {"name", "index"}:
        raise ValueError(
            "rename_channels_by must be explicitly set to 'name' or 'index' "
            "when rename_channels is provided"
        )
    if not all(isinstance(value, str) and value.strip() for value in mapping.values()):
        raise TypeError("Channel rename values must be non-empty strings")

    if by == "name":
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Name-based channel rename keys must be strings")
            # The mode is explicit, so a source channel literally named "0"
            # remains a name instead of being guessed to be an index.
            normalized[key] = value.strip()
        return normalized, by

    normalized_index: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("Index-based Python channel rename keys must be integers")
        index = int(key)
        if index < 0:
            raise ValueError("Index-based channel rename keys must be zero or greater")
        normalized_index[index] = value.strip()
    return normalized_index, by


def _apply_channel_renames(
    channel_names: Sequence[str],
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[str, ...]:
    normalized, normalized_by = _validate_channel_rename_mapping(mapping, by)
    names = tuple(str(item) for item in channel_names)
    if not normalized:
        return names

    if normalized_by == "name":
        name_mapping = normalized
        unknown = sorted(set(name_mapping) - set(names))
        if unknown:
            raise ValueError(
                "Name-based channel rename mapping contains unknown source names: "
                + ", ".join(repr(item) for item in unknown)
            )
        ambiguous = sorted(name for name in name_mapping if names.count(name) > 1)
        if ambiguous:
            raise ValueError(
                "Name-based channel rename mapping is ambiguous for duplicate source names: "
                + ", ".join(repr(item) for item in ambiguous)
                + "; use rename_channels_by='index'"
            )
        return tuple(name_mapping.get(name, name) for name in names)

    index_mapping = normalized
    out_of_range = sorted(index for index in index_mapping if index >= len(names))
    if out_of_range:
        raise IndexError(
            "Index-based channel rename mapping contains out-of-range indices: "
            + ", ".join(str(item) for item in out_of_range)
        )
    return tuple(index_mapping.get(index, name) for index, name in enumerate(names))

INPUT_TYPES: tuple[str, ...] = (
    "qptiff_mif",
    "qptiff_fusion",
    "qptiff_he",
    "svs",
    "ome_tiff",
    "component",
    "indica_mif",
)


def _hash_file(path: Path) -> dict[str, str]:
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return {"md5_checksum": md5.hexdigest(), "sha256_checksum": sha256.hexdigest()}


def _readable_runtime(seconds: float) -> str:
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:05.2f}"


def _reader_for_input(
    input_path: Path,
    *,
    input_type: InputType,
    series: int,
    channel_name_field: str | None,
    component_pixel_size: PixelSize | None,
):
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
        if component_pixel_size is None:
            raise ValueError("component input requires an explicit pixel_size")
        return AkoyaComponentTiffReader(
            input_path,
            series=series,
            pixel_size=component_pixel_size,
        )
    raise ValueError(f"Unsupported input_type {input_type!r}")


def convert(
    input_path: str | Path,
    output_path: str | Path,
    *,
    input_type: InputType,
    series: int = 0,
    channel_name_field: Literal["name", "biomarker", "auto"] | None = None,
    rename_channels: ChannelRenameMapping | None = None,
    rename_channels_by: RenameChannelsBy | None = None,
    pixel_size: PixelSize | None = None,
    compression: str | None = None,
    jpeg_quality: int = 90,
    jpeg_subsampling: JPEGSubsampling = "444",
    tile_size: int = 1024,
    pyramid_levels: int | None = None,
    downsample: DownsampleMethod = "mean",
    max_workers: int | None = None,
    display_uuid: bool = True,
    overwrite: bool = True,
    calculate_checksums: bool = True,
    cache_directory: str | Path | None = None,
) -> dict[str, object]:
    """Normalize one supported image into omeify's canonical OME-TIFF contract.

    This is the public Python counterpart to ``omeify convert``. Source-format
    readers live in :mod:`omeify.io`; this function only selects the reader,
    applies explicit channel-renaming policy, and delegates all OME-TIFF
    construction and verification to :class:`OMETiffWriter`.

    ``rename_channels`` uses native Python key types: strings in ``name`` mode
    and zero-based integers in ``index`` mode. The CLI converts JSON string keys
    into those same semantic values before calling this function.
    """

    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_file}")
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output paths must be different")
    if output_file.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_file}")
    if input_type not in INPUT_TYPES:
        raise ValueError(f"Unsupported input_type {input_type!r}")
    normalized_renames, normalized_rename_mode = _validate_channel_rename_mapping(
        rename_channels,
        rename_channels_by,
    )
    if downsample not in {"mean", "nearest"}:
        raise ValueError("downsample must be 'mean' or 'nearest'")
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize instance")

    start_epoch = time.time()
    reader = _reader_for_input(
        input_file,
        input_type=input_type,
        series=int(series),
        channel_name_field=channel_name_field,
        component_pixel_size=pixel_size if input_type == "component" else None,
    )

    with reader:
        source_pixel_size = reader.pixel_size
        effective_pixel_size = pixel_size or source_pixel_size
        if effective_pixel_size is None:
            raise ValueError(
                "Input does not provide a usable physical pixel size. Supply pixel_size="
                "PixelSize(x, y, unit) explicitly so omeify can satisfy its OME-TIFF contract."
            )

        source_channel_names = tuple(reader.channel_names)
        output_channel_names = _apply_channel_renames(
            source_channel_names,
            normalized_renames,
            normalized_rename_mode,
        )
        image_type = "rgb" if bool(getattr(reader, "is_rgb", False)) else "multichannel"
        writer = OMETiffWriter(
            output_file,
            image_type=image_type,
            channel_names=output_channel_names,
            pixel_size=effective_pixel_size,
            compression=compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,
            max_workers=max_workers,
            display_uuid=display_uuid,
            overwrite=overwrite,
            cache_directory=cache_directory,
            icc_profile=getattr(reader, "icc_profile", None),
        )
        write_report = writer.write_source(
            reader,
            axes=reader.output_axes,
            shape=reader.output_shape,
            dtype=reader.dtype,
            icc_profile=getattr(reader, "icc_profile", None),
        )

        source_channels = [
            {
                "index": channel.index,
                "id": channel.id,
                "name": channel.name,
                "source_id": channel.source_id,
                "id_is_generated": channel.id_is_generated,
                "source_metadata": dict(channel.source_metadata),
            }
            for channel in reader.channels
        ]
        source_miti_header = None
        if isinstance(reader, OMETiffReader):
            ome_summary = reader.inspection_report.get("ome")
            if ome_summary is not None:
                source_miti_header = ome_summary.get("miti")

        input_report: dict[str, object] = {
            "path": str(input_file),
            "size_bytes": input_file.stat().st_size,
            "type_description": reader.input_type_description,
            "dtype": reader.dtype.name,
            "shape": list(reader.shape),
            "normalized_shape": list(reader.output_shape),
            "source_axes": reader.source_axes,
            "output_axes": reader.output_axes,
            "byte_order": reader.source_byte_order,
            "pixel_size": (
                None if source_pixel_size is None else list(source_pixel_size.to_tuple())
            ),
            "channels": source_channels,
        }
        if source_miti_header is not None:
            input_report["source_miti_header"] = source_miti_header

    stop_epoch = time.time()
    output_size = output_file.stat().st_size
    output_report = dict(write_report["output_file"])
    image_report = dict(write_report["image"])
    pyramid_report = dict(write_report["pyramid"])
    if write_report["output_file"]["axes"] == "CYX":  # type: ignore[index]
        pyramid_report["level_shapes_cyx"] = pyramid_report["level_shapes"]
    elif write_report["output_file"]["axes"] == "YXS":  # type: ignore[index]
        pyramid_report["level_shapes_yxs"] = pyramid_report["level_shapes"]

    options = dict(write_report["options"])
    options.update(
        {
            "deidentify_ome": True,
            "input_type": input_type,
            "series": int(series),
            "rename_channels": dict(normalized_renames),
            "rename_channels_by": normalized_rename_mode,
            "channel_name_field": channel_name_field,
            "pixel_size_override": None if pixel_size is None else list(pixel_size.to_tuple()),
        }
    )

    report: dict[str, object] = {
        "ome": write_report["ome"],
        "miti_header": write_report["miti_header"],
        "input_file": input_report,
        "output_file": output_report,
        "image": image_report,
        "pyramid": pyramid_report,
        "verification": write_report["verification"],
        "options": options,
        "conversion_stats": {
            "start_time": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "stop_time": datetime.fromtimestamp(stop_epoch).strftime("%Y-%m-%d %H:%M:%S"),
            "run_time": _readable_runtime(stop_epoch - start_epoch),
            "output_to_input_size_ratio": (
                output_size / int(input_report["size_bytes"])
                if int(input_report["size_bytes"])
                else None
            ),
            "compression_ratio": (
                output_size / int(input_report["size_bytes"])
                if int(input_report["size_bytes"])
                else None
            ),
        },
        "versions": get_version_info(),
    }

    if calculate_checksums:
        input_report.update(_hash_file(input_file))
        output_report.update(_hash_file(output_file))
    else:
        input_report["md5_checksum"] = None
        input_report["sha256_checksum"] = None
        output_report["md5_checksum"] = None
        output_report["sha256_checksum"] = None
    return report
