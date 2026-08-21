from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Literal

from omeify._version import get_version_info
from omeify.io.akoya_qptiff import ChannelNameField
from omeify.io.channel import (
    ChannelRenameMapping,
    RenameChannelsBy,
    apply_channel_renames,
    validate_channel_rename_mapping,
)
from omeify.io.ome_tiff_writer import (
    CompressionSettings,
    JPEGSubsampling,
    OMETiffWriter,
    _compression_settings,
    _mean_downsample_2x,
)
from omeify.io.pixel_size import PixelSize
from omeify.utils.tiff_image_features import InputProfile, TiffMIFSource

LOGGER = logging.getLogger(__name__)

__all__ = [
    "CompressionSettings",
    "JPEGSubsampling",
    "TifffileConverter",
    "_compression_settings",
    "_mean_downsample_2x",
]

DownsampleMethod = Literal["mean", "nearest"]


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


class TifffileConverter:
    """Pure-Python, tiled, low-memory TIFF-family to OME-TIFF converter.

    Source-specific interpretation remains here. All standardized OME-TIFF
    construction, pyramid generation, metadata validation, and output
    verification are delegated to the public :class:`OMETiffWriter`.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        profile: InputProfile,
        input_type: str,
        image_name: str = "WholeSlideMIF",
        series: int = 0,
        rename_channels: ChannelRenameMapping | None = None,
        rename_channels_by: RenameChannelsBy | None = None,
        pixel_size_override: PixelSize | None = None,
        channel_name_field: ChannelNameField = "name",
        cache_directory: str | Path | None = None,
        compression: str = "LZW",
        jpeg_quality: int = 90,
        jpeg_subsampling: JPEGSubsampling = "444",
        tile_size: int = 1024,
        pyramid_levels: int | None = None,
        downsample: DownsampleMethod = "mean",
        max_workers: int | None = None,
        display_uuid: bool = True,
        overwrite: bool = True,
        calculate_checksums: bool = True,
    ) -> None:
        normalized_renames, normalized_by = validate_channel_rename_mapping(
            rename_channels,
            rename_channels_by,
        )
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.profile = profile
        self.input_type = input_type
        self.image_name = image_name
        self.series = int(series)
        self.rename_channels = normalized_renames
        self.rename_channels_by = normalized_by
        self.pixel_size_override = pixel_size_override
        self.channel_name_field: ChannelNameField = channel_name_field
        self.cache_directory = Path(cache_directory) if cache_directory is not None else None
        self.compression_name = compression
        self.jpeg_quality = int(jpeg_quality)
        self.jpeg_subsampling: JPEGSubsampling = jpeg_subsampling
        self.tile_size = int(tile_size)
        self.pyramid_levels = pyramid_levels
        if downsample not in {"mean", "nearest"}:
            raise ValueError("downsample must be 'mean' or 'nearest'")
        self.downsample: DownsampleMethod = downsample
        self.max_workers = max_workers
        self.display_uuid = bool(display_uuid)
        self.overwrite = bool(overwrite)
        self.calculate_checksums = bool(calculate_checksums)

    def convert(self) -> dict[str, object]:
        start_epoch = time.time()
        if not self.input_path.is_file():
            raise FileNotFoundError(f"Input image does not exist: {self.input_path}")
        if self.input_path.resolve() == self.output_path.resolve():
            raise ValueError("Input and output paths must be different")
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        with TiffMIFSource(
            self.input_path,
            series=self.series,
            profile=self.profile,
            input_type=self.input_type,
            image_name=self.image_name,
            pixel_size_override=self.pixel_size_override,
            channel_name_field=self.channel_name_field,
        ) as source:
            features = source.features
            renamed_channels = apply_channel_renames(
                features.channel_names,
                self.rename_channels,
                self.rename_channels_by,
            )
            writer = OMETiffWriter(
                self.output_path,
                image_type="rgb" if features.is_rgb else "multichannel",
                channel_names=renamed_channels,
                pixel_size=features.pixel_size,
                compression=self.compression_name,
                jpeg_quality=self.jpeg_quality,
                jpeg_subsampling=self.jpeg_subsampling,
                tile_size=self.tile_size,
                pyramid_levels=self.pyramid_levels,
                downsample=self.downsample,
                max_workers=self.max_workers,
                display_uuid=self.display_uuid,
                overwrite=self.overwrite,
                cache_directory=self.cache_directory,
            )
            write_report = writer.write_source(
                source,
                axes=features.output_axes,
                shape=features.output_shape,
                dtype=features.dtype,
                icc_profile=features.icc_profile,
            )
            source_channel_metadata = [
                {"index": index, **dict(item)}
                for index, item in enumerate(features.channel_source_metadata)
            ]

        input_size = self.input_path.stat().st_size
        output_size = self.output_path.stat().st_size
        stop_epoch = time.time()

        output_file = dict(write_report["output_file"])
        image = dict(write_report["image"])
        image["source_channel_metadata"] = source_channel_metadata
        pyramid = dict(write_report["pyramid"])
        options = dict(write_report["options"])
        options.update(
            {
                "deidentify_ome": True,
                "series": self.series,
                "rename_channels": dict(self.rename_channels),
                "rename_channels_by": self.rename_channels_by,
                "channel_name_field": self.channel_name_field,
            }
        )

        report: dict[str, object] = {
            "ome": write_report["ome"],
            "miti_header": write_report["miti_header"],
            "input_file": {
                "path": str(self.input_path),
                "size_bytes": input_size,
                "type_description": self.input_type,
                "dtype": features.dtype.name,
                "shape": list(features.output_shape),
                "shape_cyx": list(features.shape_cyx),
                "source_axes": features.source_axes,
                "output_axes": features.output_axes,
                "byte_order": features.source_byte_order,
                "pixel_size": list(features.pixel_size.to_tuple()),
                "channels": source_channel_metadata,
            },
            "output_file": output_file,
            "image": image,
            "pyramid": pyramid,
            "verification": write_report["verification"],
            "options": options,
            "conversion_stats": {
                "start_time": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
                "stop_time": datetime.fromtimestamp(stop_epoch).strftime("%Y-%m-%d %H:%M:%S"),
                "run_time": _readable_runtime(stop_epoch - start_epoch),
                "output_to_input_size_ratio": output_size / input_size if input_size else None,
                "compression_ratio": output_size / input_size if input_size else None,
            },
            "versions": get_version_info(),
        }
        if features.output_axes == "CYX":
            pyramid["level_shapes_cyx"] = pyramid["level_shapes"]
        else:
            pyramid["level_shapes_yxs"] = pyramid["level_shapes"]

        if self.calculate_checksums:
            report["input_file"].update(_hash_file(self.input_path))  # type: ignore[union-attr]
            report["output_file"].update(_hash_file(self.output_path))  # type: ignore[union-attr]
        else:
            report["input_file"]["md5_checksum"] = None  # type: ignore[index]
            report["input_file"]["sha256_checksum"] = None  # type: ignore[index]
            report["output_file"]["md5_checksum"] = None  # type: ignore[index]
            report["output_file"]["sha256_checksum"] = None  # type: ignore[index]
        return report
