from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from omeify.converters import JPEGSubsampling, TifffileConverter
from omeify.io.akoya_qptiff import ChannelNameField
from omeify.io.channel import (
    ChannelRenameMapping,
    RenameChannelsBy,
    validate_channel_rename_mapping,
)
from omeify.io.pixel_size import PixelSize
from omeify.utils.tiff_image_features import InputProfile, TiffImageFeatures


class GenericConversion:
    profile: ClassVar[InputProfile]
    image_name: ClassVar[str] = "WholeSlideMIF"
    default_compression: ClassVar[str] = "LZW"
    default_channel_name_field: ClassVar[ChannelNameField] = "name"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        rename_channels: ChannelRenameMapping | None = None,
        *,
        rename_channels_by: RenameChannelsBy | None = None,
        channel_name_field: ChannelNameField | None = None,
    ) -> None:
        normalized_renames, normalized_by = validate_channel_rename_mapping(
            rename_channels,
            rename_channels_by,
        )
        self.input_file_path = str(input_file_path)
        self._rename_channels = normalized_renames
        self._rename_channels_by = normalized_by
        self._series = int(series)
        self._cache_directory: str | None = None
        self.logger = logging.getLogger(__name__)
        self._pixel_size_override: PixelSize | None = None
        self.channel_name_field: ChannelNameField = (
            channel_name_field or self.default_channel_name_field
        )

    @property
    def cache_directory(self) -> str | None:
        return self._cache_directory

    @cache_directory.setter
    def cache_directory(self, value: str | Path | None) -> None:
        self._cache_directory = None if value is None else str(value)

    @property
    def series(self) -> int:
        return self._series

    @series.setter
    def series(self, value: int) -> None:
        self._series = int(value)

    @property
    def rename_channels(self) -> dict[str, str] | dict[int, str]:
        return dict(self._rename_channels)

    @property
    def rename_channels_by(self) -> RenameChannelsBy | None:
        return self._rename_channels_by

    def set_channel_renames(
        self,
        mapping: ChannelRenameMapping | None,
        *,
        by: RenameChannelsBy | None,
    ) -> None:
        normalized, normalized_by = validate_channel_rename_mapping(mapping, by)
        self._rename_channels = normalized
        self._rename_channels_by = normalized_by

    @property
    def input_type(self) -> str:
        raise NotImplementedError

    def generate_original_tiff_features(self) -> TiffImageFeatures:
        return TiffImageFeatures(
            self.input_file_path,
            series=self.series,
            profile=self.profile,
            input_type=self.input_type,
            image_name=self.image_name,
            pixel_size=self._pixel_size_override,
            channel_name_field=self.channel_name_field,
        )

    def convert(
        self,
        output_path: str | Path,
        display_uuid: bool = True,
        deidentify_ome: bool = True,
        compression: str | None = None,
        *,
        jpeg_quality: int = 90,
        jpeg_subsampling: JPEGSubsampling = "444",
        tile_size: int = 1024,
        pyramid_levels: int | None = None,
        downsample: str = "mean",
        max_workers: int | None = None,
        overwrite: bool = True,
        calculate_checksums: bool = True,
    ) -> dict[str, object]:
        if not deidentify_ome:
            raise ValueError(
                "The pure-Python conversion path intentionally writes newly constructed, "
                "deidentified OME metadata; preserving source metadata is not supported."
            )
        if downsample not in {"mean", "nearest"}:
            raise ValueError("downsample must be 'mean' or 'nearest'")

        converter = TifffileConverter(
            self.input_file_path,
            output_path,
            profile=self.profile,
            input_type=self.input_type,
            image_name=self.image_name,
            series=self.series,
            rename_channels=self.rename_channels,
            rename_channels_by=self.rename_channels_by,
            pixel_size_override=self._pixel_size_override,
            channel_name_field=self.channel_name_field,
            cache_directory=self.cache_directory,
            compression=compression or self.default_compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,  # type: ignore[arg-type]
            max_workers=max_workers,
            display_uuid=display_uuid,
            overwrite=overwrite,
            calculate_checksums=calculate_checksums,
        )
        return converter.convert()
