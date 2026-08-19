from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar

from omeify.converters import JPEGSubsampling, TifffileConverter
from omeify.utils.tiff_image_features import InputProfile, TiffImageFeatures


class GenericConversion:
    profile: ClassVar[InputProfile]
    image_name: ClassVar[str] = "WholeSlideMIF"
    default_compression: ClassVar[str] = "LZW"

    def __init__(
        self,
        input_file_path: str | Path,
        series: int = 0,
        rename_channels: dict[str, str] | None = None,
    ) -> None:
        self.input_file_path = str(input_file_path)
        self._rename_channels = dict(rename_channels or {})
        self._series = int(series)
        self._cache_directory: str | None = None
        self.logger = logging.getLogger(__name__)
        self._physical_size_x_um: float | None = None
        self._physical_size_y_um: float | None = None

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
    def rename_channels(self) -> dict[str, str]:
        return self._rename_channels

    @rename_channels.setter
    def rename_channels(self, value: dict[str, str] | None) -> None:
        self._rename_channels = dict(value or {})

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
            physical_size_x_um=self._physical_size_x_um,
            physical_size_y_um=self._physical_size_y_um,
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
            physical_size_x_um=self._physical_size_x_um,
            physical_size_y_um=self._physical_size_y_um,
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
