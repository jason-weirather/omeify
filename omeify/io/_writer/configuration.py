from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import tifffile

from omeify._version import __version__
from omeify.io.spec import ImageType

DownsampleMethod = Literal["mean", "nearest"]
JPEGSubsampling = Literal["444", "422", "420", "411"]
LossyCompressionPolicy = Literal["rgb-only", "non-label"]

OUTPUT_BYTEORDER: Literal["<", ">"] = "<"
JPEG_SUBSAMPLING_FACTORS: dict[JPEGSubsampling, tuple[int, int]] = {
    "444": (1, 1),
    "422": (2, 1),
    "420": (2, 2),
    "411": (4, 1),
}
OUTPUT_COMPRESSION_CODES = {
    "Uncompressed": int(tifffile.COMPRESSION.NONE),
    "LZW": int(tifffile.COMPRESSION.LZW),
    "JPEG": int(tifffile.COMPRESSION.JPEG),
    "Deflate": int(tifffile.COMPRESSION.DEFLATE),
    "ZSTD": int(tifffile.COMPRESSION.ZSTD),
}


@dataclass(frozen=True, slots=True)
class CompressionSettings:
    """Normalized tifffile compression options for one image series."""

    name: str
    tifffile_value: str | None
    compression_args: dict[str, object] | None
    predictor: bool | None
    lossless: bool
    subsampling: tuple[int, int] | None = None


@dataclass(frozen=True, slots=True)
class WriterSettings:
    """Shared construction settings used by both public OME-TIFF writers."""

    output_path: Path
    compression_name: str
    jpeg_quality: int
    jpeg_subsampling: JPEGSubsampling
    tile_size: int
    pyramid_levels: int | None
    max_workers: int
    display_uuid: bool
    software: str
    overwrite: bool
    cache_directory: Path | None

    @classmethod
    def from_values(
        cls,
        output_path: str | Path,
        *,
        compression: str | None,
        jpeg_quality: int,
        jpeg_subsampling: JPEGSubsampling,
        tile_size: int,
        pyramid_levels: int | None,
        max_workers: int | None,
        display_uuid: bool,
        software: str | None,
        overwrite: bool,
        cache_directory: str | Path | None,
    ) -> WriterSettings:
        """Validate public writer options into one authoritative record."""

        compression_name = "LZW" if compression is None else compression
        if not isinstance(compression_name, str) or not compression_name.strip():
            raise ValueError("compression must be a non-empty string or None")
        if isinstance(jpeg_quality, bool) or not isinstance(jpeg_quality, int):
            raise TypeError("jpeg_quality must be an integer")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if jpeg_subsampling not in JPEG_SUBSAMPLING_FACTORS:
            choices = ", ".join(JPEG_SUBSAMPLING_FACTORS)
            raise ValueError(f"jpeg_subsampling must be one of: {choices}")
        normalized_tile_size = validate_tile_size(tile_size)
        normalized_pyramid_levels = validate_pyramid_levels(pyramid_levels)
        normalized_workers = validate_max_workers(max_workers)
        if not isinstance(display_uuid, bool):
            raise TypeError("display_uuid must be a boolean")
        if not isinstance(overwrite, bool):
            raise TypeError("overwrite must be a boolean")
        return cls(
            output_path=Path(output_path),
            compression_name=compression_name.strip(),
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=normalized_tile_size,
            pyramid_levels=normalized_pyramid_levels,
            max_workers=normalized_workers,
            display_uuid=display_uuid,
            software=normalize_software_tag(software),
            overwrite=overwrite,
            cache_directory=(
                None if cache_directory is None else Path(cache_directory)
            ),
        )


def normalize_software_tag(software: str | None) -> str:
    """Return the TIFF Software tag value used by final OME-TIFF outputs."""

    if software is None:
        return f"omeify {__version__}"
    if not isinstance(software, str):
        raise TypeError("software must be a string or None")
    value = software.strip()
    if not value:
        raise ValueError("software must be a non-empty string when supplied")
    if "\x00" in value:
        raise ValueError("software must not contain NUL characters")
    return value


def compression_settings(
    name: str,
    dtype: np.dtype[object],
    *,
    is_rgb: bool,
    jpeg_quality: int,
    jpeg_subsampling: JPEGSubsampling,
):
    """Resolve one public compression name into tifffile options."""

    normalized = name.strip().lower().replace("_", "-")
    integer = np.issubdtype(dtype, np.integer)
    if normalized in {"uncompressed", "none", "no", "false"}:
        return CompressionSettings("Uncompressed", None, None, None, True)
    if normalized == "lzw":
        return CompressionSettings(
            "LZW",
            "lzw",
            None,
            True if integer else None,
            True,
        )
    if normalized in {"deflate", "zlib"}:
        return CompressionSettings(
            "Deflate",
            "deflate",
            {"level": 6},
            True if integer else None,
            True,
        )
    if normalized in {"zstd", "zstandard"}:
        return CompressionSettings(
            "ZSTD",
            "zstd",
            {"level": 3},
            True if integer else None,
            True,
        )
    if normalized in {"jpeg", "jpg"}:
        if dtype != np.dtype("uint8"):
            raise ValueError("JPEG output is restricted to uint8 images")
        return CompressionSettings(
            "JPEG",
            "jpeg",
            {"level": jpeg_quality},
            None,
            False,
            JPEG_SUBSAMPLING_FACTORS[jpeg_subsampling] if is_rgb else None,
        )
    raise ValueError(
        f"Unsupported compression {name!r}; choose LZW, Deflate, ZSTD, "
        "JPEG, or Uncompressed"
    )


def resolve_downsample(
    image_type: ImageType,
    value: DownsampleMethod | None,
) -> DownsampleMethod:
    """Resolve and validate one image's pyramid downsampling policy."""

    if value is None:
        effective = "nearest" if image_type == "label" else "mean"
    else:
        effective = value
    if effective not in {"mean", "nearest"}:
        raise ValueError("downsample must be 'mean' or 'nearest'")
    if image_type == "label" and effective != "nearest":
        raise ValueError("Label-image pyramids require nearest-neighbor downsampling")
    return effective


def validate_lossy_compression(
    *,
    image_type: ImageType,
    lossless: bool,
    policy: LossyCompressionPolicy,
) -> None:
    """Apply the one intentional compression-policy difference between writers."""

    if lossless:
        return
    if policy == "rgb-only":
        if image_type != "rgb":
            raise ValueError(
                "Lossy compression is restricted to RGB OME-TIFF output; "
                "multichannel and label output require lossless compression"
            )
        return
    if policy == "non-label":
        if image_type == "label":
            raise ValueError("Label-image OME-TIFF series require lossless compression")
        return
    raise AssertionError(f"Unhandled lossy compression policy {policy!r}")


def validate_jpeg_alignment(
    *,
    tile_size: int,
    subsampling: tuple[int, int] | None,
    jpeg_subsampling: JPEGSubsampling,
) -> None:
    if subsampling is None:
        return
    alignment = max(subsampling) * 8
    if tile_size % alignment != 0:
        raise ValueError(
            f"TIFF tile size {tile_size} is incompatible with JPEG "
            f"{jpeg_subsampling} subsampling; use a tile size divisible by "
            f"{alignment}."
        )


def validate_tile_size(tile_size: int) -> int:
    if isinstance(tile_size, bool) or not isinstance(tile_size, int):
        raise TypeError("tile_size must be an integer")
    if tile_size < 16 or tile_size % 16 != 0:
        raise ValueError("TIFF tile size must be at least 16 and divisible by 16")
    return tile_size


def validate_pyramid_levels(value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("pyramid_levels must be an integer or None")
    if value < 0:
        raise ValueError("pyramid_levels must be zero or greater")
    return value


def validate_max_workers(value: int | None) -> int:
    if value is None:
        return max(1, min(8, os.cpu_count() or 1))
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_workers must be an integer or None")
    if value < 1:
        raise ValueError("max_workers must be at least one")
    return value
