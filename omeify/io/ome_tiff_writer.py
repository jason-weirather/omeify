from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Protocol, Sequence, runtime_checkable

import numpy as np
import tifffile
from lxml import etree

from omeify._version import __version__
from omeify.io.pixel_size import PixelSize
from omeify.io.spec import ImageType, OMEImageSpec
from omeify.io.tiff import ArrayPlaneReader, PlaneReader, TiffPlaneReader
from omeify.progress import ProgressLogger
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header
from omeify.utils.ome_schema_validator import OMESchemaValidator

LOGGER = logging.getLogger(__name__)

DownsampleMethod = Literal["mean", "nearest"]
JPEGSubsampling = Literal["444", "422", "420", "411"]
_OUTPUT_BYTEORDER: Literal["<", ">"] = "<"
_JPEG_SUBSAMPLING_FACTORS: dict[JPEGSubsampling, tuple[int, int]] = {
    "444": (1, 1),
    "422": (2, 1),
    "420": (2, 2),
    "411": (4, 1),
}
_OUTPUT_COMPRESSION_CODES = {
    "Uncompressed": int(tifffile.COMPRESSION.NONE),
    "LZW": int(tifffile.COMPRESSION.LZW),
    "JPEG": int(tifffile.COMPRESSION.JPEG),
    "Deflate": int(tifffile.COMPRESSION.DEFLATE),
    "ZSTD": int(tifffile.COMPRESSION.ZSTD),
}


def _normalize_software_tag(software: str | None) -> str:
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


@runtime_checkable
class PlaneReaderSource(Protocol):
    """Source contract used by the streaming OME-TIFF writer."""

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        """Return one random-access reader for each logical TIFF plane."""


@dataclass(frozen=True)
class CompressionSettings:
    name: str
    tifffile_value: str | None
    compression_args: dict[str, object] | None
    predictor: bool | None
    lossless: bool
    subsampling: tuple[int, int] | None = None


class _ArraySource:
    def __init__(self, array: np.ndarray, spec: OMEImageSpec) -> None:
        self.array = np.asarray(array)
        self.spec = spec

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        del cache_mib
        if self.spec.axes == "CYX":
            return [ArrayPlaneReader(self.array[index]) for index in range(self.spec.plane_count)]
        return [ArrayPlaneReader(self.array)]


def _compression_settings(
    name: str,
    dtype: np.dtype,
    *,
    is_rgb: bool,
    jpeg_quality: int,
    jpeg_subsampling: JPEGSubsampling,
) -> CompressionSettings:
    normalized = name.strip().lower().replace("_", "-")
    integer = np.issubdtype(dtype, np.integer)
    if normalized in {"uncompressed", "none", "no", "false"}:
        return CompressionSettings("Uncompressed", None, None, None, True)
    if normalized == "lzw":
        return CompressionSettings("LZW", "lzw", None, True if integer else None, True)
    if normalized in {"deflate", "zlib"}:
        return CompressionSettings(
            "Deflate", "deflate", {"level": 6}, True if integer else None, True
        )
    if normalized in {"zstd", "zstandard"}:
        return CompressionSettings(
            "ZSTD", "zstd", {"level": 3}, True if integer else None, True
        )
    if normalized in {"jpeg", "jpg"}:
        if dtype != np.dtype("uint8"):
            raise ValueError("JPEG output is restricted to uint8 images")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        return CompressionSettings(
            "JPEG",
            "jpeg",
            {"level": jpeg_quality},
            None,
            False,
            _JPEG_SUBSAMPLING_FACTORS[jpeg_subsampling] if is_rgb else None,
        )
    raise ValueError(
        f"Unsupported compression {name!r}; choose LZW, Deflate, ZSTD, JPEG, or Uncompressed"
    )


def _validate_tile_size(tile_size: int) -> int:
    value = int(tile_size)
    if value < 16 or value % 16 != 0:
        raise ValueError("TIFF tile size must be at least 16 and divisible by 16")
    return value


def _yx_shape(shape: Sequence[int], axes: str) -> tuple[int, int]:
    if len(shape) != len(axes) or "Y" not in axes or "X" not in axes:
        raise ValueError(f"Shape {tuple(shape)} is incompatible with axes {axes!r}")
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def _replace_yx(
    shape: Sequence[int],
    axes: str,
    height: int,
    width: int,
) -> tuple[int, ...]:
    result = [int(value) for value in shape]
    result[axes.index("Y")] = int(height)
    result[axes.index("X")] = int(width)
    return tuple(result)


def _auto_level_shapes(
    base_shape: tuple[int, ...],
    *,
    axes: str,
    tile_size: int,
    pyramid_levels: int | None,
) -> list[tuple[int, ...]]:
    height, width = _yx_shape(base_shape, axes)
    shapes = [base_shape]
    if pyramid_levels is not None:
        if pyramid_levels < 0:
            raise ValueError("pyramid_levels must be zero or greater")
        target_levels = pyramid_levels
    else:
        target_levels = 0
        current_h, current_w = height, width
        while max(current_h, current_w) > tile_size:
            target_levels += 1
            current_h = (current_h + 1) // 2
            current_w = (current_w + 1) // 2
            if target_levels >= 32:
                break

    current_h, current_w = height, width
    for _ in range(target_levels):
        if current_h == 1 and current_w == 1:
            raise ValueError(
                f"Cannot build {target_levels} subresolution levels from base shape "
                f"{base_shape}; the pyramid already reached 1x1."
            )
        current_h = max(1, (current_h + 1) // 2)
        current_w = max(1, (current_w + 1) // 2)
        shapes.append(_replace_yx(base_shape, axes, current_h, current_w))
    return shapes


def _broadcast_counts(counts: np.ndarray, ndim: int) -> np.ndarray:
    if ndim == 2:
        return counts
    return counts[(...,) + (None,) * (ndim - 2)]


def _mean_downsample_2x(block: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    if block.ndim not in {2, 3}:
        raise ValueError(f"Expected YX or YXS block, got shape {block.shape}")

    out_h, out_w = out_shape
    dtype = block.dtype
    accumulator_shape = (out_h, out_w, *block.shape[2:])
    counts = np.zeros((out_h, out_w), dtype=np.uint8)

    if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(dtype, np.bool_):
        accumulator = np.zeros(accumulator_shape, dtype=np.uint64)
        for dy in (0, 1):
            for dx in (0, 1):
                sample = block[dy::2, dx::2, ...]
                height, width = sample.shape[:2]
                accumulator[:height, :width, ...] += sample.astype(np.uint64, copy=False)
                counts[:height, :width] += 1
        divisors = _broadcast_counts(counts, block.ndim)
        quotient, remainder = np.divmod(accumulator, divisors)
        twice_remainder = remainder * 2
        increment = (twice_remainder > divisors) | (
            (twice_remainder == divisors) & ((quotient & 1) == 1)
        )
        return (quotient + increment).astype(dtype, copy=False)

    if np.issubdtype(dtype, np.signedinteger):
        accumulator = np.zeros(accumulator_shape, dtype=np.int64)
        for dy in (0, 1):
            for dx in (0, 1):
                sample = block[dy::2, dx::2, ...]
                height, width = sample.shape[:2]
                accumulator[:height, :width, ...] += sample.astype(np.int64, copy=False)
                counts[:height, :width] += 1
        divisors = _broadcast_counts(counts, block.ndim)
        sign = np.sign(accumulator)
        magnitude = np.abs(accumulator)
        quotient, remainder = np.divmod(magnitude, divisors)
        twice_remainder = remainder * 2
        increment = (twice_remainder > divisors) | (
            (twice_remainder == divisors) & ((quotient & 1) == 1)
        )
        result = (quotient + increment) * sign
        return result.astype(dtype, copy=False)

    accumulator_dtype = (
        np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
    )
    accumulator = np.zeros(accumulator_shape, dtype=accumulator_dtype)
    for dy in (0, 1):
        for dx in (0, 1):
            sample = block[dy::2, dx::2, ...]
            height, width = sample.shape[:2]
            accumulator[:height, :width, ...] += sample.astype(
                accumulator_dtype, copy=False
            )
            counts[:height, :width] += 1
    return (accumulator / _broadcast_counts(counts, block.ndim)).astype(dtype, copy=False)


def _tile_count(
    shape: Sequence[int],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
) -> int:
    """Return the number of tiles consumed for one canonical image level."""

    height, width = _yx_shape(shape, axes)
    tiles_y = (height + tile_size - 1) // tile_size
    tiles_x = (width + tile_size - 1) // tile_size
    return int(plane_count) * tiles_y * tiles_x


def _downsample_region(
    reader: PlaneReader,
    *,
    out_y0: int,
    out_y1: int,
    out_x0: int,
    out_x1: int,
    method: DownsampleMethod,
) -> np.ndarray:
    in_y0 = out_y0 * 2
    in_y1 = min(reader.height, out_y1 * 2)
    in_x0 = out_x0 * 2
    in_x1 = min(reader.width, out_x1 * 2)
    block = reader.read_region(in_y0, in_y1, in_x0, in_x1)
    out_shape = (out_y1 - out_y0, out_x1 - out_x0)
    if method == "nearest":
        return np.ascontiguousarray(
            block[::2, ::2, ...][: out_shape[0], : out_shape[1], ...]
        )
    if method == "mean":
        return np.ascontiguousarray(_mean_downsample_2x(block, out_shape))
    raise AssertionError(f"Unhandled downsample method {method}")


def _iter_tiles(
    readers: Sequence[PlaneReader],
    shape: tuple[int, ...],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
    progress_label: str | None = None,
) -> Iterable[np.ndarray]:
    height, width = _yx_shape(shape, axes)
    if len(readers) != plane_count:
        raise ValueError(f"Expected {plane_count} page readers, got {len(readers)}")
    total_tiles = _tile_count(
        shape,
        axes=axes,
        plane_count=plane_count,
        tile_size=tile_size,
    )
    progress = (
        None
        if progress_label is None
        else ProgressLogger(LOGGER, progress_label, total_tiles, unit="tiles")
    )
    completed_tiles = 0
    yielded_tiles = 0
    try:
        for plane_index, reader in enumerate(readers):
            LOGGER.debug(
                "%s: starting plane %s/%s",
                progress_label or "Tile stream",
                plane_index + 1,
                plane_count,
            )
            try:
                for y0 in range(0, height, tile_size):
                    y1 = min(height, y0 + tile_size)
                    for x0 in range(0, width, tile_size):
                        x1 = min(width, x0 + tile_size)
                        tile = np.ascontiguousarray(
                            reader.read_region(y0, y1, x0, x1)
                        )
                        yielded_tiles += 1
                        yield tile
                        completed_tiles = yielded_tiles
                        if progress is not None:
                            progress.update(completed_tiles)
            finally:
                reader.clear_cache()
            LOGGER.debug(
                "%s: completed plane %s/%s",
                progress_label or "Tile stream",
                plane_index + 1,
                plane_count,
            )
    finally:
        # Tifffile may close a generator immediately after accepting its last
        # expected tile without resuming the code after the final ``yield``.
        # Treat a fully yielded stream as complete so the user still sees the
        # terminal 100 percent line for that stage.
        if progress is not None and yielded_tiles == total_tiles:
            progress.finish()
        for reader in readers:
            reader.clear_cache()


def _iter_downsampled_tiles(
    readers: Sequence[PlaneReader],
    output_shape: tuple[int, ...],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
    method: DownsampleMethod,
    progress_label: str | None = None,
) -> Iterable[np.ndarray]:
    height, width = _yx_shape(output_shape, axes)
    if len(readers) != plane_count:
        raise ValueError(f"Expected {plane_count} page readers, got {len(readers)}")
    total_tiles = _tile_count(
        output_shape,
        axes=axes,
        plane_count=plane_count,
        tile_size=tile_size,
    )
    progress = (
        None
        if progress_label is None
        else ProgressLogger(LOGGER, progress_label, total_tiles, unit="tiles")
    )
    completed_tiles = 0
    yielded_tiles = 0
    try:
        for plane_index, reader in enumerate(readers):
            LOGGER.debug(
                "%s: starting plane %s/%s",
                progress_label or "Downsample stream",
                plane_index + 1,
                plane_count,
            )
            try:
                for y0 in range(0, height, tile_size):
                    y1 = min(height, y0 + tile_size)
                    for x0 in range(0, width, tile_size):
                        x1 = min(width, x0 + tile_size)
                        tile = _downsample_region(
                            reader,
                            out_y0=y0,
                            out_y1=y1,
                            out_x0=x0,
                            out_x1=x1,
                            method=method,
                        )
                        yielded_tiles += 1
                        yield tile
                        completed_tiles = yielded_tiles
                        if progress is not None:
                            progress.update(completed_tiles)
            finally:
                reader.clear_cache()
            LOGGER.debug(
                "%s: completed plane %s/%s",
                progress_label or "Downsample stream",
                plane_index + 1,
                plane_count,
            )
    finally:
        # Tifffile may close a generator immediately after accepting its last
        # expected tile without resuming the code after the final ``yield``.
        # Treat a fully yielded stream as complete so the user still sees the
        # terminal 100 percent line for that stage.
        if progress is not None and yielded_tiles == total_tiles:
            progress.finish()
        for reader in readers:
            reader.clear_cache()


def _page_readers(tiff: tifffile.TiffFile, plane_count: int) -> list[TiffPlaneReader]:
    pages = [page.aspage() for page in list(tiff.pages)[:plane_count]]
    if len(pages) != plane_count:
        raise ValueError(
            f"Temporary pyramid level has {len(pages)} pages; expected {plane_count}"
        )
    lock = threading.RLock()
    return [TiffPlaneReader(page, lock=lock) for page in pages]


def _resolution(pixel_size: PixelSize, scale: int) -> tuple[float, float]:
    scaled = pixel_size.scaled(scale).converted_to("cm")
    return (1.0 / scaled.x, 1.0 / scaled.y)


def _series_layout_matches(
    actual_axes: str,
    actual_shape: Sequence[int],
    expected_axes: str,
    expected_shape: Sequence[int],
) -> bool:
    actual = tuple(int(item) for item in actual_shape)
    expected = tuple(int(item) for item in expected_shape)
    if actual_axes == expected_axes and actual == expected:
        return True
    # tifffile conventionally squeezes a singleton planar C axis when it
    # materializes a series, even though OME still declares SizeC=1.
    return (
        expected_axes == "CYX"
        and expected[0] == 1
        and actual_axes == "YX"
        and actual == expected[1:]
    )


class OMETiffWriter:
    """Write standardized, tiled, pyramidal, MITI-profiled OME-TIFF images.

    The writer accepts NumPy arrays through :meth:`write`. Advanced callers may
    provide any streaming source that implements :class:`PlaneReaderSource`
    through :meth:`write_source`. The conversion CLI uses this same class, so
    the public writer and ``omeify convert`` share one output implementation.

    ``image_type`` is a semantic input flag, not a private TIFF tag. RGB is
    encoded as one logical OME channel with three interleaved samples. Label
    images are one integer YX raster and use nearest-neighbor pyramids.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        image_type: ImageType = "multichannel",
        channel_names: Sequence[str] | None = None,
        pixel_size: PixelSize,
        compression: str | None = None,
        jpeg_quality: int = 90,
        jpeg_subsampling: JPEGSubsampling = "444",
        tile_size: int = 1024,
        pyramid_levels: int | None = None,
        downsample: DownsampleMethod | None = None,
        max_workers: int | None = None,
        display_uuid: bool = True,
        software: str | None = None,
        overwrite: bool = True,
        cache_directory: str | Path | None = None,
        icc_profile: bytes | None = None,
    ) -> None:
        if image_type not in {"multichannel", "rgb", "label"}:
            raise ValueError("image_type must be 'multichannel', 'rgb', or 'label'")
        self.output_path = Path(output_path)
        self.image_type: ImageType = image_type
        self.channel_names = (
            None if channel_names is None else tuple(str(item) for item in channel_names)
        )
        if not isinstance(pixel_size, PixelSize):
            raise TypeError("pixel_size must be a PixelSize instance")
        self.pixel_size = pixel_size
        self.compression_name = compression or "LZW"
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if jpeg_subsampling not in _JPEG_SUBSAMPLING_FACTORS:
            choices = ", ".join(_JPEG_SUBSAMPLING_FACTORS)
            raise ValueError(f"jpeg_subsampling must be one of: {choices}")
        self.jpeg_subsampling: JPEGSubsampling = jpeg_subsampling
        self.tile_size = _validate_tile_size(tile_size)
        self.pyramid_levels = pyramid_levels
        effective_downsample = downsample or ("nearest" if image_type == "label" else "mean")
        if effective_downsample not in {"mean", "nearest"}:
            raise ValueError("downsample must be 'mean' or 'nearest'")
        if image_type == "label" and effective_downsample != "nearest":
            raise ValueError("Label-image pyramids require nearest-neighbor downsampling")
        self.downsample: DownsampleMethod = effective_downsample
        if max_workers is None:
            self.max_workers = max(1, min(8, os.cpu_count() or 1))
        else:
            self.max_workers = int(max_workers)
            if self.max_workers < 1:
                raise ValueError("max_workers must be at least one")
        self.display_uuid = bool(display_uuid)
        self.software = _normalize_software_tag(software)
        self.overwrite = bool(overwrite)
        self.cache_directory = (
            Path(cache_directory) if cache_directory is not None else None
        )
        self.icc_profile = None if icc_profile is None else bytes(icc_profile)

    @property
    def path(self) -> Path:
        return self.output_path

    def write(self, image: np.ndarray, *, axes: str | None = None) -> dict[str, object]:
        """Write an in-memory array.

        Accepted layouts are ``CYX`` or ``YX`` for multichannel images, ``YXS``
        for RGB, and ``YX`` for labels. Reader-backed streaming sources use
        the explicit :meth:`write_source` contract.
        """

        array = np.asarray(image)
        inferred_axes = axes
        if inferred_axes is None:
            if self.image_type == "rgb":
                inferred_axes = "YXS"
            elif self.image_type == "label":
                inferred_axes = "YX"
            elif array.ndim == 2:
                inferred_axes = "YX"
            elif array.ndim == 3:
                inferred_axes = "CYX"
            else:
                raise ValueError(
                    "Unable to infer axes for multichannel array with shape "
                    f"{array.shape}; pass axes='YX' or axes='CYX'."
                )
        spec = self._spec(
            axes=inferred_axes,
            shape=array.shape,
            dtype=array.dtype,
            icc_profile=self.icc_profile,
        )
        return self._write_source(_ArraySource(array, spec), spec)

    def write_source(
        self,
        source: PlaneReaderSource,
        *,
        axes: str,
        shape: Sequence[int],
        dtype: np.dtype | str | type,
        icc_profile: bytes | None = None,
    ) -> dict[str, object]:
        """Write a streaming source with one reader per physical TIFF plane."""

        spec = self._spec(
            axes=axes,
            shape=shape,
            dtype=dtype,
            icc_profile=self.icc_profile if icc_profile is None else icc_profile,
        )
        return self._write_source(source, spec)

    def _spec(
        self,
        *,
        axes: str,
        shape: Sequence[int],
        dtype: np.dtype | str | type,
        icc_profile: bytes | None,
    ) -> OMEImageSpec:
        return OMEImageSpec.from_shape(
            image_type=self.image_type,
            axes=axes,
            shape=shape,
            dtype=dtype,
            channel_names=self.channel_names,
            pixel_size=self.pixel_size,
            icc_profile=icc_profile,
        )

    def _write_source(
        self,
        source: PlaneReaderSource,
        spec: OMEImageSpec,
    ) -> dict[str, object]:
        writer_started = time.monotonic()
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        LOGGER.info(
            "Writer preflight: validating %s source plane(s) for %s %s %s output",
            spec.plane_count,
            spec.image_type,
            spec.output_axes,
            spec.output_shape,
        )
        base_readers = source.plane_readers(cache_mib=64)
        try:
            self._validate_readers(base_readers, spec)
        finally:
            for reader in base_readers:
                reader.clear_cache()
        LOGGER.info("Writer preflight passed; output dtype=%s", spec.dtype)

        compression = _compression_settings(
            self.compression_name,
            spec.dtype,
            is_rgb=spec.is_rgb,
            jpeg_quality=self.jpeg_quality,
            jpeg_subsampling=self.jpeg_subsampling,
        )
        if not spec.is_rgb and not compression.lossless:
            raise ValueError(
                "Lossy compression is restricted to RGB OME-TIFF output; "
                "multichannel and label output require lossless compression"
            )
        if compression.subsampling is not None:
            jpeg_alignment = max(compression.subsampling) * 8
            if self.tile_size % jpeg_alignment != 0:
                raise ValueError(
                    f"TIFF tile size {self.tile_size} is incompatible with JPEG "
                    f"{self.jpeg_subsampling} subsampling; use a tile size divisible by "
                    f"{jpeg_alignment}."
                )

        level_shapes = _auto_level_shapes(
            spec.output_shape,
            axes=spec.output_axes,
            tile_size=self.tile_size,
            pyramid_levels=self.pyramid_levels,
        )
        LOGGER.info(
            "Writer plan: compression=%s (%s), tile=%sx%s, workers=%s, downsample=%s, "
            "subresolution-levels=%s",
            compression.name,
            "lossless" if compression.lossless else "lossy",
            self.tile_size,
            self.tile_size,
            self.max_workers,
            self.downsample,
            len(level_shapes) - 1,
        )
        LOGGER.info(
            "Pyramid level shapes: %s",
            ", ".join(f"L{index}={shape}" for index, shape in enumerate(level_shapes)),
        )

        metadata_started = time.monotonic()
        LOGGER.info("Generating and validating minimized OME-XML")
        xml_info = generate_ome_xml(
            spec,
            level_shapes,
            display_uuid=self.display_uuid,
            output_byteorder=_OUTPUT_BYTEORDER,
        )
        omexml = str(xml_info["xml_string"])
        validator = OMESchemaValidator()
        xml_is_valid = validator.validate(omexml)
        if xml_is_valid is None:
            raise RuntimeError(
                "OME-XML schema validation could not be performed because no local "
                "OME 2016-06 schema was available"
            )
        if not xml_is_valid:
            raise ValueError("Generated OME-XML failed OME 2016-06 schema validation")
        miti_header = validate_miti_ome_tiff_header(omexml)
        if not miti_header.is_valid:
            details = "; ".join(miti_header.errors)
            raise ValueError(f"Generated OME header failed omeify MITI validation: {details}")
        LOGGER.info(
            "OME-XML schema and omeify MITI header validation passed in %.2f seconds",
            time.monotonic() - metadata_started,
        )

        with tempfile.TemporaryDirectory(
            prefix="omeify-pyramid-",
            dir=str(self.cache_directory) if self.cache_directory else None,
        ) as temporary_directory:
            temp_root = Path(temporary_directory)
            LOGGER.debug("Pyramid scratch directory: %s", temp_root)
            if len(level_shapes) > 1:
                LOGGER.info(
                    "Building %s temporary uncompressed pyramid level(s) before final encoding",
                    len(level_shapes) - 1,
                )
                pyramid_started = time.monotonic()
                level_paths = self._build_pyramid(source, spec, level_shapes, temp_root)
                LOGGER.info(
                    "Temporary pyramid construction complete in %.2f seconds",
                    time.monotonic() - pyramid_started,
                )
            else:
                LOGGER.info("Pyramid construction skipped: base level only")
                level_paths = []

            file_descriptor, temporary_name = tempfile.mkstemp(
                prefix=".omeify-",
                suffix=".partial",
                dir=str(self.output_path.parent),
            )
            os.close(file_descriptor)
            temporary_output = Path(temporary_name)
            temporary_output.unlink()
            try:
                if level_paths:
                    LOGGER.info(
                        "Writing final encoded OME-TIFF; the base raster and staged pyramid "
                        "levels are streamed again here"
                    )
                else:
                    LOGGER.info("Writing final encoded OME-TIFF from the base raster")
                final_write_started = time.monotonic()
                self._write_output(
                    source,
                    spec,
                    level_shapes,
                    level_paths,
                    temporary_output,
                    omexml,
                    compression,
                )
                LOGGER.info(
                    "Final encoded temporary output complete in %.2f seconds (%s bytes)",
                    time.monotonic() - final_write_started,
                    f"{temporary_output.stat().st_size:,}",
                )

                LOGGER.info("Verifying OME metadata, TIFF layout, decoding, and spot values")
                verification_started = time.monotonic()
                verification = self._verify_output(
                    source,
                    spec,
                    level_shapes,
                    temporary_output,
                    compression,
                )
                LOGGER.info(
                    "Output verification passed in %.2f seconds",
                    time.monotonic() - verification_started,
                )

                LOGGER.info("Installing verified output atomically at %s", self.output_path)
                os.replace(temporary_output, self.output_path)
                LOGGER.info("Verified output installed")
            finally:
                temporary_output.unlink(missing_ok=True)

            LOGGER.info("Removing temporary pyramid cache")

        LOGGER.info("Temporary pyramid cache removed")
        output_size = self.output_path.stat().st_size
        LOGGER.info(
            "Writer complete in %.2f seconds; output size=%s bytes",
            time.monotonic() - writer_started,
            f"{output_size:,}",
        )
        return {
            "ome": {
                "xml_string": omexml,
                "schema_location": validator.schema_location,
                "xml_is_valid": xml_is_valid,
                "uuid": xml_info["uuid"],
            },
            "miti_header": miti_header.as_dict(),
            "output_file": {
                "path": str(self.output_path),
                "size_bytes": output_size,
                "type_description": "Pyramidal OME-TIFF",
                "dtype": spec.dtype.name,
                "shape": list(spec.output_shape),
                "shape_cyx": list(spec.shape_cyx),
                "axes": spec.output_axes,
                "byte_order": verification["output_byte_order"],
                "lossless_compression": compression.lossless,
            },
            "image": {
                "image_type": spec.image_type,
                "channel_names": list(spec.channel_names),
                "size_c": spec.size_c,
                "logical_channel_count": spec.logical_channel_count,
                "samples_per_pixel": spec.samples_per_pixel,
                "interleaved": spec.is_rgb,
                "pixel_size": list(spec.pixel_size.to_tuple()),
                "significant_bits": spec.significant_bits,
                "icc_profile_present": spec.icc_profile is not None,
            },
            "pyramid": {
                "tile_size": self.tile_size,
                "axes": spec.output_axes,
                "downsample_method": self.downsample,
                "level_shapes": [list(shape) for shape in level_shapes],
                "subresolution_count": len(level_shapes) - 1,
            },
            "verification": verification,
            "options": {
                "compression": compression.name,
                "jpeg_quality": self.jpeg_quality if compression.name == "JPEG" else None,
                "jpeg_subsampling": (
                    self.jpeg_subsampling
                    if compression.name == "JPEG" and spec.is_rgb
                    else None
                ),
                "display_uuid": self.display_uuid,
                "software": self.software,
                "max_workers": self.max_workers,
            },
        }

    @staticmethod
    def _validate_readers(readers: Sequence[PlaneReader], spec: OMEImageSpec) -> None:
        if len(readers) != spec.plane_count:
            raise ValueError(
                f"Source supplied {len(readers)} planes; expected {spec.plane_count} "
                f"for {spec.output_axes} shape {spec.output_shape}."
            )
        for index, reader in enumerate(readers):
            if reader.height != spec.size_y or reader.width != spec.size_x:
                raise ValueError(
                    f"Source plane {index} has shape {(reader.height, reader.width)}; "
                    f"expected {(spec.size_y, spec.size_x)}."
                )
            if np.dtype(reader.dtype).newbyteorder("=") != spec.dtype:
                raise TypeError(
                    f"Source plane {index} has dtype {reader.dtype}; expected {spec.dtype}."
                )
            if int(reader.samples_per_pixel) != spec.samples_per_pixel:
                raise ValueError(
                    f"Source plane {index} has SamplesPerPixel={reader.samples_per_pixel}; "
                    f"expected {spec.samples_per_pixel}."
                )

    def _build_pyramid(
        self,
        source: PlaneReaderSource,
        spec: OMEImageSpec,
        level_shapes: Sequence[tuple[int, ...]],
        temp_root: Path,
    ) -> list[Path]:
        level_paths: list[Path] = []
        total_subresolutions = len(level_shapes) - 1
        if total_subresolutions == 0:
            LOGGER.info("No subresolution levels requested; pyramid staging is skipped")
            return level_paths

        for level_index, output_shape in enumerate(level_shapes[1:], start=1):
            output_path = temp_root / f"level-{level_index}.tf8"
            LOGGER.info(
                "Building pyramid level %s/%s with shape %s from %s",
                level_index,
                total_subresolutions,
                output_shape,
                (
                    "the full-resolution source"
                    if level_index == 1
                    else f"temporary level {level_index - 1}"
                ),
            )
            level_started = time.monotonic()
            progress_label = (
                f"Building pyramid level {level_index}/{total_subresolutions}"
            )
            if level_index == 1:
                readers = source.plane_readers()
                self._write_temporary_level(
                    readers,
                    output_shape,
                    spec,
                    output_path,
                    progress_label=progress_label,
                )
            else:
                with tifffile.TiffFile(level_paths[-1]) as previous:
                    readers = _page_readers(previous, spec.plane_count)
                    self._write_temporary_level(
                        readers,
                        output_shape,
                        spec,
                        output_path,
                        progress_label=progress_label,
                    )
            level_paths.append(output_path)
            LOGGER.info(
                "Pyramid level %s/%s staged in %.2f seconds (%s bytes)",
                level_index,
                total_subresolutions,
                time.monotonic() - level_started,
                f"{output_path.stat().st_size:,}",
            )
        return level_paths

    def _write_temporary_level(
        self,
        readers: Sequence[PlaneReader],
        output_shape: tuple[int, ...],
        spec: OMEImageSpec,
        output_path: Path,
        *,
        progress_label: str,
    ) -> None:
        write_options: dict[str, object] = {
            "shape": output_shape,
            "dtype": spec.dtype,
            "photometric": spec.photometric,
            "tile": (self.tile_size, self.tile_size),
            "compression": None,
            "metadata": None,
            "software": False,
            "maxworkers": 1,
        }
        if spec.is_rgb:
            write_options["planarconfig"] = "contig"

        with tifffile.TiffWriter(
            output_path,
            bigtiff=True,
            byteorder=_OUTPUT_BYTEORDER,
            ome=False,
        ) as writer:
            writer.write(
                _iter_downsampled_tiles(
                    readers,
                    output_shape,
                    axes=spec.output_axes,
                    plane_count=spec.plane_count,
                    tile_size=self.tile_size,
                    method=self.downsample,
                    progress_label=progress_label,
                ),
                **write_options,
            )

    def _write_output(
        self,
        source: PlaneReaderSource,
        spec: OMEImageSpec,
        level_shapes: Sequence[tuple[int, ...]],
        level_paths: Sequence[Path],
        output_path: Path,
        omexml: str,
        compression: CompressionSettings,
    ) -> None:
        common_options: dict[str, object] = {
            "dtype": spec.dtype,
            "photometric": spec.photometric,
            "tile": (self.tile_size, self.tile_size),
            "compression": compression.tifffile_value,
            "compressionargs": compression.compression_args,
            "predictor": compression.predictor,
            "metadata": None,
            "maxworkers": self.max_workers,
        }
        if spec.is_rgb:
            common_options["planarconfig"] = "contig"
            if compression.subsampling is not None:
                common_options["subsampling"] = compression.subsampling
            if spec.icc_profile is not None:
                common_options["iccprofile"] = spec.icc_profile

        total_subresolutions = len(level_paths)
        with tifffile.TiffWriter(
            output_path,
            bigtiff=True,
            byteorder=_OUTPUT_BYTEORDER,
            ome=False,
        ) as writer:
            base_started = time.monotonic()
            base_readers = source.plane_readers()
            writer.write(
                _iter_tiles(
                    base_readers,
                    level_shapes[0],
                    axes=spec.output_axes,
                    plane_count=spec.plane_count,
                    tile_size=self.tile_size,
                    progress_label="Writing final full-resolution base",
                ),
                shape=level_shapes[0],
                description=omexml.encode("utf-8"),
                software=self.software,
                subifds=total_subresolutions,
                resolution=_resolution(spec.pixel_size, 1),
                resolutionunit="CENTIMETER",
                **common_options,
            )
            LOGGER.info(
                "Final full-resolution base encoded and written in %.2f seconds",
                time.monotonic() - base_started,
            )

            for level_index, (level_path, level_shape) in enumerate(
                zip(level_paths, level_shapes[1:]),
                start=1,
            ):
                level_started = time.monotonic()
                with tifffile.TiffFile(level_path) as level_tiff:
                    readers = _page_readers(level_tiff, spec.plane_count)
                    writer.write(
                        _iter_tiles(
                            readers,
                            level_shape,
                            axes=spec.output_axes,
                            plane_count=spec.plane_count,
                            tile_size=self.tile_size,
                            progress_label=(
                                f"Writing final pyramid level "
                                f"{level_index}/{total_subresolutions}"
                            ),
                        ),
                        shape=level_shape,
                        software=False,
                        subfiletype=1,
                        resolution=_resolution(spec.pixel_size, 2**level_index),
                        resolutionunit="CENTIMETER",
                        **common_options,
                    )
                LOGGER.info(
                    "Final pyramid level %s/%s encoded and written in %.2f seconds",
                    level_index,
                    total_subresolutions,
                    time.monotonic() - level_started,
                )

    def _verify_output(
        self,
        source: PlaneReaderSource,
        spec: OMEImageSpec,
        level_shapes: Sequence[tuple[int, ...]],
        output_path: Path,
        compression: CompressionSettings,
    ) -> dict[str, object]:
        verification: dict[str, object] = {
            "ome_tiff_recognized": False,
            "bigtiff": False,
            "output_byte_order": None,
            "software_tag_matches": False,
            "byte_order_metadata_matches_tiff": False,
            "significant_bits_matches_dtype": False,
            "tiff_data_mapping_matches": False,
            "channel_sample_layout_matches": False,
            "pyramid_annotation_linked": False,
            "dtype_matches_source": False,
            "axes_match": False,
            "pyramid_level_shapes_match": False,
            "top_level_ifd_count_matches": False,
            "samples_per_pixel_match": False,
            "photometric_matches": False,
            "compression_matches_requested": False,
            "jpeg_subsampling_matches_requested": None,
            "subifd_layout_matches": False,
            "all_levels_tiled": False,
            "icc_profile_preserved": None,
            "output_pixels_decodable": False,
            "base_pixel_values_checked": False,
            "base_pixel_values_match": None,
            "planes_checked": 0,
            "channels_checked": 0,
            "points_per_plane": 0,
            "points_per_channel": 0,
        }
        LOGGER.debug("Verification: opening candidate TIFF %s", output_path)
        with tifffile.TiffFile(output_path) as output:
            if not output.is_ome:
                raise ValueError("Written TIFF is not recognized as OME-TIFF")
            verification["ome_tiff_recognized"] = True

            if not output.is_bigtiff:
                raise ValueError("Written output is not BigTIFF")
            verification["bigtiff"] = True

            actual_byteorder = output.byteorder
            if actual_byteorder not in {"<", ">"}:
                raise ValueError(f"Unexpected TIFF byte order {actual_byteorder!r}")
            if actual_byteorder != _OUTPUT_BYTEORDER:
                raise ValueError(
                    f"Output TIFF byte order {actual_byteorder!r} does not match configured "
                    f"byte order {_OUTPUT_BYTEORDER!r}"
                )
            verification["output_byte_order"] = "big" if actual_byteorder == ">" else "little"

            try:
                actual_software = str(output.pages[0].aspage().tags["Software"].value)
            except KeyError as exc:
                raise ValueError("Written TIFF does not contain a Software tag") from exc
            if actual_software != self.software:
                raise ValueError(
                    f"TIFF Software tag {actual_software!r} does not match "
                    f"requested value {self.software!r}"
                )
            verification["software_tag_matches"] = True

            omexml = output.ome_metadata
            if not omexml:
                raise ValueError("Written OME-TIFF does not contain OME-XML metadata")
            parser = etree.XMLParser(resolve_entities=False, no_network=True)
            root = etree.fromstring(omexml.encode("utf-8"), parser=parser)
            namespace = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            image = root.find("./ome:Image", namespaces=namespace)
            pixels = root.find("./ome:Image/ome:Pixels", namespaces=namespace)
            if image is None or pixels is None:
                raise ValueError("Written OME-XML does not contain Image/Pixels")

            declared_big_endian = (pixels.get("BigEndian") or "").strip().lower()
            expected_big_endian = "true" if actual_byteorder == ">" else "false"
            if declared_big_endian != expected_big_endian:
                raise ValueError(
                    f"OME BigEndian={declared_big_endian!r} does not match TIFF byte order "
                    f"{actual_byteorder!r}"
                )
            verification["byte_order_metadata_matches_tiff"] = True

            try:
                declared_significant_bits = int(pixels.get("SignificantBits", ""))
            except ValueError as exc:
                raise ValueError("OME SignificantBits is missing or invalid") from exc
            if declared_significant_bits != spec.significant_bits:
                raise ValueError(
                    f"OME SignificantBits={declared_significant_bits} does not match "
                    f"dtype width {spec.significant_bits}"
                )
            verification["significant_bits_matches_dtype"] = True

            channels = pixels.findall("./ome:Channel", namespaces=namespace)
            channel_samples = [
                int(channel.get("SamplesPerPixel", "0")) for channel in channels
            ]
            expected_interleaved = "true" if spec.is_rgb else "false"
            declared_interleaved = (pixels.get("Interleaved") or "false").lower()
            if (
                len(channels) != spec.logical_channel_count
                or sum(channel_samples) != spec.size_c
                or channel_samples
                != [spec.samples_per_pixel] * spec.logical_channel_count
                or declared_interleaved != expected_interleaved
            ):
                raise ValueError(
                    "OME Channel/SamplesPerPixel layout does not match the written TIFF layout"
                )
            verification["channel_sample_layout_matches"] = True

            tiff_data = pixels.findall("./ome:TiffData", namespaces=namespace)
            if len(tiff_data) != 1:
                raise ValueError(
                    f"OME Pixels contains {len(tiff_data)} TiffData elements; expected one"
                )
            if int(tiff_data[0].get("IFD", "0")) != 0:
                raise ValueError("OME TiffData must start at IFD=0")
            if int(tiff_data[0].get("PlaneCount", "0")) != spec.plane_count:
                raise ValueError(
                    "OME TiffData PlaneCount does not match the physical top-level plane "
                    f"count ({spec.plane_count})"
                )
            verification["tiff_data_mapping_matches"] = True

            expected_subifds = len(level_shapes) - 1
            if expected_subifds:
                map_annotations = root.findall(
                    "./ome:StructuredAnnotations/ome:MapAnnotation",
                    namespaces=namespace,
                )
                pyramid_annotations = [
                    item
                    for item in map_annotations
                    if item.get("Namespace") == "openmicroscopy.org/PyramidResolution"
                ]
                annotation_refs = {
                    item.get("ID")
                    for item in image.findall("./ome:AnnotationRef", namespaces=namespace)
                }
                if len(pyramid_annotations) != 1:
                    raise ValueError(
                        "OME pyramid metadata must contain exactly one PyramidResolution "
                        "MapAnnotation"
                    )
                annotation_id = pyramid_annotations[0].get("ID")
                if not annotation_id or annotation_id not in annotation_refs:
                    raise ValueError(
                        "OME PyramidResolution MapAnnotation is not linked from Image"
                    )
            verification["pyramid_annotation_linked"] = True

            series = output.series[0]
            if np.dtype(series.dtype).newbyteorder("=") != spec.dtype:
                raise TypeError(
                    f"Output dtype {series.dtype} does not match source dtype {spec.dtype}"
                )
            verification["dtype_matches_source"] = True
            if not _series_layout_matches(
                str(series.axes),
                series.shape,
                spec.output_axes,
                spec.output_shape,
            ):
                raise ValueError(
                    f"Output axes/shape {series.axes!r} {tuple(series.shape)} do not match "
                    f"expected {spec.output_axes!r} {spec.output_shape}"
                )
            verification["axes_match"] = True
            if len(series.levels) != len(level_shapes):
                raise ValueError(
                    f"Output has {len(series.levels)} pyramid levels; expected {len(level_shapes)}"
                )
            for index, (level, expected_shape) in enumerate(zip(series.levels, level_shapes)):
                actual = tuple(int(value) for value in level.shape)
                if not _series_layout_matches(
                    str(level.axes),
                    actual,
                    spec.output_axes,
                    expected_shape,
                ):
                    raise ValueError(
                        f"Output level {index} axes/shape {level.axes!r} {actual} does not "
                        f"match expected {spec.output_axes!r} {expected_shape}"
                    )
            verification["pyramid_level_shapes_match"] = True

            if len(output.pages) != spec.plane_count:
                raise ValueError(
                    f"Output has {len(output.pages)} top-level IFDs; expected {spec.plane_count}"
                )
            verification["top_level_ifd_count_matches"] = True

            all_levels_tiled = True
            samples_match = True
            photometric_match = True
            compression_match = True
            jpeg_subsampling_match = True
            expected_compression = _OUTPUT_COMPRESSION_CODES[compression.name]
            layout_progress = ProgressLogger(
                LOGGER,
                "Verifying TIFF plane and pyramid layouts",
                spec.plane_count,
                unit="planes",
            )
            for plane_index, frame in enumerate(output.pages):
                page = frame.aspage()
                all_levels_tiled = all_levels_tiled and bool(page.is_tiled)
                compression_match = compression_match and (
                    int(page.compression) == expected_compression
                )
                samples_match = samples_match and (
                    int(page.samplesperpixel) == spec.samples_per_pixel
                )
                if spec.is_rgb:
                    photometric_match = photometric_match and int(page.photometric) in {2, 6}
                    if int(page.planarconfig) != 1:
                        raise ValueError(
                            f"Output RGB plane {plane_index} is not contiguous-sample TIFF"
                        )
                else:
                    photometric_match = photometric_match and int(page.photometric) == 1
                if compression.subsampling is not None:
                    try:
                        actual_subsampling = tuple(
                            int(value) for value in page.tags["YCbCrSubSampling"].value
                        )
                    except (KeyError, TypeError):
                        jpeg_subsampling_match = False
                    else:
                        jpeg_subsampling_match = (
                            jpeg_subsampling_match
                            and actual_subsampling == compression.subsampling
                        )

                subpages = list(page.pages) if page.pages is not None else []
                if len(subpages) != expected_subifds:
                    raise ValueError(
                        f"Output plane {plane_index} has {len(subpages)} SubIFDs; "
                        f"expected {expected_subifds}"
                    )
                for level_index, subframe in enumerate(subpages, start=1):
                    subpage = subframe.aspage()
                    all_levels_tiled = all_levels_tiled and bool(subpage.is_tiled)
                    compression_match = compression_match and (
                        int(subpage.compression) == expected_compression
                    )
                    samples_match = samples_match and (
                        int(subpage.samplesperpixel) == spec.samples_per_pixel
                    )
                    if spec.is_rgb:
                        photometric_match = photometric_match and int(
                            subpage.photometric
                        ) in {2, 6}
                        if int(subpage.planarconfig) != 1:
                            raise ValueError(
                                f"Output RGB plane {plane_index}, pyramid level "
                                f"{level_index} is not contiguous-sample TIFF"
                            )
                    else:
                        photometric_match = (
                            photometric_match and int(subpage.photometric) == 1
                        )
                    if compression.subsampling is not None:
                        try:
                            actual_subsampling = tuple(
                                int(value)
                                for value in subpage.tags["YCbCrSubSampling"].value
                            )
                        except (KeyError, TypeError):
                            jpeg_subsampling_match = False
                        else:
                            jpeg_subsampling_match = (
                                jpeg_subsampling_match
                                and actual_subsampling == compression.subsampling
                            )
                    if not (int(subpage.subfiletype) & 1):
                        raise ValueError(
                            f"Output plane {plane_index}, pyramid level {level_index} "
                            "is not marked as a reduced-resolution image"
                        )
                layout_progress.update(plane_index + 1)
            if not samples_match:
                raise ValueError("One or more output levels has the wrong SamplesPerPixel")
            verification["samples_per_pixel_match"] = True
            if not photometric_match:
                raise ValueError("One or more output levels has the wrong photometric mode")
            verification["photometric_matches"] = True
            if not compression_match:
                raise ValueError(
                    "One or more output levels does not use the requested TIFF compression"
                )
            verification["compression_matches_requested"] = True
            if compression.subsampling is not None:
                if not jpeg_subsampling_match:
                    raise ValueError(
                        "One or more output levels does not use the requested JPEG subsampling"
                    )
                verification["jpeg_subsampling_matches_requested"] = True
            verification["subifd_layout_matches"] = True
            if not all_levels_tiled:
                raise ValueError("One or more output base or pyramid planes are not tiled")
            verification["all_levels_tiled"] = True

            if spec.icc_profile is not None:
                output_icc = output.pages[0].aspage().iccprofile
                if output_icc is None or bytes(output_icc) != spec.icc_profile:
                    raise ValueError("Source ICC profile was not preserved in the RGB output")
                verification["icc_profile_preserved"] = True

            LOGGER.debug(
                "Verification: decoding %s representative point(s) from each of %s plane(s)",
                3,
                spec.plane_count,
            )
            output_readers = _page_readers(output, spec.plane_count)
            input_readers = (
                source.plane_readers(cache_mib=16) if compression.lossless else None
            )
            try:
                if input_readers is not None and len(input_readers) != spec.plane_count:
                    raise ValueError(
                        f"Source supplied {len(input_readers)} verification readers; "
                        f"expected {spec.plane_count}"
                    )
                coordinates = sorted(
                    {
                        (0, 0),
                        (spec.size_y // 2, spec.size_x // 2),
                        (spec.size_y - 1, spec.size_x - 1),
                    }
                )
                verification_progress = ProgressLogger(
                    LOGGER,
                    "Verifying representative base-plane pixels",
                    spec.plane_count,
                    unit="planes",
                )
                if input_readers is not None:
                    verification["base_pixel_values_checked"] = True
                for plane_index, output_reader in enumerate(output_readers):
                    for y, x in coordinates:
                        output_value = output_reader.read_region(
                            y, y + 1, x, x + 1
                        )[0, 0, ...]
                        if input_readers is not None:
                            source_value = input_readers[plane_index].read_region(
                                y, y + 1, x, x + 1
                            )[0, 0, ...]
                            if not np.array_equal(source_value, output_value):
                                raise ValueError(
                                    "Lossless base-image verification failed at "
                                    f"plane {plane_index}, y={y}, x={x}: "
                                    f"source={source_value}, output={output_value}"
                                )
                    verification_progress.update(plane_index + 1)
                    LOGGER.debug(
                        "Verified representative pixels for plane %s/%s",
                        plane_index + 1,
                        spec.plane_count,
                    )
            finally:
                for reader in output_readers:
                    reader.clear_cache()
                if input_readers is not None:
                    for reader in input_readers:
                        reader.clear_cache()

            verification["output_pixels_decodable"] = True
            verification["planes_checked"] = spec.plane_count
            verification["channels_checked"] = spec.size_c
            verification["points_per_plane"] = len(coordinates)
            verification["points_per_channel"] = len(coordinates)
            if input_readers is not None:
                verification["base_pixel_values_match"] = True
        return verification


class TemporaryOMETiffWriter:
    """Context-managed temporary-file wrapper around :class:`OMETiffWriter`.

    The class owns only temporary-path lifecycle. All image construction,
    validation, verification, and atomic writing remain delegated to
    ``OMETiffWriter``.
    """

    def __init__(
        self,
        *,
        directory: str | Path | None = None,
        prefix: str = "omeify-",
        suffix: str = ".ome.tif",
        **writer_options: Any,
    ) -> None:
        self.directory = None if directory is None else Path(directory)
        self.prefix = str(prefix)
        self.suffix = str(suffix)
        if not self.suffix:
            raise ValueError("Temporary OME-TIFF suffix must be non-empty")
        self.writer_options = dict(writer_options)
        self._temporary_directory: tempfile.TemporaryDirectory[str] | None = None
        self._path: Path | None = None
        self._writer: OMETiffWriter | None = None

    @property
    def path(self) -> Path:
        if self._path is None:
            raise RuntimeError("TemporaryOMETiffWriter must be entered before path is available")
        return self._path

    @property
    def writer(self) -> OMETiffWriter:
        if self._writer is None:
            raise RuntimeError("TemporaryOMETiffWriter must be entered before writing")
        return self._writer

    def __enter__(self) -> "TemporaryOMETiffWriter":
        if self._temporary_directory is not None:
            raise RuntimeError("TemporaryOMETiffWriter context is already active")
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix=self.prefix,
            dir=str(self.directory) if self.directory is not None else None,
        )
        self._temporary_directory = temporary
        self._path = Path(temporary.name) / f"image{self.suffix}"
        options = dict(self.writer_options)
        options.setdefault("cache_directory", Path(temporary.name))
        options.setdefault("overwrite", True)
        try:
            self._writer = OMETiffWriter(self._path, **options)
        except Exception:
            self.close()
            raise
        return self

    def write(self, image: np.ndarray, *, axes: str | None = None) -> dict[str, object]:
        return self.writer.write(image, axes=axes)

    def write_source(
        self,
        source: PlaneReaderSource,
        *,
        axes: str,
        shape: Sequence[int],
        dtype: np.dtype | str | type,
        icc_profile: bytes | None = None,
    ) -> dict[str, object]:
        return self.writer.write_source(
            source,
            axes=axes,
            shape=shape,
            dtype=dtype,
            icc_profile=icc_profile,
        )

    def close(self) -> None:
        self._writer = None
        self._path = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = None

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
