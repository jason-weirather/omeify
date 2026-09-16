from __future__ import annotations

import logging
import threading
import time
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import tifffile

from omeify.io.pixel_size import PixelSize
from omeify.io.tiff import PlaneReader, TiffPlaneReader
from omeify.progress import ProgressLogger

from .configuration import DownsampleMethod, OUTPUT_BYTEORDER
from .model import PreparedImage
from .precision import round_float32_mantissa

LOGGER = logging.getLogger(__name__)


def yx_shape(shape: Sequence[int], axes: str) -> tuple[int, int]:
    if len(shape) != len(axes) or "Y" not in axes or "X" not in axes:
        raise ValueError(f"Shape {tuple(shape)} is incompatible with axes {axes!r}")
    return int(shape[axes.index("Y")]), int(shape[axes.index("X")])


def replace_yx(
    shape: Sequence[int],
    axes: str,
    height: int,
    width: int,
) -> tuple[int, ...]:
    result = [int(value) for value in shape]
    result[axes.index("Y")] = int(height)
    result[axes.index("X")] = int(width)
    return tuple(result)


def auto_level_shapes(
    base_shape: tuple[int, ...],
    *,
    axes: str,
    tile_size: int,
    pyramid_levels: int | None,
) -> tuple[tuple[int, ...], ...]:
    """Return the base shape followed by every requested pyramid shape."""

    height, width = yx_shape(base_shape, axes)
    shapes = [base_shape]
    if pyramid_levels is not None:
        target_levels = pyramid_levels
    else:
        target_levels = 0
        current_height, current_width = height, width
        while max(current_height, current_width) > tile_size:
            target_levels += 1
            current_height = (current_height + 1) // 2
            current_width = (current_width + 1) // 2
            if target_levels >= 32:
                break

    current_height, current_width = height, width
    for _ in range(target_levels):
        if current_height == 1 and current_width == 1:
            raise ValueError(
                f"Cannot build {target_levels} subresolution levels from base shape "
                f"{base_shape}; the pyramid already reached 1x1."
            )
        current_height = max(1, (current_height + 1) // 2)
        current_width = max(1, (current_width + 1) // 2)
        shapes.append(
            replace_yx(
                base_shape,
                axes,
                current_height,
                current_width,
            )
        )
    return tuple(shapes)


def _broadcast_counts(counts: np.ndarray, ndim: int) -> np.ndarray:
    if ndim == 2:
        return counts
    return counts[(...,) + (None,) * (ndim - 2)]


def mean_downsample_2x(
    block: np.ndarray,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Downsample a YX or YXS block with deterministic nearest-even means."""

    if block.ndim not in {2, 3}:
        raise ValueError(f"Expected YX or YXS block, got shape {block.shape}")

    out_height, out_width = out_shape
    dtype = block.dtype
    accumulator_shape = (out_height, out_width, *block.shape[2:])
    counts = np.zeros((out_height, out_width), dtype=np.uint8)

    if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(dtype, np.bool_):
        accumulator = np.zeros(accumulator_shape, dtype=np.uint64)
        for delta_y in (0, 1):
            for delta_x in (0, 1):
                sample = block[delta_y::2, delta_x::2, ...]
                height, width = sample.shape[:2]
                accumulator[:height, :width, ...] += sample.astype(
                    np.uint64,
                    copy=False,
                )
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
        for delta_y in (0, 1):
            for delta_x in (0, 1):
                sample = block[delta_y::2, delta_x::2, ...]
                height, width = sample.shape[:2]
                accumulator[:height, :width, ...] += sample.astype(
                    np.int64,
                    copy=False,
                )
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
        np.complex128
        if np.issubdtype(dtype, np.complexfloating)
        else np.float64
    )
    accumulator = np.zeros(accumulator_shape, dtype=accumulator_dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        for delta_y in (0, 1):
            for delta_x in (0, 1):
                sample = block[delta_y::2, delta_x::2, ...]
                height, width = sample.shape[:2]
                accumulator[:height, :width, ...] += sample.astype(
                    accumulator_dtype,
                    copy=False,
                )
                counts[:height, :width] += 1
        result = accumulator / _broadcast_counts(counts, block.ndim)
    if dtype.kind == "f" and dtype.itemsize == 8 and np.any(~np.isfinite(result)):
        _recover_finite_means(block, result, counts)
    return result.astype(dtype, copy=False)


def _recover_finite_means(
    block: np.ndarray, result: np.ndarray, counts: np.ndarray,
) -> None:
    """Repair only accumulator overflow with exclusively finite contributors.

    Ordinary means keep their original arithmetic/rounding. A nonfinite input
    still propagates through IEEE arithmetic. Divide before summation only in
    overflowing groups. Counts are 1, 2 or 4, so this is binary scaling rather
    than a data-dependent normalization that could magnify cancellation errors.
    """

    finite = np.ones(result.shape, dtype=bool)
    for dy in (0, 1):
        for dx in (0, 1):
            sample = block[dy::2, dx::2, ...]
            region = (slice(0, sample.shape[0]), slice(0, sample.shape[1]))
            finite[region] &= np.isfinite(sample)
    recover = finite & ~np.isfinite(result)
    if not np.any(recover):
        return
    normalized = np.zeros(result.shape, dtype=np.float64)
    divisors = np.broadcast_to(_broadcast_counts(counts, block.ndim), result.shape)
    for dy in (0, 1):
        for dx in (0, 1):
            sample = block[dy::2, dx::2, ...]
            region = (slice(0, sample.shape[0]), slice(0, sample.shape[1]))
            selected = recover[region]
            normalized[region][selected] += sample[selected] / divisors[region][selected]
    result[recover] = normalized[recover]


def tile_count(
    shape: Sequence[int],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
) -> int:
    """Return the number of tiles consumed for one canonical image level."""

    height, width = yx_shape(shape, axes)
    tiles_y = (height + tile_size - 1) // tile_size
    tiles_x = (width + tile_size - 1) // tile_size
    return int(plane_count) * tiles_y * tiles_x


def downsample_region(
    reader: PlaneReader,
    *,
    out_y0: int,
    out_y1: int,
    out_x0: int,
    out_x1: int,
    method: DownsampleMethod,
    float32_mantissa_bits: int | None = None,
) -> np.ndarray:
    input_y0 = out_y0 * 2
    input_y1 = min(reader.height, out_y1 * 2)
    input_x0 = out_x0 * 2
    input_x1 = min(reader.width, out_x1 * 2)
    block = reader.read_region(input_y0, input_y1, input_x0, input_x1)
    out_shape = (out_y1 - out_y0, out_x1 - out_x0)
    if method == "nearest":
        result = np.ascontiguousarray(
            block[::2, ::2, ...][: out_shape[0], : out_shape[1], ...]
        )
    elif method == "mean":
        result = np.ascontiguousarray(mean_downsample_2x(block, out_shape))
    else:
        raise AssertionError(f"Unhandled downsample method {method}")
    if float32_mantissa_bits is not None:
        result = round_float32_mantissa(result, float32_mantissa_bits)
    return result


def iter_tiles(
    readers: Sequence[PlaneReader],
    shape: tuple[int, ...],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
    progress_label: str | None = None,
) -> Iterable[np.ndarray]:
    """Yield one complete tiled image level while releasing reader caches."""

    height, width = yx_shape(shape, axes)
    if len(readers) != plane_count:
        raise ValueError(f"Expected {plane_count} page readers, got {len(readers)}")
    total_tiles = tile_count(
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
                        if progress is not None:
                            progress.update(yielded_tiles)
            finally:
                reader.clear_cache()
            LOGGER.debug(
                "%s: completed plane %s/%s",
                progress_label or "Tile stream",
                plane_index + 1,
                plane_count,
            )
    finally:
        # Tifffile may close the generator after accepting the last tile
        # without resuming execution after the final yield.
        if progress is not None and yielded_tiles == total_tiles:
            progress.finish()
        for reader in readers:
            reader.clear_cache()


def iter_downsampled_tiles(
    readers: Sequence[PlaneReader],
    output_shape: tuple[int, ...],
    *,
    axes: str,
    plane_count: int,
    tile_size: int,
    method: DownsampleMethod,
    float32_mantissa_bits: int | None = None,
    progress_label: str | None = None,
) -> Iterable[np.ndarray]:
    """Yield one downsampled tiled level from readers for the preceding level."""

    height, width = yx_shape(output_shape, axes)
    if len(readers) != plane_count:
        raise ValueError(f"Expected {plane_count} page readers, got {len(readers)}")
    total_tiles = tile_count(
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
                        tile = downsample_region(
                            reader,
                            out_y0=y0,
                            out_y1=y1,
                            out_x0=x0,
                            out_x1=x1,
                            method=method,
                            float32_mantissa_bits=float32_mantissa_bits,
                        )
                        yielded_tiles += 1
                        yield tile
                        if progress is not None:
                            progress.update(yielded_tiles)
            finally:
                reader.clear_cache()
            LOGGER.debug(
                "%s: completed plane %s/%s",
                progress_label or "Downsample stream",
                plane_index + 1,
                plane_count,
            )
    finally:
        if progress is not None and yielded_tiles == total_tiles:
            progress.finish()
        for reader in readers:
            reader.clear_cache()


def page_readers(
    tiff: tifffile.TiffFile,
    plane_count: int,
) -> list[TiffPlaneReader]:
    pages = [page.aspage() for page in list(tiff.pages)[:plane_count]]
    if len(pages) != plane_count:
        raise ValueError(
            f"Temporary pyramid level has {len(pages)} pages; expected {plane_count}"
        )
    lock = threading.RLock()
    return [TiffPlaneReader(page, lock=lock) for page in pages]


def resolution(pixel_size: PixelSize, scale: int) -> tuple[float, float]:
    scaled = pixel_size.scaled(scale).converted_to("cm")
    return (1.0 / scaled.x, 1.0 / scaled.y)


def series_layout_matches(
    actual_axes: str,
    actual_shape: Sequence[int],
    expected_axes: str,
    expected_shape: Sequence[int],
) -> bool:
    actual = tuple(int(item) for item in actual_shape)
    expected = tuple(int(item) for item in expected_shape)
    if actual_axes == expected_axes and actual == expected:
        return True
    # Tifffile conventionally squeezes a singleton planar C axis when it
    # materializes a series, even though OME still declares SizeC=1.
    return (
        expected_axes == "CYX"
        and expected[0] == 1
        and actual_axes == "YX"
        and actual == expected[1:]
    )


def build_pyramid(
    prepared: PreparedImage,
    directory: Path,
    *,
    tile_size: int,
) -> tuple[Path, ...]:
    """Build one temporary pyramid without materializing the base raster."""

    paths: list[Path] = []
    total_subresolutions = len(prepared.level_shapes) - 1
    if total_subresolutions == 0:
        LOGGER.info("Pyramid construction skipped for %s: base level only", prepared.display_name)
        return ()

    directory.mkdir(parents=True, exist_ok=True)
    spec = prepared.spec
    for level_index, output_shape in enumerate(
        prepared.level_shapes[1:],
        start=1,
    ):
        output_path = directory / f"level-{level_index}.tf8"
        source_description = (
            "the full-resolution source"
            if level_index == 1
            else f"temporary level {level_index - 1}"
        )
        if prepared.name is None:
            stage_name = "pyramid"
        else:
            stage_name = prepared.display_name
        LOGGER.info(
            "Building %s level %s/%s with shape %s from %s",
            stage_name,
            level_index,
            total_subresolutions,
            output_shape,
            source_description,
        )
        level_started = time.monotonic()
        progress_label = (
            f"Building {stage_name} level "
            f"{level_index}/{total_subresolutions}"
        )
        if level_index == 1:
            readers = prepared.source.plane_readers()
            write_temporary_level(
                readers,
                output_shape,
                prepared,
                output_path,
                tile_size=tile_size,
                progress_label=progress_label,
            )
        else:
            with tifffile.TiffFile(paths[-1]) as previous:
                readers = page_readers(previous, spec.plane_count)
                write_temporary_level(
                    readers,
                    output_shape,
                    prepared,
                    output_path,
                    tile_size=tile_size,
                    progress_label=progress_label,
                )
        paths.append(output_path)
        LOGGER.info(
            "%s pyramid level %s/%s staged in %.2f seconds (%s bytes)",
            prepared.display_name.capitalize(),
            level_index,
            total_subresolutions,
            time.monotonic() - level_started,
            f"{output_path.stat().st_size:,}",
        )
    return tuple(paths)


def write_temporary_level(
    readers: Sequence[PlaneReader],
    output_shape: tuple[int, ...],
    prepared: PreparedImage,
    output_path: Path,
    *,
    tile_size: int,
    progress_label: str,
) -> None:
    """Write one uncompressed intermediate pyramid level."""

    spec = prepared.spec
    options: dict[str, object] = {
        "shape": output_shape,
        "dtype": spec.dtype,
        "photometric": spec.photometric,
        "tile": (tile_size, tile_size),
        "compression": None,
        "metadata": None,
        "software": False,
        "maxworkers": 1,
    }
    if spec.is_rgb:
        options["planarconfig"] = "contig"

    with tifffile.TiffWriter(
        output_path,
        bigtiff=True,
        byteorder=OUTPUT_BYTEORDER,
        ome=False,
    ) as writer:
        writer.write(
            iter_downsampled_tiles(
                readers,
                output_shape,
                axes=spec.output_axes,
                plane_count=spec.plane_count,
                tile_size=tile_size,
                method=prepared.downsample,
                float32_mantissa_bits=prepared.float32_mantissa_bits,
                progress_label=progress_label,
            ),
            **options,
        )
