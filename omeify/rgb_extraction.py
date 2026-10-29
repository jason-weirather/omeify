"""Recover interleaved RGB from three explicit scalar color channels.

The transform is an image *reinterpretation* with an explicit uint16-to-uint8
policy, not a fidelity-preserving ``convert``. Reads and writes stay regional;
OME-TIFF/OME-Zarr writer verification and pyramid construction are reused.
"""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np

from .io.base import Image, RGBImage
from .io.image_metadata import ImageLevel, ImageMetadata, integer
from .io.image_source import ImageSource
from .io.output import OutputFormat, image_writer
from .io.pixel_size import PixelSize
from .io.source_reader import InputType, PLANAR_INPUT_TYPES, source_reader
from .reports import complete_report
from .workflow import validate_image_paths

LOGGER = logging.getLogger(__name__)

RGBEncoding = Literal["auto", "native", "high-byte", "rounded", "low-byte"]
RGBBackground = Literal["white", "keep"]
RGBSelector = str | int

# Keep this distinct from ``convert``. RGB extraction may discard lower bits,
# and the recovery of black-as-missing pixels changes real RGB values.


def _channel_indices(image: Image, selectors: Sequence[RGBSelector]) -> tuple[int, int, int]:
    if len(selectors) != 3:
        raise ValueError("Exactly three RGB channel selectors are required")
    if image.image_type != "multichannel" or image.axes != "CYX":
        raise ValueError("RGB extraction requires scalar/multichannel CYX input")
    indices = []
    for color, selector in zip(("RED", "GREEN", "BLUE"), selectors, strict=True):
        if isinstance(selector, bool):
            raise TypeError(f"{color} channel selector must be an index or exact name")
        if isinstance(selector, (int, np.integer)):
            index = integer(selector, f"{color} channel index")
            if index >= image.channel_count:
                raise IndexError(f"{color} channel index {index} is out of range")
        elif isinstance(selector, str) and selector:
            # The Python API treats *every* string as a literal channel name.
            # The CLI explicitly parses numerical/index: selectors to integers.
            index = image.get_by_name(selector).index
        else:
            raise TypeError(f"{color} channel selector must be an index or exact name")
        indices.append(index)
    if len(set(indices)) != 3:
        raise ValueError("RED, GREEN and BLUE must be three distinct source channels")
    return tuple(indices)  # type: ignore[return-value]


def _resolved_encoding(dtype: np.dtype, encoding: RGBEncoding) -> str:
    dtype = np.dtype(dtype).newbyteorder("=")
    if encoding not in {"auto", "native", "high-byte", "rounded", "low-byte"}:
        raise ValueError("encoding must be auto, native, high-byte, rounded or low-byte")
    if dtype == np.dtype("uint8"):
        if encoding not in {"auto", "native"}:
            raise ValueError("uint8 RGB channels support only auto/native encoding")
        return "native"
    if dtype == np.dtype("uint16"):
        if encoding == "native":
            raise ValueError("native encoding requires uint8 source channels")
        return "high-byte" if encoding == "auto" else encoding
    raise TypeError(f"RGB extraction requires uint8 or uint16 scalar channels, got {dtype}")


def decode_rgb(
    values: np.ndarray, *, encoding: str, background: RGBBackground = "white",
) -> np.ndarray:
    """Convert three source CYX planes to interleaved uint8 RGB.

    ``white`` changes only original (0,0,0) triples, not values which become zero
    through 16-bit decoding. It is an explicit missing-data interpretation, not
    a stain normalization or inference of true tissue absence.
    """
    if not isinstance(values, np.ndarray) or values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("RGB recovery expects three CYX source planes")
    if background not in {"white", "keep"}:
        raise ValueError("background must be 'white' or 'keep'")
    dtype = values.dtype.newbyteorder("=")
    if encoding == "native" and dtype == np.dtype("uint8"):
        converted = values
    elif dtype == np.dtype("uint16") and encoding == "high-byte":
        converted = (values >> 8).astype(np.uint8)
    elif dtype == np.dtype("uint16") and encoding == "rounded":
        # Exact nearest-neighbor scaling of 0..65535 to 0..255 (half-up).
        # 32-bit arithmetic avoids overflow and float precision ambiguity.
        converted = ((values.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
    elif dtype == np.dtype("uint16") and encoding == "low-byte":
        if np.any(values > 255):
            raise ValueError("low-byte encoding requires *all* source values <= 255")
        converted = values.astype(np.uint8)
    else:
        raise ValueError(f"Encoding {encoding!r} cannot decode {dtype} source values")
    result = np.ascontiguousarray(np.moveaxis(converted, 0, -1))
    if background == "white":
        empty = np.all(values == 0, axis=0)
        result[empty] = 255
    return result


class _RGBRecoverySource(ImageSource):
    """A one-level, read-on-demand RGB view borrowing an already open parent."""

    def __init__(
        self, parent: Image, indices: tuple[int, int, int], *,
        encoding: str, background: RGBBackground,
        pixel_size: PixelSize, name: str,
    ) -> None:
        if not parent.is_open:
            raise RuntimeError("The source image must be open during RGB extraction")
        height, width = parent.levels[0].spatial_shape
        shape = (height, width, 3)
        super().__init__(ImageMetadata(
            axes="YXS", shape=shape, dtype=np.dtype("uint8"), image_type="rgb",
            channel_names=(name,), pixel_size=pixel_size,
            unrepresented_geometry=parent._metadata.unrepresented_geometry,
            levels=(ImageLevel(
                0, "YXS", shape, pixel_size=pixel_size, origin_yx=parent.origin_yx,
            ),),
        ))
        self._parent = parent
        self._indices = indices
        self._encoding = encoding
        self._background = background
        self._parent_session = parent._generation  # The borrowed image session may not change.

    @property
    def backing_paths(self) -> tuple[Path, ...]:
        return self._parent.backing_paths

    def _open(self) -> None:
        self._parent._ensure_session(self._parent_session)

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        self._parent._ensure_session(self._parent_session)
        block = self._parent.read_region(y0, y1, x0, x1, level=0, channels=self._indices)
        return decode_rgb(block, encoding=self._encoding, background=self._background)

    def clear_cache(self) -> None:
        self._parent._ensure_session(self._parent_session)
        self._parent.clear_cache()


def _open_rgb_source(
    image: Image, *, red: RGBSelector, green: RGBSelector, blue: RGBSelector,
    encoding: RGBEncoding, background: RGBBackground, name: str,
    pixel_size: PixelSize | None,
) -> tuple[RGBImage, tuple[int, int, int], str]:
    indices = _channel_indices(image, (red, green, blue))
    resolved = _resolved_encoding(image.dtype, encoding)
    if background not in {"white", "keep"}:
        raise ValueError("background must be 'white' or 'keep'")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("RGB output name must be a nonempty string")
    if pixel_size is not None and not isinstance(pixel_size, PixelSize):
        raise TypeError("pixel_size must be a PixelSize")
    size = pixel_size or image.pixel_size
    if size is None:
        raise ValueError("RGB extraction requires X/Y pixel calibration; supply pixel_size")
    return RGBImage(_RGBRecoverySource(
        image, indices, encoding=resolved, background=background,
        pixel_size=size, name=name.strip(),
    )), indices, resolved


def probe_rgb(
    input_path: str | Path, *, input_type: InputType = "ome_tiff", series: int = 0,
    red: RGBSelector = "RED", green: RGBSelector = "GREEN", blue: RGBSelector = "BLUE",
    group: str = "", z: int | None = None, t: int | None = None,
    intrinsic_only: bool = False, storage_options: dict | None = None,
    image_type: str | None = None, channel_name_field: str | None = None,
) -> dict:
    """Inspect five small regions only. Never claims a whole-slide encoding audit."""
    if input_type not in PLANAR_INPUT_TYPES:
        raise ValueError("extract-rgb supports only planar source types")
    reader = source_reader(input_path, input_type=input_type, series=integer(series, "series"),
                           group=group, z=z, t=t, intrinsic_only=intrinsic_only,
                           storage_options=storage_options, image_type=image_type,
                           channel_name_field=channel_name_field)
    with reader:
        indices = _channel_indices(reader, (red, green, blue))
        _resolved_encoding(reader.dtype, "auto")
        height, width = reader.levels[0].spatial_shape
        size = 96
        samples = []
        for fy, fx in ((.10, .10), (.50, .50), (.90, .90), (.10, .90), (.90, .10)):
            y0 = min(max(0, int(height * fy) - size // 2), max(0, height - size))
            x0 = min(max(0, int(width * fx) - size // 2), max(0, width - size))
            block = reader.read_region(y0, min(height, y0 + size),
                                       x0, min(width, x0 + size), channels=indices)
            samples.append(block.reshape(3, -1))
        sampled = np.concatenate(samples, axis=1)
        stats = []
        for color, index, values in zip(("RED", "GREEN", "BLUE"), indices, sampled, strict=True):
            entry = {
                "color": color, "index": index, "name": reader.channel_names[index],
                "minimum": int(values.min()), "median": float(np.median(values)),
                "maximum": int(values.max()),
            }
            if reader.dtype == np.dtype("uint16"):
                entry["byte_repeated_fraction"] = float(np.mean((values >> 8) == (values & 255)))
                entry["low_byte_zero_fraction"] = float(np.mean((values & 255) == 0))
            stats.append(entry)
        return {
            "source_dtype": reader.dtype.name, "source_size_yx": [height, width],
            "samples_examined": int(sampled.shape[1]), "sample_windows": 5,
            "all_zero_rgb_fraction": float(np.mean(np.all(sampled == 0, axis=0))),
            "channels": stats,
        }


def extract_rgb(
    input_path: str | Path, output_path: str | Path, *,
    input_type: InputType = "ome_tiff", output_format: OutputFormat = "ome_tiff",
    series: int = 0, group: str = "", z: int | None = None, t: int | None = None,
    intrinsic_only: bool = False, storage_options: dict | None = None,
    image_type: str | None = None, channel_name_field: str | None = None,
    red: RGBSelector = "RED", green: RGBSelector = "GREEN", blue: RGBSelector = "BLUE",
    encoding: RGBEncoding = "auto", background: RGBBackground = "white",
    name: str = "RGB", pixel_size: PixelSize | None = None,
    compression: str | None = None, tile_size: int | None = None,
    pyramid_levels: int | None = None, max_workers: int | None = None,
    overwrite: bool = True, cache_directory: str | Path | None = None,
) -> dict:
    """Write an 8-bit RGB image from three scalar channels with an explicit policy.

    For uint16 sources, ``auto`` chooses high-byte (also reverses x256 and x257
    expansions). For uint8 sources, it chooses native. ``white`` converts only
    exact all-zero source triples to white; ``keep`` preserves them. Compression
    is lossless (TIFF Deflate or Zarr Zstd by default), pyramids are automatic,
    and no full-slide array is materialized. Lower bits can be discarded, so this
    operation must not be confused with fidelity-preserving ``convert``.
    """
    if input_type not in PLANAR_INPUT_TYPES:
        raise ValueError("extract-rgb supports only planar source types")
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean")
    series = integer(series, "series")
    # Explicitly choose lossless compression rather than RGB's usual JPEG default.
    codecs = ({"deflate", "lzw", "zstd", "uncompressed"} if output_format == "ome_tiff"
              else {"zstd", "gzip", "uncompressed"})
    if compression is not None and (
        not isinstance(compression, str) or compression.lower() not in codecs
    ):
        raise ValueError(f"extract-rgb requires lossless {output_format} compression: {sorted(codecs)}")
    resolved_codec = compression or ("Deflate" if output_format == "ome_tiff" else "ZSTD")
    source_file, output_file = validate_image_paths(
        input_path, output_path, overwrite=overwrite, input_type=input_type,
        output_format=output_format,
    )
    reader = source_reader(source_file, input_type=input_type, series=series, group=group,
                           z=z, t=t, intrinsic_only=intrinsic_only,
                           storage_options=storage_options, image_type=image_type,
                           channel_name_field=channel_name_field)
    with reader:
        rgb, indices, resolved_encoding = _open_rgb_source(
            reader, red=red, green=green, blue=blue, encoding=encoding,
            background=background, name=name, pixel_size=pixel_size,
        )
        LOGGER.info("Recover RGB from %s source channels as %s", reader.dtype, resolved_encoding)
        for color, index in zip(("RED", "GREEN", "BLUE"), indices, strict=True):
            LOGGER.info("  %s <- channel %s (%r)", color, index, reader.channel_names[index])
        with rgb:
            result = image_writer(
                output_file, output_format=output_format, compression=resolved_codec,
                tile_size=tile_size, pyramid_levels=pyramid_levels, downsample="mean",
                max_workers=max_workers, overwrite=overwrite, cache_directory=cache_directory,
            ).write(rgb)
        source_channels = [
            {"color": color, "source_index": index, "source_name": reader.channel_names[index]}
            for color, index in zip(("RED", "GREEN", "BLUE"), indices, strict=True)
        ]
        source_dtype = reader.dtype.name
        source_size = list(reader.levels[0].spatial_shape)
    return complete_report({
        "input_path": str(source_file), "output_path": str(output_file),
        "input_type": input_type, "output_format": output_format, "source_series": series,
        "source_size_yx": source_size, "source_dtype": source_dtype,
        "source_group": group, "source_z": z, "source_t": t, "intrinsic_only": intrinsic_only,
        "channels": source_channels, "requested_encoding": encoding,
        "encoding": resolved_encoding, "background": background, "output_name": name.strip(),
        "pixel_size_override": None if pixel_size is None else list(pixel_size.to_tuple()),
        "writer_report": result,
    }, "rgb_extraction_report.schema.json")
