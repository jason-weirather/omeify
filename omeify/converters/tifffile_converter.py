from __future__ import annotations

import hashlib
import logging
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import tifffile
from lxml import etree

from omeify import __version__, get_version_info
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header
from omeify.utils.ome_schema_validator import OMESchemaValidator
from omeify.utils.tiff_image_features import (
    ImageMetadata,
    InputProfile,
    TiffMIFSource,
    TiffPlaneReader,
)

LOGGER = logging.getLogger(__name__)

DownsampleMethod = Literal["mean", "nearest"]
_OUTPUT_BYTEORDER: Literal["<", ">"] = "<"


@dataclass(frozen=True)
class CompressionSettings:
    name: str
    tifffile_value: str | None
    compression_args: dict[str, object] | None
    predictor: bool | None
    lossless: bool


def _compression_settings(name: str, dtype: np.dtype) -> CompressionSettings:
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
            raise ValueError("JPEG output is restricted to uint8; use LZW for uint16 mIF data")
        return CompressionSettings("JPEG", "jpeg", None, None, False)
    raise ValueError(
        f"Unsupported compression {name!r}; choose LZW, Deflate, ZSTD, JPEG, or Uncompressed"
    )


def _validate_tile_size(tile_size: int) -> int:
    value = int(tile_size)
    if value < 16 or value % 16 != 0:
        raise ValueError("TIFF tile size must be at least 16 and divisible by 16")
    return value


def _auto_level_shapes(
    base_shape: tuple[int, int, int],
    *,
    tile_size: int,
    pyramid_levels: int | None,
) -> list[tuple[int, int, int]]:
    channels, height, width = base_shape
    shapes = [base_shape]
    if pyramid_levels is not None:
        if pyramid_levels < 0:
            raise ValueError("pyramid_levels must be zero or greater")
        target_levels = pyramid_levels
    else:
        target_levels = 0
        current_h, current_w = height, width
        # Continue until the entire final image fits in one output tile.
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
        shapes.append((channels, current_h, current_w))
    return shapes


def _mean_downsample_2x(block: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_shape
    dtype = block.dtype

    if np.issubdtype(dtype, np.unsignedinteger) or np.issubdtype(dtype, np.bool_):
        accumulator = np.zeros((out_h, out_w), dtype=np.uint64)
        counts = np.zeros((out_h, out_w), dtype=np.uint8)
        for dy in (0, 1):
            for dx in (0, 1):
                sample = block[dy::2, dx::2]
                h, w = sample.shape
                accumulator[:h, :w] += sample.astype(np.uint64, copy=False)
                counts[:h, :w] += 1
        quotient, remainder = np.divmod(accumulator, counts)
        twice_remainder = remainder * 2
        # Match numpy.rint: nearest integer with ties rounded to even.
        increment = (twice_remainder > counts) | (
            (twice_remainder == counts) & ((quotient & 1) == 1)
        )
        return (quotient + increment).astype(dtype, copy=False)

    if np.issubdtype(dtype, np.signedinteger):
        accumulator = np.zeros((out_h, out_w), dtype=np.int64)
        counts = np.zeros((out_h, out_w), dtype=np.uint8)
        for dy in (0, 1):
            for dx in (0, 1):
                sample = block[dy::2, dx::2]
                h, w = sample.shape
                accumulator[:h, :w] += sample.astype(np.int64, copy=False)
                counts[:h, :w] += 1
        sign = np.sign(accumulator)
        magnitude = np.abs(accumulator)
        quotient, remainder = np.divmod(magnitude, counts)
        twice_remainder = remainder * 2
        increment = (twice_remainder > counts) | (
            (twice_remainder == counts) & ((quotient & 1) == 1)
        )
        result = (quotient + increment) * sign
        return result.astype(dtype, copy=False)

    accumulator_dtype = (
        np.complex128 if np.issubdtype(dtype, np.complexfloating) else np.float64
    )
    accumulator = np.zeros((out_h, out_w), dtype=accumulator_dtype)
    counts = np.zeros((out_h, out_w), dtype=np.uint8)
    for dy in (0, 1):
        for dx in (0, 1):
            sample = block[dy::2, dx::2]
            h, w = sample.shape
            accumulator[:h, :w] += sample.astype(accumulator_dtype, copy=False)
            counts[:h, :w] += 1
    return (accumulator / counts).astype(dtype, copy=False)


def _downsample_region(
    reader: TiffPlaneReader,
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
        return np.ascontiguousarray(block[::2, ::2][: out_shape[0], : out_shape[1]])
    if method == "mean":
        return np.ascontiguousarray(_mean_downsample_2x(block, out_shape))
    raise AssertionError(f"Unhandled downsample method {method}")


def _iter_tiles(
    readers: Sequence[TiffPlaneReader],
    shape_cyx: tuple[int, int, int],
    *,
    tile_size: int,
) -> Iterable[np.ndarray]:
    channels, height, width = shape_cyx
    if len(readers) != channels:
        raise ValueError(f"Expected {channels} page readers, got {len(readers)}")
    for reader in readers:
        for y0 in range(0, height, tile_size):
            y1 = min(height, y0 + tile_size)
            for x0 in range(0, width, tile_size):
                x1 = min(width, x0 + tile_size)
                yield np.ascontiguousarray(reader.read_region(y0, y1, x0, x1))
        reader.clear_cache()


def _iter_downsampled_tiles(
    readers: Sequence[TiffPlaneReader],
    output_shape_cyx: tuple[int, int, int],
    *,
    tile_size: int,
    method: DownsampleMethod,
) -> Iterable[np.ndarray]:
    channels, height, width = output_shape_cyx
    if len(readers) != channels:
        raise ValueError(f"Expected {channels} page readers, got {len(readers)}")
    for reader in readers:
        for y0 in range(0, height, tile_size):
            y1 = min(height, y0 + tile_size)
            for x0 in range(0, width, tile_size):
                x1 = min(width, x0 + tile_size)
                yield _downsample_region(
                    reader,
                    out_y0=y0,
                    out_y1=y1,
                    out_x0=x0,
                    out_x1=x1,
                    method=method,
                )
        reader.clear_cache()


def _page_readers(tiff: tifffile.TiffFile, channels: int) -> list[TiffPlaneReader]:
    # Uniform TIFF series are commonly represented as one TiffPage followed by
    # lightweight TiffFrame objects.  ``aspage`` materializes only the IFD
    # metadata and gives the random-access reader a consistent page interface.
    pages = [page.aspage() for page in list(tiff.pages)[:channels]]
    if len(pages) != channels:
        raise ValueError(f"Temporary pyramid level has {len(pages)} pages; expected {channels}")
    lock = __import__("threading").RLock()
    return [TiffPlaneReader(page, lock=lock) for page in pages]


def _resolution(pixel_size_x_um: float, pixel_size_y_um: float, scale: int) -> tuple[float, float]:
    return (
        1e4 / (pixel_size_x_um * scale),
        1e4 / (pixel_size_y_um * scale),
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


class TifffileConverter:
    """Pure-Python, tiled, low-memory mIF to pyramidal OME-TIFF converter."""

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
        *,
        profile: InputProfile,
        input_type: str,
        image_name: str = "WholeSlideMIF",
        series: int = 0,
        rename_channels: dict[str, str] | None = None,
        physical_size_x_um: float | None = None,
        physical_size_y_um: float | None = None,
        cache_directory: str | Path | None = None,
        compression: str = "LZW",
        tile_size: int = 1024,
        pyramid_levels: int | None = None,
        downsample: DownsampleMethod = "mean",
        max_workers: int | None = None,
        display_uuid: bool = True,
        overwrite: bool = True,
        calculate_checksums: bool = True,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.profile = profile
        self.input_type = input_type
        self.image_name = image_name
        self.series = int(series)
        self.rename_channels = dict(rename_channels or {})
        self.physical_size_x_um = physical_size_x_um
        self.physical_size_y_um = physical_size_y_um
        self.cache_directory = Path(cache_directory) if cache_directory is not None else None
        self.compression_name = compression
        self.tile_size = _validate_tile_size(tile_size)
        self.pyramid_levels = pyramid_levels
        self.downsample = downsample
        if max_workers is None:
            self.max_workers = max(1, min(8, os.cpu_count() or 1))
        else:
            self.max_workers = int(max_workers)
            if self.max_workers < 1:
                raise ValueError("max_workers must be at least one")
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
            physical_size_x_um=self.physical_size_x_um,
            physical_size_y_um=self.physical_size_y_um,
        ) as source:
            features = source.features
            compression = _compression_settings(self.compression_name, features.dtype)
            level_shapes = _auto_level_shapes(
                features.shape_cyx,
                tile_size=self.tile_size,
                pyramid_levels=self.pyramid_levels,
            )
            xml_info = generate_ome_xml(
                features,
                level_shapes,
                display_uuid=self.display_uuid,
                rename_channels=self.rename_channels,
                output_byteorder=_OUTPUT_BYTEORDER,
            )
            omexml = str(xml_info["xml_string"])
            validator = OMESchemaValidator()
            xml_is_valid = validator.validate(omexml)
            if xml_is_valid is False:
                raise ValueError("Generated OME-XML failed OME 2016-06 schema validation")
            miti_header = validate_miti_ome_tiff_header(omexml)
            if not miti_header.is_valid:
                details = "; ".join(miti_header.errors)
                raise ValueError(f"Generated OME header failed omeify MITI validation: {details}")

            LOGGER.info(
                "Converting %s channels of %s data at %sx%s into %s pyramid levels",
                features.size_c,
                features.dtype,
                features.size_x,
                features.size_y,
                len(level_shapes),
            )

            with tempfile.TemporaryDirectory(
                prefix="omeify-pyramid-",
                dir=str(self.cache_directory) if self.cache_directory else None,
            ) as temporary_directory:
                temp_root = Path(temporary_directory)
                level_paths = self._build_pyramid(source, features, level_shapes, temp_root)
                file_descriptor, temporary_name = tempfile.mkstemp(
                    prefix=".omeify-",
                    suffix=".partial",
                    dir=str(self.output_path.parent),
                )
                os.close(file_descriptor)
                temporary_output = Path(temporary_name)
                # TiffWriter expects to create/truncate the file itself.  The
                # reservation above guarantees a same-filesystem unique path.
                temporary_output.unlink()
                try:
                    self._write_output(
                        source,
                        features,
                        level_shapes,
                        level_paths,
                        temporary_output,
                        omexml,
                        compression,
                    )
                    verification = self._verify_output(
                        source,
                        features,
                        level_shapes,
                        temporary_output,
                        compression,
                    )
                    os.replace(temporary_output, self.output_path)
                finally:
                    temporary_output.unlink(missing_ok=True)

        input_size = self.input_path.stat().st_size
        output_size = self.output_path.stat().st_size
        stop_epoch = time.time()
        report: dict[str, object] = {
            "ome": {
                "xml_string": omexml,
                "schema_location": validator.schema_location,
                "xml_is_valid": xml_is_valid,
                "uuid": xml_info["uuid"],
            },
            "miti_header": miti_header.as_dict(),
            "input_file": {
                "path": str(self.input_path),
                "size_bytes": input_size,
                "type_description": self.input_type,
                "dtype": features.dtype.name,
                "shape_cyx": list(features.shape_cyx),
                "source_axes": features.source_axes,
                "byte_order": features.source_byte_order,
            },
            "output_file": {
                "path": str(self.output_path),
                "size_bytes": output_size,
                "type_description": "Pyramidal OME-TIFF",
                "dtype": features.dtype.name,
                "shape_cyx": list(features.shape_cyx),
                "byte_order": verification["output_byte_order"],
                "lossless_compression": compression.lossless,
            },
            "image": {
                "channel_names": [
                    self.rename_channels.get(name, name) for name in features.channel_names
                ],
                "physical_size_x_um": features.physical_size_x_um,
                "physical_size_y_um": features.physical_size_y_um,
                "significant_bits": features.significant_bits,
            },
            "pyramid": {
                "tile_size": self.tile_size,
                "downsample_method": self.downsample,
                "level_shapes_cyx": [list(shape) for shape in level_shapes],
                "subresolution_count": len(level_shapes) - 1,
            },
            "verification": verification,
            "options": {
                "compression": compression.name,
                "deidentify_ome": True,
                "display_uuid": self.display_uuid,
                "series": self.series,
                "rename_channels": self.rename_channels,
                "max_workers": self.max_workers,
            },
            "conversion_stats": {
                "start_time": datetime.fromtimestamp(start_epoch).strftime("%Y-%m-%d %H:%M:%S"),
                "stop_time": datetime.fromtimestamp(stop_epoch).strftime("%Y-%m-%d %H:%M:%S"),
                "run_time": _readable_runtime(stop_epoch - start_epoch),
                "output_to_input_size_ratio": output_size / input_size if input_size else None,
                "compression_ratio": output_size / input_size if input_size else None,
            },
            "versions": get_version_info(),
        }
        if self.calculate_checksums:
            report["input_file"].update(_hash_file(self.input_path))  # type: ignore[union-attr]
            report["output_file"].update(_hash_file(self.output_path))  # type: ignore[union-attr]
        else:
            report["input_file"]["md5_checksum"] = None  # type: ignore[index]
            report["input_file"]["sha256_checksum"] = None  # type: ignore[index]
            report["output_file"]["md5_checksum"] = None  # type: ignore[index]
            report["output_file"]["sha256_checksum"] = None  # type: ignore[index]
        return report

    def _build_pyramid(
        self,
        source: TiffMIFSource,
        features: ImageMetadata,
        level_shapes: Sequence[tuple[int, int, int]],
        temp_root: Path,
    ) -> list[Path]:
        level_paths: list[Path] = []
        for level_index, output_shape in enumerate(level_shapes[1:], start=1):
            output_path = temp_root / f"level-{level_index}.tf8"
            LOGGER.info("Building pyramid level %s with shape %s", level_index, output_shape)
            if level_index == 1:
                readers = source.plane_readers()
                self._write_temporary_level(readers, output_shape, features.dtype, output_path)
            else:
                with tifffile.TiffFile(level_paths[-1]) as previous:
                    readers = _page_readers(previous, features.size_c)
                    self._write_temporary_level(readers, output_shape, features.dtype, output_path)
            level_paths.append(output_path)
        return level_paths

    def _write_temporary_level(
        self,
        readers: Sequence[TiffPlaneReader],
        output_shape: tuple[int, int, int],
        dtype: np.dtype,
        output_path: Path,
    ) -> None:
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
                    tile_size=self.tile_size,
                    method=self.downsample,
                ),
                shape=output_shape,
                dtype=dtype,
                photometric="minisblack",
                tile=(self.tile_size, self.tile_size),
                compression=None,
                metadata=None,
                software=False,
                maxworkers=1,
            )

    def _write_output(
        self,
        source: TiffMIFSource,
        features: ImageMetadata,
        level_shapes: Sequence[tuple[int, int, int]],
        level_paths: Sequence[Path],
        output_path: Path,
        omexml: str,
        compression: CompressionSettings,
    ) -> None:
        common_options = {
            "dtype": features.dtype,
            "photometric": "minisblack",
            "tile": (self.tile_size, self.tile_size),
            "compression": compression.tifffile_value,
            "compressionargs": compression.compression_args,
            "predictor": compression.predictor,
            "metadata": None,
            "maxworkers": self.max_workers,
        }
        software = f"omeify {__version__}; tifffile {tifffile.__version__}"

        with tifffile.TiffWriter(
            output_path,
            bigtiff=True,
            byteorder=_OUTPUT_BYTEORDER,
            ome=False,
        ) as writer:
            base_readers = source.plane_readers()
            writer.write(
                _iter_tiles(base_readers, level_shapes[0], tile_size=self.tile_size),
                shape=level_shapes[0],
                description=omexml.encode("utf-8"),
                software=software,
                subifds=len(level_paths),
                resolution=_resolution(
                    features.physical_size_x_um,
                    features.physical_size_y_um,
                    1,
                ),
                resolutionunit="CENTIMETER",
                **common_options,
            )

            for level_index, (level_path, level_shape) in enumerate(
                zip(level_paths, level_shapes[1:]),
                start=1,
            ):
                with tifffile.TiffFile(level_path) as level_tiff:
                    readers = _page_readers(level_tiff, features.size_c)
                    writer.write(
                        _iter_tiles(readers, level_shape, tile_size=self.tile_size),
                        shape=level_shape,
                        software=False,
                        subfiletype=1,
                        resolution=_resolution(
                            features.physical_size_x_um,
                            features.physical_size_y_um,
                            2**level_index,
                        ),
                        resolutionunit="CENTIMETER",
                        **common_options,
                    )

    def _verify_output(
        self,
        source: TiffMIFSource,
        features: ImageMetadata,
        level_shapes: Sequence[tuple[int, int, int]],
        output_path: Path,
        compression: CompressionSettings,
    ) -> dict[str, object]:
        verification: dict[str, object] = {
            "ome_tiff_recognized": False,
            "bigtiff": False,
            "output_byte_order": None,
            "byte_order_metadata_matches_tiff": False,
            "significant_bits_matches_dtype": False,
            "tiff_data_mapping_matches": False,
            "pyramid_annotation_linked": False,
            "dtype_matches_source": False,
            "pyramid_level_shapes_match": False,
            "top_level_ifd_count_matches": False,
            "subifd_layout_matches": False,
            "all_levels_tiled": False,
            "base_pixel_values_checked": False,
            "base_pixel_values_match": None,
            "channels_checked": 0,
            "points_per_channel": 0,
        }
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
            verification["output_byte_order"] = (
                "big" if actual_byteorder == ">" else "little"
            )

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
            if declared_significant_bits != features.significant_bits:
                raise ValueError(
                    f"OME SignificantBits={declared_significant_bits} does not match "
                    f"dtype width {features.significant_bits}"
                )
            verification["significant_bits_matches_dtype"] = True

            tiff_data = pixels.findall("./ome:TiffData", namespaces=namespace)
            expected_plane_count = features.size_c
            if len(tiff_data) != 1:
                raise ValueError(
                    f"OME Pixels contains {len(tiff_data)} TiffData elements; expected one"
                )
            if int(tiff_data[0].get("IFD", "0")) != 0:
                raise ValueError("OME TiffData must start at IFD=0")
            if int(tiff_data[0].get("PlaneCount", "0")) != expected_plane_count:
                raise ValueError(
                    f"OME TiffData PlaneCount does not match channel count "
                    f"{expected_plane_count}"
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
            if np.dtype(series.dtype).newbyteorder("=") != features.dtype:
                raise TypeError(
                    f"Output dtype {series.dtype} does not match source dtype {features.dtype}"
                )
            verification["dtype_matches_source"] = True
            if len(series.levels) != len(level_shapes):
                raise ValueError(
                    f"Output has {len(series.levels)} pyramid levels; expected {len(level_shapes)}"
                )
            for index, (level, expected_shape) in enumerate(zip(series.levels, level_shapes)):
                actual = tuple(int(value) for value in level.shape)
                if actual != expected_shape:
                    raise ValueError(
                        f"Output level {index} shape {actual} does not match expected "
                        f"{expected_shape}"
                    )
            verification["pyramid_level_shapes_match"] = True

            if len(output.pages) != features.size_c:
                raise ValueError(
                    f"Output has {len(output.pages)} top-level IFDs; expected "
                    f"one per channel ({features.size_c})"
                )
            verification["top_level_ifd_count_matches"] = True

            all_levels_tiled = True
            for channel, frame in enumerate(output.pages):
                page = frame.aspage()
                all_levels_tiled = all_levels_tiled and bool(page.is_tiled)
                subpages = list(page.pages) if page.pages is not None else []
                if len(subpages) != expected_subifds:
                    raise ValueError(
                        f"Output channel {channel} has {len(subpages)} SubIFDs; "
                        f"expected {expected_subifds}"
                    )
                for level_index, subframe in enumerate(subpages, start=1):
                    subpage = subframe.aspage()
                    all_levels_tiled = all_levels_tiled and bool(subpage.is_tiled)
                    if not (int(subpage.subfiletype) & 1):
                        raise ValueError(
                            f"Output channel {channel}, pyramid level {level_index} "
                            "is not marked as a reduced-resolution image"
                        )
            verification["subifd_layout_matches"] = True
            if not all_levels_tiled:
                raise ValueError("One or more output base or pyramid planes are not tiled")
            verification["all_levels_tiled"] = True

            if compression.lossless:
                input_readers = source.plane_readers(cache_mib=16)
                output_readers = _page_readers(output, features.size_c)
                coordinates = sorted(
                    {
                        (0, 0),
                        (features.size_y // 2, features.size_x // 2),
                        (features.size_y - 1, features.size_x - 1),
                    }
                )
                verification["base_pixel_values_checked"] = True
                verification["channels_checked"] = features.size_c
                verification["points_per_channel"] = len(coordinates)
                for channel in range(features.size_c):
                    for y, x in coordinates:
                        source_value = input_readers[channel].read_region(
                            y, y + 1, x, x + 1
                        )[0, 0]
                        output_value = output_readers[channel].read_region(
                            y, y + 1, x, x + 1
                        )[0, 0]
                        if source_value != output_value:
                            raise ValueError(
                                "Lossless base-image verification failed at "
                                f"channel {channel}, y={y}, x={x}: "
                                f"source={source_value}, output={output_value}"
                            )
                verification["base_pixel_values_match"] = True
        return verification
