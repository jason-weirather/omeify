from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any

from omeify.io._writer import (
    DownsampleMethod,
    JPEGSubsampling,
    PlaneReaderSource,
    WriterEngine,
    WriterSettings,
    prepare_image,
    validate_ome_xml,
)
from omeify.io._writer.configuration import (
    DEFAULT_JPEG_QUALITY,
    DEFAULT_JPEG_SUBSAMPLING,
    OUTPUT_BYTEORDER,
)
from omeify.io._writer.single_verification import verify_single_output
from omeify.io.base import Image
from omeify.io.image_planes import ImagePlaneSource, protect_source_paths
from omeify.io.spec import OMEImageSpec
from omeify.utils.generate_ome_xml import generate_ome_xml

LOGGER = logging.getLogger(__name__)


class OMETiffWriter:
    """Write one standardized, tiled, pyramidal, MITI-profiled OME Image.

    The public writer owns the one-image contract and report. Shared source
    validation, compression resolution, pyramid construction, TIFF writing,
    verification primitives, and atomic installation live in Omeify's internal
    writer engine and are also used by :class:`OMEMultiSeriesWriter`.

    ``image_type`` is a semantic input flag, not a private TIFF tag. RGB is
    encoded as one logical OME channel with three interleaved samples. Label
    images are one integer YX raster and use nearest-neighbor pyramids.

    Automatic storage (``compression=None``, ``tile_size=None``) is lossy JPEG
    quality 90 / 4:2:2 with 256-pixel tiles for RGB, and lossless LZW with
    1024-pixel tiles for scalar/multichannel/label images. Explicit settings win.
    Choose a lossless codec explicitly when exact RGB sample values are needed.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        compression: str | None = None,
        jpeg_quality: int = DEFAULT_JPEG_QUALITY,
        jpeg_subsampling: JPEGSubsampling = DEFAULT_JPEG_SUBSAMPLING,
        tile_size: int | None = None,
        pyramid_levels: int | None = None,
        downsample: DownsampleMethod | None = None,
        max_workers: int | None = None,
        display_uuid: bool = True,
        software: str | None = None,
        float32_mantissa_bits: int | None = None,
        overwrite: bool = True,
        cache_directory: str | Path | None = None,
    ) -> None:
        settings = WriterSettings.from_values(
            output_path,
            compression=compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            max_workers=max_workers,
            display_uuid=display_uuid,
            software=software,
            float32_mantissa_bits=float32_mantissa_bits,
            overwrite=overwrite,
            cache_directory=cache_directory,
        )
        self._settings = settings
        self._downsample = downsample

    @property
    def path(self) -> Path:
        return self._settings.output_path

    def write(self, image: Image, *, level: int = 0) -> dict[str, object]:
        """Borrow an open Image and materialize its selected level, tile by tile.

        Metadata belongs to the image. Storage settings belong to this writer.
        Pyramids are rebuilt from the selected level; source arrays are not mutated.
        """
        source = ImagePlaneSource(image, level=level)
        return self._write_source(source, source.output_spec())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"

    def _write_source(
        self,
        source: PlaneReaderSource,
        spec: OMEImageSpec,
    ) -> dict[str, object]:
        protect_source_paths(source, self._settings.output_path)
        prepared = prepare_image(
            source,
            spec,
            name=None,
            downsample=self._downsample,
            compression_name=self._settings.compression_name,
            settings=self._settings,
            lossy_policy="rgb-only",
        )
        spec = prepared.spec

        metadata_started = time.monotonic()
        xml_info = generate_ome_xml(
            spec,
            prepared.level_shapes,
            display_uuid=self._settings.display_uuid,
            output_byteorder=OUTPUT_BYTEORDER,
        )
        omexml = str(xml_info["xml_string"])
        metadata = validate_ome_xml(omexml, context="single-series")
        LOGGER.info(
            "OME-XML schema and omeify MITI header validation passed in "
            "%.2f seconds",
            time.monotonic() - metadata_started,
        )

        result = WriterEngine(self._settings).write(
            (prepared,),
            omexml,
            verify=lambda path: verify_single_output(
                prepared,
                path,
                software=self._settings.software,
            ),
        )
        compression = prepared.compression
        return {
            "ome": {
                "xml_string": omexml,
                "schema_location": metadata.validator.schema_location,
                "xml_is_valid": metadata.xml_is_valid,
                "uuid": xml_info["uuid"],
            },
            "miti_header": metadata.miti_header.as_dict(),
            "output_file": {
                "path": str(self._settings.output_path),
                "size_bytes": result.output_size,
                "type_description": "Pyramidal OME-TIFF",
                "dtype": spec.dtype.name,
                "shape": list(spec.output_shape),
                "shape_cyx": list(spec.shape_cyx),
                "axes": spec.output_axes,
                "byte_order": result.verification["output_byte_order"],
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
                "float32_mantissa_bits": prepared.float32_mantissa_bits,
                "float_precision_bits": (
                    None
                    if prepared.float32_mantissa_bits is None
                    else prepared.float32_mantissa_bits + 1
                ),
                "icc_profile_present": spec.icc_profile is not None,
            },
            "pyramid": {
                "tile_size": prepared.tile_size,
                "axes": spec.output_axes,
                "downsample_method": prepared.downsample,
                "level_shapes": [
                    list(shape) for shape in prepared.level_shapes
                ],
                "subresolution_count": len(prepared.level_shapes) - 1,
            },
            "verification": result.verification,
            "options": {
                "compression": compression.name,
                "jpeg_quality": (
                    self._settings.jpeg_quality if compression.name == "JPEG" else None
                ),
                "jpeg_subsampling": (
                    self._settings.jpeg_subsampling
                    if compression.name == "JPEG" and spec.is_rgb
                    else None
                ),
                "display_uuid": self._settings.display_uuid,
                "software": self._settings.software,
                "float32_mantissa_bits": prepared.float32_mantissa_bits,
                "max_workers": self._settings.max_workers,
            },
        }


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
            raise RuntimeError(
                "TemporaryOMETiffWriter must be entered before path is available"
            )
        return self._path

    def __enter__(self) -> TemporaryOMETiffWriter:
        if self._temporary_directory is not None:
            raise RuntimeError("TemporaryOMETiffWriter context is already active")
        if self.directory is not None:
            self.directory.mkdir(parents=True, exist_ok=True)
        temporary = tempfile.TemporaryDirectory(
            prefix=self.prefix,
            dir=(str(self.directory) if self.directory is not None else None),
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

    def write(self, image: Image, *, level: int = 0) -> dict[str, object]:
        if self._writer is None:
            raise RuntimeError("Enter TemporaryOMETiffWriter before writing")
        return self._writer.write(image, level=level)

    def close(self) -> None:
        self._writer = None
        self._path = None
        if self._temporary_directory is not None:
            self._temporary_directory.cleanup()
        self._temporary_directory = None

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
