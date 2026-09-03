from __future__ import annotations

import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from omeify.io._writer import (
    ArraySource,
    DownsampleMethod,
    JPEGSubsampling,
    PlaneReaderSource,
    PredictorMode,
    WriterEngine,
    WriterSettings,
    prepare_image,
    resolve_downsample,
    validate_ome_xml,
)
from omeify.io._writer.configuration import OUTPUT_BYTEORDER
from omeify.io._writer.single_verification import verify_single_output
from omeify.io.pixel_size import PixelSize
from omeify.io.spec import ImageType, OMEImageSpec
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
        predictor: PredictorMode = "auto",
        float32_mantissa_bits: int | None = None,
        overwrite: bool = True,
        cache_directory: str | Path | None = None,
        icc_profile: bytes | None = None,
    ) -> None:
        if image_type not in {"multichannel", "rgb", "label"}:
            raise ValueError("image_type must be 'multichannel', 'rgb', or 'label'")
        if not isinstance(pixel_size, PixelSize):
            raise TypeError("pixel_size must be a PixelSize instance")

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
            predictor=predictor,
            float32_mantissa_bits=float32_mantissa_bits,
            overwrite=overwrite,
            cache_directory=cache_directory,
        )
        self._settings = settings
        self.output_path = settings.output_path
        self.image_type: ImageType = image_type
        self.channel_names = (
            None
            if channel_names is None
            else tuple(str(item) for item in channel_names)
        )
        self.pixel_size = pixel_size
        self.compression_name = settings.compression_name
        self.jpeg_quality = settings.jpeg_quality
        self.jpeg_subsampling = settings.jpeg_subsampling
        self.tile_size = settings.tile_size
        self.pyramid_levels = settings.pyramid_levels
        self.downsample = resolve_downsample(image_type, downsample)
        self.max_workers = settings.max_workers
        self.display_uuid = settings.display_uuid
        self.software = settings.software
        self.predictor = settings.predictor
        self.float32_mantissa_bits = settings.float32_mantissa_bits
        self.overwrite = settings.overwrite
        self.cache_directory = settings.cache_directory
        self.icc_profile = None if icc_profile is None else bytes(icc_profile)

    @property
    def path(self) -> Path:
        return self.output_path

    def write(
        self,
        image: np.ndarray,
        *,
        axes: str | None = None,
    ) -> dict[str, object]:
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
        return self._write_source(ArraySource(array, spec), spec)

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
            icc_profile=(
                self.icc_profile if icc_profile is None else icc_profile
            ),
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
        prepared = prepare_image(
            source,
            spec,
            name=None,
            downsample=self.downsample,
            compression_name=self.compression_name,
            settings=self._settings,
            lossy_policy="rgb-only",
        )
        spec = prepared.spec

        metadata_started = time.monotonic()
        xml_info = generate_ome_xml(
            spec,
            prepared.level_shapes,
            display_uuid=self.display_uuid,
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
                software=self.software,
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
                "path": str(self.output_path),
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
                "tile_size": self.tile_size,
                "axes": spec.output_axes,
                "downsample_method": self.downsample,
                "level_shapes": [
                    list(shape) for shape in prepared.level_shapes
                ],
                "subresolution_count": len(prepared.level_shapes) - 1,
            },
            "verification": result.verification,
            "options": {
                "compression": compression.name,
                "jpeg_quality": (
                    self.jpeg_quality if compression.name == "JPEG" else None
                ),
                "jpeg_subsampling": (
                    self.jpeg_subsampling
                    if compression.name == "JPEG" and spec.is_rgb
                    else None
                ),
                "display_uuid": self.display_uuid,
                "software": self.software,
                "predictor": compression.predictor_name,
                "float32_mantissa_bits": prepared.float32_mantissa_bits,
                "max_workers": self.max_workers,
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

    @property
    def writer(self) -> OMETiffWriter:
        if self._writer is None:
            raise RuntimeError(
                "TemporaryOMETiffWriter must be entered before writing"
            )
        return self._writer

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

    def write(
        self,
        image: np.ndarray,
        *,
        axes: str | None = None,
    ) -> dict[str, object]:
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
