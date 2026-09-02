from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from omeify.io.ome_tiff_writer import (
    JPEGSubsampling,
    _JPEG_SUBSAMPLING_FACTORS,
    _OUTPUT_BYTEORDER,
    _normalize_software_tag,
    _validate_tile_size,
)
from omeify.utils.generate_ome_xml import (
    OMEIFY_PROVENANCE_NAMESPACE,
    generate_multi_series_ome_xml,
)
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header
from omeify.utils.ome_schema_validator import OMESchemaValidator

from .image_series import OMEImageSeries
from .model import PreparedSeries
from .verification import verify_output
from .writing import build_pyramid, prepare_series, write_output


class OMEMultiSeriesWriter:
    """Write named heterogeneous OME Images into one atomic OME-TIFF file.

    This writer is deliberately separate from the one-image
    :class:`~omeify.OMETiffWriter`. Each derived OME Image may have its own
    dtype, image type, channel names, and pyramid downsampling policy.
    """

    def __init__(
        self,
        output_path: str | Path,
        *,
        compression: str | None = None,
        jpeg_quality: int = 90,
        jpeg_subsampling: JPEGSubsampling = "444",
        tile_size: int = 1024,
        pyramid_levels: int | None = None,
        max_workers: int | None = None,
        display_uuid: bool = True,
        software: str | None = None,
        overwrite: bool = True,
        cache_directory: str | Path | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.compression_name = compression or "LZW"
        self.jpeg_quality = int(jpeg_quality)
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if jpeg_subsampling not in _JPEG_SUBSAMPLING_FACTORS:
            choices = ", ".join(_JPEG_SUBSAMPLING_FACTORS)
            raise ValueError(f"jpeg_subsampling must be one of: {choices}")
        self.jpeg_subsampling: JPEGSubsampling = jpeg_subsampling
        self.tile_size = _validate_tile_size(tile_size)
        if pyramid_levels is not None:
            if isinstance(pyramid_levels, bool) or not isinstance(pyramid_levels, int):
                raise TypeError("pyramid_levels must be an integer or None")
            if pyramid_levels < 0:
                raise ValueError("pyramid_levels must be zero or greater")
        self.pyramid_levels = pyramid_levels
        if max_workers is None:
            self.max_workers = max(1, min(8, os.cpu_count() or 1))
        else:
            if isinstance(max_workers, bool) or not isinstance(max_workers, int):
                raise TypeError("max_workers must be an integer or None")
            if max_workers < 1:
                raise ValueError("max_workers must be at least one")
            self.max_workers = max_workers
        self.display_uuid = bool(display_uuid)
        self.software = _normalize_software_tag(software)
        self.overwrite = bool(overwrite)
        self.cache_directory = (
            None if cache_directory is None else Path(cache_directory)
        )

    @property
    def path(self) -> Path:
        return self.output_path

    def write(
        self,
        series: Sequence[OMEImageSeries],
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Write all series and return one structured construction report."""

        images = self._normalize_series(series)
        provenance_json, normalized_provenance = _canonical_provenance(provenance)
        if self.output_path.exists() and not self.overwrite:
            raise FileExistsError(f"Output already exists: {self.output_path}")
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_directory is not None:
            self.cache_directory.mkdir(parents=True, exist_ok=True)

        prepared = tuple(
            prepare_series(
                image,
                compression_name=image.compression or self.compression_name,
                jpeg_quality=self.jpeg_quality,
                jpeg_subsampling=self.jpeg_subsampling,
                tile_size=self.tile_size,
                pyramid_levels=self.pyramid_levels,
            )
            for image in images
        )
        xml_info = generate_multi_series_ome_xml(
            tuple(
                (item.image.name, item.image.spec, item.level_shapes)
                for item in prepared
            ),
            provenance_json=provenance_json,
            display_uuid=self.display_uuid,
            output_byteorder=_OUTPUT_BYTEORDER,
        )
        omexml = str(xml_info["xml_string"])
        validator, xml_is_valid, miti_header = self._validate_xml(omexml)
        verification = self._write_atomic(
            prepared,
            omexml,
            provenance_json=provenance_json,
            software=self.software,
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
                "size_bytes": self.output_path.stat().st_size,
                "type_description": "Multi-series pyramidal OME-TIFF",
                "series_count": len(prepared),
                "top_level_ifd_count": sum(
                    item.image.spec.plane_count for item in prepared
                ),
                "byte_order": verification["output_byte_order"],
                "lossless_compression": all(
                    item.compression.lossless for item in prepared
                ),
            },
            "series": [
                _series_report(index, item)
                for index, item in enumerate(prepared)
            ],
            "provenance": (
                None
                if normalized_provenance is None
                else {
                    "namespace": OMEIFY_PROVENANCE_NAMESPACE,
                    "value": normalized_provenance,
                }
            ),
            "verification": verification,
            "options": {
                "compression": self.compression_name,
                "jpeg_quality": self.jpeg_quality,
                "jpeg_subsampling": self.jpeg_subsampling,
                "tile_size": self.tile_size,
                "pyramid_levels": self.pyramid_levels,
                "display_uuid": self.display_uuid,
                "software": self.software,
                "max_workers": self.max_workers,
            },
        }

    @staticmethod
    def _normalize_series(
        series: Sequence[OMEImageSeries],
    ) -> tuple[OMEImageSeries, ...]:
        if isinstance(series, (str, bytes)) or not isinstance(series, Sequence):
            raise TypeError("series must be a sequence of OMEImageSeries values")
        values = tuple(series)
        if not values:
            raise ValueError("multi-series output requires at least one OMEImageSeries")
        if not all(isinstance(item, OMEImageSeries) for item in values):
            raise TypeError("series must contain only OMEImageSeries values")
        names = tuple(item.name for item in values)
        if len(names) != len(set(names)):
            raise ValueError("OME series names must be unique")
        return values

    @staticmethod
    def _validate_xml(omexml: str):
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
            raise ValueError(
                f"Generated multi-series OME header failed omeify MITI validation: {details}"
            )
        return validator, xml_is_valid, miti_header

    def _write_atomic(
        self,
        prepared: Sequence[PreparedSeries],
        omexml: str,
        *,
        provenance_json: str | None,
        software: str,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(
            prefix="omeify-multiseries-",
            dir=str(self.cache_directory) if self.cache_directory else None,
        ) as temporary_directory:
            temp_root = Path(temporary_directory)
            level_paths = tuple(
                build_pyramid(
                    item,
                    temp_root / f"series-{index}",
                    tile_size=self.tile_size,
                )
                for index, item in enumerate(prepared)
            )
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".omeify-",
                suffix=".partial",
                dir=str(self.output_path.parent),
            )
            os.close(descriptor)
            temporary_output = Path(temporary_name)
            temporary_output.unlink()
            try:
                write_output(
                    prepared,
                    level_paths,
                    temporary_output,
                    omexml,
                    tile_size=self.tile_size,
                    max_workers=self.max_workers,
                    software=software,
                )
                verification = verify_output(
                    prepared,
                    temporary_output,
                    provenance_json=provenance_json,
                    software=software,
                )
                os.replace(temporary_output, self.output_path)
            finally:
                temporary_output.unlink(missing_ok=True)
        return verification


def _canonical_provenance(
    provenance: Mapping[str, object] | None,
) -> tuple[str | None, dict[str, object] | None]:
    if provenance is None:
        return None, None
    if not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a mapping or None")
    value = dict(provenance)
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "provenance must contain only finite JSON-serializable values"
        ) from exc
    normalized = json.loads(rendered)
    if not isinstance(normalized, dict):
        raise RuntimeError("canonical provenance did not resolve to a JSON object")
    return rendered, normalized


def _series_report(
    index: int,
    prepared: PreparedSeries,
) -> dict[str, object]:
    spec = prepared.image.spec
    return {
        "index": index,
        "name": prepared.image.name,
        "image_type": spec.image_type,
        "dtype": spec.dtype.name,
        "shape": list(spec.output_shape),
        "axes": spec.output_axes,
        "channel_names": list(spec.channel_names),
        "size_c": spec.size_c,
        "logical_channel_count": spec.logical_channel_count,
        "samples_per_pixel": spec.samples_per_pixel,
        "pixel_size": list(spec.pixel_size.to_tuple()),
        "significant_bits": spec.significant_bits,
        "compression": prepared.compression.name,
        "downsample_method": prepared.image.downsample,
        "level_shapes": [list(shape) for shape in prepared.level_shapes],
        "subresolution_count": len(prepared.level_shapes) - 1,
    }
