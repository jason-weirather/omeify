from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from omeify.io._writer import (
    JPEGSubsampling,
    PreparedImage,
    WriterEngine,
    WriterSettings,
    prepare_image,
    validate_ome_xml,
)
from omeify.io._writer.configuration import OUTPUT_BYTEORDER
from omeify.io.image_planes import protect_source_paths
from omeify.utils.generate_ome_xml import (
    OMEIFY_PROVENANCE_NAMESPACE,
    generate_multi_series_ome_xml,
)

from .image_series import OMEImageSeries
from .verification import verify_output


class OMEMultiSeriesWriter:
    """Write named heterogeneous OME Images into one atomic OME-TIFF file.

    This public writer deliberately exposes a different input contract from
    :class:`~omeify.OMETiffWriter`: every OME Image may have its own dtype,
    image type, channel vocabulary, compression, and downsampling policy.
    Both writers delegate source validation, pyramid staging, TIFF encoding,
    verification primitives, and atomic installation to the same internal
    writer engine.
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

    @property
    def path(self) -> Path:
        return self._settings.output_path

    def __repr__(self) -> str:
        return f"{type(self).__name__}(path={str(self.path)!r})"

    def write(
        self,
        series: Sequence[OMEImageSeries],
        *,
        provenance: Mapping[str, object] | None = None,
    ) -> dict[str, object]:
        """Write all series and return one structured construction report."""

        images = self._normalize_series(series)
        for image in images:
            protect_source_paths(image._source, self._settings.output_path)
        provenance_json, normalized_provenance = _canonical_provenance(provenance)
        prepared = tuple(
            prepare_image(
                image._source,
                image._source.output_spec(),
                name=image.name,
                downsample=image.downsample,
                compression_name=image.compression or self._settings.compression_name,
                settings=self._settings,
                lossy_policy="non-label",
            )
            for image in images
        )
        xml_info = generate_multi_series_ome_xml(
            tuple(
                (_required_name(item), item.spec, item.level_shapes)
                for item in prepared
            ),
            provenance_json=provenance_json,
            display_uuid=self._settings.display_uuid,
            output_byteorder=OUTPUT_BYTEORDER,
        )
        omexml = str(xml_info["xml_string"])
        metadata = validate_ome_xml(omexml, context="multi-series")
        result = WriterEngine(self._settings).write(
            prepared,
            omexml,
            verify=lambda path: verify_output(
                prepared,
                path,
                provenance_json=provenance_json,
                software=self._settings.software,
            ),
        )

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
                "type_description": "Multi-series pyramidal OME-TIFF",
                "series_count": len(prepared),
                "top_level_ifd_count": sum(
                    item.spec.plane_count for item in prepared
                ),
                "byte_order": result.verification["output_byte_order"],
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
            "verification": result.verification,
            "options": {
                "compression": self._settings.compression_name,
                "jpeg_quality": self._settings.jpeg_quality,
                "jpeg_subsampling": self._settings.jpeg_subsampling,
                "tile_size": self._settings.tile_size,
                "pyramid_levels": self._settings.pyramid_levels,
                "display_uuid": self._settings.display_uuid,
                "software": self._settings.software,
                "float32_mantissa_bits": self._settings.float32_mantissa_bits,
                "max_workers": self._settings.max_workers,
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


def _required_name(prepared: PreparedImage) -> str:
    if prepared.name is None:
        raise RuntimeError("multi-series prepared images require a name")
    return prepared.name


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
    prepared: PreparedImage,
) -> dict[str, object]:
    spec = prepared.spec
    if prepared.name is None:
        raise RuntimeError("multi-series prepared images require a name")
    return {
        "index": index,
        "name": prepared.name,
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
        "float32_mantissa_bits": prepared.float32_mantissa_bits,
        "float_precision_bits": (
            None
            if prepared.float32_mantissa_bits is None
            else prepared.float32_mantissa_bits + 1
        ),
        "compression": prepared.compression.name,
        "downsample_method": prepared.downsample,
        "level_shapes": [list(shape) for shape in prepared.level_shapes],
        "subresolution_count": len(prepared.level_shapes) - 1,
    }
