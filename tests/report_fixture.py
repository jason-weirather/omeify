"""Synthetic successful report data for export-spy tests, not a TIFF/XSD oracle.

These records let orchestration tests replace export while retaining the real
report schema boundary. Real writer/XSD/codec checks live in integration tests.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from omeify import OMEImageSeries
from omeify.reports import complete_report
from omeify.utils.generate_ome_xml import OMEIFY_PROVENANCE_NAMESPACE


def multi_report_fixture(
    path: Path, entries: Sequence[OMEImageSeries], provenance: dict[str, Any],
) -> dict[str, Any]:
    series = []
    for index, entry in enumerate(entries):
        image = entry.image
        series.append({
            "index": index, "name": entry.name, "image_type": image.image_type,
            "dtype": image.dtype.name, "shape": list(image.shape), "axes": image.axes,
            "channel_names": list(image.channel_names), "size_c": image.sample_count,
            "logical_channel_count": image.channel_count,
            "samples_per_pixel": 3 if image.image_type == "rgb" else 1,
            "pixel_size": list(image.pixel_size.to_tuple()),
            "significant_bits": image.dtype.itemsize * 8,
            "float32_mantissa_bits": None, "float_precision_bits": None,
            "compression": "Uncompressed", "tile_size": 16,
            "downsample_method": "nearest" if image.image_type == "label" else "mean",
            "level_shapes": [list(image.shape)], "subresolution_count": 0,
        })
    planes = sum(item["logical_channel_count"] for item in series)
    verification = dict.fromkeys((
        "ome_tiff_recognized", "bigtiff", "software_tag_matches", "image_count_matches",
        "series_names_match", "metadata_matches_specs", "tiff_data_mapping_matches",
        "pyramid_annotations_linked", "provenance_annotation_linked", "series_layouts_match",
        "top_level_ifd_count_matches", "storage_matches_requested", "output_pixels_decodable",
        "base_pixel_values_checked", "base_pixel_values_match", "ome_physical_sizes_match_specs",
        "tiff_calibration_matches_specs", "pyramid_calibration_matches_specs",
    ), True)
    verification.update({
        "output_byte_order": "little", "series_checked": len(series), "planes_checked": planes,
        "points_per_plane": 1, "calibration_ifds_checked": planes,
        "calibration_relative_tolerance": 1e-5,
        "pixel_verification": {
            "schema_version": "1.0", "mode": "sampled",
            "sampling": "top_left_center_bottom_right_per_plane_per_level",
            "all_pixels_checked": False, "plane_levels_checked": planes,
            "base_points_decoded": planes, "lossless_base_points_compared": planes,
            "pyramid_points_decoded": 0, "lossless_pyramid_points_compared": 0,
            "base_comparison": "bitwise_after_byte_order_normalization",
            "pyramid_comparison": "nearest_bitwise_or_mean_numeric_equal_nan",
            "pyramid_reference": "preceding_decoded_output_level_for_lossless_series",
            "source_reader_independent": False,
        },
    })
    return complete_report({
        "ome": {"xml_string": "<fixture/>", "schema_location": "synthetic report fixture",
                "xml_is_valid": True, "uuid": None},
        "miti_header": {"profile": "omeify.miti_ome_tiff_header", "source": "synthetic fixture",
                        "schema_resource": "omeify.schemas/miti_ome_tiff_header.schema.json",
                        "status": "pass", "is_valid": True, "errors": [], "warnings": [],
                        "missing_fields": [], "has_extra_metadata": False, "extra_metadata": []},
        "output_file": {"path": str(path), "size_bytes": 1,
                        "type_description": "Multi-series pyramidal OME-TIFF",
                        "series_count": len(series), "top_level_ifd_count": planes,
                        "byte_order": "little", "lossless_compression": True},
        "series": series, "provenance": {"namespace": OMEIFY_PROVENANCE_NAMESPACE,
                                         "value": provenance},
        "verification": verification,
        "options": {"metadata_minimization": True, "compression": "Uncompressed",
                    "jpeg_quality": 90, "jpeg_subsampling": "422", "tile_size": 16,
                    "pyramid_levels": 0, "display_uuid": False, "software": "export-spy",
                    "float32_mantissa_bits": None, "max_workers": 1},
    }, "multi_series_report.schema.json")
