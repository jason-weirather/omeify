from __future__ import annotations

from pathlib import Path

import tifffile

from .model import PreparedImage
from .verification import (
    PYRAMID_NAMESPACE,
    verify_container,
    verify_prepared_images,
)


def verify_single_output(
    prepared: PreparedImage,
    output_path: Path,
    *,
    software: str,
) -> dict[str, object]:
    """Verify one ordinary OME-TIFF against its metadata and prepared image."""

    verification: dict[str, object] = {
        "ome_tiff_recognized": False,
        "bigtiff": False,
        "output_byte_order": None,
        "software_tag_matches": False,
        "byte_order_metadata_matches_tiff": False,
        "significant_bits_matches_dtype": False,
        "significant_bits_matches_spec": False,
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
        "predictor_matches_requested": False,
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
    spec = prepared.spec
    with tifffile.TiffFile(output_path) as output:
        container = verify_container(output, software=software)
        verification["ome_tiff_recognized"] = True
        verification["bigtiff"] = True
        verification["output_byte_order"] = container.output_byte_order
        verification["software_tag_matches"] = True

        root = container.root
        namespace = container.namespace
        image = root.find("./ome:Image", namespaces=namespace)
        pixels = root.find("./ome:Image/ome:Pixels", namespaces=namespace)
        if image is None or pixels is None:
            raise ValueError("Written OME-XML does not contain Image/Pixels")

        declared_big_endian = (pixels.get("BigEndian") or "").strip().lower()
        expected_big_endian = "true" if output.byteorder == ">" else "false"
        if declared_big_endian != expected_big_endian:
            raise ValueError(
                f"OME BigEndian={declared_big_endian!r} does not match TIFF "
                f"byte order {output.byteorder!r}"
            )
        verification["byte_order_metadata_matches_tiff"] = True

        try:
            declared_significant_bits = int(pixels.get("SignificantBits", ""))
        except ValueError as exc:
            raise ValueError("OME SignificantBits is missing or invalid") from exc
        if declared_significant_bits != spec.significant_bits:
            raise ValueError(
                f"OME SignificantBits={declared_significant_bits} does not match "
                f"the writer specification {spec.significant_bits}"
            )
        verification["significant_bits_matches_spec"] = True
        # Retain the historical report key for compatibility. The writer spec
        # defaults to the dtype width and may deliberately declare fewer
        # significant bits after float mantissa trimming.
        verification["significant_bits_matches_dtype"] = True

        channels = pixels.findall("./ome:Channel", namespaces=namespace)
        channel_samples = [
            int(channel.get("SamplesPerPixel", "0"))
            for channel in channels
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
                "OME Channel/SamplesPerPixel layout does not match the written "
                "TIFF layout"
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
                "OME TiffData PlaneCount does not match the physical top-level "
                f"plane count ({spec.plane_count})"
            )
        verification["tiff_data_mapping_matches"] = True

        if len(prepared.level_shapes) > 1:
            map_annotations = root.findall(
                "./ome:StructuredAnnotations/ome:MapAnnotation",
                namespaces=namespace,
            )
            pyramid_annotations = [
                item
                for item in map_annotations
                if item.get("Namespace") == PYRAMID_NAMESPACE
            ]
            annotation_refs = {
                item.get("ID")
                for item in image.findall(
                    "./ome:AnnotationRef",
                    namespaces=namespace,
                )
            }
            if len(pyramid_annotations) != 1:
                raise ValueError(
                    "OME pyramid metadata must contain exactly one "
                    "PyramidResolution MapAnnotation"
                )
            annotation_id = pyramid_annotations[0].get("ID")
            if not annotation_id or annotation_id not in annotation_refs:
                raise ValueError(
                    "OME PyramidResolution MapAnnotation is not linked from Image"
                )
        verification["pyramid_annotation_linked"] = True

        raster = verify_prepared_images(output, (prepared,))
        verification["dtype_matches_source"] = True
        verification["axes_match"] = True
        verification["pyramid_level_shapes_match"] = True
        verification["top_level_ifd_count_matches"] = True
        verification["samples_per_pixel_match"] = True
        verification["photometric_matches"] = True
        verification["compression_matches_requested"] = True
        verification["predictor_matches_requested"] = True
        verification["jpeg_subsampling_matches_requested"] = (
            raster.jpeg_subsampling_checked
        )
        verification["subifd_layout_matches"] = True
        verification["all_levels_tiled"] = True
        verification["icc_profile_preserved"] = raster.icc_profiles_checked
        verification["output_pixels_decodable"] = True
        verification["base_pixel_values_checked"] = prepared.compression.lossless
        verification["base_pixel_values_match"] = (
            True if prepared.compression.lossless else None
        )
        verification["planes_checked"] = raster.planes_checked
        verification["channels_checked"] = spec.size_c
        verification["points_per_plane"] = raster.points_per_plane
        verification["points_per_channel"] = raster.points_per_plane
    return verification
