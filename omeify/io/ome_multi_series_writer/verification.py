from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import tifffile
from lxml import etree

from omeify.io._writer import PreparedImage
from omeify.io._writer.verification import (
    PYRAMID_NAMESPACE,
    verify_container,
    verify_physical_calibration,
    verify_prepared_images,
)
from omeify.utils.generate_ome_xml import OMEIFY_PROVENANCE_NAMESPACE


def verify_output(
    prepared: Sequence[PreparedImage],
    output_path: Path,
    *,
    provenance_json: str | None,
    software: str,
) -> dict[str, object]:
    """Verify multi-image OME metadata plus shared TIFF storage invariants."""

    images_expected = tuple(prepared)
    verification: dict[str, object] = {
        "ome_tiff_recognized": False,
        "bigtiff": False,
        "output_byte_order": None,
        "software_tag_matches": False,
        "image_count_matches": False,
        "series_names_match": False,
        "metadata_matches_specs": False,
        "tiff_data_mapping_matches": False,
        "pyramid_annotations_linked": False,
        "provenance_annotation_linked": None,
        "series_layouts_match": False,
        "top_level_ifd_count_matches": False,
        "storage_matches_requested": False,
        "output_pixels_decodable": False,
        "base_pixel_values_checked": False,
        "base_pixel_values_match": None,
        "series_checked": 0,
        "planes_checked": 0,
        "points_per_plane": 0,
    }
    with tifffile.TiffFile(output_path) as output:
        container = verify_container(output, software=software)
        verification["ome_tiff_recognized"] = True
        verification["bigtiff"] = True
        verification["output_byte_order"] = container.output_byte_order
        verification["software_tag_matches"] = True

        root = container.root
        namespace = container.namespace
        images = root.findall("./ome:Image", namespaces=namespace)
        if len(images) != len(images_expected):
            raise ValueError(
                f"OME metadata contains {len(images)} Images; "
                f"expected {len(images_expected)}"
            )
        verification["image_count_matches"] = True

        expected_names = tuple(_required_name(item) for item in images_expected)
        names = tuple(image.get("Name") for image in images)
        if names != expected_names:
            raise ValueError(
                f"OME Image names {names} do not match expected {expected_names}"
            )
        verification["series_names_match"] = True

        annotations = {
            item.get("ID"): item
            for item in root.findall(
                "./ome:StructuredAnnotations/ome:MapAnnotation",
                namespaces=namespace,
            )
            if item.get("ID")
        }
        expected_ifd = 0
        pyramid_links_match = True
        for image_index, (image, item) in enumerate(
            zip(images, images_expected, strict=True)
        ):
            pixels = image.find("./ome:Pixels", namespaces=namespace)
            if pixels is None:
                raise ValueError(f"OME Image {image_index} has no Pixels element")
            _verify_pixels_metadata(
                pixels,
                item,
                image_index=image_index,
                byteorder=output.byteorder,
                namespace=namespace,
            )
            expected_ifd = _verify_tiff_data_mapping(
                pixels,
                item,
                image_index=image_index,
                expected_ifd=expected_ifd,
                namespace=namespace,
            )
            if len(item.level_shapes) > 1:
                pyramid_links_match = pyramid_links_match and _has_pyramid_link(
                    image,
                    annotations,
                    namespace,
                )
        verification["metadata_matches_specs"] = True
        verification["tiff_data_mapping_matches"] = True
        if not pyramid_links_match:
            raise ValueError(
                "One or more OME pyramid annotations is missing or unlinked"
            )
        verification["pyramid_annotations_linked"] = True

        if provenance_json is not None:
            _verify_provenance_annotation(
                images,
                annotations,
                namespace,
                provenance_json,
            )
            verification["provenance_annotation_linked"] = True

        verification.update(verify_physical_calibration(output, root, images_expected))
        raster = verify_prepared_images(output, images_expected)
        verification["series_layouts_match"] = True
        verification["top_level_ifd_count_matches"] = True
        verification["storage_matches_requested"] = True
        verification["output_pixels_decodable"] = True
        verification["base_pixel_values_checked"] = raster.all_lossless
        verification["base_pixel_values_match"] = (
            True if raster.all_lossless else None
        )
        verification["series_checked"] = raster.series_checked
        verification["pixel_verification"] = raster.pixel_coverage()
        verification["planes_checked"] = raster.planes_checked
        verification["points_per_plane"] = raster.points_per_plane
    return verification


def _verify_pixels_metadata(
    pixels: etree._Element,
    prepared: PreparedImage,
    *,
    image_index: int,
    byteorder: str,
    namespace: dict[str, str],
) -> None:
    spec = prepared.spec
    declared_big_endian = (pixels.get("BigEndian") or "").lower()
    expected_big_endian = "true" if byteorder == ">" else "false"
    channels = pixels.findall("./ome:Channel", namespaces=namespace)
    samples = [int(channel.get("SamplesPerPixel", "0")) for channel in channels]
    matches = all(
        (
            declared_big_endian == expected_big_endian,
            int(pixels.get("SignificantBits", "0")) == spec.significant_bits,
            int(pixels.get("SizeX", "0")) == spec.size_x,
            int(pixels.get("SizeY", "0")) == spec.size_y,
            int(pixels.get("SizeC", "0")) == spec.size_c,
            len(channels) == spec.logical_channel_count,
            samples == [spec.samples_per_pixel] * spec.logical_channel_count,
            (pixels.get("Interleaved") or "false").lower()
            == ("true" if spec.is_rgb else "false"),
        )
    )
    if not matches:
        raise ValueError(
            f"OME Image {image_index} does not match its image specification"
        )


def _verify_tiff_data_mapping(
    pixels: etree._Element,
    prepared: PreparedImage,
    *,
    image_index: int,
    expected_ifd: int,
    namespace: dict[str, str],
) -> int:
    tiff_data = pixels.findall("./ome:TiffData", namespaces=namespace)
    if len(tiff_data) != 1:
        raise ValueError(
            f"OME Image {image_index} contains {len(tiff_data)} TiffData "
            "elements; expected one"
        )
    if (
        int(tiff_data[0].get("IFD", "-1")) != expected_ifd
        or int(tiff_data[0].get("PlaneCount", "0"))
        != prepared.spec.plane_count
    ):
        raise ValueError(
            f"OME Image {image_index} TiffData does not map IFD "
            f"{expected_ifd} with PlaneCount={prepared.spec.plane_count}"
        )
    return expected_ifd + prepared.spec.plane_count


def _has_pyramid_link(
    image: etree._Element,
    annotations: dict[str | None, etree._Element],
    namespace: dict[str, str],
) -> bool:
    references = {
        item.get("ID")
        for item in image.findall("./ome:AnnotationRef", namespaces=namespace)
    }
    pyramids = [
        annotation
        for annotation_id, annotation in annotations.items()
        if annotation_id in references
        and annotation.get("Namespace") == PYRAMID_NAMESPACE
    ]
    return len(pyramids) == 1


def _verify_provenance_annotation(
    images: Sequence[etree._Element],
    annotations: dict[str | None, etree._Element],
    namespace: dict[str, str],
    provenance_json: str,
) -> None:
    provenance_annotations = [
        annotation
        for annotation in annotations.values()
        if annotation.get("Namespace") == OMEIFY_PROVENANCE_NAMESPACE
    ]
    if len(provenance_annotations) != 1:
        raise ValueError(
            "OME metadata must contain exactly one omeify provenance annotation"
        )
    annotation = provenance_annotations[0]
    value = annotation.find("./ome:Value", namespaces=namespace)
    entries = [] if value is None else value.findall("./ome:M", namespaces=namespace)
    payloads = [entry.text or "" for entry in entries if entry.get("K") == "json"]
    if payloads != [provenance_json]:
        raise ValueError("OME provenance annotation does not match the supplied JSON")
    annotation_id = annotation.get("ID")
    for image in images:
        references = {
            item.get("ID")
            for item in image.findall("./ome:AnnotationRef", namespaces=namespace)
        }
        if annotation_id not in references:
            raise ValueError(
                "OME provenance annotation is not linked from every Image"
            )


def _required_name(prepared: PreparedImage) -> str:
    if prepared.name is None:
        raise RuntimeError("multi-series prepared images require a name")
    return prepared.name
