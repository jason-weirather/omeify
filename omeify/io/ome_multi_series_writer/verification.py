from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import tifffile
from lxml import etree

from omeify.io.ome_tiff_writer import (
    _OUTPUT_BYTEORDER,
    _OUTPUT_COMPRESSION_CODES,
    _series_layout_matches,
)
from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import TiffPlaneReader
from omeify.utils.generate_ome_xml import OMEIFY_PROVENANCE_NAMESPACE

from .model import PreparedSeries

_OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_PYRAMID_NAMESPACE = "openmicroscopy.org/PyramidResolution"


def verify_output(
    prepared: Sequence[PreparedSeries],
    output_path: Path,
    *,
    provenance_json: str | None,
) -> dict[str, object]:
    """Verify OME metadata, TIFF storage, decoding, and lossless spot values."""

    verification: dict[str, object] = {
        "ome_tiff_recognized": False,
        "bigtiff": False,
        "output_byte_order": None,
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
        if not output.is_ome:
            raise ValueError("Written TIFF is not recognized as OME-TIFF")
        verification["ome_tiff_recognized"] = True
        if not output.is_bigtiff:
            raise ValueError("Written output is not BigTIFF")
        verification["bigtiff"] = True
        if output.byteorder != _OUTPUT_BYTEORDER:
            raise ValueError(
                f"Output TIFF byte order {output.byteorder!r} does not match "
                f"configured byte order {_OUTPUT_BYTEORDER!r}"
            )
        verification["output_byte_order"] = (
            "big" if output.byteorder == ">" else "little"
        )

        omexml = output.ome_metadata
        if not omexml:
            raise ValueError("Written OME-TIFF does not contain OME-XML metadata")
        parser = etree.XMLParser(resolve_entities=False, no_network=True)
        root = etree.fromstring(omexml.encode("utf-8"), parser=parser)
        namespace = {"ome": _OME_NAMESPACE}
        images = root.findall("./ome:Image", namespaces=namespace)
        if len(images) != len(prepared):
            raise ValueError(
                f"OME metadata contains {len(images)} Images; expected {len(prepared)}"
            )
        verification["image_count_matches"] = True
        names = tuple(image.get("Name") for image in images)
        expected_names = tuple(item.image.name for item in prepared)
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
        metadata_matches = True
        pyramid_links_match = True
        for image_index, (image, item) in enumerate(zip(images, prepared)):
            spec = item.image.spec
            pixels = image.find("./ome:Pixels", namespaces=namespace)
            if pixels is None:
                raise ValueError(f"OME Image {image_index} has no Pixels element")
            declared_big_endian = (pixels.get("BigEndian") or "").lower()
            expected_big_endian = "true" if output.byteorder == ">" else "false"
            channels = pixels.findall("./ome:Channel", namespaces=namespace)
            samples = [
                int(channel.get("SamplesPerPixel", "0"))
                for channel in channels
            ]
            metadata_matches = metadata_matches and all(
                (
                    declared_big_endian == expected_big_endian,
                    int(pixels.get("SignificantBits", "0")) == spec.significant_bits,
                    int(pixels.get("SizeX", "0")) == spec.size_x,
                    int(pixels.get("SizeY", "0")) == spec.size_y,
                    int(pixels.get("SizeC", "0")) == spec.size_c,
                    len(channels) == spec.logical_channel_count,
                    samples
                    == [spec.samples_per_pixel] * spec.logical_channel_count,
                    (pixels.get("Interleaved") or "false").lower()
                    == ("true" if spec.is_rgb else "false"),
                )
            )
            tiff_data = pixels.findall("./ome:TiffData", namespaces=namespace)
            if len(tiff_data) != 1:
                raise ValueError(
                    f"OME Image {image_index} contains {len(tiff_data)} TiffData "
                    "elements; expected one"
                )
            if (
                int(tiff_data[0].get("IFD", "-1")) != expected_ifd
                or int(tiff_data[0].get("PlaneCount", "0")) != spec.plane_count
            ):
                raise ValueError(
                    f"OME Image {image_index} TiffData does not map IFD "
                    f"{expected_ifd} with PlaneCount={spec.plane_count}"
                )
            expected_ifd += spec.plane_count

            if len(item.level_shapes) > 1:
                refs = {
                    ref.get("ID")
                    for ref in image.findall(
                        "./ome:AnnotationRef",
                        namespaces=namespace,
                    )
                }
                pyramid = [
                    annotation
                    for annotation_id, annotation in annotations.items()
                    if annotation_id in refs
                    and annotation.get("Namespace") == _PYRAMID_NAMESPACE
                ]
                pyramid_links_match = pyramid_links_match and len(pyramid) == 1
        if not metadata_matches:
            raise ValueError("One or more OME Images does not match its image specification")
        verification["metadata_matches_specs"] = True
        verification["tiff_data_mapping_matches"] = True
        if not pyramid_links_match:
            raise ValueError("One or more OME pyramid annotations is missing or unlinked")
        verification["pyramid_annotations_linked"] = True

        if provenance_json is not None:
            _verify_provenance_annotation(
                images,
                annotations,
                namespace,
                provenance_json,
            )
            verification["provenance_annotation_linked"] = True

        if len(output.series) != len(prepared):
            raise ValueError(
                f"Output contains {len(output.series)} TIFF series; expected {len(prepared)}"
            )
        top_level_count = sum(item.image.spec.plane_count for item in prepared)
        if len(output.pages) != top_level_count:
            raise ValueError(
                f"Output contains {len(output.pages)} top-level IFDs; expected "
                f"{top_level_count}"
            )
        verification["top_level_ifd_count_matches"] = True

        _verify_series_pixels(output, prepared, verification)
    return verification


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
        refs = {
            ref.get("ID")
            for ref in image.findall("./ome:AnnotationRef", namespaces=namespace)
        }
        if annotation_id not in refs:
            raise ValueError("OME provenance annotation is not linked from every Image")


def _verify_series_pixels(
    output: tifffile.TiffFile,
    prepared: Sequence[PreparedSeries],
    verification: dict[str, object],
) -> None:
    page_offset = 0
    all_lossless = all(item.compression.lossless for item in prepared)
    if all_lossless:
        verification["base_pixel_values_checked"] = True
    total_planes = 0
    points_per_plane = 0
    top_level_pages = list(output.pages)

    for series_index, (series, item) in enumerate(zip(output.series, prepared)):
        spec = item.image.spec
        if series.name != item.image.name:
            raise ValueError(
                f"TIFF series {series_index} name {series.name!r} does not match "
                f"{item.image.name!r}"
            )
        if np.dtype(series.dtype).newbyteorder("=") != spec.dtype:
            raise TypeError(
                f"TIFF series {series_index} dtype {series.dtype} does not match "
                f"{spec.dtype}"
            )
        if not _series_layout_matches(
            str(series.axes),
            series.shape,
            spec.output_axes,
            spec.output_shape,
        ):
            raise ValueError(
                f"TIFF series {series_index} axes/shape {series.axes!r} "
                f"{tuple(series.shape)} do not match {spec.output_axes!r} "
                f"{spec.output_shape}"
            )
        if len(series.levels) != len(item.level_shapes):
            raise ValueError(
                f"TIFF series {series_index} has {len(series.levels)} levels; "
                f"expected {len(item.level_shapes)}"
            )
        for level_index, (level, expected_shape) in enumerate(
            zip(series.levels, item.level_shapes)
        ):
            if not _series_layout_matches(
                str(level.axes),
                level.shape,
                spec.output_axes,
                expected_shape,
            ):
                raise ValueError(
                    f"TIFF series {series_index} level {level_index} shape "
                    f"{tuple(level.shape)} does not match {expected_shape}"
                )

        frames = top_level_pages[page_offset : page_offset + spec.plane_count]
        page_offset += spec.plane_count
        _verify_storage(frames, item)
        output_readers = _frame_readers(frames)
        coordinates = sorted(
            {
                (0, 0),
                (spec.size_y // 2, spec.size_x // 2),
                (spec.size_y - 1, spec.size_x - 1),
            }
        )
        points_per_plane = max(points_per_plane, len(coordinates))
        for reader in output_readers:
            for y, x in coordinates:
                reader.read_region(y, y + 1, x, x + 1)
        if item.compression.lossless:
            input_readers = item.image.source.plane_readers(cache_mib=16)
            try:
                for plane_index in range(spec.plane_count):
                    for y, x in coordinates:
                        source_value = input_readers[plane_index].read_region(
                            y, y + 1, x, x + 1
                        )[0, 0, ...]
                        output_value = output_readers[plane_index].read_region(
                            y, y + 1, x, x + 1
                        )[0, 0, ...]
                        if not np.array_equal(source_value, output_value):
                            raise ValueError(
                                "Lossless base-image verification failed at "
                                f"series {series_index}, plane {plane_index}, "
                                f"y={y}, x={x}: source={source_value}, "
                                f"output={output_value}"
                            )
            finally:
                for reader in input_readers:
                    reader.clear_cache()
        for reader in output_readers:
            reader.clear_cache()
        total_planes += spec.plane_count

    verification["series_layouts_match"] = True
    verification["storage_matches_requested"] = True
    verification["output_pixels_decodable"] = True
    verification["base_pixel_values_match"] = True if all_lossless else None
    verification["series_checked"] = len(prepared)
    verification["planes_checked"] = total_planes
    verification["points_per_plane"] = points_per_plane


def _verify_storage(
    frames: Sequence[tifffile.TiffPage | tifffile.TiffFrame],
    prepared: PreparedSeries,
) -> None:
    spec = prepared.image.spec
    compression = prepared.compression
    expected_compression = _OUTPUT_COMPRESSION_CODES[compression.name]
    expected_subifds = len(prepared.level_shapes) - 1
    if len(frames) != spec.plane_count:
        raise ValueError(
            f"Series {prepared.image.name!r} has {len(frames)} top-level pages; "
            f"expected {spec.plane_count}"
        )
    for plane_index, frame in enumerate(frames):
        page = frame.aspage()
        _verify_page_layout(
            page,
            spec=spec,
            expected_compression=expected_compression,
            expected_subsampling=compression.subsampling,
            reduced=False,
            context=f"series {prepared.image.name!r} plane {plane_index}",
        )
        subpages = list(page.pages) if page.pages is not None else []
        if len(subpages) != expected_subifds:
            raise ValueError(
                f"Series {prepared.image.name!r} plane {plane_index} has "
                f"{len(subpages)} SubIFDs; expected {expected_subifds}"
            )
        for level_index, subframe in enumerate(subpages, start=1):
            _verify_page_layout(
                subframe.aspage(),
                spec=spec,
                expected_compression=expected_compression,
                expected_subsampling=compression.subsampling,
                reduced=True,
                context=(
                    f"series {prepared.image.name!r} plane {plane_index} "
                    f"level {level_index}"
                ),
            )
    if spec.icc_profile is not None:
        output_icc = frames[0].aspage().iccprofile
        if output_icc is None or bytes(output_icc) != spec.icc_profile:
            raise ValueError(
                f"Series {prepared.image.name!r} did not preserve its ICC profile"
            )


def _frame_readers(
    frames: Sequence[tifffile.TiffPage | tifffile.TiffFrame],
) -> list[TiffPlaneReader]:
    lock = threading.RLock()
    return [TiffPlaneReader(frame.aspage(), lock=lock) for frame in frames]


def _verify_page_layout(
    page: tifffile.TiffPage,
    *,
    spec: OMEImageSpec,
    expected_compression: int,
    expected_subsampling: tuple[int, int] | None,
    reduced: bool,
    context: str,
) -> None:
    if not page.is_tiled:
        raise ValueError(f"{context} is not tiled")
    if int(page.compression) != expected_compression:
        raise ValueError(f"{context} does not use the requested TIFF compression")
    if int(page.samplesperpixel) != spec.samples_per_pixel:
        raise ValueError(f"{context} has the wrong SamplesPerPixel")
    if spec.is_rgb:
        if int(page.photometric) not in {2, 6} or int(page.planarconfig) != 1:
            raise ValueError(f"{context} does not use contiguous RGB storage")
    elif int(page.photometric) != 1:
        raise ValueError(f"{context} does not use grayscale photometric storage")
    if expected_subsampling is not None:
        try:
            actual = tuple(int(value) for value in page.tags["YCbCrSubSampling"].value)
        except (KeyError, TypeError) as exc:
            raise ValueError(f"{context} has no readable JPEG subsampling tag") from exc
        if actual != expected_subsampling:
            raise ValueError(
                f"{context} JPEG subsampling {actual} does not match "
                f"{expected_subsampling}"
            )
    if reduced and not (int(page.subfiletype) & 1):
        raise ValueError(f"{context} is not marked as reduced resolution")
