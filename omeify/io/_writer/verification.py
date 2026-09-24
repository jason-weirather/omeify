from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import tifffile
from lxml import etree

from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import TiffPlaneReader

from .configuration import OUTPUT_BYTEORDER, OUTPUT_COMPRESSION_CODES
from .model import PreparedImage
from .pyramid import downsample_region, series_layout_matches

OME_NAMESPACE = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
PYRAMID_NAMESPACE = "openmicroscopy.org/PyramidResolution"


@dataclass(frozen=True, slots=True)
class ContainerVerification:
    """Verified container identity plus parsed OME metadata."""

    output_byte_order: str
    omexml: str
    root: etree._Element
    namespace: dict[str, str]


@dataclass(frozen=True, slots=True)
class RasterVerification:
    """Successful common TIFF-series, storage, and pixel verification."""

    all_lossless: bool
    jpeg_subsampling_checked: bool | None
    icc_profiles_checked: bool | None
    series_checked: int
    planes_checked: int
    points_per_plane: int
    base_points_checked: int
    base_points_compared: int
    pyramid_points_checked: int
    pyramid_points_compared: int
    plane_levels_checked: int

    def pixel_coverage(self) -> dict[str, object]:
        """Describe the bounds of successful raster checks, not a full-image certificate."""

        return {
            "schema_version": "1.0",
            "mode": "sampled",
            "sampling": "top_left_center_bottom_right_per_plane_per_level",
            "all_pixels_checked": False,
            "plane_levels_checked": self.plane_levels_checked,
            "base_points_decoded": self.base_points_checked,
            "lossless_base_points_compared": self.base_points_compared,
            "pyramid_points_decoded": self.pyramid_points_checked,
            "lossless_pyramid_points_compared": self.pyramid_points_compared,
            "base_comparison": "bitwise_after_byte_order_normalization",
            "pyramid_comparison": "nearest_bitwise_or_mean_numeric_equal_nan",
            "pyramid_reference": "preceding_decoded_output_level_for_lossless_series",
            "source_reader_independent": False,
        }


def verify_container(
    output: tifffile.TiffFile,
    *,
    software: str,
) -> ContainerVerification:
    """Verify shared BigTIFF identity, byte order, Software tag, and OME XML."""

    if not output.is_ome:
        raise ValueError("Written TIFF is not recognized as OME-TIFF")
    if not output.is_bigtiff:
        raise ValueError("Written output is not BigTIFF")
    actual_byteorder = output.byteorder
    if actual_byteorder not in {"<", ">"}:
        raise ValueError(f"Unexpected TIFF byte order {actual_byteorder!r}")
    if actual_byteorder != OUTPUT_BYTEORDER:
        raise ValueError(
            f"Output TIFF byte order {actual_byteorder!r} does not match "
            f"configured byte order {OUTPUT_BYTEORDER!r}"
        )
    try:
        actual_software = str(output.pages[0].aspage().tags["Software"].value)
    except KeyError as exc:
        raise ValueError("Written TIFF does not contain a Software tag") from exc
    if actual_software != software:
        raise ValueError(
            f"TIFF Software tag {actual_software!r} does not match "
            f"requested value {software!r}"
        )
    omexml = output.ome_metadata
    if not omexml:
        raise ValueError("Written OME-TIFF does not contain OME-XML metadata")
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    root = etree.fromstring(omexml.encode("utf-8"), parser=parser)
    return ContainerVerification(
        output_byte_order=("big" if actual_byteorder == ">" else "little"),
        omexml=omexml,
        root=root,
        namespace={"ome": OME_NAMESPACE},
    )


def verify_prepared_images(
    output: tifffile.TiffFile,
    prepared: Sequence[PreparedImage],
) -> RasterVerification:
    """Verify layouts/storage and sampled base/pyramid values, without a full raster scan."""

    images = tuple(prepared)
    if len(output.series) != len(images):
        raise ValueError(
            f"Output contains {len(output.series)} TIFF series; expected {len(images)}"
        )
    top_level_count = sum(item.spec.plane_count for item in images)
    if len(output.pages) != top_level_count:
        raise ValueError(
            f"Output contains {len(output.pages)} top-level IFDs; expected "
            f"{top_level_count}"
        )

    page_offset = 0
    total_planes = 0
    points_per_plane = 0
    base_points_checked = 0
    base_points_compared = 0
    pyramid_points_checked = 0
    pyramid_points_compared = 0
    plane_levels_checked = 0
    all_lossless = all(item.compression.lossless for item in images)
    any_subsampling = any(
        item.compression.subsampling is not None for item in images
    )
    any_icc = any(item.spec.icc_profile is not None for item in images)
    top_level_pages = list(output.pages)

    for series_index, (series, item) in enumerate(
        zip(output.series, images, strict=True)
    ):
        spec = item.spec
        if item.name is not None and series.name != item.name:
            raise ValueError(
                f"TIFF series {series_index} name {series.name!r} does not "
                f"match {item.name!r}"
            )
        if np.dtype(series.dtype).newbyteorder("=") != spec.dtype:
            raise TypeError(
                f"TIFF series {series_index} dtype {series.dtype} does not "
                f"match {spec.dtype}"
            )
        if not series_layout_matches(
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
            zip(series.levels, item.level_shapes, strict=True)
        ):
            if not series_layout_matches(
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
        verify_storage(frames, item)
        coordinates = representative_coordinates(spec.size_y, spec.size_x)
        points_per_plane = max(points_per_plane, len(coordinates))
        output_readers = frame_readers(frames)
        input_readers = (
            item.source.plane_readers(cache_mib=16)
            if item.compression.lossless
            else None
        )
        try:
            if input_readers is not None and len(input_readers) != spec.plane_count:
                raise ValueError(
                    f"Source supplied {len(input_readers)} verification readers; "
                    f"expected {spec.plane_count}"
                )
            for plane_index, output_reader in enumerate(output_readers):
                for y, x in coordinates:
                    output_value = output_reader.read_region(
                        y,
                        y + 1,
                        x,
                        x + 1,
                    )[0, 0, ...]
                    base_points_checked += 1
                    if input_readers is None:
                        continue
                    source_value = input_readers[plane_index].read_region(
                        y,
                        y + 1,
                        x,
                        x + 1,
                    )[0, 0, ...]
                    base_points_compared += 1
                    if not same_sample_bits(source_value, output_value):
                        raise ValueError(
                            "Lossless base-image verification failed at "
                            f"series {series_index}, plane {plane_index}, "
                            f"y={y}, x={x}: source={source_value}, "
                            f"output={output_value}"
                        )
        finally:
            for reader in output_readers:
                reader.clear_cache()
            if input_readers is not None:
                for reader in input_readers:
                    reader.clear_cache()
        checked, compared, plane_levels = _verify_pyramid_pixels(series, item)
        pyramid_points_checked += checked
        pyramid_points_compared += compared
        plane_levels_checked += spec.plane_count + plane_levels
        total_planes += spec.plane_count

    return RasterVerification(
        all_lossless=all_lossless,
        jpeg_subsampling_checked=(True if any_subsampling else None),
        icc_profiles_checked=(True if any_icc else None),
        series_checked=len(images),
        planes_checked=total_planes,
        points_per_plane=points_per_plane,
        base_points_checked=base_points_checked,
        base_points_compared=base_points_compared,
        pyramid_points_checked=pyramid_points_checked,
        pyramid_points_compared=pyramid_points_compared,
        plane_levels_checked=plane_levels_checked,
    )


def same_sample_bits(source: np.ndarray, output: np.ndarray) -> bool:
    """Compare storage values including NaN payloads and signed zeros, not endianness."""

    left, right = np.asarray(source), np.asarray(output)
    if left.shape != right.shape or left.dtype.newbyteorder("=") != right.dtype.newbyteorder("="):
        return False

    def raw(value: np.ndarray) -> np.ndarray:
        if not value.dtype.isnative:
            value = value.byteswap().view(value.dtype.newbyteorder("="))
        return np.ascontiguousarray(value).view(np.uint8)

    return bool(np.array_equal(raw(left), raw(right)))


def _verify_pyramid_pixels(
    series: tifffile.TiffPageSeries, item: PreparedImage,
) -> tuple[int, int, int]:
    """Decode bounded samples at every level; compare lossless downsampling.

    JPEG samples are only decoded: a previously JPEG-encoded level is not the
    uncompressed source from which the staged pyramid was constructed.
    """

    checked = compared = plane_levels = 0
    for level_index in range(1, len(series.levels)):
        previous = series.levels[level_index - 1]
        current = series.levels[level_index]
        if len(current.pages) != item.spec.plane_count:
            raise ValueError("Pyramid level does not contain the expected physical planes")
        for plane_index in range(item.spec.plane_count):
            output_reader = TiffPlaneReader(current.pages[plane_index].aspage(), cache_mib=16)
            reference = (
                TiffPlaneReader(previous.pages[plane_index].aspage(), cache_mib=16)
                if item.compression.lossless else None
            )
            try:
                for y, x in representative_coordinates(output_reader.height, output_reader.width):
                    actual = output_reader.read_region(y, y + 1, x, x + 1)
                    checked += 1
                    if reference is None:
                        continue
                    expected = downsample_region(
                        reference, out_y0=y, out_y1=y + 1, out_x0=x, out_x1=x + 1,
                        method=item.downsample,
                        float32_mantissa_bits=item.float32_mantissa_bits,
                    )
                    compared += 1
                    matches = (
                        np.array_equal(actual, expected, equal_nan=True)
                        if item.downsample == "mean" and item.spec.dtype.kind == "f"
                        else same_sample_bits(expected, actual)
                    )
                    if not matches:
                        raise ValueError(
                            "Lossless pyramid verification failed at "
                            f"{item.display_name}, plane {plane_index}, "
                            f"level {level_index}, y={y}, x={x}"
                        )
            finally:
                output_reader.clear_cache()
                if reference is not None:
                    reference.clear_cache()
            plane_levels += 1
    return checked, compared, plane_levels


def representative_coordinates(
    height: int,
    width: int,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(
            {
                (0, 0),
                (height // 2, width // 2),
                (height - 1, width - 1),
            }
        )
    )


def verify_storage(
    frames: Sequence[tifffile.TiffPage | tifffile.TiffFrame],
    prepared: PreparedImage,
) -> None:
    """Verify tiled base and SubIFD storage for one prepared image."""

    spec = prepared.spec
    compression = prepared.compression
    expected_compression = OUTPUT_COMPRESSION_CODES[compression.name]
    expected_subifds = len(prepared.level_shapes) - 1
    if len(frames) != spec.plane_count:
        raise ValueError(
            f"{prepared.display_name.capitalize()} has {len(frames)} top-level "
            f"pages; expected {spec.plane_count}"
        )
    for plane_index, frame in enumerate(frames):
        page = frame.aspage()
        verify_page_layout(
            page,
            spec=spec,
            expected_compression=expected_compression,
            expected_subsampling=compression.subsampling,
            expected_tile_size=prepared.tile_size,
            reduced=False,
            context=f"{prepared.display_name} plane {plane_index}",
        )
        subpages = list(page.pages) if page.pages is not None else []
        if len(subpages) != expected_subifds:
            raise ValueError(
                f"{prepared.display_name.capitalize()} plane {plane_index} has "
                f"{len(subpages)} SubIFDs; expected {expected_subifds}"
            )
        for level_index, subframe in enumerate(subpages, start=1):
            verify_page_layout(
                subframe.aspage(),
                spec=spec,
                expected_compression=expected_compression,
                expected_subsampling=compression.subsampling,
                expected_tile_size=prepared.tile_size,
                reduced=True,
                context=(
                    f"{prepared.display_name} plane {plane_index} "
                    f"level {level_index}"
                ),
            )
    if spec.icc_profile is not None:
        output_icc = frames[0].aspage().iccprofile
        if output_icc is None or bytes(output_icc) != spec.icc_profile:
            raise ValueError(
                f"{prepared.display_name.capitalize()} did not preserve its ICC profile"
            )


def frame_readers(
    frames: Sequence[tifffile.TiffPage | tifffile.TiffFrame],
) -> list[TiffPlaneReader]:
    lock = threading.RLock()
    return [TiffPlaneReader(frame.aspage(), lock=lock) for frame in frames]


def verify_page_layout(
    page: tifffile.TiffPage,
    *,
    spec: OMEImageSpec,
    expected_compression: int,
    expected_subsampling: tuple[int, int] | None,
    expected_tile_size: int,
    reduced: bool,
    context: str,
) -> None:
    if not page.is_tiled:
        raise ValueError(f"{context} is not tiled")
    if (int(page.tilelength), int(page.tilewidth)) != (expected_tile_size, expected_tile_size):
        raise ValueError(f"{context} does not use the requested {expected_tile_size}-pixel tiles")
    if int(page.compression) != expected_compression:
        raise ValueError(f"{context} does not use the requested TIFF compression")
    if int(page.samplesperpixel) != spec.samples_per_pixel:
        raise ValueError(f"{context} has the wrong SamplesPerPixel")
    if spec.is_rgb:
        if int(page.planarconfig) != 1:
            raise ValueError(f"{context} does not use contiguous RGB storage")
        # Check the on-disk encoding policy, not merely whether this reader can
        # turn either RGB or YCbCr into an RGB array. Accepting both concealed
        # the 4:4:4 YCbCr/Bio-Formats double-conversion interoperability failure.
        expected_photometric = (
            tifffile.PHOTOMETRIC.YCBCR
            if expected_compression == int(tifffile.COMPRESSION.JPEG)
            and expected_subsampling != (1, 1)
            else tifffile.PHOTOMETRIC.RGB
        )
        if int(page.photometric) != int(expected_photometric):
            raise ValueError(
                f"{context} PhotometricInterpretation={int(page.photometric)} "
                f"does not match the RGB encoding policy "
                f"({expected_photometric.name}={int(expected_photometric)})"
            )
    elif int(page.photometric) != 1:
        raise ValueError(f"{context} does not use grayscale photometric storage")
    if expected_subsampling is not None:
        try:
            actual = tuple(
                int(value) for value in page.tags["YCbCrSubSampling"].value
            )
        except (KeyError, TypeError) as exc:
            raise ValueError(
                f"{context} has no readable JPEG subsampling tag"
            ) from exc
        if actual != expected_subsampling:
            raise ValueError(
                f"{context} JPEG subsampling {actual} does not match "
                f"{expected_subsampling}"
            )
    if reduced and not (int(page.subfiletype) & 1):
        raise ValueError(f"{context} is not marked as reduced resolution")


def verify_physical_calibration(
    output: tifffile.TiffFile,
    root: etree._Element,
    prepared: Sequence[PreparedImage],
) -> dict[str, object]:
    """Re-read OME and every base/SubIFD calibration against the writer inputs.

    The writer knows the sampling policy: scale is 2**level even when odd image
    dimensions are rounded upward. Do not substitute base_width/level_width.
    This verifies emitted metadata independently of the resolution encoder.
    """

    from omeify._calibration import (
        CALIBRATION_REL_TOL,
        compare_axis,
        ome_calibration,
        tiff_calibration,
    )

    images = root.findall(f"{{{OME_NAMESPACE}}}Image")
    if len(images) != len(prepared):
        raise ValueError("OME image count does not match calibration specifications")
    ifd = 0
    checked = 0
    for image_index, (image, item) in enumerate(zip(images, prepared, strict=True)):
        pixels = image.find(f"{{{OME_NAMESPACE}}}Pixels")
        declared = ome_calibration(pixels)
        expected = item.spec.pixel_size.converted_to("µm")
        for axis in ("x", "y"):
            value = declared[axis]
            if value["unit_defaulted"] or compare_axis(
                value["pixel_size_um"], getattr(expected, axis),
            )["status"] != "consistent":
                raise ValueError(
                    f"OME Image {image_index} PhysicalSize{axis.upper()} calibration "
                    "does not match the writer specification with explicit units"
                )
        for plane in range(item.spec.plane_count):
            page = output.pages[ifd].aspage()
            ifd += 1
            subpages = list(page.pages) if page.pages is not None else []
            if len(subpages) != len(item.level_shapes) - 1:
                raise ValueError("SubIFD count does not match calibration level specifications")
            for level, frame in enumerate([page, *subpages]):
                actual = tiff_calibration(frame.aspage())
                if actual["resolution_unit"] != 3:
                    raise ValueError(
                        f"TIFF calibration for image {image_index}, plane {plane}, level {level} "
                        "must explicitly use ResolutionUnit=CENTIMETER"
                    )
                for axis in ("x", "y"):
                    if compare_axis(
                        actual[axis]["pixel_size_um"], getattr(expected, axis) * (2**level),
                    )["status"] != "consistent":
                        raise ValueError(
                            f"TIFF {axis.upper()} calibration for image {image_index}, "
                            f"plane {plane}, level {level} does not match the writer specification"
                        )
                checked += 1
    return {
        "ome_physical_sizes_match_specs": True,
        "tiff_calibration_matches_specs": True,
        "pyramid_calibration_matches_specs": True,
        "calibration_ifds_checked": checked,
        "calibration_relative_tolerance": CALIBRATION_REL_TOL,
    }
