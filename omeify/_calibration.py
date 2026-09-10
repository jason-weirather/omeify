"""Metadata-only X/Y calibration arithmetic and conservative OME/IFD matching.

Inspection compares only full-resolution planes: OME PhysicalSizeX/Y describes
those planes, not every SubIFD. Writer verification can additionally compare its
known pyramid scales. No reader calibration policy or stored units are changed.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from numbers import Integral
from typing import Any

import tifffile
from lxml import etree

# Ten parts per million accommodates six-significant-digit source normalization.
# This is an encoding-consistency tolerance, not scanner measurement uncertainty.
CALIBRATION_REL_TOL = 1e-5
CALIBRATION_ABS_TOL_UM = 0.0
_MAX_PAGES = 4096
_MAX_OME_CHARS = 8_388_608
_OME_2016 = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
_ENCODING = re.compile(r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2", re.I)


def _positive(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) and result > 0 else None


def _integer(value: Any, default: int | None = None) -> int | None:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _children(node: Any, name: str) -> list[Any]:
    if node is None:
        return []
    namespace = etree.QName(node).namespace
    tag = f"{{{namespace}}}{name}" if namespace else name
    return [child for child in node if child.tag == tag]


def _tag(page: Any, name: str) -> tuple[Any, str | None]:
    try:
        tag = page.tags.get(name)
        if tag is None:
            return None, "missing_tag"
        return tag.value, None
    except Exception:
        # Provider/file exception text may contain identifying source content.
        return None, "unreadable_tag"


def tiff_calibration(page: Any) -> dict[str, Any]:
    """Read actual TIFF rational tags without reader/display rounding.

    An absent ResolutionUnit uses TIFF's inch default, explicitly reported as a
    default, never inherited from another page. Missing resolution values are not
    filled from another axis, page, or OME. Unit 1 cannot supply an absolute scale.
    """

    raw_unit, unit_issue = _tag(page, "ResolutionUnit")
    unit = _integer(raw_unit)
    defaulted = unit_issue == "missing_tag"
    effective = 2 if defaulted else unit
    if unit_issue == "unreadable_tag":
        problem = unit_issue
    elif effective == 1:
        problem = "nonphysical_unit"
    elif effective not in (2, 3):
        problem = "unsupported_unit"
    else:
        problem = None
    result: dict[str, Any] = {
        "resolution_unit": unit, "effective_resolution_unit": effective,
        "unit_defaulted": defaulted,
    }
    for axis in ("x", "y"):
        raw, issue = _tag(page, axis.upper() + "Resolution")
        rational = None
        density = None
        if issue is None:
            if isinstance(raw, (tuple, list)) and len(raw) == 2:
                numerator, denominator = raw
                if all(isinstance(v, Integral) and not isinstance(v, bool) for v in raw):
                    rational = [int(numerator), int(denominator)]
                    if numerator > 0 and denominator > 0:
                        density = _positive(numerator / denominator)
            else:
                density = _positive(raw)
            if density is None:
                issue = "invalid_resolution"
        issue = issue or problem
        size = None if issue else _positive((10000.0 if effective == 3 else 25400.0) / density)
        if issue is None and size is None:
            issue = "invalid_resolution"
        result[axis] = {
            "rational": rational, "pixels_per_unit": density,
            "pixel_size_um": size, "issue": issue,
        }
    return result


def ome_calibration(pixels: Any) -> dict[str, Any]:
    """Normalize independent OME X/Y lengths, retaining explicit/defaulted units."""

    # Lazy import avoids the inspection -> io -> inspection import cycle.
    from omeify.io.pixel_size import PixelSize, canonical_length_unit

    result = {}
    modern = pixels is not None and etree.QName(pixels).namespace == _OME_2016
    for axis in ("x", "y"):
        attribute = "PhysicalSize" + axis.upper()
        raw = None if pixels is None else pixels.get(attribute)
        unit = None if pixels is None else pixels.get(attribute + "Unit")
        value = _positive(raw)
        defaulted = raw is not None and unit is None and modern
        effective = "µm" if defaulted else unit
        size = None
        if raw is None:
            issue = "missing_value"
        elif value is None:
            issue = "invalid_value"
        elif effective is None:
            issue = "missing_unit"
        elif effective in ("pixel", "reference frame"):
            issue = "nonphysical_unit"
        else:
            try:
                size = PixelSize(value, value, effective).converted_to("µm").x
                effective = canonical_length_unit(effective)
                issue = None
            except (ValueError, OverflowError):
                issue = "unsupported_unit"
        result[axis] = {
            "value": value, "unit": unit, "effective_unit": effective,
            "unit_defaulted": defaulted, "pixel_size_um": size, "issue": issue,
        }
    return result


def compare_axis(actual: float | None, expected: float | None) -> dict[str, Any]:
    """Compare positive, normalized lengths with a symmetric relative tolerance."""

    if actual is None or expected is None:
        return {"status": "not_comparable", "relative_difference": None}
    difference = abs(actual - expected) / max(actual, expected)
    matches = math.isclose(
        actual, expected, rel_tol=CALIBRATION_REL_TOL, abs_tol=CALIBRATION_ABS_TOL_UM,
    )
    return {
        "status": "consistent" if matches else "mismatch",
        "relative_difference": difference,
    }


def _status(values: Sequence[str], *, complete: bool = True) -> str:
    if "mismatch" in values:
        return "mismatch"
    if values and all(value == "consistent" for value in values) and complete:
        return "consistent"
    if any(value in ("consistent", "partial") for value in values):
        return "partial"
    return "not_comparable"


def _ome_images(tiff: Any) -> tuple[list[dict[str, Any]], str]:
    """Read local TiffData ranges without opening UUID/FileName companions."""

    try:
        xml = tiff.ome_metadata
        if not xml:
            return [], "no_ome"
        if len(xml) > _MAX_OME_CHARS:
            return [], "ome_scan_limit"
        if isinstance(xml, str):
            xml = _ENCODING.sub(r"\1", xml, count=1).encode("utf-8")
        parser = etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True)
        root = etree.fromstring(xml, parser)
        if root.getroottree().docinfo.doctype:
            return [], "unsafe_ome"
        if etree.QName(root).localname != "OME":
            return [], "unreadable_ome"
    except Exception:
        return [], "unreadable_ome"
    images = []
    for index, image in enumerate(_children(root, "Image")):
        pixels_elements = _children(image, "Pixels")
        pixels = pixels_elements[0] if len(pixels_elements) == 1 else None
        ranges = []
        unresolved = False
        if pixels is not None:
            channels = _children(pixels, "Channel")
            size_c = _integer(pixels.get("SizeC"))
            samples = [_integer(channel.get("SamplesPerPixel"), 1) for channel in channels]
            valid_samples = bool(samples) and all(v is not None and v > 0 for v in samples)
            logical_c = len(channels) if valid_samples and sum(samples) == size_c else size_c
            z, t = _integer(pixels.get("SizeZ")), _integer(pixels.get("SizeT"))
            plane_count = (
                logical_c * z * t
                if all(v is not None and v > 0 for v in (logical_c, z, t)) else None
            )
            for data in _children(pixels, "TiffData"):
                uuid = _children(data, "UUID")
                if uuid and (
                    len(uuid) != 1 or not root.get("UUID")
                    or (uuid[0].text or "").strip() != root.get("UUID")
                ):
                    unresolved = True
                    continue  # No filename guessing, companion reads, or directory searches.
                start = _integer(data.get("IFD"), 0)
                count = _integer(data.get("PlaneCount"), 1 if "IFD" in data.attrib else None)
                if count is None and "PlaneCount" not in data.attrib:
                    # A bare TiffData maps the image's planes, not unrelated tail IFDs.
                    if any(_integer(data.get("First" + a), 0) != 0 for a in ("C", "Z", "T")):
                        unresolved = True
                        continue
                    count = min(len(tiff.pages), plane_count) if plane_count is not None else None
                if (
                    start is None or start < 0 or count is None or count < 0
                    or start + count > len(tiff.pages)
                ):
                    unresolved = True
                    continue
                if count:
                    ranges.append((start, start + count))
        images.append({
            "index": index, "pixels": pixels, "ranges": ranges,
            "unresolved": unresolved,
        })
    return images, "parsed_ome"


def _local_ifd(tiff: Any, page: Any) -> int | None:
    """Confirm a top-level physical IFD; SubIFDs may reuse integer page indices."""

    index = page.index
    if not isinstance(index, Integral) or page.parent is not tiff:
        return None
    index = int(index)
    if not 0 <= index < len(tiff.pages):
        return None
    return index if int(tiff.pages[index].offset) == int(page.offset) else None


def _match_image(
    pages: list[tuple[int, Any]], images: list[dict[str, Any]], *, complete: bool,
) -> tuple[dict[str, Any] | None, str]:
    if not complete or not pages:
        return None, "incomplete_local_pages"
    candidates = []
    for index, page in pages:
        owners = [
            image for image in images
            if any(start <= index < end for start, end in image["ranges"])
        ]
        if len(owners) > 1:
            return None, "ambiguous_tiff_data"
        if not owners:
            return None, "unmapped_ifd"
        candidates.append(owners[0])
    chosen = candidates[0]
    if any(image is not chosen for image in candidates) or chosen["unresolved"]:
        return None, "incomplete_tiff_data"
    pixels = chosen["pixels"]
    if any(
        int(page.imagewidth) != _integer(pixels.get("SizeX"))
        or int(page.imagelength) != _integer(pixels.get("SizeY"))
        or bool(page.is_reduced)
        for _, page in pages
    ):
        return None, "not_full_resolution_geometry"
    return chosen, "local_tiff_data"


def inspect_calibration(
    tiff: tifffile.TiffFile, series: Sequence[tifffile.TiffPageSeries],
) -> dict[str, Any]:
    """Compare raw TIFF tags to OME on proven local full-resolution plane mappings.

    Inspect every base-plane entry, including real directories behind frames, up
    to a file-wide 4,096-entry limit. Never pair images by ordinal, name, or shape
    alone. Missing/ambiguous data remains not-comparable or partial, not a pass.
    Does not enumerate ``tiff.series`` itself (that can open companion files).
    """

    images, ome_state = _ome_images(tiff)
    remaining = _MAX_PAGES
    results = []
    for series_index, item in enumerate(series):
        local_pages = []
        try:
            count = len(item.pages)
        except Exception:
            count = 0
        attempted = min(count, remaining)
        remaining -= attempted
        for position in range(attempted):
            try:
                page = item.pages[position].aspage()
                index = _local_ifd(tiff, page)
                if index is not None:
                    local_pages.append((index, page))
            except Exception:
                continue
        complete = count > 0 and len(local_pages) == count
        image, reason = _match_image(local_pages, images, complete=complete)
        if ome_state != "parsed_ome":
            reason = ome_state
        declared = ome_calibration(image["pixels"]) if image is not None else None
        checks = []
        for ifd, page in local_pages:
            actual = tiff_calibration(page)
            axes = {
                axis: compare_axis(
                    actual[axis]["pixel_size_um"],
                    declared[axis]["pixel_size_um"] if declared else None,
                ) for axis in ("x", "y")
            }
            checks.append({
                "ifd": ifd, "tiff": actual, "axes": axes,
                "status": _status([axes[a]["status"] for a in ("x", "y")]),
            })
        results.append({
            "series_index": series_index,
            "ome_image_index": None if image is None else image["index"],
            "mapping_reason": reason,
            "status": _status([check["status"] for check in checks], complete=complete),
            "ome": declared, "pages_total": count, "pages_checked": len(checks),
            "scan_limited": attempted < count, "checks": checks,
        })
    return {
        "schema_version": "1.0", "scope": "full_resolution_xy", "comparison_unit": "µm",
        "relative_tolerance": CALIBRATION_REL_TOL, "absolute_tolerance_um": CALIBRATION_ABS_TOL_UM,
        "max_page_entries": _MAX_PAGES, "series": results,
    }


def calibration_context(calibration: dict[str, Any]) -> list[str]:
    """Create bounded, numeric-only derived evidence, never arbitrary file prose."""

    result = []
    for item in calibration["series"]:
        index, status = item["series_index"], item["status"]
        parts = [
            f"Computed calibration check for TIFF series {index}, "
            f"full-resolution X/Y only: {status}.",
            f"OME Image index: {item['ome_image_index']}; mapping: {item['mapping_reason']}.",
            f"Read {item['pages_checked']}/{item['pages_total']} local base-plane entries.",
            f"Numerical tolerance: relative {calibration['relative_tolerance']:g}, "
            f"absolute {calibration['absolute_tolerance_um']:g} µm.",
        ]
        counts = {s: sum(c["status"] == s for c in item["checks"])
                  for s in ("consistent", "mismatch", "partial", "not_comparable")}
        parts.append("Plane results: " + ", ".join(f"{key}={n}" for key, n in counts.items()) + ".")
        if item["ome"]:
            for axis in ("x", "y"):
                declared = item["ome"][axis]
                parts.append(f"OME {axis.upper()}={declared['pixel_size_um']} µm/pixel "
                             f"(unit defaulted={declared['unit_defaulted']}).")
        # Include one representative worst case, not thousands of page records.
        if item["checks"]:
            priority = {"mismatch": 0, "partial": 1, "not_comparable": 2, "consistent": 3}
            check = min(item["checks"], key=lambda c: priority[c["status"]])
            actual = check["tiff"]
            parts.append(f"Example IFD {check['ifd']}: effective TIFF ResolutionUnit="
                         f"{actual['effective_resolution_unit']} "
                         f"(defaulted={actual['unit_defaulted']}).")
            for axis in ("x", "y"):
                parts.append(f"TIFF {axis.upper()}={actual[axis]['pixel_size_um']} µm/pixel; "
                             f"comparison={check['axes'][axis]['status']}; "
                             f"relative difference={check['axes'][axis]['relative_difference']}; "
                             f"tag issue={actual[axis]['issue']}.")
        parts.append("SubIFDs were not compared to base OME sizes. "
                     "Numerical agreement does not establish acquisition accuracy or compliance.")
        result.append(" ".join(parts))
    return result
