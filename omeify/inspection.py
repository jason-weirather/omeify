from __future__ import annotations

import html
import json
import math
import re
import textwrap
from collections.abc import Collection
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tifffile
from jsonschema import Draft202012Validator
from lxml import etree

from omeify.intelligence import DEFAULT_ALLOWED_SCOPES, DEFAULT_MAX_METADATA_CHARS
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header

if TYPE_CHECKING:
    from sheetbend import Registry

_INSPECTION_SCHEMA_RESOURCE = "omeify.schemas/tiff_inspection.schema.json"
_INSPECTION_SCHEMA_VERSION = "1.4"
_DEFAULT_DETAIL = 1
_DEFAULT_MAX_TEXT_LENGTH = 240
_MAX_XML_CHILDREN = 100
_MAX_XML_DEPTH = 12
_XML_ENCODING_DECLARATION = re.compile(
    r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2",
    flags=re.IGNORECASE,
)


def _xml_bytes(value: str) -> bytes:
    """Encode decoded XML text without retaining a stale encoding declaration."""

    normalized = _XML_ENCODING_DECLARATION.sub(r"\1", value, count=1)
    return normalized.encode("utf-8")


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _optional_text(value: Any) -> str | None:
    if value is None or value == "":
        return None
    return str(value)


def _safe_attr(value: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(value, name)
    except Exception:
        return default


def _safe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _safe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    return None


def _json_index(value: Any) -> int | list[int] | str | None:
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (tuple, list)):
        converted: list[int] = []
        for item in value:
            try:
                converted.append(int(item))
            except (TypeError, ValueError, OverflowError):
                return str(value)
        return converted
    return str(value)


def _enum_name(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        return value.name
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _shape(value: Any) -> list[int]:
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return []


def _truncate_text(value: str, max_length: int | None) -> tuple[str, bool]:
    if max_length is None or len(value) <= max_length:
        return value, False
    if max_length <= 1:
        return value[:max_length], True
    return value[: max_length - 1] + "…", True


def _value_preview(value: Any, max_length: int | None) -> tuple[str, int | None, bool]:
    """Return a bounded, human-readable TIFF tag value preview."""

    if isinstance(value, bytes):
        if max_length is None:
            rendered = "0x" + value.hex()
            return rendered, len(value), False
        byte_limit = max(0, (max_length - 3) // 2)
        prefix = value[:byte_limit]
        rendered = "0x" + prefix.hex()
        if len(prefix) < len(value):
            rendered += "…"
        return rendered, len(value), len(prefix) < len(value)

    if isinstance(value, np.ndarray):
        flattened = value.reshape(-1)
        item_limit = len(flattened) if max_length is None else min(len(flattened), 32)
        preview = flattened[:item_limit].tolist()
        rendered = (
            f"array(shape={list(value.shape)}, dtype={value.dtype}, values={preview}"
            + (", …)" if item_limit < len(flattened) else ")")
        )
        bounded, text_truncated = _truncate_text(rendered, max_length)
        return bounded, int(value.size), item_limit < len(flattened) or text_truncated

    if isinstance(value, (tuple, list)):
        item_limit = len(value) if max_length is None else min(len(value), 32)
        preview = list(value[:item_limit])
        rendered = repr(preview)
        if item_limit < len(value):
            rendered = rendered[:-1] + ", …]"
        bounded, text_truncated = _truncate_text(rendered, max_length)
        return bounded, len(value), item_limit < len(value) or text_truncated

    if isinstance(value, Enum):
        rendered = value.name
    elif isinstance(value, np.generic):
        rendered = repr(value.item())
    else:
        rendered = str(value)
    bounded, truncated = _truncate_text(rendered, max_length)
    return bounded, len(rendered), truncated


def _xml_node_summary(
    element: etree._Element,
    *,
    max_text_length: int | None,
    depth: int = 0,
) -> dict[str, Any]:
    attributes: dict[str, str] = {}
    for key, value in element.attrib.items():
        attributes[_local_name(key)] = _truncate_text(value, max_text_length)[0]

    text = (element.text or "").strip() or None
    text_truncated = False
    if text is not None:
        text, text_truncated = _truncate_text(text, max_text_length)

    children = list(element)
    if depth >= _MAX_XML_DEPTH:
        summarized_children: list[dict[str, Any]] = []
        omitted_children = len(children)
    else:
        summarized_children = [
            _xml_node_summary(
                child,
                max_text_length=max_text_length,
                depth=depth + 1,
            )
            for child in children[:_MAX_XML_CHILDREN]
        ]
        omitted_children = max(0, len(children) - len(summarized_children))

    return {
        "tag": _local_name(element.tag),
        "attributes": attributes,
        "text": text,
        "text_truncated": text_truncated,
        "children": summarized_children,
        "omitted_children": omitted_children,
    }


def _description_summary(value: str, max_text_length: int | None) -> dict[str, Any]:
    rendered, truncated = _truncate_text(value, max_text_length)
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    try:
        root = etree.fromstring(_xml_bytes(value), parser=parser)
    except (etree.XMLSyntaxError, ValueError, UnicodeEncodeError):
        return {
            "format": "text",
            "length": len(value),
            "value": rendered,
            "truncated": truncated,
            "xml_root": None,
            "parsed_xml": None,
        }
    return {
        "format": "xml",
        "length": len(value),
        "value": rendered,
        "truncated": truncated,
        "xml_root": _local_name(root.tag),
        "parsed_xml": _xml_node_summary(root, max_text_length=max_text_length),
    }


def _physical_size(value: Any, unit: Any) -> dict[str, Any] | None:
    numeric = _safe_float(value)
    if numeric is None:
        return None
    return {"value": numeric, "unit": str(unit) if unit not in {None, ""} else None}


def _pixel_size_summary(pixel_size: Any) -> dict[str, Any]:
    return {
        "x": {"value": float(pixel_size.x), "unit": str(pixel_size.unit)},
        "y": {"value": float(pixel_size.y), "unit": str(pixel_size.unit)},
        "z": None,
    }


def _ome_summary(xml: str, max_text_length: int | None) -> dict[str, Any]:
    parser = etree.XMLParser(resolve_entities=False, no_network=True, recover=False)
    root = etree.fromstring(_xml_bytes(xml), parser=parser)
    namespace = etree.QName(root).namespace

    image_elements = [child for child in root if _local_name(child.tag) == "Image"]
    images: list[dict[str, Any]] = []
    for image_index, image in enumerate(image_elements):
        pixels = next(
            (child for child in image if _local_name(child.tag) == "Pixels"),
            None,
        )
        if pixels is None:
            images.append(
                {
                    "index": image_index,
                    "id": image.get("ID"),
                    "name": image.get("Name"),
                    "pixels_id": None,
                    "dimension_order": None,
                    "pixel_type": None,
                    "significant_bits": None,
                    "interleaved": None,
                    "size": {"x": None, "y": None, "z": None, "c": None, "t": None},
                    "physical_size": {"x": None, "y": None, "z": None},
                    "channels": [],
                    "tiff_data": [],
                }
            )
            continue

        channels: list[dict[str, Any]] = []
        tiff_data: list[dict[str, Any]] = []
        for child in pixels:
            local = _local_name(child.tag)
            if local == "Channel":
                channels.append(
                    {
                        "index": len(channels),
                        "id": child.get("ID"),
                        "name": child.get("Name"),
                        "samples_per_pixel": _safe_int(child.get("SamplesPerPixel")),
                        "color": _safe_int(child.get("Color")),
                    }
                )
            elif local == "TiffData":
                tiff_data.append(
                    {
                        "ifd": _safe_int(child.get("IFD")),
                        "plane_count": _safe_int(child.get("PlaneCount")),
                        "first_c": _safe_int(child.get("FirstC")),
                        "first_z": _safe_int(child.get("FirstZ")),
                        "first_t": _safe_int(child.get("FirstT")),
                    }
                )

        images.append(
            {
                "index": image_index,
                "id": image.get("ID"),
                "name": image.get("Name"),
                "pixels_id": pixels.get("ID"),
                "dimension_order": pixels.get("DimensionOrder"),
                "pixel_type": pixels.get("Type"),
                "significant_bits": _safe_int(pixels.get("SignificantBits")),
                "interleaved": _safe_bool(pixels.get("Interleaved")),
                "size": {
                    "x": _safe_int(pixels.get("SizeX")),
                    "y": _safe_int(pixels.get("SizeY")),
                    "z": _safe_int(pixels.get("SizeZ")),
                    "c": _safe_int(pixels.get("SizeC")),
                    "t": _safe_int(pixels.get("SizeT")),
                },
                "physical_size": {
                    "x": _physical_size(
                        pixels.get("PhysicalSizeX"),
                        pixels.get("PhysicalSizeXUnit"),
                    ),
                    "y": _physical_size(
                        pixels.get("PhysicalSizeY"),
                        pixels.get("PhysicalSizeYUnit"),
                    ),
                    "z": _physical_size(
                        pixels.get("PhysicalSizeZ"),
                        pixels.get("PhysicalSizeZUnit"),
                    ),
                },
                "channels": channels,
                "tiff_data": tiff_data,
            }
        )

    xml_value, xml_truncated = _truncate_text(xml, max_text_length)
    return {
        "namespace": namespace,
        "version": namespace.rsplit("/", 1)[-1] if namespace else None,
        "creator": root.get("Creator"),
        "uuid": root.get("UUID"),
        "image_count": len(image_elements),
        "images": images,
        "miti": validate_miti_ome_tiff_header(xml).as_dict(),
        "xml": {
            "length": len(xml),
            "value": xml_value,
            "truncated": xml_truncated,
        },
    }


def _channel_labels(ome_image: dict[str, Any] | None) -> list[str]:
    if ome_image is None:
        return []
    labels: list[str] = []
    for channel in ome_image["channels"]:
        labels.append(
            channel.get("name")
            or channel.get("id")
            or f"Channel {int(channel['index']) + 1}"
        )
    return labels


def _tag_summary(tag: tifffile.TiffTag, max_text_length: int | None) -> dict[str, Any]:
    try:
        value = tag.value
        rendered, value_length, truncated = _value_preview(value, max_text_length)
    except Exception as exc:  # A malformed tag should not abort inspection of the TIFF.
        rendered = f"<unreadable: {type(exc).__name__}: {exc}>"
        value_length = None
        truncated = False
    dtype = getattr(tag.dtype, "name", None) or str(tag.dtype)
    return {
        "code": int(tag.code),
        "name": str(tag.name),
        "dtype": str(dtype),
        "count": int(tag.count),
        "value": rendered,
        "value_length": value_length,
        "truncated": truncated,
    }


def _frame_summary(frame: tifffile.TiffFrame) -> dict[str, Any]:
    return {
        "index": _json_index(_safe_attr(frame, "index")),
        "axes": str(_safe_attr(frame, "axes", "")),
        "shape": _shape(_safe_attr(frame, "shape", ())),
        "dtype": str(_safe_attr(frame, "dtype", "unknown")),
    }


def _page_summary(
    page: tifffile.TiffPage,
    *,
    detail: int,
    max_text_length: int | None,
) -> dict[str, Any]:
    page = page.aspage()
    tile_shape = None
    if bool(page.is_tiled):
        tile_shape = [int(page.tilelength), int(page.tilewidth)]

    description = None
    tags: list[dict[str, Any]] = []
    if detail >= 3:
        raw_description = page.description
        if raw_description:
            description = _description_summary(str(raw_description), max_text_length)
        tags = [_tag_summary(tag, max_text_length) for tag in page.tags]

    return {
        "index": _json_index(getattr(page, "index", None)),
        "axes": str(getattr(page, "axes", "")),
        "shape": _shape(getattr(page, "shape", ())),
        "dtype": str(getattr(page, "dtype", "unknown")),
        "flags": sorted(str(item) for item in getattr(page, "flags", set())),
        "width": int(page.imagewidth),
        "height": int(page.imagelength),
        "samples_per_pixel": int(page.samplesperpixel),
        "planar_configuration": _enum_name(getattr(page, "planarconfig", None)),
        "photometric": _enum_name(getattr(page, "photometric", None)),
        "compression": _enum_name(getattr(page, "compression", None)),
        "is_tiled": bool(page.is_tiled),
        "tile_shape": tile_shape,
        "rows_per_strip": _safe_int(getattr(page, "rowsperstrip", None)),
        "subifd_count": len(getattr(page, "subifds", None) or ()),
        "uncompressed_bytes": _safe_int(getattr(page, "nbytes", None)),
        "stored_bytes": sum(int(value) for value in getattr(page, "databytecounts", ()) or ()),
        "description": description,
        "tags": tags,
        "frames": [],
    }


def _level_summary(
    level: tifffile.TiffPageSeries,
    *,
    level_index: int,
    detail: int,
    max_text_length: int | None,
    warnings: list[str],
) -> dict[str, Any]:
    page_items: list[Any] = []
    if detail >= 2:
        try:
            page_items = list(level.pages)
        except Exception as exc:
            warnings.append(
                f"Unable to enumerate pages for series level {level_index}: "
                f"{type(exc).__name__}: {exc}"
            )
    tiff_page_count = sum(isinstance(item, tifffile.TiffPage) for item in page_items)
    tiff_frame_count = sum(isinstance(item, tifffile.TiffFrame) for item in page_items)
    if detail < 2:
        try:
            page_count = len(level.pages)
        except Exception:
            page_count = 0
        tiff_page_count = None
        tiff_frame_count = None
    else:
        page_count = len(page_items)

    pages: list[dict[str, Any]] = []
    current_page: dict[str, Any] | None = None
    if detail >= 2:
        for item in page_items:
            try:
                if isinstance(item, tifffile.TiffFrame):
                    frame = _frame_summary(item)
                    if current_page is None:
                        current_page = _page_summary(
                            item.aspage(),
                            detail=detail,
                            max_text_length=max_text_length,
                        )
                        pages.append(current_page)
                    current_page["frames"].append(frame)
                    continue
                current_page = _page_summary(
                    item.aspage(),
                    detail=detail,
                    max_text_length=max_text_length,
                )
                pages.append(current_page)
            except Exception as exc:
                warnings.append(
                    f"Unable to summarize series level {level_index} page "
                    f"{getattr(item, 'index', '?')}: {type(exc).__name__}: {exc}"
                )

    return {
        "index": int(level_index),
        "name": _optional_text(_safe_attr(level, "name")),
        "axes": str(_safe_attr(level, "axes", "")),
        "shape": _shape(_safe_attr(level, "shape", ())),
        "dtype": str(_safe_attr(level, "dtype", "unknown")),
        "page_count": int(page_count),
        "tiff_page_count": tiff_page_count,
        "tiff_frame_count": tiff_frame_count,
        "pages": pages,
    }


def _series_summary(
    series: tifffile.TiffPageSeries,
    *,
    series_index: int,
    detail: int,
    max_text_length: int | None,
    ome_image: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    try:
        levels = list(series.levels)
    except Exception as exc:
        warnings.append(
            f"Unable to enumerate levels for series {series_index}: "
            f"{type(exc).__name__}: {exc}"
        )
        levels = [series]
    if not levels:
        warnings.append(f"Series {series_index} reported no levels; using the series itself")
        levels = [series]

    tiff_resolution_pixel_size = None
    try:
        from omeify.io.pixel_size import pixel_size_from_tiff_resolution

        for item in levels[0].pages:
            page = item.aspage()
            if not all(
                tag_name in page.tags
                for tag_name in ("XResolution", "YResolution", "ResolutionUnit")
            ):
                continue
            pixel_size = pixel_size_from_tiff_resolution(page)
            if pixel_size is not None:
                tiff_resolution_pixel_size = _pixel_size_summary(pixel_size)
                break
    except Exception:
        # Inspection reports what can be read from the TIFF tags. Missing,
        # malformed, or unusable calibration is represented as N/A rather
        # than promoted to a diagnostic warning.
        tiff_resolution_pixel_size = None

    rendered_levels = []
    if detail >= 1:
        rendered_levels = [
            _level_summary(
                level,
                level_index=level_index,
                detail=detail,
                max_text_length=max_text_length,
                warnings=warnings,
            )
            for level_index, level in enumerate(levels)
        ]

    series_name = _optional_text(_safe_attr(series, "name"))
    if ome_image is not None and ome_image.get("name"):
        series_name = str(ome_image["name"])

    return {
        "index": int(series_index),
        "name": series_name,
        "kind": _optional_text(_safe_attr(series, "kind")),
        "axes": str(_safe_attr(series, "axes", "")),
        "shape": _shape(_safe_attr(series, "shape", ())),
        "dtype": str(_safe_attr(series, "dtype", "unknown")),
        "is_pyramidal": bool(_safe_attr(series, "is_pyramidal", len(levels) > 1)),
        "level_count": len(levels),
        "ome_image_index": ome_image.get("index") if ome_image is not None else None,
        "ome_image_id": ome_image.get("id") if ome_image is not None else None,
        "channel_names": _channel_labels(ome_image),
        "physical_size": (
            ome_image.get("physical_size") if ome_image is not None else None
        ),
        "tiff_resolution_pixel_size": tiff_resolution_pixel_size,
        "levels": rendered_levels,
    }


def _format_name(tiff: tifffile.TiffFile, flags: list[str]) -> str:
    if bool(getattr(tiff, "is_ome", False)):
        return "OME-TIFF"
    preferred = [
        ("svs", "Aperio SVS TIFF"),
        ("qpi", "Akoya/PerkinElmer QPI TIFF"),
        ("indica", "Indica Labs TIFF"),
        ("lsm", "Zeiss LSM TIFF"),
        ("ndpi", "Hamamatsu NDPI TIFF"),
        ("imagej", "ImageJ TIFF"),
        ("shaped", "tifffile shaped TIFF"),
    ]
    flag_set = set(flags)
    for flag, label in preferred:
        if flag in flag_set:
            return label
    return "TIFF"


@lru_cache(maxsize=1)
def _inspection_validator() -> Draft202012Validator:
    resource = files("omeify.schemas").joinpath("tiff_inspection.schema.json")
    schema = json.loads(resource.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    # Resolve the optional report schema from package resources, never the web.
    from referencing import Registry, Resource

    intelligence = json.loads(
        files("omeify.schemas").joinpath("metadata_intelligence.schema.json")
        .read_text(encoding="utf-8")
    )
    calibration = json.loads(
        files("omeify.schemas").joinpath("calibration.schema.json").read_text(encoding="utf-8")
    )
    registry = Registry().with_resources(
        (item["$id"], Resource.from_contents(item)) for item in (intelligence, calibration)
    )
    return Draft202012Validator(schema, registry=registry)


@dataclass
class _TreeNode:
    label: str
    children: list["_TreeNode"] = field(default_factory=list)


def _human_bytes(size: int | None) -> str:
    if size is None:
        return "unknown size"
    value = float(size)
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def _format_shape(shape: list[int]) -> str:
    return " × ".join(str(item) for item in shape) if shape else "unknown shape"


def _format_physical_size(physical: dict[str, Any] | None) -> str | None:
    if not physical:
        return None
    x = physical.get("x")
    y = physical.get("y")
    if not x and not y:
        return None
    values: list[str] = []
    for axis, item in (("X", x), ("Y", y)):
        if item:
            unit = f" {item['unit']}" if item.get("unit") else ""
            values.append(f"{axis}={item['value']:g}{unit}")
    return ", ".join(values)


def _render_tree(root: _TreeNode) -> str:
    lines = [root.label]

    def visit(node: _TreeNode, prefix: str, is_last: bool) -> None:
        label_lines = node.label.splitlines() or [""]
        lines.append(prefix + ("└── " if is_last else "├── ") + label_lines[0])
        child_prefix = prefix + ("    " if is_last else "│   ")
        lines.extend(child_prefix + line for line in label_lines[1:])
        for index, child in enumerate(node.children):
            visit(child, child_prefix, index == len(node.children) - 1)

    for index, child in enumerate(root.children):
        visit(child, "", index == len(root.children) - 1)
    return "\n".join(lines)


def _calibration_tree(check: dict[str, Any], *, detail: int) -> _TreeNode:
    status = check["status"].upper().replace("_", " ")
    root = _TreeNode(f"TIFF/OME calibration (full-resolution X/Y): {status}")
    reasons = {
        "no_ome": "No OME calibration to compare; this does not mean the TIFF is uncalibrated.",
        "local_tiff_data": "Local full-resolution planes matched through OME TiffData.",
        "ambiguous_tiff_data": "Multiple OME Images reference the same IFD; no pairing guessed.",
        "unmapped_ifd": "Base IFDs could not be mapped to one local OME Image.",
        "incomplete_local_pages": "Not all local base-plane entries could be checked.",
        "incomplete_tiff_data": "OME mapping is incomplete, external, or spans multiple Images.",
        "not_full_resolution_geometry": "Mapped pages do not match full-resolution OME geometry.",
        "unreadable_ome": "OME calibration metadata could not be parsed.",
        "unsafe_ome": "OME XML with a DTD was not used for calibration comparison.",
        "ome_scan_limit": "OME metadata exceeded the calibration parser limit.",
    }
    reason = reasons[check["mapping_reason"]]
    if detail == 0:
        if check["status"] != "consistent":
            root.children.append(_TreeNode(reason))
        return root
    root.children.append(_TreeNode(
        f"{check['pages_checked']}/{check['pages_total']} local base-plane entries checked; "
        f"{reason}"
    ))
    declared = check["ome"]
    if declared is not None:
        sizes = []
        for axis in ("x", "y"):
            value = declared[axis]["pixel_size_um"]
            text = "N/A" if value is None else f"{value:.9g} µm/pixel"
            if declared[axis]["unit_defaulted"]:
                text += " (OME unit default)"
            if declared[axis]["issue"]:
                text += f" ({declared[axis]['issue'].replace('_', ' ')})"
            sizes.append(f"{axis.upper()}={text}")
        root.children.append(_TreeNode(
            f"OME Image {check['ome_image_index']}: " + ", ".join(sizes)
        ))
    # Show distinct encodings, with problematic planes before matching examples.
    priority = {"mismatch": 0, "partial": 1, "not_comparable": 2, "consistent": 3}
    seen = set()
    for page in sorted(check["checks"], key=lambda item: priority[item["status"]]):
        actual = page["tiff"]
        key = json.dumps(actual, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 3:
            continue
        unit = {2: "inch", 3: "cm"}.get(actual["effective_resolution_unit"], "unit")
        parts = []
        for axis in ("x", "y"):
            value = actual[axis]["pixel_size_um"]
            density = actual[axis]["pixels_per_unit"]
            text = "N/A" if value is None else f"{density:.9g} pixels/{unit} = {value:.9g} µm/pixel"
            if page["axes"][axis]["status"] == "mismatch":
                difference = page["axes"][axis]["relative_difference"]
                text += f" (MISMATCH, relative difference {difference:.6g})"
            elif actual[axis]["issue"]:
                text += f" ({actual[axis]['issue'].replace('_', ' ')})"
            parts.append(f"{axis.upper()}={text}")
        node = _TreeNode(f"TIFF IFD {page['ifd']}: " + ", ".join(parts))
        if actual["unit_defaulted"]:
            node.children.append(_TreeNode("ResolutionUnit absent: using the TIFF inch default."))
        root.children.append(node)
    if len(seen) > 3:
        root.children.append(_TreeNode(
            f"{len(seen) - 3} additional encodings are retained in JSON."
        ))
    root.children.append(_TreeNode(
        "Compared after unit conversion; SubIFDs excluded. "
        "Agreement is not proof of acquisition calibration accuracy."
    ))
    return root


def _intelligence_tree(report: dict[str, Any], max_text_length: int | None) -> _TreeNode:
    """Render the validated JSON view, never parse model-authored display text."""

    def display(value: str) -> str:
        # Make terminal escapes, control codes and bidi controls visible. Leave
        # scientific Unicode (including µm) intact. JSON retains original values.
        return "".join(
            char if char.isprintable() else char.encode("unicode_escape").decode("ascii")
            for char in value
        )

    def node(text: str) -> _TreeNode:
        return _TreeNode(textwrap.fill(display(text), width=96, break_long_words=False))

    source = report["source"]
    root = node(
        f"Metadata intelligence (advisory): {source['name']} [{source['scope']}] / {source['model']}"
    )
    catalog = {record["id"]: record for record in report["records"]}

    def add_evidence(parent: _TreeNode, statement: dict[str, Any]) -> None:
        ids = dict.fromkeys(evidence["record_id"] for evidence in statement["evidence"])
        for record_id in ids:
            record = catalog[record_id]
            origin = record["locations"][0]
            repeats = f" ({record['occurrences']} occurrences)" if record["occurrences"] > 1 else ""
            kind = "Computed evidence" if record.get("origin") == "computed" else "Evidence"
            parent.children.append(node(f"{kind} {record_id}: {origin}{repeats}"))

    summary = report["summary"]
    overview = node("Overview: " + summary["overview"]["text"])
    add_evidence(overview, summary["overview"])
    root.children.append(overview)
    categories = {
        "date": "Dates", "identifier": "Identifiers", "path": "Embedded paths / filenames",
        "acquisition": "Acquisition / instrument", "channel": "Channels / markers",
        "calibration": "Spatial calibration", "processing": "Software / processing",
        "other": "Other useful metadata",
    }
    for category, label in categories.items():
        findings = [item for item in summary["findings"] if item["category"] == category]
        if not findings:
            if category in {"date", "identifier", "path"}:
                root.children.append(node(label + ": not reported in the supplied metadata"))
            continue
        group = node(f"{label} ({len(findings)})")
        for finding in findings:
            value = _truncate_text(finding["value"], max_text_length)[0]
            item = node(f"{finding['label']}: {value}")
            item.children.append(node(finding["interpretation"]))
            add_evidence(item, finding)
            group.children.append(item)
        root.children.append(group)
    if summary["cautions"]:
        cautions = node("Points to review")
        for statement in summary["cautions"]:
            item = node(statement["text"])
            add_evidence(item, statement)
            cautions.children.append(item)
        root.children.append(cautions)
    coverage = report["coverage"]
    omitted = coverage["omitted"]
    incomplete = (
        coverage["scan_limited"] or coverage["records_truncated"]
        or coverage["records_included"] < coverage["records_available"]
        or any(value for key, value in omitted.items() if key != "binary_or_large_arrays")
    )
    coverage_node = node(
        f"Coverage{' (LIMITED)' if incomplete else ''}: "
        f"{coverage['records_included']}/{coverage['records_available']} collected records supplied, "
        f"{coverage['ifds_scanned']} TIFF directories, {coverage['metadata_chars']:,} metadata characters"
    )
    if coverage.get("computed_records_available"):
        coverage_node.children.append(node(
            f"Computed calibration context: {coverage['computed_records_included']}/"
            f"{coverage['computed_records_available']} series checks supplied."
        ))
    if coverage["records_truncated"]:
        coverage_node.children.append(node(
            f"{coverage['records_truncated']} supplied records contain value excerpts."
        ))
    if coverage["scan_limited"]:
        coverage_node.children.append(node("Directory or metadata scan was incomplete."))
    for key, value in omitted.items():
        if value:
            coverage_node.children.append(node(f"Omitted {key.replace('_', ' ')}: {value}"))
    coverage_node.children.extend(node(warning) for warning in coverage["warnings"])
    root.children.append(coverage_node)
    root.children.append(node(
        "Values and quotations come from cited records; interpretation may be wrong or incomplete. "
        "No raster pixels inspected. This is not a deidentification or sharing clearance."
    ))
    return root


class TiffInspector:
    """Build a schema-backed, tree-like summary of any TIFF readable by tifffile.

    Detail levels are cumulative:

    * 0: file and series summaries
    * 1: pyramid levels and OME image/channel metadata
    * 2: TIFF pages and frames
    * 3: TIFF tags plus parsed XML/plain-text descriptions
    """

    def __init__(
        self,
        file_path: str | Path,
        *,
        detail: int = _DEFAULT_DETAIL,
        max_text_length: int | None = _DEFAULT_MAX_TEXT_LENGTH,
        _tiff: tifffile.TiffFile | None = None,
    ) -> None:
        if not 0 <= int(detail) <= 3:
            raise ValueError("detail must be between 0 and 3")
        if max_text_length is not None and max_text_length < 1:
            raise ValueError("max_text_length must be positive or None")
        self.file_path = Path(file_path)
        self.detail = int(detail)
        self.max_text_length = max_text_length
        self._tiff = _tiff
        self._report: dict[str, Any] | None = None

    @classmethod
    def from_tiff(
        cls,
        tiff: tifffile.TiffFile,
        *,
        file_path: str | Path,
        detail: int = _DEFAULT_DETAIL,
        max_text_length: int | None = _DEFAULT_MAX_TEXT_LENGTH,
    ) -> "TiffInspector":
        """Create an inspector that reuses an already-open ``TiffFile``."""

        return cls(
            file_path,
            detail=detail,
            max_text_length=max_text_length,
            _tiff=tiff,
        )

    @property
    def report(self) -> dict[str, Any]:
        if self._report is None:
            if self._tiff is not None:
                self._report = self._build_report(self._tiff)
            else:
                with tifffile.TiffFile(self.file_path, _multifile=False) as tiff:
                    self._report = self._build_report(tiff)
        return self._report

    def _build_report(self, tiff: tifffile.TiffFile) -> dict[str, Any]:
        warnings: list[str] = []
        ome = None
        if bool(getattr(tiff, "is_ome", False)):
            xml = tiff.ome_metadata
            if xml:
                try:
                    xml_text = (
                        xml.decode("utf-8", errors="replace")
                        if isinstance(xml, bytes)
                        else str(xml)
                    )
                    ome = _ome_summary(xml_text, self.max_text_length)
                except Exception as exc:
                    warnings.append(
                        f"OME-TIFF was detected, but its OME-XML could not be summarized: "
                        f"{type(exc).__name__}: {exc}"
                    )
            else:
                warnings.append("OME-TIFF was detected, but no OME-XML metadata was available")

        try:
            series_collection = list(tiff.series)
        except Exception as exc:
            warnings.append(
                f"Unable to enumerate TIFF series: {type(exc).__name__}: {exc}"
            )
            series_collection = []

        # Some tifffile flags are established lazily while series are parsed.
        # ``uniform`` is an internal cache/layout observation rather than a
        # file-format identity, so omit it to keep CLI and open-reader reports
        # stable regardless of which properties were accessed first.
        flags = sorted(
            str(item)
            for item in getattr(tiff, "flags", set())
            if str(item) != "uniform"
        )
        from omeify._calibration import inspect_calibration

        calibration = inspect_calibration(tiff, series_collection)
        ome_images = ome["images"] if ome is not None else []
        matched_images = [item["ome_image_index"] for item in calibration["series"]]
        series = [
            _series_summary(
                item,
                series_index=index,
                detail=self.detail,
                max_text_length=self.max_text_length,
                ome_image=(
                    ome_images[matched_images[index]]
                    if matched_images[index] is not None
                    and matched_images[index] < len(ome_images) else None
                ),
                warnings=warnings,
            )
            for index, item in enumerate(series_collection)
        ]

        try:
            size_bytes = self.file_path.stat().st_size
        except OSError:
            size_bytes = _safe_int(getattr(tiff.filehandle, "size", None))

        return {
            "schema": _INSPECTION_SCHEMA_RESOURCE,
            "schema_version": _INSPECTION_SCHEMA_VERSION,
            "detail": self.detail,
            "file": {
                "path": str(self.file_path),
                "name": self.file_path.name,
                "format": _format_name(tiff, flags),
                "size_bytes": size_bytes,
                "byte_order": "big" if tiff.byteorder == ">" else "little",
                "is_bigtiff": bool(tiff.is_bigtiff),
                "is_ome": bool(tiff.is_ome),
                "flags": flags,
                "series_count": len(series_collection),
                "top_level_ifd_count": len(tiff.pages),
            },
            "ome": ome,
            "series": series,
            "calibration": calibration,
            "warnings": warnings,
        }

    def summarize_metadata(
        self,
        *,
        registry: Registry | None = None,
        source_name: str | None = None,
        model_name: str | None = None,
        allowed_scopes: Collection[str] = DEFAULT_ALLOWED_SCOPES,
        max_metadata_chars: int = DEFAULT_MAX_METADATA_CHARS,
        max_output_tokens: int = 4096,
    ) -> dict[str, Any]:
        """Explicitly infer and attach one validated metadata summary.

        Collection ignores display detail/preview limits and never decodes raster
        pixels. Configuration, credentials, capabilities, and request lifetime
        belong to Sheetbend. A call makes one inference request; later rendering
        or serialization reuses the attached result without further inference.
        Failures raise and leave the deterministic report intact.

        A supplied open TIFF remains caller-owned. For a path-backed inspector,
        both inspection and metadata collection happen on one open file before
        inference; external OME companion files are not opened.
        """

        from omeify.intelligence import collect_metadata, summarize_metadata

        if self._tiff is not None:
            self._report = self._build_report(self._tiff)
            packet = collect_metadata(
                self._tiff, max_chars=max_metadata_chars, calibration=self._report["calibration"],
            )
        else:
            with tifffile.TiffFile(self.file_path, _multifile=False) as tiff:
                # Refresh local diagnostics with the same handle used to gather
                # evidence, rather than attaching a summary to stale file data.
                self._report = self._build_report(tiff)
                packet = collect_metadata(
                    tiff, max_chars=max_metadata_chars, calibration=self._report["calibration"],
                )
        summary = summarize_metadata(
            packet, registry=registry, source_name=source_name, model_name=model_name,
            allowed_scopes=allowed_scopes, max_output_tokens=max_output_tokens,
        )
        self._report["intelligence"] = summary
        return summary

    def validation_errors(self) -> tuple[str, ...]:
        """Return JSON Schema validation errors for the generated report."""

        errors = sorted(
            _inspection_validator().iter_errors(self.report),
            key=lambda item: tuple(str(part) for part in item.path),
        )
        rendered: list[str] = []
        for error in errors:
            path = "$"
            for part in error.path:
                path += f"[{part}]" if isinstance(part, int) else f".{part}"
            rendered.append(f"{path}: {error.message}")
        return tuple(rendered)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.report, indent=indent, ensure_ascii=False)

    def render_text(self) -> str:
        report = self.report
        file_info = report["file"]
        root = _TreeNode(file_info["name"])
        container = "BigTIFF" if file_info["is_bigtiff"] else "classic TIFF"
        flags = f"; flags: {', '.join(file_info['flags'])}" if file_info["flags"] else ""
        root.children.append(
            _TreeNode(
                f"{file_info['format']}: {container}, {file_info['byte_order']}-endian, "
                f"{_human_bytes(file_info['size_bytes'])}, "
                f"{file_info['series_count']} series, "
                f"{file_info['top_level_ifd_count']} top-level IFDs{flags}"
            )
        )

        ome = report["ome"]
        if ome is not None:
            ome_bits = [f"OME {ome['version'] or 'unknown'}", f"{ome['image_count']} image(s)"]
            if ome.get("creator"):
                ome_bits.append(f"Creator={ome['creator']}")
            if ome.get("uuid"):
                ome_bits.append(f"UUID={ome['uuid']}")
            ome_node = _TreeNode(", ".join(ome_bits))
            miti = ome["miti"]
            miti_node = _TreeNode(
                f"MITI header profile: {str(miti['status']).upper()}"
            )
            if self.detail >= 1:
                if miti["missing_fields"]:
                    missing_node = _TreeNode(
                        f"Missing fields ({len(miti['missing_fields'])})"
                    )
                    missing_node.children.extend(
                        _TreeNode(item) for item in miti["missing_fields"]
                    )
                    miti_node.children.append(missing_node)
                if miti["errors"]:
                    errors_node = _TreeNode(f"Validation findings ({len(miti['errors'])})")
                    errors_node.children.extend(_TreeNode(item) for item in miti["errors"])
                    miti_node.children.append(errors_node)
                if miti["extra_metadata"]:
                    extra_node = _TreeNode(
                        f"Additional OME metadata ({len(miti['extra_metadata'])})"
                    )
                    extra_node.children.extend(
                        _TreeNode(item) for item in miti["extra_metadata"]
                    )
                    miti_node.children.append(extra_node)
            ome_node.children.append(miti_node)
            if self.detail >= 1:
                for image in ome["images"]:
                    size = image["size"]
                    dimensions = (
                        f"X={size['x']}, Y={size['y']}, Z={size['z']}, "
                        f"C={size['c']}, T={size['t']}"
                    )
                    image_label = image.get("name") or image.get("id") or "unnamed image"
                    image_node = _TreeNode(
                        f"Image {image['index']}: {image_label}; "
                        f"{image.get('dimension_order') or 'unknown order'}, "
                        f"{image.get('pixel_type') or 'unknown dtype'}; {dimensions}"
                    )
                    physical = _format_physical_size(image.get("physical_size"))
                    if physical:
                        image_node.children.append(_TreeNode(f"Physical pixel size: {physical}"))
                    channel_labels = _channel_labels(image)
                    if channel_labels:
                        image_node.children.append(
                            _TreeNode(
                                f"OME channels ({len(channel_labels)}): "
                                + ", ".join(channel_labels)
                            )
                        )
                    ome_node.children.append(image_node)
            root.children.append(ome_node)

        for series in report["series"]:
            name = f" {series['name']!r}" if series.get("name") else ""
            pyramid = (
                f", {series['level_count']} levels"
                if series["is_pyramidal"]
                else ", 1 level"
            )
            series_node = _TreeNode(
                f"Series {series['index']}{name}: {series['axes']} {series['dtype']} "
                f"({_format_shape(series['shape'])}){pyramid}"
            )
            if series["channel_names"]:
                series_node.children.append(
                    _TreeNode(
                        f"Channels ({len(series['channel_names'])}): "
                        + ", ".join(series["channel_names"])
                    )
                )
            physical = _format_physical_size(series.get("physical_size"))
            if physical:
                series_node.children.append(_TreeNode(f"Physical pixel size: {physical}"))
            tiff_resolution = _format_physical_size(
                series.get("tiff_resolution_pixel_size")
            )
            series_node.children.append(
                _TreeNode(
                    "TIFF resolution pixel size: "
                    + (tiff_resolution if tiff_resolution else "N/A")
                )
            )

            calibration = report["calibration"]["series"][series["index"]]
            series_node.children.append(_calibration_tree(calibration, detail=self.detail))

            if self.detail >= 1:
                for level in series["levels"]:
                    level_node = _TreeNode(
                        f"Level {level['index']}: {level['axes']} {level['dtype']} "
                        f"({_format_shape(level['shape'])}), "
                        f"{level['page_count']} page/frame entries"
                    )
                    if self.detail >= 2:
                        for page in level["pages"]:
                            if page["tile_shape"]:
                                layout = (
                                    f"tiled {page['tile_shape'][0]}×{page['tile_shape'][1]}"
                                )
                            elif page["rows_per_strip"] is not None:
                                layout = f"strips ({page['rows_per_strip']} rows)"
                            else:
                                layout = "stripped"
                            page_node = _TreeNode(
                                f"Page {page['index']}: {page['axes']} {page['dtype']} "
                                f"({_format_shape(page['shape'])}); {layout}; "
                                f"{page['compression']}; {page['photometric']}; "
                                f"SamplesPerPixel={page['samples_per_pixel']}; "
                                f"stored={_human_bytes(page['stored_bytes'])}"
                            )
                            for frame in page["frames"]:
                                page_node.children.append(
                                    _TreeNode(
                                        f"Frame {frame['index']}: {frame['axes']} {frame['dtype']} "
                                        f"({_format_shape(frame['shape'])})"
                                    )
                                )
                            if self.detail >= 3:
                                if page["description"] is not None:
                                    description = page["description"]
                                    page_node.children.append(
                                        _TreeNode(
                                            f"ImageDescription: {description['format']} "
                                            f"({description['length']} characters)"
                                        )
                                    )
                                tags_node = _TreeNode(f"TIFF tags ({len(page['tags'])})")
                                for tag in page["tags"]:
                                    tags_node.children.append(
                                        _TreeNode(
                                            f"{tag['code']} {tag['name']} "
                                            f"[{tag['dtype']} × {tag['count']}] = {tag['value']}"
                                        )
                                    )
                                page_node.children.append(tags_node)
                            level_node.children.append(page_node)
                    series_node.children.append(level_node)
            root.children.append(series_node)

        if report["warnings"]:
            warning_node = _TreeNode(f"Warnings ({len(report['warnings'])})")
            warning_node.children.extend(_TreeNode(item) for item in report["warnings"])
            root.children.append(warning_node)
        if "intelligence" in report:
            root.children.append(_intelligence_tree(report["intelligence"], self.max_text_length))
        return _render_tree(root)

    def __str__(self) -> str:
        return self.render_text()

    def _repr_html_(self) -> str:
        return f"<pre>{html.escape(self.render_text())}</pre>"

    def __repr__(self) -> str:
        return self.render_text()
