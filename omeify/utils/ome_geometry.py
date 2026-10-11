"""A small, linked OME annotation for derived image coordinates, not stage positions.

OME-XML has no general intrinsic-raster transform. This supplemental annotation
preserves the origin of a selected NGFF image through OME-TIFF without pretending
that a local coordinate is an acquisition-stage measurement. Generic OME readers
still see an ordinary calibrated image; readers ignoring the annotation do not
recover its nonlocal origin.
"""
from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import replace

from lxml import etree

NAMESPACE = "https://github.com/jason-weirather/omeify/ns/geometry/1"
RECORD_SCHEMA = "omeify.geometry/1"


def _tag(root, name):
    return f"{{{etree.QName(root).namespace}}}{name}"


def record_for_spec(spec):
    if getattr(spec, "origin_yx", None) is None:
        return None
    return {"schema": RECORD_SCHEMA, "origin_yx_um": list(spec.origin_yx),
            "pyramid_method": spec.pyramid_method}


def attach_geometry(root, images: Sequence, specs: Sequence) -> None:
    for i, (image, spec) in enumerate(zip(images, specs)):
        record = record_for_spec(spec)
        if record is None:
            continue
        structured = root.find(_tag(root, "StructuredAnnotations"))
        if structured is None:
            structured = etree.SubElement(root, _tag(root, "StructuredAnnotations"))
        identifier = f"Annotation:Geometry:{i}"
        annotation = etree.SubElement(structured, _tag(root, "MapAnnotation"),
                                      ID=identifier, Namespace=NAMESPACE)
        value = etree.SubElement(annotation, _tag(root, "Value"))
        entry = etree.SubElement(value, _tag(root, "M"), K="geometry")
        entry.text = json.dumps(record, allow_nan=False, separators=(",", ":"))
        etree.SubElement(image, _tag(root, "AnnotationRef"), ID=identifier)


def read_geometry(root, image):
    references = {n.get("ID") for n in image.findall(_tag(root, "AnnotationRef"))}
    matches = [n for n in root.findall(f"{_tag(root, 'StructuredAnnotations')}/{_tag(root, 'MapAnnotation')}")
               if n.get("ID") in references and n.get("Namespace") == NAMESPACE]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Multiple image-coordinate annotations are ambiguous")
    items = matches[0].findall(f"{_tag(root, 'Value')}/{_tag(root, 'M')}")
    if len(items) != 1 or items[0].get("K") != "geometry":
        raise ValueError("Malformed image-coordinate annotation")
    text = items[0].text or ""
    if len(text) > 4096:
        raise ValueError("Image-coordinate annotation exceeds its bound")
    try:
        record = json.loads(text)
    except ValueError as exc:
        raise ValueError("Malformed image-coordinate JSON") from exc
    if (not isinstance(record, dict) or set(record) != {"schema", "origin_yx_um", "pyramid_method"}
            or record["schema"] != RECORD_SCHEMA or record["pyramid_method"] not in {"mean", "nearest"}):
        raise ValueError("Unsupported image-coordinate annotation")
    origin = record["origin_yx_um"]
    if (not isinstance(origin, list) or len(origin) != 2 or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
            for v in origin)):
        raise ValueError("Image origin must contain two finite micrometer coordinates")
    return record


def calibrated_levels(levels, record):
    """Recover the writer's regular sampling grid, including nearest-neighbor offsets."""
    if record is None:
        return levels
    base = levels[0]
    if base.pixel_size is None:
        raise ValueError("Image coordinates require physical calibration")
    size = base.pixel_size.converted_to("µm")
    origin = tuple(record["origin_yx_um"])
    result = []
    for item in levels:
        if item.downsample_yx is None:
            raise ValueError("Image-coordinate annotation requires calibrated pyramid levels")
        factors = item.downsample_yx
        offset = (0., 0.) if record["pyramid_method"] == "mean" else tuple((1 - s) / 2 for s in factors)
        centers = tuple(origin[j] + (offset[j] + (factors[j] - 1) / 2) * (size.y, size.x)[j]
                        for j in range(2))
        result.append(replace(item, offset_yx=offset, origin_yx=centers))
    return tuple(result)
