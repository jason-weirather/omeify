"""Allowlisted intrinsic Z/T origins, deliberately separate from stage positions."""
from __future__ import annotations

import json
import math

from lxml import etree

from omeify.reports import validate_report

from .ome_geometry import _tag, _unique_keys

NAMESPACE = "https://github.com/jason-weirather/omeify/ns/stack-geometry/1"


def stack_record(spec):
    values = {key: getattr(spec, key, None) for key in ("z_origin_um", "time_origin_s")}
    values = {key: value for key, value in values.items() if value is not None}
    if not values:
        return None
    record = {"schema": "omeify.stack_geometry/1", **values}
    _validate(record)
    return record


def _validate(record):
    validate_report(record, "stack_geometry.schema.json")
    if any(not math.isfinite(value) for key, value in record.items() if key != "schema"):
        raise ValueError("Stack-coordinate origins must be finite")


def attach_stack_geometry(root, images, specs):
    for i, (image, spec) in enumerate(zip(images, specs, strict=True)):
        record = stack_record(spec)
        if record is None:
            continue
        structured = root.find(_tag(root, "StructuredAnnotations"))
        if structured is None:
            structured = etree.SubElement(root, _tag(root, "StructuredAnnotations"))
        identifier = f"Annotation:StackGeometry:{i}"
        annotation = etree.SubElement(structured, _tag(root, "MapAnnotation"),
                                      ID=identifier, Namespace=NAMESPACE)
        entry = etree.SubElement(etree.SubElement(annotation, _tag(root, "Value")),
                                 _tag(root, "M"), K="geometry")
        entry.text = json.dumps(record, allow_nan=False, separators=(",", ":"))
        etree.SubElement(image, _tag(root, "AnnotationRef"), ID=identifier)


def read_stack_geometry(root, image):
    refs = {node.get("ID") for node in image.findall(_tag(root, "AnnotationRef"))}
    matches = [node for node in root.findall(
        f"{_tag(root, 'StructuredAnnotations')}/{_tag(root, 'MapAnnotation')}")
        if node.get("ID") in refs and node.get("Namespace") == NAMESPACE]
    if not matches:
        return None
    if len(matches) != 1:
        raise ValueError("Ambiguous stack-coordinate annotations")
    items = matches[0].findall(f"{_tag(root, 'Value')}/{_tag(root, 'M')}")
    if len(items) != 1 or items[0].get("K") != "geometry":
        raise ValueError("Malformed stack-coordinate annotation")
    text = items[0].text or ""
    if len(text.encode("utf-8")) > 4096:
        raise ValueError("Stack-coordinate annotation exceeds its bound")
    record = json.loads(text, object_pairs_hook=_unique_keys)
    _validate(record)
    return record
