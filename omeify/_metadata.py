"""Bounded, metadata-only evidence collection for optional inspection summaries.

This is not a format reader or a privacy classifier. TIFF owns storage, lxml
owns XML parsing, and the inference layer owns interpretation. No pixels,
sidecars, schema URLs, or paths found in metadata are opened here.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict, deque
from collections.abc import Iterator
from enum import Enum
from typing import Any

import numpy as np
import tifffile
from lxml import etree

# Bound collection independently of the much smaller inference payload. tifffile
# may have decoded metadata while opening the file; these are not process RSS caps.
_MAX_IFDS = 4096
_MAX_TAG_BYTES = 1_048_576
_MAX_SCAN_BYTES = 8_388_608
_MAX_RECORDS = 20_000
_MAX_VALUE_CHARS = 2048
_MAX_LOCATIONS = 8
_MAX_DEPTH = 32
# Pixel offsets, encoded pixels, lookup tables, profiles, and opaque binary blocks.
_SKIP_TAGS = {273, 279, 288, 289, 320, 324, 325, 330, 347, 513, 514, 34675}
_BINARY_ELEMENTS = {"bindata", "binarydata", "binaryfile", "thumbnail"}
_ENCODING = re.compile(r"(<\?xml\b[^>]*?)\s+encoding=(['\"])[^'\"]+\2", re.I)
_IMPORTANT = re.compile(
    r"date|time|uuid|(?:^|[/@_])(?:id|guid)(?:\[\d+\])?(?:/text\(\))?$|"
    r"identifier|sample|slide|specimen|patient|"
    r"barcode|accession|path|file|directory|user|operator|artist|name|host", re.I,
)
_SCIENTIFIC = re.compile(
    r"channel|biomarker|pixel|physical|resolution|unit|software|creator|scanner|"
    r"instrument|objective|exposure|wavelength|processing|magnification", re.I,
)
_PATH_VALUE = re.compile(r"(?:[A-Za-z]:[\\/]|\\\\|(?<![:\w/])/(?!/)[^\s/]+/)")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _ifds(tiff: tifffile.TiffFile, coverage: dict[str, Any]) -> Iterator[tuple[Any, str]]:
    """Walk physical directories, including real IFDs behind lightweight frames."""

    seen: set[int] = set()

    def walk(pages: Any, prefix: str, depth: int) -> Iterator[tuple[Any, str]]:
        if depth > _MAX_DEPTH:
            coverage["scan_limited"] = True
            return
        try:
            count = len(pages)
        except Exception:
            coverage["scan_limited"] = True
            coverage["warnings"].append(f"Cannot enumerate {prefix} directories.")
            return
        for index in range(count):
            if coverage["ifds_scanned"] >= _MAX_IFDS:
                coverage["scan_limited"] = True
                return
            location = f"{prefix}[{index}]"
            try:
                page = pages[index].aspage()
                offset = int(page.offset)
                if offset in seen:
                    continue
                seen.add(offset)
                coverage["ifds_scanned"] += 1
                yield page, location
                if page.subifds:
                    yield from walk(page.pages, location + "/SubIFD", depth + 1)
            except Exception:
                coverage["scan_limited"] = True
                # Exception text can include paths or arbitrary source content.
                coverage["warnings"].append(f"Cannot read {location} metadata.")

    yield from walk(tiff.pages, "IFD", 0)


def _xml_fields(text: str, coverage: dict[str, Any]) -> Iterator[tuple[str, str]]:
    """Flatten attributes/text, retaining XML paths and OME Image indices.

    XML entity spelling is decoded by lxml, but external and DTD entities are
    never expanded. Unparseable XML remains explicitly unparsed text.
    """

    if not text.lstrip().startswith("<"):
        yield "", text
        return
    parser = etree.XMLParser(
        resolve_entities=False, load_dtd=False, no_network=True, recover=False,
        remove_comments=False, remove_pis=False,
    )
    try:
        root = etree.fromstring(_ENCODING.sub(r"\1", text, count=1).encode(), parser)
    except (etree.XMLSyntaxError, ValueError, UnicodeError):
        # Do not forward possibly encoded pixel blocks from malformed XML.
        if re.search(r"<!DOCTYPE|<(?:\w+:)?(?:BinData|BinaryData|Thumbnail)\b", text, re.I):
            coverage["omitted"]["embedded_binary_or_unsafe_xml"] += 1
        else:
            yield "/unparsed", text
        return
    if root.getroottree().docinfo.doctype:
        coverage["omitted"]["embedded_binary_or_unsafe_xml"] += 1
        return

    def walk(node: Any, path: str, depth: int) -> Iterator[tuple[str, str]]:
        if depth > _MAX_DEPTH:
            coverage["scan_limited"] = True
            return
        name = etree.QName(node).localname
        for key, value in node.attrib.items():
            yield path + "/@" + etree.QName(key).localname, value
        if name.lower() in _BINARY_ELEMENTS:
            coverage["omitted"]["embedded_binary_or_unsafe_xml"] += 1
            return
        value = (node.text or "").strip()
        if value:
            if (
                len(value) > 512 and re.fullmatch(r"[A-Za-z0-9+/=\s]+", value)
                and not re.search(r"\s", value.strip())
            ):
                coverage["omitted"]["embedded_binary_or_unsafe_xml"] += 1
            else:
                yield path + "/text()", value
        counts: Counter[str] = Counter()
        for child in node:
            if not isinstance(child.tag, str):
                if isinstance(child, (etree._Comment, etree._ProcessingInstruction)):
                    text = (child.text or "").strip()
                    if text:
                        yield path + "/comment-or-instruction()", text
                continue
            child_name = etree.QName(child).localname
            position = counts[child_name]
            counts[child_name] += 1
            yield from walk(child, f"{path}/{child_name}[{position}]", depth + 1)
            tail = (child.tail or "").strip()
            if tail:
                yield f"{path}/{child_name}[{position}]/tail()", tail

    yield from walk(root, "/" + etree.QName(root).localname, 0)


def _fields(value: Any, coverage: dict[str, Any], depth: int = 0) -> Iterator[tuple[str, str]]:
    if depth > _MAX_DEPTH:
        coverage["scan_limited"] = True
        return
    if isinstance(value, str):
        if value.lstrip().startswith(("{", "[")):
            try:
                parsed = json.loads(value)
            except (ValueError, RecursionError):
                pass
            else:
                yield from _fields(parsed, coverage, depth + 1)
                return
        yield from _xml_fields(value, coverage)
    elif isinstance(value, dict):
        for key, child in value.items():
            for suffix, text in _fields(child, coverage, depth + 1):
                yield f"/{key}{suffix}", text
    elif isinstance(value, Enum):
        yield "", f"{value.name} ({value.value})"
    elif isinstance(value, (tuple, list, np.ndarray)):
        if len(value) <= 16 and all(isinstance(v, (int, float, np.number)) for v in value):
            values = [v.item() if isinstance(v, np.generic) else v for v in value]
            if all(math.isfinite(v) for v in values):
                yield "", _json(values)
        elif isinstance(value, (list, tuple)) and any(
            isinstance(item, (str, dict)) for item in value
        ):
            for index, child in enumerate(value):
                for suffix, text in _fields(child, coverage, depth + 1):
                    yield f"[{index}]{suffix}", text
        else:
            coverage["omitted"]["binary_or_large_arrays"] += 1
    elif isinstance(value, (int, float, np.number)):
        scalar = value.item() if isinstance(value, np.generic) else value
        if math.isfinite(scalar):
            yield "", _json(scalar)
    else:
        coverage["omitted"]["binary_or_large_arrays"] += 1


def _evidence_fields(
    value: Any, coverage: dict[str, Any],
) -> Iterator[tuple[str, str, bool]]:
    for suffix, text in _fields(value, coverage):
        if len(text) <= _MAX_VALUE_CHARS:
            yield suffix, text, False
            continue
        # Contiguous overlapping excerpts retain late plain-text/JSON findings
        # without constructing fake head/tail strings that could pass quote checks.
        for start in range(0, len(text), _MAX_VALUE_CHARS - 128):
            end = min(len(text), start + _MAX_VALUE_CHARS)
            yield f"{suffix}/characters[{start}:{end}]", text[start:end], True
            if end == len(text):
                break


def _priority(record: dict[str, Any]) -> int:
    location = record["locations"][0]
    if _IMPORTANT.search(location) or _PATH_VALUE.search(record["value"]):
        return 0
    if _SCIENTIFIC.search(location):
        return 1
    return 2


def collect_metadata(
    tiff: tifffile.TiffFile, *, max_chars: int = 32_000,
    calibration: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect an inspectable inference packet without reading raster pixels.

    ``max_chars`` caps the serialized records array, including JSON escaping and
    record locations. It is a character budget, not a token/context guarantee.
    Repeated field/value pairs share evidence records. Bounded XML leaf extraction
    precedes prioritization so identifiers late in a header are not discarded
    merely because the ordinary inspection preview is short. Per-IFD round-robin
    selection avoids spending the entire budget on the first description.

    ``calibration=`` can supply the same deterministic calibration report built
    by TiffInspector. Its bounded numeric summaries become explicitly computed
    evidence records, never raw TIFF metadata. At most one quarter of max_chars
    is reserved for these records, with mismatches first and omissions counted.
    Without this argument the packet contains raw metadata only; this function
    never enumerates series or opens OME companion files to obtain context.

    This function is offline and requires no intelligence dependencies. The
    packet includes identifying source metadata and must be handled accordingly.
    """

    if isinstance(max_chars, bool) or not isinstance(max_chars, int) or max_chars < 4096:
        raise ValueError("max_chars must be an integer of at least 4096")
    coverage: dict[str, Any] = {
        "ifds_scanned": 0, "tags_scanned": 0, "records_available": 0,
        "records_included": 0, "records_truncated": 0, "metadata_chars": 0,
        "max_metadata_chars": max_chars, "scan_limited": False,
        "omitted": {
            "binary_or_large_arrays": 0, "oversized_tags": 0, "unreadable_tags": 0,
            "embedded_binary_or_unsafe_xml": 0,
        },
        "warnings": [],
    }
    records: dict[tuple[str, str], dict[str, Any]] = {}
    scanned_bytes = 0
    for page, location in _ifds(tiff, coverage):
        for tag in page.tags:
            coverage["tags_scanned"] += 1
            if int(tag.code) in _SKIP_TAGS:
                coverage["omitted"]["binary_or_large_arrays"] += 1
                continue
            size = int(tag.valuebytecount)
            if size > _MAX_TAG_BYTES:
                coverage["omitted"]["oversized_tags"] += 1
                continue
            scanned_bytes += size
            if scanned_bytes > _MAX_SCAN_BYTES or len(records) >= _MAX_RECORDS:
                coverage["scan_limited"] = True
                break
            try:
                value = tag.value
                # XMP is defined as a byte tag containing XML, not raster data.
                if int(tag.code) == 700 and isinstance(value, bytes):
                    value = value.decode("utf-8")
                for suffix, text, truncated in _evidence_fields(value, coverage):
                    if not text:
                        continue
                    field = str(tag.name) + suffix
                    key = (field, text)
                    origin = f"{location}/{field}"
                    if key in records:
                        record = records[key]
                        record["occurrences"] += 1
                        if len(record["locations"]) < _MAX_LOCATIONS:
                            record["locations"].append(origin)
                        continue
                    if len(records) >= _MAX_RECORDS:
                        coverage["scan_limited"] = True
                        break
                    records[key] = {
                        "id": f"m{len(records) + 1}", "locations": [origin],
                        "value": text, "truncated": truncated, "occurrences": 1,
                    }
            except Exception:
                coverage["omitted"]["unreadable_tags"] += 1
        if coverage["scan_limited"] and (
            scanned_bytes > _MAX_SCAN_BYTES or len(records) >= _MAX_RECORDS
        ):
            break

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records.values():
        # Within an OME document, give each Image its own turn. An OME Image
        # index is deliberately not asserted to be the same as a TIFF series.
        origin = record["locations"][0]
        image = re.search(r"/OME/Image\[\d+\]", origin)
        group = origin[:image.end()] if image else origin.split("/", 1)[0]
        groups[group].append(record)
    queues = deque(deque(sorted(group, key=_priority)) for group in groups.values())
    selected: list[dict[str, Any]] = []
    used = 2  # JSON array brackets
    computed = []
    if calibration is not None:
        from ._calibration import calibration_context

        summaries = calibration_context(calibration)
        priority = {"mismatch": 0, "partial": 1, "not_comparable": 2, "consistent": 3}
        for index in sorted(
            range(len(summaries)),
            key=lambda i: priority[calibration["series"][i]["status"]],
        ):
            computed.append({
                "id": f"m{len(records) + index + 1}", "origin": "computed",
                "locations": [f"omeify/calibration/series[{index}]"],
                "value": summaries[index], "truncated": False, "occurrences": 1,
            })
        for record in computed:
            size = len(_json(record)) + (1 if selected else 0)
            if len(record["value"]) <= _MAX_VALUE_CHARS and used + size <= max_chars // 4:
                selected.append(record)
                used += size
    computed_included = len(selected)
    while queues:
        queue = queues.popleft()
        record = queue.popleft()
        size = len(_json(record)) + (1 if selected else 0)
        if used + size <= max_chars:
            selected.append(record)
            used += size
        if queue:
            queues.append(queue)
    coverage.update(
        records_available=len(records) + len(computed), records_included=len(selected),
        records_truncated=sum(record["truncated"] for record in selected),
        metadata_chars=used,
    )
    if calibration is not None:
        coverage["computed_records_available"] = len(computed)
        coverage["computed_records_included"] = computed_included
    if len(coverage["warnings"]) > 20:
        coverage["warnings"] = coverage["warnings"][:20] + ["Further directory warnings omitted."]
    return {"records": selected, "coverage": coverage}
