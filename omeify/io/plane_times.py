"""The one permitted per-plane acquisition measurement in converted datasets.

Times are relative to the source experiment's zero, NOT presentation FPS.
The normalized record is sparse and indexed by logical output (T, C, Z).
"""
from __future__ import annotations

import math

from .dataset_metadata import time_unit_seconds

MAX_DELTA_T_RECORDS = 250_000


def plane_delta_t(dataset):
    """Return sorted (t, channel, z, seconds) records, or an empty tuple.

    Skip traversal when the selected OME source has no DeltaT and when an
    NGFF source has no recognized curated timing extension. This is a
    metadata-only pass and does not decode image planes.
    """
    dataset._ensure_open()
    root = dataset._source
    while hasattr(root, "parent"):
        root = root.parent._source
    mapping = getattr(root, "_plane_map", None)
    if mapping is not None:
        from omeify._calibration import _children
        if not any("DeltaT" in node.attrib for node in _children(mapping.pixels, "Plane")):
            return ()
    elif not (isinstance(getattr(root, "attributes", None), dict)
              and isinstance(root.attributes.get("omeify"), dict)
              and "plane_delta_t" in root.attributes["omeify"]):
        return ()

    zs = dataset.axis_info("Z").size
    ts = dataset.axis_info("T").size
    with dataset.plane(z=0 if zs > 1 else None, t=0 if ts > 1 else None) as image:
        count = image.channel_count
    if zs * ts * count > MAX_DELTA_T_RECORDS:
        raise ValueError("Per-plane DeltaT inspection exceeds the supported metadata budget")
    records = []
    for t in range(ts):
        for c in range(count):
            for z in range(zs):
                measurement = dataset.acquisition_plane(
                    z=z if zs > 1 else None, t=t if ts > 1 else None, channel=c,
                ).get("DeltaT")
                if measurement is None:
                    continue
                value, unit = measurement
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("OME Plane DeltaT must be a finite numeric measurement")
                seconds = float(value) * time_unit_seconds(unit)
                if not math.isfinite(seconds):
                    raise ValueError("OME Plane DeltaT must be finite in seconds")
                records.append((t, c, z, seconds))
    return tuple(records)


def validate_plane_delta_t(records, *, size_t, size_c, size_z):
    """Validate sorted unique logical coordinates and finite relative seconds."""
    if len(records) > MAX_DELTA_T_RECORDS:
        raise ValueError("Too many curated Plane DeltaT measurements")
    validated = []
    seen = set()
    for entry in records:
        if not isinstance(entry, (list, tuple)) or len(entry) != 4:
            raise ValueError("Plane DeltaT entry must be (T,C,Z,seconds)")
        t, c, z, seconds = entry
        for value, size, name in ((t, size_t, "T"), (c, size_c, "C"), (z, size_z, "Z")):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
                raise ValueError(f"Plane DeltaT {name} index is outside the image")
        if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds):
            raise ValueError("Plane DeltaT seconds must be finite")
        key = (t, c, z)
        if key in seen:
            raise ValueError("Duplicate Plane DeltaT coordinate")
        seen.add(key)
        validated.append((t, c, z, float(seconds)))
    validated.sort()
    return tuple(validated)


def verify_tiff_delta_t(path, expected):
    """Check exact declared timing after TIFF encoding, before publication."""
    import tifffile
    from lxml import etree

    with tifffile.TiffFile(path, _multifile=False) as tif:
        raw = tif.ome_metadata
        if not raw:
            raise ValueError("Missing OME metadata when verifying DeltaT")
        root = etree.fromstring(raw.encode("utf-8") if isinstance(raw, str) else raw,
            etree.XMLParser(resolve_entities=False, load_dtd=False, no_network=True))
        images = root.findall("./{*}Image")
        if len(images) != 1:
            raise ValueError("Expected one OME Image when verifying Plane DeltaT")
        pixels = images[0].find("./{*}Pixels")
        observed = []
        for plane in pixels.findall("./{*}Plane"):
            if set(plane.attrib) != {"TheT", "TheC", "TheZ", "DeltaT", "DeltaTUnit"}:
                raise ValueError("Output contains an unexpected OME Plane acquisition field")
            if plane.get("DeltaTUnit") != "s":
                raise ValueError("Output Plane DeltaT unit differs from seconds")
            observed.append((int(plane.get("TheT")), int(plane.get("TheC")),
                             int(plane.get("TheZ")), float(plane.get("DeltaT"))))
        if tuple(sorted(observed)) != tuple(expected):
            raise ValueError("Written OME Plane DeltaT differs from the selected source")
