"""Pixel-coordinate GeoJSON and rectangular crop planning (not GIS or masking)."""
from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Literal

MAX_GEOJSON_CHARS = 8_388_608
MAX_REGIONS = 4096
ShatterMode = Literal["by_index", "by_name"]


def _constant(value: str) -> None:
    raise ValueError("Non-finite JSON numbers are not supported")


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON keys are not supported")
        result[key] = value
    return result


def load_geojson(text: str) -> dict[str, Any] | list[Any]:
    """Read a bounded JSON document, rejecting duplicate keys and non-finite constants."""
    if not isinstance(text, str) or len(text) > MAX_GEOJSON_CHARS:
        raise ValueError("GeoJSON exceeds the 8 Mi-character input limit")
    try:
        value = json.loads(text, parse_constant=_constant, object_pairs_hook=_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("GeoJSON is not valid, finite, duplicate-free JSON") from exc
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected a GeoJSON object or a list of Features")
    return value


def _number(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("Coordinates must be finite numbers, not booleans or strings")
    try:
        result = float(value)
    except OverflowError as exc:
        raise ValueError("Coordinate is too large") from exc
    if not math.isfinite(result):
        raise ValueError("Coordinates must be finite")
    return result


def pixel_bounds(
    bounds: Sequence[Real], width: int, height: int, *, clip: bool = False,
) -> tuple[int, int, int, int]:
    """Round XYXY outward to integer pixel edges; out-of-bounds is an explicit policy."""
    if len(bounds) != 4:
        raise ValueError("bounds must contain X0 Y0 X1 Y1")
    x0, y0, x1, y1 = map(_number, bounds)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("Region bounds must have positive width and height")
    if not clip and (x0 < 0 or y0 < 0 or x1 > width or y1 > height):
        raise ValueError("Region is outside the image; use --clip / clip=True explicitly")
    x0, y0 = max(0, math.floor(x0)), max(0, math.floor(y0))
    x1, y1 = min(width, math.ceil(x1)), min(height, math.ceil(y1))
    if x0 >= x1 or y0 >= y1:
        raise ValueError("Region does not intersect the image")
    return x0, y0, x1, y1


def rectangle_feature(bounds: Sequence[Real], *, name: str, **properties: Any) -> dict[str, Any]:
    """Construct a closed rectangle; coordinates are XY level-zero pixel edges."""
    x0, y0, x1, y1 = bounds
    return {
        "type": "Feature", "properties": {"name": name, **properties},
        "geometry": {"type": "Polygon", "coordinates": [[
            [x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0],
        ]]},
    }


def _geometry_points(geometry: Any, *, depth: int = 0) -> list[tuple[float, float]]:
    if depth > 16 or not isinstance(geometry, dict) or "crs" in geometry:
        raise ValueError("Expected bounded pixel-coordinate geometry without a CRS")
    kind = geometry.get("type")
    if kind in {"GeometryCollection", "MultiPolygon"}:
        children = geometry.get("geometries" if kind == "GeometryCollection" else "coordinates")
        if not isinstance(children, list) or not children:
            raise ValueError("Empty or invalid geometry collection")
        if kind == "MultiPolygon":
            children = [{"type": "Polygon", "coordinates": c} for c in children]
        return [p for child in children for p in _geometry_points(child, depth=depth + 1)]
    if kind != "Polygon":
        raise ValueError("Cropping supports Polygon, MultiPolygon, and polygon GeometryCollection")
    rings = geometry.get("coordinates")
    if not isinstance(rings, list) or not rings:
        raise ValueError("Polygon requires at least one ring")
    points = []
    for ring in rings:
        if not isinstance(ring, list) or len(ring) < 4:
            raise ValueError("Polygon rings require at least four positions, including closure")
        positions = []
        for p in ring:
            if not isinstance(p, list) or len(p) != 2:
                raise ValueError("Positions must be two-dimensional [x, y] pixel coordinates")
            positions.append((_number(p[0]), _number(p[1])))
        if positions[0] != positions[-1]:
            raise ValueError("Polygon rings must be closed")
        points.extend(positions)
    return points


@dataclass(frozen=True)
class Region:
    index: int  # One-based input order, never a model-assigned identifier.
    name: str | None
    bounds: tuple[float, float, float, float]


def parse_regions(value: Mapping[str, Any] | list[Any]) -> tuple[Region, ...]:
    """Extract envelopes, not masks. A multipart Feature stays one ROI.

    A top-level GeometryCollection or Feature array supplies one ROI per member.
    Only properties.name (or a bare geometry's name) controls grouping, never class labels.
    """
    # Snapshot and bound caller-owned Python data, rejecting cycles and non-JSON values.
    try:
        text = json.dumps(value, allow_nan=False, ensure_ascii=False)
    except (TypeError, ValueError, RecursionError) as exc:
        raise ValueError("GeoJSON must contain finite JSON data") from exc
    value = load_geojson(text)
    if isinstance(value, list):
        items = value
    else:
        if "crs" in value:
            raise ValueError("Geographic CRS declarations are not supported; use level-zero pixels")
        kind = value.get("type")
        items = (value.get("features") if kind == "FeatureCollection" else
                 value.get("geometries") if kind == "GeometryCollection" else [value])
    if not isinstance(items, list) or not 1 <= len(items) <= MAX_REGIONS:
        raise ValueError(f"Expected between 1 and {MAX_REGIONS} regions")
    result = []
    for index, item in enumerate(items, 1):
        if not isinstance(item, dict) or "crs" in item:
            raise ValueError(f"Region {index} is not a pixel-coordinate GeoJSON object")
        props = item.get("properties")
        if props is None:
            props = {}
        if not isinstance(props, dict):
            raise ValueError(f"Region {index} properties must be an object or null")
        name = props.get("name", item.get("name"))
        if name is not None and not isinstance(name, str):
            raise ValueError(f"Region {index} name must be a string")
        name = name.strip() if name else None
        geometry = item.get("geometry") if item.get("type") == "Feature" else item
        points = _geometry_points(geometry)
        xs, ys = zip(*points)
        bounds = (min(xs), min(ys), max(xs), max(ys))
        if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
            raise ValueError(f"Region {index} has an empty bounding rectangle")
        result.append(Region(index, name or None, bounds))
    return tuple(result)


def group_outputs(
    regions: Sequence[Region], output_path: str | Path, *, shatter: ShatterMode | None = None,
) -> list[tuple[Path, tuple[Region, ...]]]:
    """Keep all ROIs in one file unless explicitly shattered by input index or name.

    Without shatter, output_path is the exact filename, even without an extension.
    Shattered outputs insert a suffix before the supplied TIFF extension, or append
    .ome.tiff when no TIFF extension was supplied. Input order is retained within
    every file; name groups follow their first occurrence, never alphabetical order.
    """
    if shatter not in (None, "by_index", "by_name"):
        raise ValueError("shatter must be None, 'by_index', or 'by_name'")
    if not regions:
        raise ValueError("At least one region is required")
    base = Path(output_path)
    if not base.name or base.name in {".", ".."}:
        raise ValueError("output_path must include a filename")
    if base.is_dir():
        raise IsADirectoryError(base)
    if shatter is None:
        return [(base, tuple(regions))]

    extension = ".ome.tiff"
    for suffix in (".ome.tiff", ".ome.tif", ".tiff", ".tif"):
        if base.name.lower().endswith(suffix):
            stem, extension = base.name[:-len(suffix)], base.name[-len(suffix):]
            if not stem or stem in {".", ".."}:
                raise ValueError("Shattered output requires a filename prefix before the TIFF suffix")
            base = base.with_name(stem)
            break
    width = max(2, len(str(len(regions))))
    groups: dict[tuple[str, str | int], list[Region]] = {}
    for region in regions:
        key = ("name", region.name) if shatter == "by_name" and region.name else ("index", region.index)
        groups.setdefault(key, []).append(region)
    used = set()
    outputs = []
    for (kind, key), members in groups.items():
        if kind == "name":
            stem = re.sub(r"[^\w.-]+", "-", unicodedata.normalize("NFKC", str(key))).strip("._-")
            if not stem or len(stem.encode("utf-8")) > 120:
                raise ValueError(
                    "A region name cannot form a safe filename; use --shatter by_index "
                    "or omit shatter"
                )
        else:
            stem = f"{int(key):0{width}d}"
        if stem.casefold() in used:
            raise ValueError(
                "Distinct region names collide as filenames; use --shatter by_index "
                "or omit shatter"
            )
        used.add(stem.casefold())
        outputs.append((base.with_name(f"{base.name}-{stem}{extension}"), tuple(members)))
    return outputs
