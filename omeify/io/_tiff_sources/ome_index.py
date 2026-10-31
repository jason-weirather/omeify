"""Operational local OME plane mapping, independent of diagnostic scan limits.

No companion UUID or filename is opened. Logical runs remain compact; IFD
identity, not image ordinal/name/shape, associates a tifffile series with OME.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from math import prod

from omeify._calibration import _children, _local_ifd, _ome_images
from omeify.inspection import _ome_summary


def _number(node, key, default=None):
    raw = node.get(key)
    if raw is None:
        if default is None:
            raise ValueError(f"Missing OME {key}")
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"Negative OME {key}")
    return value


@dataclass(frozen=True)
class PlaneRun:
    start: int
    stop: int
    ifd: int


class PlaneMap:
    def __init__(self, item, *, page_count):
        self.index = item["index"]
        self.pixels = pixels = item["pixels"]
        if pixels is None or item["unresolved"]:
            raise ValueError("incomplete_tiff_data (external or invalid reference)")
        order = pixels.get("DimensionOrder", "")
        if order[:2] != "XY" or len(order) != 5 or set(order[2:]) != set("ZCT"):
            raise ValueError("Unsupported OME DimensionOrder")
        self.order = order[2:]
        channels = _children(pixels, "Channel")
        samples = tuple(_number(c, "SamplesPerPixel", 1) for c in channels)
        size_c = _number(pixels, "SizeC")
        if not samples or any(n not in (1, 3) for n in samples) or sum(samples) != size_c:
            raise ValueError("OME channel sample counts disagree with SizeC")
        self.samples = samples
        self.sizes = {"C": len(channels), "Z": _number(pixels, "SizeZ"),
                      "T": _number(pixels, "SizeT")}
        if any(n < 1 for n in self.sizes.values()):
            raise ValueError("OME dimensions must be positive")
        self.count = prod(self.sizes.values())
        self.strides = {}
        stride = 1
        for axis in self.order:
            self.strides[axis] = stride
            stride *= self.sizes[axis]
        runs = []
        for data in _children(pixels, "TiffData"):
            first = {a: _number(data, "First" + a, 0) for a in "ZCT"}
            start = self.linear(**{a.lower(): v for a, v in first.items()})
            ifd = _number(data, "IFD", 0)
            count = _number(data, "PlaneCount", 1 if "IFD" in data.attrib else
                            min(page_count - ifd, self.count - start))
            if count == 0:
                continue
            if ifd + count > page_count or start + count > self.count:
                raise ValueError("OME TiffData range exceeds image or file")
            runs.append(PlaneRun(start, start + count, ifd))
        runs.sort(key=lambda r: r.start)
        cursor = 0
        for run in runs:
            if run.start != cursor:
                raise ValueError("OME plane mapping has missing or duplicate logical planes")
            cursor = run.stop
        if cursor != self.count:
            raise ValueError("OME plane mapping is incomplete")
        physical = sorted((r.ifd, r.ifd + r.stop - r.start) for r in runs)
        if any(a[1] > b[0] for a, b in zip(physical, physical[1:])):
            raise ValueError("OME plane mapping has overlapping physical IFDs")
        self.runs = tuple(runs)
        self.starts = tuple(r.start for r in runs)

    def linear(self, *, z, c, t):
        values = {"Z": z, "C": c, "T": t}
        if any(not 0 <= values[a] < self.sizes[a] for a in values):
            raise IndexError("OME plane coordinate outside source dimensions")
        return sum(values[a] * self.strides[a] for a in values)

    def physical(self, logical):
        if not 0 <= logical < self.count:
            raise IndexError("OME plane ordinal outside source")
        run = self.runs[bisect_right(self.starts, logical) - 1]
        return run.ifd + logical - run.start

    def ifd(self, *, z, c, t):
        return self.physical(self.linear(z=z, c=c, t=t))


class OMEPlaneIndex:
    def __init__(self, tiff):
        self.tiff = tiff
        self.items, state = _ome_images(tiff)
        if state != "parsed_ome":
            raise ValueError(f"Cannot associate TIFF series: {state}")
        self._summary = _ome_summary(tiff.ome_metadata, None)
        self._mapped = {}

    def map_series(self, series_index):
        if series_index in self._mapped:
            return self._mapped[series_index]
        series = self.tiff.series[series_index]
        if not series.pages or series.pages[0] is None:
            raise ValueError("Cannot associate TIFF series: missing first physical plane")
        first = _local_ifd(self.tiff, series.pages[0])
        candidates = [item for item in self.items if first is not None and
                      any(lo <= first < hi for lo, hi in item["ranges"])]
        if len(candidates) != 1:
            raise ValueError("Cannot associate TIFF series: ambiguous_tiff_data or unmapped_ifd")
        item = candidates[0]
        try:
            mapping = PlaneMap(item, page_count=len(self.tiff.pages))
            if len(series.pages) != mapping.count:
                raise ValueError("logical plane count disagrees with TIFF series")
            # Compare lightweight IFD identities. Do not decode pixels or collect
            # all page objects into a second unbounded list.
            for ordinal, page in enumerate(series.pages):
                if page is None or _local_ifd(self.tiff, page) != mapping.physical(ordinal):
                    raise ValueError("TIFF series disagrees with logical OME plane mapping")
                if any(other is not item and any(lo <= int(page.index) < hi
                       for lo, hi in other["ranges"]) for other in self.items):
                    raise ValueError("ambiguous_tiff_data")
            base = series.pages[0].aspage()
            pixels = item["pixels"]
            if (base.is_reduced or base.imagewidth != _number(pixels, "SizeX") or
                    base.imagelength != _number(pixels, "SizeY")):
                raise ValueError("not_full_resolution_geometry")
        except (ValueError, IndexError, TypeError) as error:
            raise ValueError(f"Cannot associate TIFF series {series_index}: {error}") from error
        self._mapped[series_index] = mapping
        return mapping

    def summary(self, series_index):
        return self._summary["images"][self.map_series(series_index).index]
