"""Bounded, pixel-center polygon masking and optional off-slide crop padding.

GeoJSON coordinates describe level-zero pixel *edges*. Each output pixel is
selected when its center lies within a polygon exterior and outside its holes.
Only the requested writer block is rasterized, never a whole-slide mask.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Literal

import numpy as np

from .io.base import Image, LabelImage, MultichannelImage, RGBImage
from .io.image_metadata import ImageLevel
from .io.image_views import BorrowedSource
from .regions import PolygonGeometry

MaskBackground = Literal["auto", "black", "white", "keep"]


def _ring_mask(
    ring: tuple[tuple[float, float], ...], x0: int, y0: int, width: int, height: int,
) -> np.ndarray:
    """Even-odd scanline fill at pixel centers; half-open crossings exclude vertices.

    The scanline is compared to each edge only where that edge crosses the current
    pixel center. Horizontal edges do not contribute. The leftmost crossing is
    included and the rightmost excluded, ensuring an integer-edged rectangle
    includes exactly its enclosed pixels, without one-pixel halos.
    """
    result = np.zeros((height, width), dtype=bool)
    if not width or not height:
        return result
    vertices = np.asarray(ring, dtype=np.float64)
    xs0, ys0 = vertices[:-1, 0], vertices[:-1, 1]
    xs1, ys1 = vertices[1:, 0], vertices[1:, 1]
    low = np.minimum(ys0, ys1)
    high = np.maximum(ys0, ys1)
    for row in range(height):
        cy = y0 + row + 0.5
        crossed = (low <= cy) & (cy < high)
        if not np.any(crossed):
            continue
        intersections = xs0[crossed] + (cy - ys0[crossed]) * (
            (xs1[crossed] - xs0[crossed]) / (ys1[crossed] - ys0[crossed])
        )
        intersections.sort()
        if len(intersections) % 2:
            # Malformed/self-intersecting geometry must not silently leave pixels
            # outside the intended selection. Nonzero-area polygons produce pairs.
            raise ValueError("Polygon has an unmatched scanline intersection")
        for left, right in intersections.reshape(-1, 2):
            start = max(0, math.ceil(float(left) - x0 - 0.5))
            stop = min(width, math.ceil(float(right) - x0 - 0.5))
            if start < stop:
                result[row, start:stop] = True
    return result


def polygon_mask(
    polygons: PolygonGeometry, x0: int, y0: int, width: int, height: int,
) -> np.ndarray:
    """Union multipolygon exteriors; subtract holes only from their own polygon."""
    inside = np.zeros((height, width), dtype=bool)
    for rings in polygons:
        part = _ring_mask(rings[0], x0, y0, width, height)
        for hole in rings[1:]:
            if np.any(part):
                part &= ~_ring_mask(hole, x0, y0, width, height)
        inside |= part
    return inside


def mask_fill_value(image: Image, mode: MaskBackground) -> int | float:
    """Return a *representable* value; never cast or renormalize image intensities."""
    if mode not in ("auto", "black", "white", "keep"):
        raise ValueError("mask_background must be auto, black, white, or keep")
    if image.image_type == "label":
        if mode == "white":
            raise ValueError("White padding would invent a label ID; use auto/black/background")
        return int(image._metadata.background_label)
    if mode in ("auto", "keep"):
        mode = "white" if image.image_type == "rgb" else "black"
    if mode == "black":
        return 0
    if np.issubdtype(image.dtype, np.bool_):
        return 1
    if np.issubdtype(image.dtype, np.integer):
        return int(np.iinfo(image.dtype).max)
    # For arbitrary quantitative floats there is no universal display-white.
    # One is explicit and reproducible, but not necessarily visually saturated.
    return 1.0


class GeometryCropSource(BorrowedSource):
    """Borrow an image and emit cropped, padded, geometry-masked regional pixels."""

    def __init__(
        self, image: Image, bounds: tuple[int, int, int, int], *,
        polygons: PolygonGeometry, mask: bool, fill_value: int | float,
    ) -> None:
        x0, y0, x1, y1 = bounds
        if x0 >= x1 or y0 >= y1:
            raise ValueError("Crop bounds must be nonempty")
        shape = list(image.shape)
        shape[image.axes.index("Y")] = y1 - y0
        shape[image.axes.index("X")] = x1 - x0
        origin = image.origin_yx
        if origin is not None:
            if image.pixel_size is None:
                raise ValueError("A located crop requires calibrated pixel size")
            size = image.pixel_size.converted_to("µm")
            origin = (origin[0] + y0 * size.y, origin[1] + x0 * size.x)
        level = ImageLevel(0, image.axes, tuple(shape), image.pixel_size,
                           (1., 1.), origin_yx=origin)
        metadata = replace(image._metadata, shape=tuple(shape), levels=(level,))
        super().__init__(metadata, (image,))
        self._image = image
        self._bounds = bounds
        self._polygons = polygons
        self._mask = mask
        self._fill_value = fill_value

    def _read_region(
        self, y0: int, y1: int, x0: int, x1: int, *,
        level: int, channels: tuple[int, ...],
    ) -> np.ndarray:
        height, width = y1 - y0, x1 - x0
        axes = self.metadata.axes
        shape = ((len(channels), height, width) if axes == "CYX"
                 else (height, width, 3) if axes == "YXS" else (height, width))
        out = np.full(shape, self._fill_value, dtype=self.metadata.dtype)
        px0, py0 = self._bounds[:2]
        ox0, oy0 = px0 + x0, py0 + y0
        ox1, oy1 = px0 + x1, py0 + y1
        source_height, source_width = self._image.levels[0].spatial_shape
        sx0, sy0 = max(0, ox0), max(0, oy0)
        sx1, sy1 = min(source_width, ox1), min(source_height, oy1)
        if sx0 < sx1 and sy0 < sy1:
            block = self._image.read_region(sy0, sy1, sx0, sx1, channels=channels)
            dy0, dy1 = sy0 - oy0, sy1 - oy0
            dx0, dx1 = sx0 - ox0, sx1 - ox0
            if axes == "CYX":
                out[:, dy0:dy1, dx0:dx1] = block
            elif axes == "YXS":
                out[dy0:dy1, dx0:dx1, :] = block
            else:
                out[dy0:dy1, dx0:dx1] = block
        if self._mask:
            inside = polygon_mask(self._polygons, ox0, oy0, width, height)
            if axes == "CYX":
                out[:, ~inside] = self._fill_value
            else:
                out[~inside] = self._fill_value
        return out


def geometry_crop(
    image: Image, bounds: tuple[int, int, int, int], *,
    polygons: PolygonGeometry, mask: bool, fill_value: int | float,
) -> Image:
    """Create an unopened semantic image without rasterizing until a writer reads."""
    cls = {"multichannel": MultichannelImage, "rgb": RGBImage, "label": LabelImage}
    return cls[image.image_type](GeometryCropSource(
        image, bounds, polygons=polygons, mask=mask, fill_value=fill_value,
    ))
