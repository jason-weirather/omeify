"""Bounded, display-only whole-image overviews for explicit visual intelligence."""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from .io.base import Image
from .io.image_metadata import integer
from .progress import ProgressLogger

LOGGER = logging.getLogger(__name__)
DEFAULT_PREVIEW_SIZE = 1536
_READ_TILE = 1024


def build_preview(
    image: Image, *, max_size: int = DEFAULT_PREVIEW_SIZE,
    channel: int | None = None, display_range: tuple[float, float] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a uint8 overview and the exact canvas-to-base coordinate contract.

    The canvas covers [0,W] x [0,H], preserving orientation and aspect ratio to
    integer rounding. Source pixels are assigned to canvas bins by their centers
    using the advertised sampling scale, NOT rounded pyramid dimension ratios.
    A suitable calibrated pyramid is preferred. Without one, scan the selected
    base channel in bounded blocks. No full slide array is ever requested.

    Binned means and scalar contrast mapping are DISPLAY ONLY, not quantitative
    normalization. Returned pixels own their storage and outlive the input session.
    """
    max_size = integer(max_size, "max_size", minimum=128)
    if max_size > 2048:
        raise ValueError("max_size must be at most 2048")
    if image.image_type == "label":
        raise ValueError("Visual localization requires intensity or RGB, not categorical labels")
    rgb = image.image_type == "rgb"
    if rgb and (channel is not None or display_range is not None):
        raise ValueError(
            "RGB previews use all three samples; channel/range overrides are scalar-only"
        )
    if rgb and image.dtype != np.dtype("uint8"):
        raise ValueError("RGB previews require uint8 samples")
    selection = "explicit"
    if channel is None:
        candidates = [i for i, name in enumerate(image.channel_names)
                      if name.strip().casefold() == "dapi"]
        channel = candidates[0] if len(candidates) == 1 else 0
        selection = "rgb" if rgb else "unique_dapi" if len(candidates) == 1 else "channel_zero"
    channel = integer(channel, "channel")
    if channel >= image.channel_count:
        raise ValueError("Preview channel is outside the image channel range")
    if display_range is not None:
        if len(display_range) != 2 or any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
            or not math.isfinite(v) for v in display_range
        ) or display_range[0] >= display_range[1]:
            raise ValueError("display_range must contain finite LOW < HIGH")
        if not math.isfinite(float(display_range[1]) - float(display_range[0])):
            raise ValueError("display_range span is too large")
    height, width = image.levels[0].spatial_shape
    factor = min(1., max_size / max(height, width))
    ph, pw = max(1, round(height * factor)), max(1, round(width * factor))
    canvas_scale = (height / ph, width / pw)  # Our defined canvas transform, not source sampling.
    candidates = []
    for descriptor in image.levels:
        scale = descriptor.downsample_yx
        if scale is None or any(a > b for a, b in zip(scale, canvas_scale)):
            continue
        # Only use a level covering the same base extent, within metadata encoding tolerance.
        if all((n - 1) * step < full * (1 + 1e-5)
               and n * step >= full * (1 - 1e-5)
               for n, step, full in zip(descriptor.spatial_shape, scale, (height, width))):
            candidates.append(descriptor)
    level = max(candidates, key=lambda item: math.prod(item.downsample_yx))
    lh, lw = level.spatial_shape
    sy, sx = level.downsample_yx
    if level.index == 0 and height * width > 16_777_216:
        LOGGER.warning(
            "No suitable calibrated pyramid: scanning the selected base channel in blocks"
        )
    sums = np.zeros((ph, pw, 3) if rgb else (ph, pw), dtype=np.float64)
    counts = np.zeros((ph, pw), dtype=np.uint64)
    total = math.ceil(lh / _READ_TILE) * math.ceil(lw / _READ_TILE)
    progress = ProgressLogger(LOGGER, "Building visual overview", total, unit="blocks")
    completed, nonfinite = 0, 0
    try:
        for y0 in range(0, lh, _READ_TILE):
            y1 = min(lh, y0 + _READ_TILE)
            ybins = np.minimum(ph - 1, ((np.arange(y0, y1) + .5) * sy / canvas_scale[0])
                               .astype(np.int64))
            uy, iy = np.unique(ybins, return_index=True)
            for x0 in range(0, lw, _READ_TILE):
                x1 = min(lw, x0 + _READ_TILE)
                xbins = np.minimum(pw - 1, ((np.arange(x0, x1) + .5) * sx / canvas_scale[1])
                                   .astype(np.int64))
                ux, ix = np.unique(xbins, return_index=True)
                values = image.read_region(y0, y1, x0, x1, level=level.index, channels=[channel])
                if image.axes == "CYX":
                    values = values[0]
                values = values.astype(np.float64)
                finite = np.ones(values.shape[:2], dtype=bool) if rgb else np.isfinite(values)
                nonfinite += int(finite.size - np.count_nonzero(finite))
                values[~finite] = 0
                with np.errstate(over="raise", invalid="raise"):
                    grouped = np.add.reduceat(np.add.reduceat(values, ix, axis=1), iy, axis=0)
                    sums[np.ix_(uy, ux)] += grouped
                counts[np.ix_(uy, ux)] += np.add.reduceat(
                    np.add.reduceat(finite.astype(np.uint64), ix, axis=1), iy, axis=0,
                )
                completed += 1
                progress.update(completed)
    except FloatingPointError as exc:
        raise ValueError("Source values overflowed the display-only overview mean") from exc
    finally:
        image.clear_cache()
    progress.finish()
    valid = counts > 0
    if not np.any(valid):
        raise ValueError("No finite image samples are available for a visual overview")
    np.divide(sums, counts[..., None] if rgb else counts, out=sums,
              where=valid[..., None] if rgb else valid)
    contrast: dict[str, Any] | None = None
    if rgb:
        pixels = np.clip(np.rint(sums), 0, 255).astype(np.uint8)
    else:
        if display_range is None:
            low, high = np.percentile(sums[valid], [1., 99.5])
            if high <= low:
                low, high = np.min(sums[valid]), np.max(sums[valid])
            mode = "overview_p1_p99.5_with_minmax_fallback"
        else:
            low, high = display_range
            mode = "explicit"
        low, high = float(low), float(high)
        if not math.isfinite(high - low):
            raise ValueError("Overview intensity span is too large for display")
        pixels = np.zeros((ph, pw), dtype=np.uint8)
        if high > low:
            # Clip before subtracting to avoid huge finite outliers overflowing the scale.
            values = (np.clip(sums[valid], low, high) - low) / (high - low)
            pixels[valid] = np.rint(values * 255).astype(np.uint8)
        contrast = {"mode": mode, "low": low, "high": high}
    context = {
        "width": pw, "height": ph, "base_width": width, "base_height": height,
        "base_pixels_per_preview_pixel_xy": [width / pw, height / ph],
        "origin": "top_left", "x_direction": "right", "y_direction": "down",
        "source_level": level.index, "source_downsample_yx": [sy, sx],
        "resampling": "pixel_center_binned_mean", "image_type": image.image_type,
        "channel_index": channel, "channel_name": image.channel_names[channel][:256],
        "channel_name_truncated": len(image.channel_names[channel]) > 256,
        "channel_selection": selection, "contrast": contrast,
        "nonfinite_source_samples_omitted": nonfinite,
        "empty_preview_bins": int(valid.size - np.count_nonzero(valid)),
    }
    return pixels, context
