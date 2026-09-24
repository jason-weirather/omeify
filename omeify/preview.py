"""Bounded, display-only whole-image overviews for explicit visual intelligence."""
from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from .io.base import Image
from .io.image_metadata import integer
from .progress import ProgressLogger

LOGGER = logging.getLogger(__name__)
DEFAULT_PREVIEW_SIZE = 1536
DEFAULT_PREVIEW_QUANTILE = 0.999
_READ_TILE = 1024
_COLOR_NAMES = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "yellow": (255, 255, 0),
    "orange": (255, 165, 0),
    "purple": (160, 32, 240),
    "white": (255, 255, 255),
    "gray": (160, 160, 160),
    "grey": (160, 160, 160),
}


def _validate_range(display_range: tuple[float, float] | None) -> tuple[float, float] | None:
    if display_range is None:
        return None
    if len(display_range) != 2 or any(
        isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.number))
        or not math.isfinite(v) for v in display_range
    ) or display_range[0] >= display_range[1]:
        raise ValueError("display_range must contain finite LOW < HIGH")
    if not math.isfinite(float(display_range[1]) - float(display_range[0])):
        raise ValueError("display_range span is too large")
    return float(display_range[0]), float(display_range[1])


def _validate_quantile(clip_quantile: float) -> float:
    if (
        isinstance(clip_quantile, (bool, np.bool_))
        or not isinstance(clip_quantile, (int, float, np.number))
        or not math.isfinite(clip_quantile)
    ):
        raise ValueError("clip_quantile must be a finite float in the interval (0, 1]")
    clip_quantile = float(clip_quantile)
    if clip_quantile <= 0 or clip_quantile > 1:
        raise ValueError("clip_quantile must be in the interval (0, 1]")
    return clip_quantile


def _normalize_color(color: str | Sequence[int]) -> tuple[tuple[int, int, int], str]:
    if isinstance(color, str):
        token = color.strip()
        key = token.casefold()
        if key in _COLOR_NAMES:
            return _COLOR_NAMES[key], token
        hex_value = token[1:] if token.startswith("#") else token
        if len(hex_value) == 6 and all(c in "0123456789abcdefABCDEF" for c in hex_value):
            return tuple(int(hex_value[i:i + 2], 16) for i in (0, 2, 4)), token
        parts = [part.strip() for part in token.split(",")]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            rgb = tuple(integer(part, "color component", minimum=0, maximum=255) for part in parts)
            return rgb, token
        raise ValueError(
            "Preview color must be a named color, #RRGGBB hex, or R,G,B integers",
        )
    if not isinstance(color, Sequence) or len(color) != 3:
        raise TypeError("Preview color must be a string or a 3-element integer sequence")
    rgb = tuple(integer(value, "color component", minimum=0, maximum=255) for value in color)
    return rgb, ",".join(str(v) for v in rgb)


def _normalize_composite_channels(
    composite_channels: Sequence[tuple[int, str | Sequence[int]]] | None,
) -> list[tuple[int, tuple[int, int, int], str]] | None:
    if composite_channels is None:
        return None
    normalized = []
    seen = set()
    for index, color in composite_channels:
        channel = integer(index, "channel")
        if channel in seen:
            raise ValueError("Each preview composite channel may appear only once")
        seen.add(channel)
        rgb, color_text = _normalize_color(color)
        normalized.append((channel, rgb, color_text))
    if not normalized:
        raise ValueError("At least one preview composite channel is required")
    return normalized


def _auto_display_range(values: np.ndarray, valid: np.ndarray, clip_quantile: float) -> tuple[float, float]:
    low = float(np.min(values[valid]))
    high = float(np.quantile(values[valid], clip_quantile)) if clip_quantile < 1 else float(np.max(values[valid]))
    if high <= low:
        high = float(np.max(values[valid]))
    return low, high


def build_preview(
    image: Image, *, max_size: int = DEFAULT_PREVIEW_SIZE,
    channel: int | None = None,
    composite_channels: Sequence[tuple[int, str | Sequence[int]]] | None = None,
    display_range: tuple[float, float] | None = None,
    clip_quantile: float = DEFAULT_PREVIEW_QUANTILE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a uint8 overview and the exact canvas-to-base coordinate contract.

    The canvas covers [0,W] x [0,H], preserving orientation and aspect ratio to
    integer rounding. Source pixels are assigned to canvas bins by their centers
    using the advertised sampling scale, NOT rounded pyramid dimension ratios.
    A suitable calibrated pyramid is preferred. Without one, scan the selected
    base channel(s) in bounded blocks. No full slide array is ever requested.

    Binned means and scalar/composite contrast mapping are DISPLAY ONLY, not
    quantitative normalization. Returned pixels own their storage and outlive
    the input session.
    """
    max_size = integer(max_size, "max_size", minimum=128)
    if max_size > 2048:
        raise ValueError("max_size must be at most 2048")
    clip_quantile = _validate_quantile(clip_quantile)
    display_range = _validate_range(display_range)
    if image.image_type == "label":
        raise ValueError("Visual localization requires intensity or RGB, not categorical labels")
    rgb = image.image_type == "rgb"
    if channel is not None and composite_channels is not None:
        raise ValueError("Specify either one preview channel or a composite, not both")
    composite = _normalize_composite_channels(composite_channels)
    if rgb and (channel is not None or composite is not None or display_range is not None):
        raise ValueError(
            "RGB previews use their original colors; channel/range/composite overrides are scalar-only",
        )
    selection = "explicit"
    requested_channels: tuple[int, ...]
    if composite is not None:
        requested_channels = tuple(index for index, _, _ in composite)
        selection = "explicit_composite"
    else:
        if channel is None:
            candidates = [i for i, name in enumerate(image.channel_names)
                          if name.strip().casefold() == "dapi"]
            channel = candidates[0] if len(candidates) == 1 else 0
            selection = "rgb" if rgb else "unique_dapi" if len(candidates) == 1 else "channel_zero"
        channel = integer(channel, "channel")
        requested_channels = (channel,)
    if max(requested_channels, default=0) >= image.channel_count:
        raise ValueError("Preview channel is outside the image channel range")
    height, width = image.levels[0].spatial_shape
    factor = min(1., max_size / max(height, width))
    ph, pw = max(1, round(height * factor)), max(1, round(width * factor))
    canvas_scale = (height / ph, width / pw)
    candidates = []
    for descriptor in image.levels:
        scale = descriptor.downsample_yx
        if scale is None or any(a > b for a, b in zip(scale, canvas_scale)):
            continue
        if all((n - 1) * step < full * (1 + 1e-5)
               and n * step >= full * (1 - 1e-5)
               for n, step, full in zip(descriptor.spatial_shape, scale, (height, width))):
            candidates.append(descriptor)
    level = max(candidates, key=lambda item: math.prod(item.downsample_yx))
    lh, lw = level.spatial_shape
    sy, sx = level.downsample_yx
    if level.index == 0 and height * width > 16_777_216:
        LOGGER.warning(
            "No suitable calibrated pyramid: scanning the selected base channel in blocks",
        )
    mode = "rgb" if rgb else "composite" if composite is not None else "scalar"
    if mode == "rgb":
        sums = np.zeros((ph, pw, 3), dtype=np.float64)
        counts = np.zeros((ph, pw), dtype=np.uint64)
    elif mode == "composite":
        sums = np.zeros((len(requested_channels), ph, pw), dtype=np.float64)
        counts = np.zeros((len(requested_channels), ph, pw), dtype=np.uint64)
    else:
        sums = np.zeros((ph, pw), dtype=np.float64)
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
                values = image.read_region(
                    y0, y1, x0, x1, level=level.index, channels=list(requested_channels),
                )
                if mode == "rgb":
                    values = values.astype(np.float64)
                    finite = np.ones(values.shape[:2], dtype=bool)
                    with np.errstate(over="raise", invalid="raise"):
                        grouped = np.add.reduceat(np.add.reduceat(values, ix, axis=1), iy, axis=0)
                        sums[np.ix_(uy, ux)] += grouped
                    counts[np.ix_(uy, ux)] += np.add.reduceat(
                        np.add.reduceat(finite.astype(np.uint64), ix, axis=1), iy, axis=0,
                    )
                else:
                    if values.ndim == 2:
                        values = values[..., None]
                    elif image.axes == "CYX":
                        values = np.moveaxis(values, 0, -1)
                    values = values.astype(np.float64)
                    finite = np.isfinite(values)
                    nonfinite += int(finite.size - np.count_nonzero(finite))
                    values[~finite] = 0
                    for i in range(values.shape[-1]):
                        with np.errstate(over="raise", invalid="raise"):
                            grouped = np.add.reduceat(
                                np.add.reduceat(values[..., i], ix, axis=1), iy, axis=0,
                            )
                        grouped_counts = np.add.reduceat(
                            np.add.reduceat(finite[..., i].astype(np.uint64), ix, axis=1), iy, axis=0,
                        )
                        if mode == "composite":
                            sums[i][np.ix_(uy, ux)] += grouped
                            counts[i][np.ix_(uy, ux)] += grouped_counts
                        else:
                            sums[np.ix_(uy, ux)] += grouped
                            counts[np.ix_(uy, ux)] += grouped_counts
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
    if mode == "rgb":
        np.divide(sums, counts[..., None], out=sums, where=valid[..., None])
        pixels = np.clip(np.rint(sums), 0, 255).astype(np.uint8)
        contrast: dict[str, Any] | None = None
        channel_fields = {
            "channel_index": 0,
            "channel_name": image.channel_names[0][:256],
            "channel_name_truncated": len(image.channel_names[0]) > 256,
        }
    elif mode == "scalar":
        np.divide(sums, counts, out=sums, where=valid)
        if display_range is None:
            low, high = _auto_display_range(sums, valid, clip_quantile)
            mode_name = "overview_min_upper_quantile_with_max_fallback"
        else:
            low, high = display_range
            mode_name = "explicit"
        if not math.isfinite(high - low):
            raise ValueError("Overview intensity span is too large for display")
        pixels = np.zeros((ph, pw), dtype=np.uint8)
        if high > low:
            values = (np.clip(sums[valid], low, high) - low) / (high - low)
            pixels[valid] = np.rint(values * 255).astype(np.uint8)
        contrast = {"mode": mode_name, "low": float(low), "high": float(high)}
        if display_range is None:
            contrast["upper_quantile"] = clip_quantile
        index = requested_channels[0]
        channel_fields = {
            "channel_index": index,
            "channel_name": image.channel_names[index][:256],
            "channel_name_truncated": len(image.channel_names[index]) > 256,
        }
    else:
        means = np.zeros_like(sums)
        np.divide(sums, counts, out=means, where=valid)
        pixels = np.zeros((ph, pw, 3), dtype=np.float64)
        channels_context = []
        for i, (index, color_rgb, color_text) in enumerate(composite):
            cvalid = valid[i]
            scaled = np.zeros((ph, pw), dtype=np.float64)
            if np.any(cvalid):
                low, high = _auto_display_range(means[i], cvalid, clip_quantile)
                if high > low:
                    scaled[cvalid] = (
                        np.clip(means[i][cvalid], low, high) - low
                    ) / (high - low)
            else:
                low, high = 0., 1.
            pixels += scaled[..., None] * np.asarray(color_rgb, dtype=np.float64)
            channels_context.append({
                "index": index,
                "name": image.channel_names[index][:256],
                "name_truncated": len(image.channel_names[index]) > 256,
                "color": list(color_rgb),
                "color_spec": color_text,
                "contrast": {
                    "mode": "overview_min_upper_quantile_with_max_fallback",
                    "low": float(low),
                    "high": float(high),
                    "upper_quantile": clip_quantile,
                },
            })
        pixels = np.clip(np.rint(pixels), 0, 255).astype(np.uint8)
        contrast = None
        channel_fields = {"channels": channels_context}
    physical_context = {
        "microns_per_pixel_xy": None,
        "field_of_view_um_xy": None,
    }
    if image.pixel_size is not None:
        size_um = image.pixel_size.converted_to("µm")
        physical_context = {
            "microns_per_pixel_xy": [size_um.x, size_um.y],
            "field_of_view_um_xy": [width * size_um.x, height * size_um.y],
        }
    context = {
        "width": pw, "height": ph, "base_width": width, "base_height": height,
        "base_pixels_per_preview_pixel_xy": [width / pw, height / ph],
        "origin": "top_left", "x_direction": "right", "y_direction": "down",
        "source_level": level.index, "source_downsample_yx": [sy, sx],
        "resampling": "pixel_center_binned_mean", "image_type": image.image_type,
        "channel_selection": selection, "contrast": contrast,
        "clip_quantile": None if rgb or display_range is not None else clip_quantile,
        "nonfinite_source_samples_omitted": nonfinite,
        "empty_preview_bins": int(valid.size - np.count_nonzero(valid)),
        **physical_context,
        **channel_fields,
    }
    return pixels, context
