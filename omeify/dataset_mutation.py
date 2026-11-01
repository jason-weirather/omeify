"""Whole-selection mutation planning with one fixed mapping per logical channel.

The private scan raster concatenates rows only to reuse the existing bounded
statistics implementation. It is never exposed as an Image or physical geometry.
"""
from __future__ import annotations

import math
from dataclasses import replace

import numpy as np

from .dtype_mutation import (
    DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    DTypeChannelPlan,
    RANGE_MODES,
    _mapping_for_scan,
    _scan_plane,
    _target_dtype,
)
from .io.dataset import ImageDataset
from .io.dataset_views import DatasetMetadataSource
from .io.image_metadata import integer


class _ChannelStackScan:
    def __init__(self, dataset, channel):
        dataset._ensure_open()
        self.dataset, self.generation = dataset, dataset.generation
        self.channel = channel
        self.zs = range(dataset.axis_info("Z").size) if dataset.axis_info("Z") else (None,)
        self.ts = range(dataset.axis_info("T").size) if dataset.axis_info("T") else (None,)
        self.plane_height, self.width = dataset.levels[0].plane_metadata.levels[0].spatial_shape
        self.height = len(self.zs) * len(self.ts) * self.plane_height
        self.dtype, self.samples_per_pixel = dataset.dtype, 1

    def read_region(self, y0, y1, x0, x1):
        self.dataset._ensure_session(self.generation)
        out = np.empty((y1-y0, x1-x0), dtype=self.dtype)
        cursor = y0
        while cursor < y1:
            plane_index, row = divmod(cursor, self.plane_height)
            ti, zi = divmod(plane_index, len(self.zs))
            stop = min(y1, cursor + self.plane_height-row)
            with self.dataset.plane(t=self.ts[ti], z=self.zs[zi]) as plane:
                out[cursor-y0:stop-y0] = plane[self.channel].read_region(
                    row, row+stop-cursor, x0, x1)
            cursor = stop
        return out

    def clear_cache(self):
        # Statistics do not own the dataset's shared storage cache.
        pass


def analyze_dataset_dtype_mutation(dataset, *, dtype, range_mode="auto",
        sample_pixels_per_channel=DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
        auto_max_normalized_rmse=DEFAULT_AUTO_MAX_NORMALIZED_RMSE):
    """Scan the selected T/Z extent; keep one fixed mapping per logical channel."""
    if not isinstance(dataset, ImageDataset):
        raise TypeError("dataset must be an open ImageDataset")
    dataset._ensure_open()
    if dataset.image_type != "multichannel" or dataset.dtype.kind != "f":
        raise TypeError("Dataset dtype mutation requires scalar floating-point intensity data")
    if range_mode not in RANGE_MODES:
        raise ValueError("range_mode must be auto, preserve, or full")
    samples = integer(sample_pixels_per_channel, "sample_pixels_per_channel", minimum=1024)
    loss = float(auto_max_normalized_rmse)
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("auto_max_normalized_rmse must be finite and nonnegative")
    target = _target_dtype(dtype)
    plans = []
    for index, name in enumerate(dataset.channel_names):
        scan = _scan_plane(_ChannelStackScan(dataset, index), channel_index=index,
                           channel_name=name, sample_pixels=samples,
                           progress_label=f"Scanning stack mutation channel {index+1}",
                           sample_layout="linear")
        plan = _mapping_for_scan(scan, source_dtype=dataset.dtype, target_dtype=target,
                                range_mode=range_mode, auto_max_normalized_rmse=loss)
        plans.append(plan)
    return tuple(plans)


class _QuantizedDatasetSource(DatasetMetadataSource):
    def __init__(self, parent, plans):
        super().__init__(parent)
        self.plans = tuple(plans)
        if (parent.image_type != "multichannel" or not self.plans or
                len(self.plans) != len(parent.channel_names) or
                any(not isinstance(p, DTypeChannelPlan) or p.channel_index != i or
                    p.source_dtype != parent.dtype or p.channel_name != parent.channel_names[i]
                    for i, p in enumerate(self.plans))):
            raise ValueError("Mutation plans do not match dataset channel identity or dtype")
        dtype = self.plans[0].target_dtype
        if any(p.target_dtype != dtype for p in self.plans):
            raise ValueError("All mutation plans must use the same output dtype")
        # A transformed dataset has its own native-level metadata, not a second
        # ImageSource-style descriptor. Its pixels are still read lazily from the
        # parent and quantized with one fixed mapping across the selected stack.
        metadata = self._dataset_levels[0].plane_metadata
        base = replace(metadata.levels[0], index=0)
        transformed = replace(metadata, dtype=dtype, levels=(base,))
        self.plane_level_indices = (0,)
        self._dataset_levels = (replace(self._dataset_levels[0], plane_metadata=transformed),)

    def read_plane_region(self, *bounds, z, t, level, channels, speculative=False):
        data = super().read_plane_region(*bounds, z=z, t=t, level=level, channels=channels,
                                         speculative=speculative)
        if self._dataset_levels[0].plane_metadata.axes == "YX":
            return self.plans[0].quantize(data)
        return np.stack([self.plans[c].quantize(data[i]) for i, c in enumerate(channels)])


def apply_dataset_dtype_mutation(dataset, plans):
    """Return an unopened borrowed dataset with explicit precomputed channel mappings."""
    return ImageDataset(_QuantizedDatasetSource(dataset, plans))
