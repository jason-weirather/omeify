"""Explicit multidimensional arrays as source sessions, without changing Image."""
from __future__ import annotations

from threading import RLock

import numpy as np

from .dataset_metadata import AxisInfo, DatasetLevel, canonical_axis, selected_index
from .image_metadata import ImageLevel, ImageMetadata
from .image_source import ImageSource
from .pixel_size import PixelSize


class DatasetArraySource(ImageSource):
    supports_prefetch = True

    def __init__(self, array, *, axes, image_type="multichannel", channel_names=None,
                 pixel_size=None, axis_info=(), copy=False, icc_profile=None,
                 background_label=0):
        if not isinstance(array, np.ndarray) or np.ma.isMaskedArray(array):
            raise TypeError("array must be an unmasked NumPy ndarray")
        if not isinstance(copy, bool):
            raise TypeError("copy must be boolean")
        if (not isinstance(axes, str) or len(axes) != array.ndim or
                len(set(axes)) != len(axes) or not set(axes).issubset("TCZYXS") or
                not {"Y", "X"}.issubset(axes) or any(n < 1 for n in array.shape)):
            raise ValueError("Explicit unique T/C/Z/Y/X/S axes must match the array")
        if image_type == "rgb":
            if "S" not in axes or array.shape[axes.index("S")] != 3 or "C" in axes:
                raise ValueError("RGB requires S=3 and one implicit logical channel")
            canonical = "YXS"
        elif image_type in {"multichannel", "label"}:
            if "S" in axes:
                raise ValueError("Scalar arrays cannot contain an RGB sample axis")
            if image_type == "label" and "C" in axes and array.shape[axes.index("C")] != 1:
                raise ValueError("Labels require one logical channel")
            canonical = "CYX" if "C" in axes and image_type != "label" else "YX"
        else:
            raise ValueError("image_type must be multichannel, rgb, or label")
        self.array = np.array(array, copy=True) if copy else array
        self.native_axes = axes
        self._read_lock = RLock()
        supplied = tuple(axis_info)
        if any(not isinstance(a, AxisInfo) for a in supplied) or len({a.name for a in supplied}) != len(supplied):
            raise TypeError("axis_info must contain unique AxisInfo values")
        supplied = {a.name: canonical_axis(a) for a in supplied}
        if not set(supplied).issubset(axes):
            raise ValueError("Axis calibration supplied for an absent axis")
        if pixel_size is not None and not isinstance(pixel_size, PixelSize):
            raise TypeError("pixel_size must be PixelSize")
        size = None if pixel_size is None else pixel_size.converted_to("µm")
        infos = []
        for name, length in zip(axes, array.shape):
            info = supplied.get(name)
            if info is None:
                scale = (size.x if name == "X" else size.y) if size and name in "XY" else None
                info = AxisInfo(name, length, "time" if name == "T" else
                                "channel" if name in "CS" else "space",
                                "µm" if scale else None, scale)
            if info.size != length:
                raise ValueError("Axis calibration size disagrees with array")
            if name in "XY" and size and (info.unit != "µm" or
                    info.scale != (size.x if name == "X" else size.y)):
                raise ValueError("X/Y axis calibration disagrees with pixel_size")
            infos.append(info)
        coordinates = {a.name: a for a in infos}
        if size is None and all(coordinates[a].unit == "µm" and coordinates[a].scale for a in "YX"):
            size = PixelSize(coordinates["X"].scale, coordinates["Y"].scale, "µm")
        origin = tuple(coordinates[a].origin for a in "YX")
        if any(v is None for v in origin) or size is None:
            origin = None
        count = array.shape[axes.index("C")] if canonical == "CYX" else 1
        shape = tuple(array.shape[axes.index(a)] for a in canonical)
        level = ImageLevel(0, canonical, shape, size, origin_yx=origin)
        metadata = ImageMetadata(axes=canonical, shape=shape, dtype=array.dtype,
            image_type=image_type, channel_names=channel_names, pixel_size=size, levels=(level,),
            icc_profile=icc_profile, background_label=background_label)
        if count != metadata.logical_channel_count:
            raise ValueError("Logical channel count mismatch")
        super().__init__(metadata)
        self._dataset_levels = (DatasetLevel(0, tuple(infos), metadata),)

    @property
    def backing_paths(self):
        values, seen = self.array, set()
        while values is not None and id(values) not in seen:
            seen.add(id(values))
            if isinstance(values, np.memmap) and values.filename is not None:
                from pathlib import Path
                return (Path(values.filename),)
            values = getattr(values, "base", None)
        return ()

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._dataset_levels

    def plane_metadata(self, *, z=None, t=None, level=0):
        if level != 0:
            raise IndexError("Array dataset has one native level")
        descriptor = self.dataset_levels[0]
        selected_index(descriptor.axis("Z"), z, name="Z")
        selected_index(descriptor.axis("T"), t, name="T")
        return self.metadata

    def read_plane_region(self, y0, y1, x0, x1, *, z, t, level, channels, speculative=False):
        self.plane_metadata(z=z, t=t, level=level)
        indices = {"Z": z, "T": t, "Y": slice(y0, y1), "X": slice(x0, x1),
                   "S": slice(None)}
        wanted = channels if self.metadata.axes == "CYX" else (0,)
        result = []
        for c in wanted:
            indices["C"] = c
            selection = tuple(indices[a] for a in self.native_axes)
            values = self.array[selection]
            remaining = [a for a, v in zip(self.native_axes, selection) if isinstance(v, slice)]
            axes = "YXS" if self.metadata.image_type == "rgb" else "YX"
            result.append(np.transpose(values, [remaining.index(a) for a in axes]))
        return np.stack(result) if self.metadata.axes == "CYX" else np.array(result[0], copy=True)

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        descriptor = self.dataset_levels[0]
        z = selected_index(descriptor.axis("Z"), None, name="Z")
        t = selected_index(descriptor.axis("T"), None, name="T")
        return self.read_plane_region(y0,y1,x0,x1,z=z,t=t,level=level,channels=channels)

    def cache_info(self):
        return {"max_bytes": 0, "resident_bytes": 0, "entries": 0, "in_flight": 0}
