"""Multidimensional source sessions that yield ordinary immutable 2D Images.

This is a source/navigation surface, not a replacement Image or a renderer.
Existing Image axes/shapes/read_region/asarray/writer contracts are unchanged.
"""
from __future__ import annotations

from concurrent.futures import CancelledError
from contextlib import contextmanager
from dataclasses import dataclass
from threading import Condition, RLock
from typing import Self
from weakref import WeakSet

import numpy as np

from ._selection import normalize_channel_indices
from .base import _semantic_image
from .dataset_metadata import selected_index
from .image_metadata import integer
from .image_source import ImageSource
from .prefetch import PlaneRequest, PrefetchPolicy, PrefetchQueue


@dataclass(frozen=True, slots=True)
class PlaneSelection:
    z: int | None
    t: int | None
    source_level: int
    session: tuple[int, int]


class _DatasetPlaneSource(ImageSource):
    def __init__(self, dataset, z, t, level, metadata):
        super().__init__(metadata)
        self.dataset = dataset
        self._session = dataset.generation
        self.plane_selection = PlaneSelection(z, t, level, dataset.identity)
        self._level_map = dataset._plane_level_map(level)

    @property
    def is_open(self):
        return (self._is_open and self.dataset.is_open and
                self._session == self.dataset.generation)

    @property
    def backing_paths(self):
        return self.dataset.backing_paths

    def _open(self):
        self.dataset._ensure_session(self._session)

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        self.dataset._ensure_session(self._session)
        selected = self.plane_selection
        return self.dataset._read(PlaneRequest((y0, y1, x0, x1), selected.z, selected.t,
                                               self._level_map[level], channels))

    def clear_cache(self):
        self.dataset.clear_cache()


class _ReadGate:
    def __init__(self):
        self.condition = Condition()
        self.foreground = 0
        self.sessions = WeakSet()


class ImageDataset:
    """One source session with Z/T metadata and shared regional I/O.

    ``plane()`` returns an unopened, borrowed ordinary Image. Close the dataset
    only after all consumers are finished. ``levels`` describes *source* levels;
    a plane Image's levels include only plane-preserving resolutions. No pixels
    are read merely to open or select a plane. Background work is opt-in through
    ``prefetch`` or ``hint_navigation``; it is never enabled by construction.
    """
    def __init__(self, source, *, owns_source=True):
        if not isinstance(source, ImageSource):
            raise TypeError("source must implement ImageSource")
        if not isinstance(owns_source, bool):
            raise TypeError("owns_source must be boolean")
        self._source = source
        self._owns_source = owns_source
        self._active = False
        self._generation = 0
        self._source_generation = -1
        self._context_active = False
        self._levels = ()
        self._prefetch = None
        parent = getattr(source, "parent", None)
        gate = getattr(source, "_dataset_gate", None)
        if gate is None:
            gate = parent._gate if parent is not None else _ReadGate()
            source._dataset_gate = gate
        self._gate = gate
        gate.sessions.add(self)
        self._condition = gate.condition
        self._lock = getattr(source, "_read_lock", RLock())

    @property
    def generation(self):
        return self._generation

    @property
    def identity(self):
        self._ensure_open()
        return id(self._source), self._source_generation

    @property
    def is_open(self):
        return (self._active and self._source.is_open and
                self._source_generation == self._source.generation)

    def _ensure_open(self):
        if not self.is_open:
            raise RuntimeError("Dataset is closed or belongs to an earlier source session")

    def _ensure_session(self, generation):
        self._ensure_open()
        if generation != self._generation:
            raise RuntimeError("Plane belongs to an earlier dataset session")

    def open(self) -> Self:
        if self._active:
            self._ensure_open()
            return self
        if self._owns_source:
            self._source.open()
        elif not self._source.is_open:
            raise RuntimeError("Borrowed source must already be open")
        try:
            self._levels = tuple(self._source.dataset_levels)
            if not self._levels:
                raise ValueError("Source exposes no dataset levels")
        except BaseException:
            if self._owns_source:
                self._source.close()
            raise
        self._source_generation = self._source.generation
        self._active = True
        self._generation += 1
        return self

    def _stop_prefetch(self, *, shared=False):
        sessions = tuple(self._gate.sessions) if shared else (self,)
        for session in sessions:
            queue, session._prefetch = session._prefetch, None
            if queue is not None:
                queue.cancel()
                with self._condition:
                    self._condition.notify_all()
                queue.close()

    def close(self):
        # Only a physical owner releases shared resources. Borrowed dataset
        # closure stops its own predictor without ending an unrelated session.
        physical_owner = self._owns_source and not hasattr(self._source, "parent")
        self._stop_prefetch(shared=physical_owner)
        with self._lock:
            if self._active:
                self._active = False
                self._generation += 1
                if self._owns_source:
                    self._source.close()
                self._levels = ()

    def __enter__(self):
        if self._context_active:
            raise RuntimeError("Nested contexts on the same dataset are not supported")
        self.open()
        self._context_active = True
        return self

    def __exit__(self, *args):
        self._context_active = False
        self.close()

    @property
    def levels(self):
        self._ensure_open()
        return self._levels

    @property
    def axes(self):
        return self.levels[0].axis_names

    @property
    def shape(self):
        return self.levels[0].shape

    @property
    def dtype(self):
        return self.levels[0].plane_metadata.dtype

    @property
    def image_type(self):
        return self.levels[0].plane_metadata.image_type

    @property
    def channel_names(self):
        return self.levels[0].plane_metadata.channel_names

    @property
    def backing_paths(self):
        return self._source.backing_paths

    def axis_info(self, name: str, *, level=0):
        level = integer(level, "source level")
        if level >= len(self.levels):
            raise IndexError("Source level outside pyramid")
        return self.levels[level].axis(name)

    def _plane_level_map(self, level):
        if level != 0:
            return (level,)
        parsed = getattr(self._source, "parsed", None)
        return parsed.level_indices if parsed is not None else tuple(range(len(self.levels)))

    def plane(self, *, z=None, t=None, level=0):
        """Select indices in a stored level; level=0 keeps its safe XY pyramid.

        Choosing level>0 is an explicit native-level observation. Its Z index is
        local to that level, not a silently mapped full-resolution optical slice.
        """
        self._ensure_open()
        level = integer(level, "source level")
        z = selected_index(self.axis_info("Z", level=level), z, name="Z")
        t = selected_index(self.axis_info("T", level=level), t, name="T")
        with self._lock:
            self._ensure_open()
            metadata = self._source.plane_metadata(z=z, t=t, level=level)
        return _semantic_image(_DatasetPlaneSource(self, z, t, level, metadata))

    @contextmanager
    def _demand(self):
        with self._condition:
            self._gate.foreground += 1
        try:
            with self._lock:
                self._ensure_open()
                yield
        finally:
            with self._condition:
                self._gate.foreground -= 1
                self._condition.notify_all()

    def _read(self, request, *, speculative=False, cancelled=None):
        if not speculative:
            with self._demand():
                return self._source.read_plane_region(*request.bounds, z=request.z, t=request.t,
                    level=request.level, channels=request.channels, speculative=False)
        with self._condition:
            self._condition.wait_for(lambda: not self._gate.foreground or cancelled())
        if cancelled():
            raise CancelledError()
        with self._lock:
            self._ensure_open()
            if cancelled():
                raise CancelledError()
            return self._source.read_plane_region(*request.bounds, z=request.z, t=request.t,
                level=request.level, channels=request.channels, speculative=True)

    def _normalize_request(self, request):
        with self.plane(z=request.z, t=request.t, level=request.level) as plane:
            y0, y1, x0, x1 = request.bounds
            h, w = plane.levels[0].spatial_shape
            if not (0 <= y0 <= y1 <= h and 0 <= x0 <= x1 <= w):
                raise ValueError("Prefetch region is outside the selected plane")
            channels = tuple(normalize_channel_indices(request.channels, plane.channel_count))
            if plane.axes != "CYX" and channels != (0,):
                raise ValueError("RGB/label/single-plane images allow only logical channel zero")
            selection = plane._source.plane_selection
            return PlaneRequest(request.bounds, selection.z, selection.t, request.level, channels)

    def _estimate(self, request):
        request = self._normalize_request(request)
        y0, y1, x0, x1 = request.bounds
        if y0 == y1 or x0 == x1:
            return 0
        estimate = getattr(self._source, "estimate_read", None)
        if estimate is not None:
            return estimate(request)
        arrays = getattr(self._source, "arrays", None)
        if arrays is not None:
            array = arrays[request.level]
            native = self._source.parsed.native_axes
            extent = dict(y=(y0, y1), x=(x0, x1))
            count = 1
            for i, name in enumerate(native):
                chunk = array.chunks[i]
                if name in extent:
                    lo, hi = extent[name]
                    count *= (hi - 1) // chunk - lo // chunk + 1
            if "c" in native:
                width = array.chunks[native.index("c")]
                wanted = range(3) if self.image_type == "rgb" else request.channels
                count *= len({c // width for c in wanted})
            return count * int(np.prod(array.chunks, dtype=object)) * self.dtype.itemsize
        pages_for = getattr(self._source, "_pages_for", None)
        if pages_for is not None:
            pages = pages_for(request.z, request.t, request.level)
            total = 0
            for c in request.channels:
                page = pages[c]
                bh = page.tilelength if page.is_tiled else page.rowsperstrip or page.imagelength
                bw = page.tilewidth if page.is_tiled else page.imagewidth
                total += (((y1 - 1) // bh - y0 // bh + 1) *
                          ((x1 - 1) // bw - x0 // bw + 1) * bh * bw *
                          page.samplesperpixel * self.dtype.itemsize)
            return total
        return (y1-y0) * (x1-x0) * len(request.channels) * self.dtype.itemsize * (3 if self.image_type == "rgb" else 1)

    def prefetch(self, requests, *, policy=None):
        """Explicitly enable bounded read-ahead; returns completion Futures.

        Each call supersedes older pending predictions. Errors remain available in
        the Futures and prefetch_info(); demanded reads still receive real errors.
        """
        self._ensure_open()
        if not getattr(self._source, "supports_prefetch", False):
            raise NotImplementedError("This source has not opted into worker-thread prefetch")
        policy = PrefetchPolicy() if policy is None else policy
        if not isinstance(policy, PrefetchPolicy):
            raise TypeError("policy must be PrefetchPolicy")
        if self._prefetch is not None and self._prefetch.policy != policy:
            self._prefetch.close()
            self._prefetch = None
        if self._prefetch is None:
            def execute(request, cancelled):
                self._read(self._normalize_request(request), speculative=True, cancelled=cancelled)
            self._prefetch = PrefetchQueue(execute, self._estimate, policy)
        return self._prefetch.submit(requests)

    def hint_navigation(self, request: PlaneRequest, *, axis, direction=1, count=1, policy=None):
        """Predict adjacent Z/T viewports from caller intent, not whole planes."""
        if axis.upper() not in {"Z", "T"}:
            raise ValueError("Navigation axis must be Z or T")
        if isinstance(direction, bool) or direction not in {-1, 1}:
            raise ValueError("direction must be -1 or 1")
        count = integer(count, "lookahead", minimum=1)
        policy = PrefetchPolicy() if policy is None else policy
        if count > policy.max_pending:
            raise ValueError("lookahead exceeds max_pending")
        request = self._normalize_request(request)
        info = self.axis_info(axis, level=request.level)
        if info is None or info.size == 1:
            return ()
        current = request.z if axis.upper() == "Z" else request.t
        plans = []
        for step in range(1, count + 1):
            target = current + direction * step
            if 0 <= target < info.size:
                plans.append(PlaneRequest(request.bounds,
                    target if axis.upper() == "Z" else request.z,
                    target if axis.upper() == "T" else request.t,
                    request.level, request.channels))
        return self.prefetch(plans, policy=policy)

    def hint_pan(self, request: PlaneRequest, *, dx=0, dy=0, distance=256, policy=None):
        """Explicitly prefetch leading XY strips; ordinary pan/read calls do not predict."""
        if any(isinstance(v, bool) or v not in (-1, 0, 1) for v in (dx, dy)):
            raise ValueError("Pan directions must be -1, 0, or 1")
        distance = integer(distance, "distance", minimum=1)
        request = self._normalize_request(request)
        level = self.levels[request.level]
        h, w = level.plane_metadata.levels[0].spatial_shape
        y0,y1,x0,x1 = request.bounds
        bounds = []
        if dx > 0 and x1 < w: bounds.append((y0,y1,x1,min(w,x1+distance)))
        if dx < 0 and x0 > 0: bounds.append((y0,y1,max(0,x0-distance),x0))
        if dy > 0 and y1 < h: bounds.append((y1,min(h,y1+distance),x0,x1))
        if dy < 0 and y0 > 0: bounds.append((max(0,y0-distance),y0,x0,x1))
        return self.prefetch([PlaneRequest(b,request.z,request.t,request.level,request.channels)
                              for b in bounds],policy=policy)

    def locate_plane(self, *, z=None, t=None, level):
        """Report an explicit nearest native-level Z sample, never select it implicitly.

        Returns requested/actual indices and coordinates. A coarse voxel's support
        or kernel is not inferred merely from its coordinate transform.
        """
        import math
        from types import MappingProxyType
        level = integer(level, "source level")
        z = selected_index(self.axis_info("Z"), z, name="Z")
        t = selected_index(self.axis_info("T"), t, name="T")
        target = self.levels[level]  # range validation is intentional
        source_z, target_z = self.axis_info("Z"), target.axis("Z")
        actual, requested, coordinate = z, None, None
        if source_z is not None:
            requested = source_z.coordinate(z)
            if not target.plane_preserving:
                if (requested is None or target_z is None or target_z.scale is None or
                        target_z.origin is None or target_z.unit != source_z.unit):
                    raise ValueError("Cannot map a coarse plane without comparable declared Z coordinates")
                fractional = (requested-target_z.origin)/target_z.scale
                actual = min(target_z.size-1,max(0,int(math.floor(fractional+.5))))
            coordinate = target_z.coordinate(actual)
        return MappingProxyType({"level":level,"requested_z":z,"z":actual,"t":t,
            "requested_z_coordinate":requested,"sampled_z_coordinate":coordinate,
            "z_displacement":None if requested is None or coordinate is None else coordinate-requested,
            "unit":None if source_z is None else source_z.unit,
            "plane_preserving":target.plane_preserving})

    def with_metadata(self, **overrides):
        """Borrow an explicit calibration/naming override; pixel values are unchanged."""
        from .dataset_views import DatasetMetadataSource
        return ImageDataset(DatasetMetadataSource(self, **overrides))

    def cancel_prefetch(self):
        if self._prefetch is not None:
            self._prefetch.cancel()
        with self._condition:
            self._condition.notify_all()

    def prefetch_info(self):
        return {"enabled": False} if self._prefetch is None else {"enabled": True, **self._prefetch.info()}

    def cache_info(self):
        self._ensure_open()
        return self._source.cache_info()

    def clear_cache(self):
        # Payload retention is source-wide, so pending predictions in borrowed
        # views must not refill it immediately after the owner's clear.
        for session in tuple(self._gate.sessions):
            session.cancel_prefetch()
        with self._lock:
            self._ensure_open()
            self._source.clear_cache()

    def _indices(self, name, selection, level):
        info = self.axis_info(name, level=level)
        if info is None:
            if selection is not None:
                raise ValueError(f"Source has no {name} axis")
            return (None,)
        if isinstance(selection, slice):
            if selection.step not in (None, 1):
                raise ValueError("Block ranges require unit stride")
            lo = 0 if selection.start is None else integer(selection.start, name + " start")
            hi = info.size if selection.stop is None else integer(selection.stop, name + " stop")
            if not 0 <= lo < hi <= info.size:
                raise ValueError(f"Invalid {name} range")
            return range(lo, hi)
        return (selected_index(info, selection, name=name),)

    def iter_planes(self, *, z=None, t=None, level=0):
        """Yield (selection, unopened Image); explicit slices select whole/ranged axes."""
        generation = self.generation
        zs, ts = self._indices("Z", z, level), self._indices("T", t, level)
        for ti in ts:
            for zi in zs:
                self._ensure_session(generation)
                plane = self.plane(z=zi, t=ti, level=level)
                yield plane._source.plane_selection, plane

    def read_block(self, y0, y1, x0, x1, *, z=None, t=None, channels=None,
                   level=0, max_bytes=64 * 1024 * 1024):
        """Explicit bounded materialization; output axes are TZ + plane.axes.

        T/Z remain singleton dimensions even when absent or selected by integer.
        Use plane().read_region() for the unchanged 2D result convention.
        """
        max_bytes = integer(max_bytes, "max_bytes", minimum=1)
        zs, ts = self._indices("Z", z, level), self._indices("T", t, level)
        with self.plane(z=zs[0], t=ts[0], level=level) as first:
            bounds = tuple(integer(v, "bound") for v in (y0, y1, x0, x1))
            y0, y1, x0, x1 = bounds
            h, w = first.levels[0].spatial_shape
            if not (y0 <= y1 <= h and x0 <= x1 <= w):
                raise ValueError("Block bounds are outside the source")
            selected = tuple(normalize_channel_indices(channels, first.channel_count))
            if first.axes != "CYX" and selected != (0,):
                raise ValueError("Non-channel image permits only logical channel zero")
            shape = (y1-y0, x1-x0)
            if first.axes == "CYX":
                shape = (len(selected), *shape)
            elif first.axes == "YXS":
                shape = (*shape, 3)
            shape = (len(ts), len(zs), *shape)
            if int(np.prod(shape, dtype=object)) * first.dtype.itemsize > max_bytes:
                raise ValueError("Requested block exceeds max_bytes")
            output = np.empty(shape, dtype=first.dtype)
        for it, ti in enumerate(ts):
            for iz, zi in enumerate(zs):
                with self.plane(z=zi, t=ti, level=level) as plane:
                    output[it, iz] = plane.read_region(*bounds, channels=selected)
        return output

    def acquisition_plane(self, *, z=None, t=None, channel=0):
        """Read declared numeric acquisition fields separately from intrinsic axes."""
        from types import MappingProxyType
        with self.plane(z=z, t=t) as plane:
            channel = normalize_channel_indices(channel, plane.channel_count)[0]
            selected = plane.plane_selection
        function = getattr(self._source, "acquisition_plane", None)
        if function is None:
            return MappingProxyType({})
        return function(z=selected.z, t=selected.t, channel=channel)

    def require_regular_export(self):
        """Refuse to discard per-plane acquisition coordinates in stack output."""
        self._ensure_open()
        source = self._source
        while hasattr(source, "parent"):
            source = source.parent._source
        mapping = getattr(source, "_plane_map", None)
        if mapping is not None:
            from omeify._calibration import _children
            if any(any(key in node.attrib for key in ("DeltaT", "ExposureTime", "PositionX",
                                                      "PositionY", "PositionZ"))
                   for node in _children(mapping.pixels, "Plane")):
                raise NotImplementedError("Stack export cannot preserve per-plane acquisition "
                    "coordinates yet; select an explicit 2D plane instead. No uniform grid is inferred.")

    @classmethod
    def from_array(cls, array, *, axes, **metadata):
        from .dataset_array import DatasetArraySource
        return cls(DatasetArraySource(array, axes=axes, **metadata))

    def select(self, *, z=None, t=None, channels=None, bounds=None):
        from .dataset_views import SelectedDatasetSource
        return ImageDataset(SelectedDatasetSource(self, z=z, t=t, channels=channels, bounds=bounds))


def open_dataset(path, *, input_type, series=0, group="", image_type=None,
                 intrinsic_only=False, storage_options=None, cache_bytes=64*1024*1024):
    """Construct an unopened OME source session without requiring a Z/T selection."""
    if input_type == "ome_zarr":
        from .ome_zarr.reader import ZarrSource
        return ImageDataset(ZarrSource(path, series=series, group=group,
            image_type=image_type, intrinsic_only=intrinsic_only, storage_options=storage_options,
            cache_bytes=cache_bytes, _allow_default_selection=True))
    if input_type == "ome_tiff":
        if group or intrinsic_only or storage_options is not None or "://" in str(path):
            raise ValueError("TIFF dataset access requires a local, self-contained file")
        if image_type not in (None, "label"):
            raise ValueError("OME-TIFF permits only an explicit label override")
        from ._tiff_sources.ome import OMESource
        return ImageDataset(OMESource(path, series=series, labels=image_type == "label",
            cache_bytes=cache_bytes, _allow_default_selection=True))
    raise ValueError("Dataset access requires input_type='ome_tiff' or 'ome_zarr'")


class PlaneReaderMixin:
    """Navigation access for existing 2D OME reader constructors."""
    @property
    def plane_selection(self):
        self._ensure_open()
        parsed = getattr(self._source, "parsed", None)
        if parsed is None:
            z, t = self._source.z, self._source.t
        else:
            z = parsed.selections[parsed.native_axes.index("z")] if "z" in parsed.native_axes else None
            t = parsed.selections[parsed.native_axes.index("t")] if "t" in parsed.native_axes else None
        return PlaneSelection(z, t, 0, (id(self._source), self._source.generation))

    @property
    def dataset(self):
        self._ensure_open()
        view = getattr(self, "_dataset_view", None)
        if view is None or not view.is_open:
            view = ImageDataset(self._source, owns_source=False).open()
            self._dataset_view = view
        return view

    def plane(self, *, z=None, t=None, level=0):
        return self.dataset.plane(z=z, t=t, level=level)

    def axis_info(self, name, *, level=0):
        return self.dataset.axis_info(name, level=level)

    @property
    def source_levels(self):
        return self.dataset.levels

    def cache_info(self):
        return self.dataset.cache_info()

    def clear_cache(self):
        view = getattr(self, "_dataset_view", None)
        if view is not None and view.is_open:
            view.clear_cache()
        else:
            super().clear_cache()

    def close(self):
        view = getattr(self, "_dataset_view", None)
        if view is not None:
            view._stop_prefetch(shared=True)
            view.close()
            self._dataset_view = None
        super().close()
