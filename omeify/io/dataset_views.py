"""Borrowed Z/T/channel/rectangle selections implemented by the existing Image API."""
from __future__ import annotations

from dataclasses import replace

from ._selection import normalize_channel_indices
from .base import MultichannelImage
from .dataset_metadata import DatasetLevel, selected_index
from .image_metadata import integer
from .dataset_source import DatasetSourceBase


class SelectedDatasetSource(DatasetSourceBase):
    """One base-level view; no materialized volume and no second crop implementation."""
    def __init__(self, parent, *, z=None, t=None, channels=None, bounds=None):
        parent._ensure_open()
        self.parent, self._parent_generation = parent, parent.generation
        self.supports_prefetch = getattr(parent._source, "supports_prefetch", False)
        self._read_lock = parent._lock
        self._zs = parent._indices("Z", slice(None) if z is None and parent.axis_info("Z") else z, 0)
        self._ts = parent._indices("T", slice(None) if t is None and parent.axis_info("T") else t, 0)
        with parent.plane(z=self._zs[0], t=self._ts[0]) as plane:
            self._channels = tuple(normalize_channel_indices(channels, plane.channel_count))
            if plane.axes != "CYX" and self._channels != (0,):
                raise ValueError("RGB/labels permit only logical channel zero")
            h, w = plane.levels[0].spatial_shape
            if bounds is None:
                bounds = (0, h, 0, w)
            bounds = tuple(integer(v, "crop bound") for v in bounds)
            if len(bounds) != 4:
                raise ValueError("bounds must be (y0, y1, x0, x1)")
            self._bounds = bounds
            with plane.crop(*bounds) as cropped:
                if cropped.axes == "CYX":
                    with MultichannelImage.from_channels([cropped[c] for c in self._channels]) as selected:
                        metadata = selected._metadata
                else:
                    metadata = cropped._metadata
        super().__init__()
        axes = []
        for info in parent.levels[0].axes:
            if info.name in "TZ":
                indices = self._ts if info.name == "T" else self._zs
                axes.append(replace(info, size=len(indices), origin=(None if info.origin is None or
                                    info.scale is None else info.origin + indices[0]*info.scale)))
            elif info.name == "C":
                # Zarr stores RGB as three C components, not three logical
                # channels. A logical-channel selection cannot split those samples.
                axes.append(info if metadata.image_type == "rgb" else
                            replace(info, size=len(self._channels)))
            elif info.name in "YX":
                j = 0 if info.name == "Y" else 2
                axes.append(replace(info, size=bounds[j+1]-bounds[j],
                    origin=None if info.origin is None or info.scale is None else
                    info.origin + bounds[j]*info.scale))
            else:
                axes.append(info)
        self._dataset_levels = (DatasetLevel(0, tuple(axes), metadata),)

    @property
    def is_open(self):
        return (self._is_open and self.parent.is_open and
                self._parent_generation == self.parent.generation)

    def _open(self):
        self.parent._ensure_session(self._parent_generation)

    @property
    def backing_paths(self):
        return self.parent.backing_paths

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._dataset_levels

    def plane_metadata(self, *, z=None, t=None, level=0):
        if level != 0:
            raise IndexError("Selected dataset views have one level")
        selected_index(self.dataset_levels[0].axis("Z"), z, name="Z")
        selected_index(self.dataset_levels[0].axis("T"), t, name="T")
        return self.dataset_levels[0].plane_metadata

    def read_plane_region(self, y0, y1, x0, x1, *, z, t, level, channels, speculative=False):
        self.plane_metadata(z=z, t=t, level=level)
        zi = self._zs[0 if z is None else z]
        ti = self._ts[0 if t is None else t]
        y, _, x, _ = self._bounds
        self.parent._ensure_session(self._parent_generation)
        return self.parent._source.read_plane_region(y+y0, y+y1, x+x0, x+x1,
            z=zi, t=ti, level=0, channels=tuple(self._channels[c] for c in channels),
            speculative=speculative)

    def estimate_read(self, request):
        from .prefetch import PlaneRequest
        y, _, x, _ = self._bounds
        y0,y1,x0,x1=request.bounds
        return self.parent._estimate(PlaneRequest((y+y0,y+y1,x+x0,x+x1),
            self._zs[0 if request.z is None else request.z],
            self._ts[0 if request.t is None else request.t],0,
            tuple(self._channels[c] for c in request.channels)))

    def acquisition_plane(self, *, z, t, channel=0):
        self.parent._ensure_session(self._parent_generation)
        return self.parent.acquisition_plane(z=self._zs[0 if z is None else z],
            t=self._ts[0 if t is None else t],channel=self._channels[channel])

    def clear_cache(self):
        self.parent.clear_cache()

    def cache_info(self):
        return self.parent.cache_info()


class DatasetMetadataSource(DatasetSourceBase):
    """Metadata-only native pyramid view, retaining every compatible resolution.

    The underlying data, per-native-level dimensions, and level selection stay
    unchanged. Calibrations are updated from declared native sampling ratios,
    never from rounded array-shape ratios. An unknown ratio stays unknown.
    """

    def __init__(self, parent, *, pixel_size=None, channel_names=None,
                 z_spacing_um=None, time_increment_s=None, z_origin_um=None,
                 time_origin_s=None):
        parent._ensure_open()
        from .pixel_size import PixelSize

        if pixel_size is not None and not isinstance(pixel_size, PixelSize):
            raise TypeError("pixel_size must be PixelSize")
        self.parent, self._parent_generation = parent, parent.generation
        self.supports_prefetch = getattr(parent._source, "supports_prefetch", False)
        self._read_lock = parent._lock
        self.plane_level_indices = parent._plane_level_map(0)
        base = parent.levels[0]
        values = {"Z": (z_spacing_um, z_origin_um, "µm"),
                  "T": (time_increment_s, time_origin_s, "s")}
        for name, (scale, origin, _) in values.items():
            old = base.axis(name)
            if old is None and (scale is not None or origin is not None):
                raise ValueError(f"Cannot override absent {name} axis")
            if old is not None and (scale is not None or origin is not None):
                if old.unit not in (None, values[name][2]):
                    raise ValueError(f"Cannot override incompatible {name} units")
                if old.unit is None and scale is None:
                    raise ValueError(f"Uncalibrated {name} origin requires explicit sampling")
                if (old.unit is None and old.origin not in (None, 0)
                        and origin is None):
                    raise ValueError(
                        f"Calibrating a shifted unitless {name} axis requires an explicit origin")

        def axis_update(old, original, *, base_scale, base_origin, unit, at_base):
            """Project only declared base-relative sampling into new physical units."""
            ratio = (1.0 if at_base else
                     old.scale / original.scale
                     if old.scale is not None and original.scale is not None else None)
            new_scale = None if ratio is None or base_scale is None else base_scale * ratio
            if base_origin is None:
                new_origin = None
            elif at_base:
                new_origin = base_origin
            elif (old.origin is not None and original.origin is not None
                  and original.scale is not None and base_scale is not None):
                new_origin = base_origin + (
                    (old.origin - original.origin) / original.scale * base_scale)
            else:
                new_origin = None  # Unknown sampling phase is not zero.
            return replace(old, unit=unit, scale=new_scale, origin=new_origin)

        # The metadata override is also an ordinary 2D Image operation; use
        # that authoritative calculation for the compatible XY plane pyramid.
        z0 = 0 if base.axis("Z") else None
        t0 = 0 if base.axis("T") else None
        with parent.plane(z=z0, t=t0) as plane, plane.with_metadata(
                pixel_size=pixel_size, channel_names=channel_names) as changed:
            complete_base = changed._metadata
        super().__init__()
        native_levels = []

        for i, descriptor in enumerate(parent.levels):
            axis_infos = []
            for old in descriptor.axes:
                original = base.axis(old.name)
                if old.name in values:
                    override_scale, override_origin, unit = values[old.name]
                    if override_scale is not None or override_origin is not None:
                        base_scale = original.scale if override_scale is None else override_scale
                        base_origin = original.origin if override_origin is None else override_origin
                        old = axis_update(old, original, base_scale=base_scale,
                            base_origin=base_origin, unit=unit, at_base=i == 0)
                elif old.name in "YX" and pixel_size is not None:
                    size = pixel_size.converted_to("µm")
                    base_scale = size.y if old.name == "Y" else size.x
                    origin_yx = complete_base.levels[0].origin_yx
                    base_origin = (None if origin_yx is None else origin_yx[
                                   0 if old.name == "Y" else 1])
                    old = axis_update(old, original, base_scale=base_scale,
                        base_origin=base_origin, unit="µm", at_base=i == 0)
                axis_infos.append(old)

            if i == 0:
                # Existing 2D metadata rules preserve/recalibrate all compatible
                # XY levels, including their offsets and original unknowns.
                metadata = complete_base
            else:
                with parent.plane(z=0 if descriptor.axis("Z") else None,
                                  t=0 if descriptor.axis("T") else None,
                                  native_level=i) as plane:
                    old_metadata = plane._metadata
                y, x = (next(a for a in axis_infos if a.name == name) for name in "YX")
                new_size = (PixelSize(x.scale, y.scale, "µm")
                            if x.unit == y.unit == "µm" and
                            x.scale is not None and y.scale is not None else None)
                old_origin = old_metadata.levels[0].origin_yx
                # An unchanged metadata view must not infer a previously unknown
                # origin; a calibration override only rescales known positions.
                new_origin = (y.origin, x.origin) if (
                    old_origin is not None and y.origin is not None and x.origin is not None
                ) else old_origin
                current_level = replace(old_metadata.levels[0],
                    pixel_size=new_size, origin_yx=new_origin)
                metadata = replace(old_metadata,
                    channel_names=old_metadata.channel_names if channel_names is None
                                  else channel_names,
                    pixel_size=new_size, shape=current_level.shape,
                    levels=(current_level,))
            native_levels.append(DatasetLevel(i, tuple(axis_infos), metadata,
                                             descriptor.plane_preserving))
        self._dataset_levels = tuple(native_levels)

    @property
    def is_open(self):
        return (self._is_open and self.parent.is_open and
                self._parent_generation == self.parent.generation)

    def _open(self):
        self.parent._ensure_session(self._parent_generation)

    @property
    def backing_paths(self):
        return self.parent.backing_paths

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._dataset_levels

    def plane_metadata(self, *, z=None, t=None, level=0):
        self.parent._ensure_session(self._parent_generation)
        level = integer(level, "native level")
        if level >= len(self.dataset_levels):
            raise IndexError("Native level outside pyramid")
        descriptor = self.dataset_levels[level]
        selected_index(descriptor.axis("Z"), z, name="Z")
        selected_index(descriptor.axis("T"), t, name="T")
        # Retain parent's per-plane TIFF/NGFF checks, not just first-plane metadata.
        self.parent._source.plane_metadata(z=z, t=t, level=level)
        return descriptor.plane_metadata

    def read_plane_region(self, *bounds, z, t, level, channels, speculative=False):
        self.plane_metadata(z=z, t=t, level=level)
        self.parent._ensure_session(self._parent_generation)
        return self.parent._source.read_plane_region(*bounds, z=z, t=t,
            level=level, channels=channels, speculative=speculative)

    def estimate_read(self, request):
        return self.parent._estimate(request)

    def acquisition_plane(self, *, z, t, channel=0):
        self.parent._ensure_session(self._parent_generation)
        return self.parent.acquisition_plane(z=z, t=t, channel=channel)

    def clear_cache(self):
        self.parent.clear_cache()

    def cache_info(self):
        return self.parent.cache_info()
