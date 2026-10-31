"""Borrowed Z/T/channel/rectangle selections implemented by the existing Image API."""
from __future__ import annotations

from dataclasses import replace

from ._selection import normalize_channel_indices
from .base import MultichannelImage
from .dataset_metadata import DatasetLevel, selected_index
from .image_metadata import integer
from .image_source import ImageSource


class SelectedDatasetSource(ImageSource):
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
        super().__init__(metadata)
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
        return self.metadata

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

    def _read_region(self, *bounds, level, channels):
        return self.read_plane_region(*bounds, z=selected_index(self.dataset_levels[0].axis("Z"),None,name="Z"),
            t=selected_index(self.dataset_levels[0].axis("T"),None,name="T"), level=level, channels=channels)

    def clear_cache(self):
        self.parent.clear_cache()

    def cache_info(self):
        return self.parent.cache_info()


class DatasetMetadataSource(ImageSource):
    """A base-level metadata-only view; no hidden calibration inference or mutation."""
    def __init__(self,parent,*,pixel_size=None,channel_names=None,z_spacing_um=None,
                 time_increment_s=None,z_origin_um=None,time_origin_s=None):
        parent._ensure_open()
        self.parent, self._parent_generation = parent,parent.generation
        self.supports_prefetch = getattr(parent._source, "supports_prefetch", False)
        self._read_lock = parent._lock
        z=0 if parent.axis_info('Z') else None
        t=0 if parent.axis_info('T') else None
        with parent.plane(z=z,t=t) as plane, plane.with_metadata(
                pixel_size=pixel_size,channel_names=channel_names) as modified:
            # Dataset views expose their explicitly selected base observation.
            metadata=replace(modified._metadata,levels=(modified.levels[0],))
        super().__init__(metadata)
        axes=[]
        values={'Z':(z_spacing_um,z_origin_um,'µm'),'T':(time_increment_s,time_origin_s,'s')}
        for name,(scale,origin,_) in values.items():
            if parent.axis_info(name) is None and (scale is not None or origin is not None):
                raise ValueError(f"Cannot override absent {name} axis")
        size=None if metadata.pixel_size is None else metadata.pixel_size.converted_to('µm')
        for axis in parent.levels[0].axes:
            if axis.name in values:
                scale,origin,unit=values[axis.name]
                if (scale is not None or origin is not None) and axis.unit != unit:
                    if scale is None:
                        raise ValueError("A physical/time origin override also needs explicit sampling")
                    if origin is None and axis.origin not in (None, 0):
                        raise ValueError("Calibrating a shifted unitless axis requires an explicit origin")
                axis=replace(axis,scale=axis.scale if scale is None else scale,
                    origin=axis.origin if origin is None else origin,
                    unit=axis.unit if scale is None and origin is None else unit)
                if axis.origin is not None and axis.scale is None:
                    raise ValueError("An explicit physical/time origin needs known sampling")
            elif axis.name in 'YX' and pixel_size is not None:
                origin = metadata.levels[0].origin_yx
                axis=replace(axis,unit='µm',scale=size.y if axis.name=='Y' else size.x,
                             origin=None if origin is None else origin[0 if axis.name=='Y' else 1])
            axes.append(axis)
        self._dataset_levels=(DatasetLevel(0,tuple(axes),metadata),)

    @property
    def is_open(self):
        return self._is_open and self.parent.is_open and self._parent_generation==self.parent.generation

    def _open(self):
        self.parent._ensure_session(self._parent_generation)

    @property
    def backing_paths(self):
        return self.parent.backing_paths

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._dataset_levels

    def plane_metadata(self,*,z=None,t=None,level=0):
        if level!=0:raise IndexError('Metadata dataset view has one level')
        selected_index(self.dataset_levels[0].axis('Z'),z,name='Z')
        selected_index(self.dataset_levels[0].axis('T'),t,name='T')
        return self.metadata

    def read_plane_region(self,*bounds,z,t,level,channels,speculative=False):
        self.plane_metadata(z=z,t=t,level=level)
        self.parent._ensure_session(self._parent_generation)
        return self.parent._source.read_plane_region(*bounds,z=z,t=t,level=0,
                                                     channels=channels,speculative=speculative)

    def estimate_read(self, request):
        return self.parent._estimate(request)

    def acquisition_plane(self, *, z, t, channel=0):
        self.parent._ensure_session(self._parent_generation)
        return self.parent.acquisition_plane(z=z,t=t,channel=channel)

    def _read_region(self,*bounds,level,channels):
        return self.read_plane_region(*bounds,z=selected_index(self.dataset_levels[0].axis('Z'),None,name='Z'),
            t=selected_index(self.dataset_levels[0].axis('T'),None,name='T'),level=level,channels=channels)

    def clear_cache(self):self.parent.clear_cache()
    def cache_info(self):return self.parent.cache_info()
