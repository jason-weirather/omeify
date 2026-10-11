"""Regional OME-Zarr image readers over the existing ImageSource contract."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from ..base import Image, LabelImage
from ..image_metadata import integer
from ..image_source import ImageSource
from .metadata import ArrayInfo, ome_attributes, parse_image, relative_path
from .store import is_remote, location, read_store

LOGGER = logging.getLogger(__name__)


def open_group(store, path=""):
    import zarr

    return zarr.open_group(store=store, path=path, mode="r", use_consolidated=False)


def array_info(array) -> ArrayInfo:
    meta = array.metadata.to_dict()
    names = meta.get("dimension_names")
    return ArrayInfo(tuple(array.shape), np.dtype(array.dtype), meta["zarr_format"],
                     None if names is None else tuple(names))


def group_attributes(group):
    attributes = group.attrs.asdict()
    version, ome = ome_attributes(attributes, group.metadata.zarr_format)
    return attributes, version, ome


def discover_images(store, group_path="", *, metadata_nodes=None) -> tuple[tuple[str, int, str | None], ...]:
    """Discover declared images, optionally retaining the group metadata consulted.

    This records collection roots/manifests as well as image attributes without
    recursive listing or reading chunks. It is not a whole-store metadata audit.
    """
    import zarr

    def describe(path):
        group = open_group(store, path)
        attributes = group.attrs.asdict()
        if metadata_nodes is not None:
            metadata_nodes[path] = attributes
        return ome_attributes(attributes, group.metadata.zarr_format)

    def image_name(item):
        if not isinstance(item, dict):
            raise ValueError("A multiscale image entry must be an object")
        name = item.get("name")
        if name is not None and not isinstance(name, str):
            raise ValueError("An image name must be a string or absent")
        return name

    version, ome = describe(group_path)
    if "multiscales" in ome:
        ms = ome["multiscales"]
        if not isinstance(ms, list) or not ms or len(ms) > 4096:
            raise ValueError("Invalid or excessive multiscale image list")
        return tuple((group_path, i, image_name(item)) for i, item in enumerate(ms))
    if any(key in ome for key in ("plate", "well", "scene")):
        raise NotImplementedError("Select an image within this plate/well/scene using group= / --group")
    if ome.get("bioformats2raw.layout") != 3:
        raise ValueError("No multiscale image or supported image collection in selected group")
    prefix = group_path + "/" if group_path else ""
    try:
        manifest_version, manifest = describe(prefix + "OME")
        if manifest_version != version:
            raise ValueError("Collection metadata versions disagree")
        paths = manifest.get("series")
    except zarr.errors.GroupNotFoundError:
        paths = None
    if paths is None:
        # Standard transitional fallback: consecutively numbered groups, bounded.
        paths = []
        for i in range(4097):
            try:
                describe(prefix + str(i))
            except zarr.errors.GroupNotFoundError:
                break
            paths.append(str(i))
    if (not isinstance(paths, list) or not 1 <= len(paths) <= 4096
            or any(not isinstance(p, str) for p in paths) or len(set(paths)) != len(paths)):
        raise ValueError("Invalid or excessive collection series manifest")
    entries = []
    for path in paths:
        path = prefix + relative_path(path)
        child_version, child = describe(path)
        if child_version != version:
            raise ValueError("Collection image metadata versions disagree")
        ms = child.get("multiscales")
        if not isinstance(ms, list) or len(ms) != 1:
            raise ValueError("Each collection series must describe exactly one multiscale image")
        entries.append((path, 0, image_name(ms[0])))
    return tuple(entries)


class ZarrSource(ImageSource):
    def __init__(self, path: str | Path, *, series=0, group="", z=None, t=None,
                 image_type=None, intrinsic_only=False, storage_options=None,
                 cache_bytes=64 * 1024 * 1024):
        super().__init__()
        self.path = location(path)
        self.series_index = integer(series, "series")
        self.group_path = relative_path(group, allow_empty=True)
        self.z, self.t, self.image_type = z, t, image_type
        if not isinstance(intrinsic_only, bool):
            raise TypeError("intrinsic_only must be boolean")
        self.intrinsic_only = intrinsic_only
        self.storage_options = dict(storage_options or {})
        self.cache_bytes = integer(cache_bytes, "cache_bytes")
        self.store = None
        self.arrays = ()
        self.entries = ()
        self.parsed = None
        self.attributes = {}
        self.discovery_metadata = {}
        self.zarr_format = None
        self.array_metadata = ()

    @property
    def backing_paths(self):
        return () if is_remote(self.path) else (Path(self.path),)

    def _open(self):
        import zarr

        self.store = read_store(self.path, storage_options=self.storage_options,
                                cache_bytes=self.cache_bytes)
        self.discovery_metadata = {}
        self.entries = discover_images(self.store, self.group_path,
                                       metadata_nodes=self.discovery_metadata)
        if self.series_index >= len(self.entries):
            raise IndexError(f"Series {self.series_index} is outside {len(self.entries)} declared images")
        if len(self.entries) > 1:
            LOGGER.info("Selected image %s of %s declared images", self.series_index, len(self.entries))
        path, index, _ = self.entries[self.series_index]
        self.selected_group = path
        group = open_group(self.store, path)
        self.attributes, _, ome = group_attributes(group)
        self.zarr_format = group.metadata.zarr_format
        datasets = ome["multiscales"][index].get("datasets", [])
        if not isinstance(datasets, list) or not 1 <= len(datasets) <= 64:
            raise ValueError("Expected 1 to 64 image resolutions")
        self.arrays = tuple(zarr.open_array(
            store=self.store, path=(path + "/" if path else "") + relative_path(d["path"]),
            mode="r",
        ) for d in datasets)
        self.array_metadata = tuple(a.metadata.to_dict() for a in self.arrays)
        self.parsed = parse_image(
            self.attributes, group.metadata.zarr_format, [array_info(a) for a in self.arrays],
            series=index, z=self.z, t=self.t, image_type=self.image_type,
            intrinsic_only=self.intrinsic_only,
        )
        self._metadata = self.parsed.metadata
        if self.parsed.additional_spatial_context and not self.intrinsic_only:
            LOGGER.warning("Reading the selected intrinsic raster, not additional coordinate systems "
                           "or scene registration; export requires explicit intrinsic_only=True")

    def _close(self):
        self.arrays = ()
        self.entries = ()
        self.parsed = None
        self.attributes = {}
        self.discovery_metadata = {}
        self.zarr_format = None
        self.array_metadata = ()
        self._metadata = None
        if self.store is not None:
            self.store.close()
            self.store = None

    def clear_cache(self):
        self._ensure_open()
        self.store.clear_cache()

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        parsed = self.parsed
        native = parsed.native_axes
        array = self.arrays[level]
        selection = list(parsed.selections)
        selection[native.index("y")] = slice(y0, y1)
        selection[native.index("x")] = slice(x0, x1)
        remaining = [axis for axis, index in zip(native, selection) if isinstance(index, slice)]
        if "c" in native:
            # Scalar reads preserve requested ordering/duplicates without allocating all channels.
            values = []
            wanted = (0, 1, 2) if self.metadata.image_type == "rgb" else channels
            for channel in wanted:
                selection[native.index("c")] = channel
                plane = np.asarray(array[tuple(selection)])
                spatial_order = [a for a in remaining if a != "c"]
                values.append(np.transpose(plane, [spatial_order.index("y"), spatial_order.index("x")]))
            if self.metadata.image_type == "rgb":
                return np.stack(values, axis=-1)
            if self.metadata.axes == "YX":
                return values[0]
            return np.stack(values)
        plane = np.asarray(array[tuple(selection)])
        return np.transpose(plane, [remaining.index("y"), remaining.index("x")])


class OMEZarrReader(Image):
    """Open a local/HTTP NGFF 0.4, 0.5 or 0.6 image as a regional 2D observation.

    Non-singleton Z/T axes require explicit z=/t=. A group selects an image within
    a plate/scene; series selects a declared collection image or multiscale entry.
    Unannotated three-channel arrays remain multichannel, never inferred RGB.
    """
    input_type_description = "OME-Zarr"

    def __init__(self, path: str | Path, *, series=0, group="", z=None, t=None,
                 image_type=None, intrinsic_only=False, storage_options=None,
                 cache_bytes=64 * 1024 * 1024):
        super().__init__(ZarrSource(
            path, series=series, group=group, z=z, t=t, image_type=image_type,
            intrinsic_only=intrinsic_only, storage_options=storage_options, cache_bytes=cache_bytes,
        ))

    @property
    def path(self):
        return self._source.path

    @property
    def series_index(self):
        return self._source.series_index

    @property
    def series_names(self):
        self._ensure_open()
        return tuple(e[2] for e in self._source.entries)

    @property
    def series_name(self):
        return self.series_names[self.series_index]

    @property
    def native_axes(self):
        self._ensure_open()
        return "".join(self._source.parsed.native_axes).upper()

    @property
    def native_shape(self):
        self._ensure_open()
        return tuple(self._source.arrays[0].shape)

    @property
    def source_byte_order(self):
        return "not_applicable"  # No single TIFF-style container byte order.

    @property
    def ngff_version(self):
        self._ensure_open()
        return self._source.parsed.version

    def inspect(self):
        from omeify.zarr_inspection import ZarrInspector
        self._ensure_open()
        return ZarrInspector.from_source(self._source)


class OMEZarrLabelReader(OMEZarrReader, LabelImage):
    def __init__(self, path: str | Path, **kwargs):
        super().__init__(path, image_type="label", **kwargs)
