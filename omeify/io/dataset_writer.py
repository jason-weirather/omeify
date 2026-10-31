"""Stack writer adapter: streams ordinary Image planes through the shared engine."""
from __future__ import annotations

from contextlib import ExitStack

from ._reader_sequence import ReaderSequence
from .dataset import ImageDataset
from .rgb_color import SRGB_ICC, canonical_rgb, color_management_report
from .spec import OMEImageSpec


def _physical_axis(dataset, name, unit):
    axis = dataset.axis_info(name)
    if axis is None:
        return None, None
    if axis.scale is None:
        if axis.origin is not None:
            raise ValueError(f"Dataset {name} origin cannot be exported without its sampling")
        return None, None
    if axis.unit is None and axis.scale == 1 and axis.origin in (None, 0):
        return None, None  # Uncalibrated identity index grid, not 1 second/micrometer.
    if axis.unit != unit:
        raise ValueError(f"Dataset {name} output requires {unit} calibration; supply an explicit override")
    return axis.scale, axis.origin


class DatasetPlaneSource:
    """Private lazy sequence in canonical TIFF order T, C, Z, Y, X[,S]."""
    def __init__(self, dataset):
        if not isinstance(dataset, ImageDataset):
            raise TypeError("write_dataset requires an open ImageDataset, not an ndarray or Image")
        dataset._ensure_open()
        self.dataset, self.generation = dataset, dataset.generation
        self.zs = range(dataset.axis_info("Z").size) if dataset.axis_info("Z") else (None,)
        self.ts = range(dataset.axis_info("T").size) if dataset.axis_info("T") else (None,)
        self.color_management = None
        with dataset.plane(z=self.zs[0], t=self.ts[0]) as plane, canonical_rgb(plane) as normalized:
            metadata = normalized._metadata
            self.color_management = color_management_report(normalized)
            if metadata.unrepresented_geometry:
                raise ValueError("Additional spatial relationships require explicit intrinsic_only=True")
            if metadata.pixel_size is None:
                raise ValueError("Dataset output requires calibrated X/Y pixels")
            h, w = metadata.levels[0].spatial_shape
            names = metadata.channel_names
            kind = metadata.image_type
            if kind == "rgb":
                axes, shape = "TZYXS", (len(self.ts), len(self.zs), h, w, 3)
            elif kind == "label":
                axes, shape = "TZYX", (len(self.ts), len(self.zs), h, w)
            else:
                axes, shape = "TCZYX", (len(self.ts), len(names), len(self.zs), h, w)
            dz, oz = _physical_axis(dataset, "Z", "µm")
            dt, ot = _physical_axis(dataset, "T", "s")
            self.spec = OMEImageSpec(kind, axes, shape, metadata.dtype, names,
                metadata.pixel_size, SRGB_ICC if kind == "rgb" else None,
                origin_yx=metadata.levels[0].origin_yx,
                z_spacing_um=dz, time_increment_s=dt, z_origin_um=oz, time_origin_s=ot)
            self.background_label = metadata.background_label

    @property
    def backing_paths(self):
        return self.dataset.backing_paths

    def coordinates(self, index):
        t, rem = divmod(index, self.spec.logical_channel_count * self.spec.size_z)
        c, z = divmod(rem, self.spec.size_z)
        return self.ts[t], c, self.zs[z]

    def plane_readers(self):
        self.dataset._ensure_session(self.generation)
        return ReaderSequence(self.spec.plane_count, lambda i: _PhysicalPlane(self, i))


class _PhysicalPlane:
    def __init__(self, source, index):
        self.source, self.index = source, index
        self.height, self.width = source.spec.size_y, source.spec.size_x
        self.dtype, self.samples_per_pixel = source.spec.dtype, source.spec.samples_per_pixel
        self._stack = None
        self._channel = None

    def _open(self):
        self.source.dataset._ensure_session(self.source.generation)
        if self._stack is not None:
            return
        stack = ExitStack()
        try:
            t, c, z = self.source.coordinates(self.index)
            plane = stack.enter_context(self.source.dataset.plane(z=z, t=t))
            normalized = stack.enter_context(canonical_rgb(plane))
            if (normalized.dtype != self.dtype or normalized.channel_names != self.source.spec.channel_names
                    or normalized.image_type != self.source.spec.image_type
                    or normalized.pixel_size is None
                    or normalized.pixel_size.converted_to("µm") != self.source.spec.pixel_size
                    or normalized.origin_yx != self.source.spec.origin_yx):
                raise ValueError("Dataset plane interpretation changed during writing")
            self._channel = normalized[c]
            self._stack = stack
        except BaseException:
            stack.close()
            raise

    def read_region(self, y0, y1, x0, x1):
        self._open()
        return self._channel.read_region(y0, y1, x0, x1)

    def clear_cache(self):
        # Release just this view/ICC transform. Shared demand caches remain useful
        # across adjacent planes and verification passes under their own budget.
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._channel = None


def write_tiff_dataset(writer, dataset):
    from ._writer import WriterEngine, prepare_image, validate_ome_xml
    from ._writer.configuration import OUTPUT_BYTEORDER
    from ._writer.single_verification import verify_single_output
    from .image_planes import protect_source_paths
    from omeify.reports import complete_report
    from omeify.utils.generate_ome_xml import generate_ome_xml

    source = DatasetPlaneSource(dataset)
    # Per-plane measured coordinates are not silently collapsed into a regular
    # axis. This initial stack output contract accepts regular intrinsic rasters.
    dataset.require_regular_export()
    protect_source_paths(source, writer.path)
    settings = writer._settings
    from omeify.path_safety import check_output_path
    check_output_path(writer.path, protected=source.backing_paths, overwrite=settings.overwrite)
    item = prepare_image(source, source.spec, name=None, downsample=writer._downsample,
        compression_name=settings.compression_name, settings=settings, lossy_policy="rgb-only")
    xml = generate_ome_xml(item.spec, item.level_shapes,
                          display_uuid=settings.display_uuid, output_byteorder=OUTPUT_BYTEORDER)
    validation = validate_ome_xml(xml["xml_string"], context="Z/T dataset")
    result = WriterEngine(settings).write((item,), xml["xml_string"],
        verify=lambda path: verify_single_output(item, path, software=settings.software))
    return complete_report({
        "output_file": {"path": str(writer.path), "format": "ome_tiff",
            "size_bytes": result.output_size, "axes": item.spec.axes,
            "shape": list(item.spec.shape), "dtype": item.spec.dtype.name},
        "image": {"image_type": item.spec.image_type, "channel_names": list(item.spec.channel_names),
            "pixel_size": list(item.spec.pixel_size.to_tuple()),
            "z_spacing_um": item.spec.z_spacing_um, "time_increment_s": item.spec.time_increment_s,
            "z_origin_um": item.spec.z_origin_um, "time_origin_s": item.spec.time_origin_s},
        "pyramid": {"axes": "xy", "method": item.downsample,
            "level_shapes": [list(s) for s in item.level_shapes]},
        "verification": result.verification,
        "metadata": {"ome_xml_valid": validation.xml_is_valid,
                     "miti_header_valid": validation.miti_header.is_valid},
        "options": {"compression": item.compression.name, "tile_size": item.tile_size,
            "lossless_compression": item.compression.lossless,
            "float32_mantissa_bits": item.float32_mantissa_bits,
            "color_management": source.color_management},
    }, "dataset_write_report.schema.json")
