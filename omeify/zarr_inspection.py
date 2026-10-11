"""Metadata-only OME-Zarr diagnostics, separate from the stricter image reader."""
from __future__ import annotations

import html
import json

from .io.image_metadata import integer
from .io.ome_zarr.metadata import ome_attributes, parse_image, profile_errors, relative_path
from .io.ome_zarr.reader import array_info, discover_images, open_group
from .io.ome_zarr.store import location, read_store
from .reports import complete_report, report_validator
from .terminal import escape_terminal


def json_value(value):
    """Detach metadata into JSON containers; never coerce arbitrary Python objects."""
    import math
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {k: json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    return json.loads(json.dumps(value, allow_nan=False))


class ZarrInspector:
    """Inspect node metadata without decoding chunks or measuring a remote store.

    Inspection reports raw attributes, including fields that would be minimized on
    writing. Passing the production profile check is not a whole-store privacy audit
    or a claim of exhaustive NGFF conformance. Unsupported image semantics remain
    visible as diagnostics rather than being silently interpreted.
    """
    def __init__(self, path, *, group="", series=0, z=None, t=None, image_type=None,
                 intrinsic_only=False, storage_options=None, detail=1, max_text_length=240):
        self.path = location(path)
        self.group = relative_path(group, allow_empty=True)
        self.series = integer(series, "series")
        self.z, self.t, self.image_type = z, t, image_type
        self.intrinsic_only = intrinsic_only
        self.storage_options = storage_options
        self.detail = integer(detail, "detail")
        if self.detail > 3:
            raise ValueError("detail must be between 0 and 3")
        if max_text_length is not None:
            max_text_length = integer(max_text_length, "max_text_length", minimum=1)
        self.max_text_length = max_text_length
        self._report = None

    @classmethod
    def from_source(cls, source):
        source._ensure_open()
        result = cls(source.path, group=source.group_path, series=source.series_index,
                     z=source.z, t=source.t, intrinsic_only=source.intrinsic_only,
                     image_type=source.image_type)
        # Copy current session metadata, without reopening or retaining its handles.
        result._report = result._describe(
            source.attributes, source.arrays, source.selected_group,
            source.entries, source.parsed,
        )
        return result

    def _describe(self, attributes, arrays, selected_group, entries, parsed=None, errors=()):
        errors = list(errors)
        version = None
        try:
            # Arrays and their owning group have the same storage version for a
            # supported observation. A metadata-only failure may have no arrays.
            fmt = arrays[0].metadata.zarr_format if arrays else (3 if "ome" in attributes else 2)
            version, _ = ome_attributes(attributes, fmt)
            if parsed is None and arrays:
                index = entries[self.series][1]
                parsed = parse_image(attributes, fmt, [array_info(a) for a in arrays],
                                     series=index, z=self.z, t=self.t,
                                     image_type=self.image_type, intrinsic_only=self.intrinsic_only)
        except (ValueError, TypeError, KeyError, IndexError, NotImplementedError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        image = None
        if parsed is not None:
            m = parsed.metadata
            image = {"axes": m.axes, "shape": list(m.shape), "dtype": m.dtype.name,
                     "image_type": m.image_type, "channel_names": list(m.channel_names),
                     "pixel_size": None if m.pixel_size is None else list(m.pixel_size.to_tuple()),
                     "origin_yx_um": m.levels[0].origin_yx,
                     "additional_spatial_context": parsed.additional_spatial_context,
                     "export_geometry_representable": not m.unrepresented_geometry,
                     "levels": [{"index": l.index, "shape": list(l.shape),
                                 "downsample_yx": l.downsample_yx, "offset_yx": l.offset_yx,
                                 "origin_yx_um": l.origin_yx,
                                 "pixel_size": None if l.pixel_size is None else list(l.pixel_size.to_tuple())}
                                for l in m.levels]}
        problems = profile_errors(attributes)
        return complete_report(json_value({
            "path": str(self.path), "group": selected_group, "series_index": self.series,
            "format": "OME-Zarr", "ngff_version": version, "detail": self.detail,
            "series": [{"index": i, "group": e[0], "multiscale_index": e[1], "name": e[2]}
                       for i, e in enumerate(entries)],
            "image": image, "arrays": [a.metadata.to_dict() for a in arrays],
            "attributes": attributes, "diagnostics": errors,
            "profile": {"name": "omeify.ome_zarr/1", "matches_image_attributes": not problems,
                        "findings": list(problems), "whole_store_audited": False},
            "raster_pixels_examined": False,
        }), "zarr_inspection.schema.json")

    @property
    def report(self):
        import zarr

        if self._report is None:
            store = read_store(self.path, storage_options=self.storage_options)
            try:
                group = open_group(store, self.group)
                attributes = group.attrs.asdict()
                arrays, entries = (), ()
                selected = self.group
                problems = []
                try:
                    entries = discover_images(store, self.group)
                    if self.series >= len(entries):
                        raise IndexError("Selected series is outside the declared image collection")
                    selected, index, _ = entries[self.series]
                    group = open_group(store, selected)
                    attributes = group.attrs.asdict()
                    _, ome = ome_attributes(attributes, group.metadata.zarr_format)
                    datasets = ome["multiscales"][index]["datasets"]
                    if not isinstance(datasets, list) or not 1 <= len(datasets) <= 64:
                        raise ValueError("Expected 1 to 64 multiscale datasets")
                    arrays = tuple(zarr.open_array(store=store,
                                   path=(selected + "/" if selected else "") + relative_path(d["path"]),
                                   mode="r") for d in datasets)
                except (ValueError, TypeError, KeyError, IndexError, NotImplementedError) as exc:
                    problems.append(f"{type(exc).__name__}: {exc}")
                self._report = self._describe(attributes, arrays, selected, entries, errors=problems)
            finally:
                store.close()
        return self._report

    def validation_errors(self):
        return tuple(f"{e.json_path}: {e.validator}" for e in
                     report_validator("zarr_inspection.schema.json").iter_errors(self.report))

    def to_json(self, *, indent=2):
        return json.dumps(self.report, indent=indent, ensure_ascii=False, allow_nan=False)

    def render_text(self):
        r = self.report
        lines = [str(self.path), f"  OME-Zarr {r['ngff_version'] or 'unknown'}; "
                 f"selected group={r['group']!r}, series={r['series_index']}"]
        if r["image"] is not None:
            m = r["image"]
            lines += [f"  {m['axes']} {m['dtype']} {tuple(m['shape'])}; {m['image_type']}",
                      "  Channels: " + ", ".join(m["channel_names"]),
                      f"  Pixel size: {m['pixel_size']}; pixel-center origin (Y,X) µm: {m['origin_yx_um']}"]
            if m["additional_spatial_context"]:
                lines.append("  Additional spatial context exists; no registration/resampling applied.")
            if self.detail >= 1:
                for l in m["levels"]:
                    lines.append(f"  Level {l['index']}: {tuple(l['shape'])}; "
                                 f"scale={l['downsample_yx']}, edge offset={l['offset_yx']}")
        lines.append("  Image-attribute production profile: " +
                     ("PASS" if r["profile"]["matches_image_attributes"] else "NOT MATCHED"))
        lines.append("  Metadata only; no pixel checks or whole-store minimization audit.")
        if self.detail >= 2:
            for i, a in enumerate(r["arrays"]):
                lines.append(f"  Array {i}: Zarr v{a['zarr_format']}; "
                             f"chunks={a.get('chunks', a.get('chunk_grid'))}; "
                             f"encoding={a.get('codecs', a.get('compressor'))}")
        if self.detail >= 3:
            value = json.dumps(r["attributes"], ensure_ascii=False)
            if self.max_text_length is not None and len(value) > self.max_text_length:
                value = value[:self.max_text_length] + "… (full attributes in JSON)"
            lines.append("  Raw attributes: " + value)
        lines.extend("  Diagnostic: " + d for d in r["diagnostics"])
        return "\n".join(escape_terminal(line) for line in lines)

    def __str__(self):
        return self.render_text()

    def _repr_html_(self):
        return "<pre>" + html.escape(self.render_text()) + "</pre>"
