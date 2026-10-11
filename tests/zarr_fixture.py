"""Small, independently specified NGFF fixtures; no Omeify writer is used."""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np


def attributes(version, axes, paths, scales, translations, names=None):
    axis_meta = [{"name": a, "type": "channel" if a == "c" else "time" if a == "t" else "space",
                  **({"unit": "micrometer"} if a in "xyz" else {})} for a in axes]
    datasets = []
    for path, scale, shift in zip(paths, scales, translations):
        transforms = [{"type": "scale", "scale": list(scale)}]
        if any(shift):
            transforms.append({"type": "translation", "translation": list(shift)})
        if version == "0.6":
            outer = (dict(transforms[0]) if len(transforms) == 1 else
                     {"type": "sequence", "transformations": transforms})
            outer.update(input={"path": path}, output={"name": "physical"})
            transforms = [outer]
        datasets.append({"path": path, "coordinateTransformations": transforms})
    ms = {"datasets": datasets}
    if version == "0.6":
        ms["coordinateSystems"] = [{"name": "physical", "axes": axis_meta}]
    else:
        ms["axes"] = axis_meta
    ome = {"multiscales": [ms]}
    if names is not None:
        ome["omero"] = {"channels": [{"label": n,"color":"FFFFFF","window":{"min":0,"max":65535,"start":0,"end":65535}} for n in names]}
    if version == "0.4":
        ms["version"] = version
        return ome
    ome["version"] = version
    return {"ome": ome}


def fixture(path: Path, *, version="0.5", data=None, axes="cyx", scale=None,
            origin=None, levels=2, fill_value=0, compressor=None):
    import zarr
    if data is None:
        data = np.arange(3 * 65 * 81, dtype=np.uint16).reshape(3, 65, 81)
    n = data.ndim
    if scale is None:
        scale = [0.6 if a == "y" else 0.5 if a == "x" else 1.0 for a in axes]
    if origin is None:
        origin = [11.25 if a == "y" else -9.5 if a == "x" else 0.0 for a in axes]
    fmt = 2 if version == "0.4" else 3
    root = zarr.open_group(str(path), mode="w", zarr_format=fmt)
    arrays, scales, shifts = [], [], []
    for level in range(levels):
        slicer = tuple(slice(None, None, 2**level) if a in "xy" else slice(None) for a in axes)
        values = data[slicer]
        kwargs = {"compressor": compressor} if fmt == 2 else {
            "serializer": zarr.codecs.BytesCodec(endian="little"),
            "compressors": [] if compressor is None else [compressor],
            "dimension_names": tuple(axes)}
        a = root.create_array(str(level), shape=values.shape, dtype=data.dtype,
                              chunks=tuple(16 if a in "xy" else 1 for a in axes),
                              fill_value=fill_value, **kwargs)
        a[:] = values
        arrays.append(values)
        scales.append([s * (2**level if a in "xy" else 1) for a, s in zip(axes, scale)])
        shifts.append(list(origin))
    names = [f"Marker {c}" for c in range(data.shape[axes.index("c")])] if "c" in axes else None
    attrs = attributes(version, axes, list(map(str, range(levels))), scales, shifts, names)
    root.attrs.update(attrs)
    root.store.close()
    return data, arrays, copy.deepcopy(attrs)
