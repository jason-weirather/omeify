"""Versioned NGFF interpretation, independent of stores and pixel decoding.

The supported image product is a selected 2D plane, not a registered scene.
Version-specific parsing is deliberately separate from the single output profile.
See docs/ome_zarr.md for the standard references and capability boundaries.
"""
from __future__ import annotations

import base64
import copy
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..image_metadata import ImageLevel, ImageMetadata, integer
from ..pixel_size import PixelSize

READ_VERSIONS = ("0.4", "0.5", "0.6")
WRITE_VERSION = "0.5"
PROFILE = "omeify.ome_zarr/1"
MAX_LEVELS = 64
MAX_CHANNELS = 4096
MAX_METADATA_BYTES = 8 * 1024 * 1024

# NGFF's unit spellings differ from OME-XML's symbols. Never write 'µm' in NGFF axes.
UNIT_TO_UM = {
    "angstrom": 1e-4, "attometer": 1e-12, "centimeter": 1e4,
    "decimeter": 1e5, "exameter": 1e24, "femtometer": 1e-9,
    "foot": 304800., "gigameter": 1e15, "hectometer": 1e8,
    "inch": 25400., "kilometer": 1e9, "megameter": 1e12,
    "meter": 1e6, "micrometer": 1., "mile": 1609344000.,
    "millimeter": 1e3, "nanometer": 1e-3, "parsec": 3.0856775814913673e22,
    "petameter": 1e21, "picometer": 1e-6, "terameter": 1e18,
    "yard": 914400., "yoctometer": 1e-18, "yottameter": 1e30,
    "zeptometer": 1e-15, "zettameter": 1e27,
}
SUPPORTED_DTYPES = frozenset(("uint8", "uint16", "uint32", "int8", "int16", "int32",
                             "float32", "float64"))


def relative_path(value: Any, *, allow_empty: bool = False) -> str:
    """Only descendant store keys; metadata cannot redirect reads outside the store."""
    if not isinstance(value, str):
        raise TypeError("An NGFF path must be a string")
    if value == "" and allow_empty:
        return value
    if (not value or len(value) > 4096 or any(c in value for c in "\\:%?#")
            or any(ord(c) < 32 or ord(c) == 127 for c in value)
            or any(part in {"", ".", ".."} for part in value.split("/"))):
        raise ValueError("NGFF image/array paths must be safe relative descendant keys")
    return value


def ome_attributes(attributes: Mapping[str, Any], zarr_format: int) -> tuple[str, dict]:
    """Distinguish the NGFF metadata version from the Zarr storage version."""
    if not isinstance(attributes, Mapping):
        raise TypeError("Zarr group attributes must be an object")
    if zarr_format == 2:
        ome = dict(attributes)
        multiscales = ome.get("multiscales", [])
        versions = {m.get("version") for m in multiscales if isinstance(m, dict)}
        # Collection groups in 0.4 do not carry an independent NGFF version.
        if not multiscales and any(k in ome for k in ("bioformats2raw.layout", "plate", "well", "series", "labels")):
            versions = {"0.4"}
        if versions != {"0.4"} or "ome" in ome:
            raise ValueError("Zarr v2 requires consistently declared OME-Zarr 0.4 metadata")
        version = "0.4"
    elif zarr_format == 3:
        ome = attributes.get("ome")
        if not isinstance(ome, dict) or ome.get("version") not in ("0.5", "0.6"):
            raise ValueError("Zarr v3 requires an ome object declaring OME-Zarr 0.5 or 0.6")
        version = ome["version"]
        for ms in ome.get("multiscales", []):
            if "version" in ms and ms["version"] != version:
                raise ValueError("Conflicting OME-Zarr metadata versions")
    else:
        raise ValueError("Only Zarr storage versions 2 and 3 are supported")
    return version, copy.deepcopy(ome)


def _vector(value: Any, ndim: int, field: str) -> np.ndarray:
    if (not isinstance(value, (list, tuple)) or len(value) != ndim
            or any(isinstance(x, bool) or not isinstance(x, (int, float))
                   or not math.isfinite(x) for x in value)):
        raise ValueError(f"{field} must contain {ndim} finite numbers")
    result = np.asarray(value, dtype=np.float64)
    if field == "scale" and np.any(result <= 0):
        raise NotImplementedError("Image scales must be positive; reflection/resampling is not implicit")
    return result


def scale_translation(transforms: Any, ndim: int, *, version: str,
                      dataset_path: str | None = None) -> tuple[np.ndarray, np.ndarray, str | None]:
    """Parse the mandatory array-to-intrinsic transforms, in their declared order."""
    if not isinstance(transforms, list) or not transforms:
        raise ValueError("coordinateTransformations must be a nonempty array")
    output_name = None
    if version == "0.6":
        if len(transforms) != 1 or not isinstance(transforms[0], dict):
            raise ValueError("OME-Zarr 0.6 datasets require one scale, identity, or sequence")
        outer = transforms[0]
        if dataset_path is not None:
            if outer.get("input") != {"path": dataset_path}:
                raise ValueError("Transform input must reference its own dataset path")
            output = outer.get("output")
            if not isinstance(output, dict) or set(output) != {"name"} or not output["name"]:
                raise ValueError("Dataset transform output must name a local coordinate system")
            output_name = output["name"]
        if outer.get("type") == "identity":
            return np.ones(ndim), np.zeros(ndim), output_name
        if outer.get("type") == "sequence":
            transforms = outer.get("transformations")
            if not isinstance(transforms, list) or len(transforms) != 2:
                raise ValueError("Dataset sequence must be scale followed by translation")
        elif outer.get("type") != "scale":
            raise NotImplementedError("Dataset transform must be scale, identity, or scale/translation sequence")
    if ([x.get("type") for x in transforms if isinstance(x, dict)]
            not in (["scale"], ["scale", "translation"])):
        raise ValueError("Dataset transforms must be scale followed by an optional translation")
    for transform in transforms:
        if "path" in transform:
            raise NotImplementedError("Array-valued transform parameters are not supported")
    scale = _vector(transforms[0].get("scale"), ndim, "scale")
    translation = (_vector(transforms[1].get("translation"), ndim, "translation")
                   if len(transforms) == 2 else np.zeros(ndim))
    return scale, translation, output_name


def axes_for_multiscale(ms: dict, version: str) -> tuple[list[dict], str | None]:
    if version != "0.6":
        axes, intrinsic = ms.get("axes"), None
    else:
        systems = ms.get("coordinateSystems")
        if not isinstance(systems, list) or not systems:
            raise ValueError("OME-Zarr 0.6 requires coordinateSystems")
        names = [s.get("name") for s in systems if isinstance(s, dict)]
        if (len(names) != len(systems) or any(not isinstance(n, str) or not n for n in names)
                or len(set(names)) != len(names)):
            raise ValueError("Coordinate system names must be nonempty and unique")
        datasets = ms.get("datasets")
        if not isinstance(datasets, list) or not datasets:
            raise ValueError("A multiscale image requires datasets")
        try:
            intrinsic = datasets[0]["coordinateTransformations"][0]["output"]["name"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError("The intrinsic coordinate system must be explicitly referenced") from exc
        if intrinsic not in names:
            raise ValueError("The intrinsic coordinate system is not defined")
        axes = systems[names.index(intrinsic)].get("axes")
    if not isinstance(axes, list) or not 2 <= len(axes) <= 5 or any(not isinstance(a, dict) for a in axes):
        raise ValueError("Image axes must be an array of 2 to 5 objects")
    names = [a.get("name") for a in axes]
    if len(set(names)) != len(names) or not {"x", "y"}.issubset(names):
        raise NotImplementedError("Image reads require unique axes named x and y")
    if not set(names).issubset("tczyx"):
        raise NotImplementedError("Supported axis names are t, c, z, y, x; no axis-role guessing")
    rank = {"t": 0, "c": 1, "z": 2, "y": 2, "x": 2}
    if sorted(rank[n] for n in names) != [rank[n] for n in names]:
        raise ValueError("NGFF axes must put time before channel before spatial dimensions")
    for axis in axes:
        expected = "time" if axis["name"] == "t" else "channel" if axis["name"] == "c" else "space"
        if axis.get("type", expected) != expected:
            raise NotImplementedError("Axis type conflicts with its supported semantic role")
    return axes, intrinsic


@dataclass(frozen=True)
class ArrayInfo:
    shape: tuple[int, ...]
    dtype: np.dtype
    zarr_format: int
    dimension_names: tuple[str, ...] | None = None


@dataclass(frozen=True)
class ParsedImage:
    version: str
    paths: tuple[str, ...]
    native_axes: tuple[str, ...]
    selections: tuple[int | slice, ...]
    metadata: ImageMetadata
    intrinsic: str | None
    additional_spatial_context: bool


def parse_image(attributes: Mapping[str, Any], zarr_format: int, arrays: Sequence[ArrayInfo], *,
                series: int = 0, z: int | None = None, t: int | None = None,
                image_type: str | None = None, intrinsic_only: bool = False) -> ParsedImage:
    version, ome = ome_attributes(attributes, zarr_format)
    multiscales = ome.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError("Selected group does not describe an NGFF multiscale image")
    series = integer(series, "series")
    if series >= len(multiscales):
        raise IndexError("NGFF multiscale index is outside the selected group")
    ms = multiscales[series]
    axes, intrinsic = axes_for_multiscale(ms, version)
    native = tuple(a["name"] for a in axes)
    datasets = ms.get("datasets")
    if not isinstance(datasets, list) or not 1 <= len(datasets) <= MAX_LEVELS:
        raise ValueError(f"Expected between 1 and {MAX_LEVELS} resolution levels")
    paths = tuple(relative_path(d["path"]) for d in datasets)
    if len(set(paths)) != len(paths) or len(arrays) != len(paths):
        raise ValueError("Dataset paths must be unique and match the supplied array descriptions")
    shape = tuple(integer(n, "array dimension", minimum=1) for n in arrays[0].shape)
    dtype = np.dtype(arrays[0].dtype).newbyteorder("=")
    if dtype.name not in SUPPORTED_DTYPES:
        raise NotImplementedError(f"Unsupported image dtype {dtype}; no implicit casting")
    if len(shape) != len(native):
        raise ValueError("NGFF axes do not match the array dimensionality")
    selection: list[int | slice] = [slice(None)] * len(native)
    for axis, index in (("z", z), ("t", t)):
        if index is not None:
            index = integer(index, f"{axis} index")
        if axis not in native:
            if index is not None:
                raise ValueError(f"The source has no {axis} axis to select")
            continue
        pos = native.index(axis)
        if index is None:
            if shape[pos] != 1:
                raise ValueError(f"Non-singleton {axis}={shape[pos]} requires an explicit {axis} index")
            index = 0
        if index >= shape[pos]:
            raise IndexError(f"{axis} index is outside the source array")
        selection[pos] = index
    c = shape[native.index("c")] if "c" in native else 1
    if c > MAX_CHANNELS:
        raise ValueError(f"Channel count exceeds {MAX_CHANNELS}")
    extension = attributes.get("omeify", {})
    if not isinstance(extension, dict):
        raise ValueError("omeify attributes must be an object")
    declared = extension.get("image_type", "label" if "image-label" in ome else "multichannel")
    kind = image_type or declared
    if kind not in {"multichannel", "rgb", "label"}:
        raise ValueError("image_type must be multichannel, rgb, or label")
    if kind == "rgb" and (c != 3 or dtype != np.dtype("uint8")):
        raise ValueError("Declared RGB requires exactly three uint8 channel components")
    if kind == "label" and (c != 1 or dtype.kind not in "ui"):
        raise ValueError("A label image requires one integer channel")
    if image_type is not None and declared != "multichannel" and image_type != declared:
        raise ValueError("Explicit image type conflicts with embedded image semantics")
    channels = ome.get("omero", {}).get("channels")
    if channels is not None and (not isinstance(channels, list) or len(channels) != c
                                 or any(not isinstance(v, dict) for v in channels)):
        raise ValueError("omero channel metadata must match the stored channel count")
    records = channels or [{} for _ in range(c)]
    names = tuple(r.get("label") or f"Channel {i + 1}" for i, r in enumerate(records))
    if kind == "rgb":
        names, records = (extension.get("logical_channel_name", "RGB"),), [{}]
    if kind == "label" and "channel_name" in extension:
        names = (extension["channel_name"],)
    icc = None
    if "icc_profile_base64" in extension:
        if kind != "rgb":
            raise ValueError("ICC metadata requires declared RGB samples")
        value = extension["icc_profile_base64"]
        if not isinstance(value, str) or len(value) > 2 * 1024 * 1024:
            raise ValueError("ICC payload is invalid or exceeds its bound")
        try:
            icc = base64.b64decode(value, validate=True)
        except (ValueError, TypeError) as exc:
            raise ValueError("Invalid ICC base64") from exc
    canonical = "YXS" if kind == "rgb" else "YX" if kind == "label" or "c" not in native else "CYX"
    spatial_axes = (native.index("y"), native.index("x"))
    units = tuple(axes[i].get("unit") for i in spatial_axes)
    for unit in units:
        if unit is not None and unit not in UNIT_TO_UM:
            raise NotImplementedError(f"Unsupported NGFF spatial unit {unit!r}")
    physical = all(u is not None for u in units)
    extra_context = (version == "0.6" and bool(ms.get("coordinateTransformations")))
    transforms = []
    for dataset in datasets:
        scale, translation, target = scale_translation(
            dataset.get("coordinateTransformations"), len(native), version=version,
            dataset_path=dataset["path"] if version == "0.6" else None,
        )
        if version == "0.6" and target != intrinsic:
            raise ValueError("All resolution levels must map to the same intrinsic coordinate system")
        if version != "0.6" and "coordinateTransformations" in ms:
            global_scale, global_shift, _ = scale_translation(
                ms["coordinateTransformations"], len(native), version=version,
            )
            translation = global_scale * translation + global_shift
            scale = global_scale * scale
        if not np.all(np.isfinite(scale)) or not np.all(np.isfinite(translation)):
            raise ValueError("Coordinate composition overflowed")
        transforms.append((scale, translation))
    base_scale, base_shift = transforms[0]
    levels = []
    for level, (info, (scale, translation)) in enumerate(zip(arrays, transforms)):
        if (len(info.shape) != len(native) or np.dtype(info.dtype).newbyteorder("=") != dtype
                or info.zarr_format != zarr_format):
            raise ValueError("Pyramid arrays must have common dimensionality, dtype and storage version")
        if version == "0.5" and info.dimension_names != native:
            raise ValueError("Zarr v3 dimension_names must match the NGFF axes")
        for i, axis in enumerate(native):
            if axis not in "xy" and (info.shape[i] != shape[i] or scale[i] != base_scale[i]
                                     or translation[i] != base_shift[i]):
                raise NotImplementedError("Selected-plane pyramids must retain channel, Z and T sampling")
        height, width = (int(info.shape[i]) for i in spatial_axes)
        normalized_shape = ((height, width, 3) if canonical == "YXS" else
                            (c, height, width) if canonical == "CYX" else (height, width))
        ratios = tuple(float(scale[i] / base_scale[i]) for i in spatial_axes)
        offsets = tuple(float((translation[i] - base_shift[i]) / base_scale[i]
                              + (1 - scale[i] / base_scale[i]) / 2) for i in spatial_axes)
        pixel_size = origin = None
        if physical:
            sy, sx = (float(scale[i] * UNIT_TO_UM[axes[i]["unit"]]) for i in spatial_axes)
            pixel_size = PixelSize(sx, sy, "µm")
            origin = tuple(float(translation[i] * UNIT_TO_UM[axes[i]["unit"]]) for i in spatial_axes)
        elif any(translation[i] != 0 for i in spatial_axes):
            extra_context = True
        levels.append(ImageLevel(level, canonical, normalized_shape, pixel_size,
                                 ratios, offsets, origin))
    metadata = ImageMetadata(
        axes=canonical, shape=levels[0].shape, dtype=dtype, image_type=kind,
        channel_names=names, pixel_size=levels[0].pixel_size, levels=tuple(levels),
        channel_metadata=records, icc_profile=icc,
        background_label=extension.get("background_label", 0),
        unrepresented_geometry=extra_context and not intrinsic_only,
    )
    return ParsedImage(version, paths, native, tuple(selection), metadata, intrinsic, extra_context)


def output_attributes(metadata: ImageMetadata, levels: Sequence[ImageLevel], *,
                      method: str, windows: Sequence[dict], name: str | None = None) -> dict:
    """Build fresh 0.5 attributes. No arbitrary source attributes cross this boundary."""
    kind = metadata.image_type
    stored_axes = "cyx" if metadata.axes in {"CYX", "YXS"} else "yx"
    axes = [{"name": a, "type": "channel"} if a == "c" else
            {"name": a, "type": "space", "unit": "micrometer"} for a in stored_axes]
    datasets = []
    for i, level in enumerate(levels):
        size = level.pixel_size.converted_to("µm")
        origin = level.origin_yx or (0.0, 0.0)
        scale = ([1.] if "c" in stored_axes else []) + [size.y, size.x]
        shift = ([0.] if "c" in stored_axes else []) + list(origin)
        transforms = [{"type": "scale", "scale": scale}]
        if any(shift):
            transforms.append({"type": "translation", "translation": shift})
        datasets.append({"path": str(i), "coordinateTransformations": transforms})
    ms = {"axes": axes, "datasets": datasets, "type": method,
          "metadata": {"method": f"omeify.{method}_2x", "edge_policy": "available_samples"}}
    if name is not None:
        ms["name"] = name
    labels = ("Red", "Green", "Blue") if kind == "rgb" else metadata.channel_names
    if len(windows) != len(labels):
        raise ValueError("Display windows must match the stored channels")
    channels = []
    for i, (label, window) in enumerate(zip(labels, windows)):
        channels.append({"label": label, "color": ("FF0000", "00FF00", "0000FF")[i]
                         if kind == "rgb" else "FFFFFF", "window": window,
                         "active": kind == "rgb" or i == 0})
    ome = {"version": WRITE_VERSION, "multiscales": [ms], "omero": {"channels": channels}}
    attributes: dict[str, Any] = {"ome": ome}
    # No private metadata is needed for ordinary intensity images. These supplemental
    # declarations retain sample/label semantics absent from the core NGFF raster model.
    if kind in {"rgb", "label"}:
        extension: dict[str, Any] = {"image_type": kind}
        if kind == "rgb":
            from ..rgb_color import check_canonical_icc

            check_canonical_icc(metadata.icc_profile)
            extension["logical_channel_name"] = metadata.channel_names[0]
            extension["icc_profile_base64"] = base64.b64encode(metadata.icc_profile).decode("ascii")
        else:
            extension.update(background_label=metadata.background_label,
                             channel_name=metadata.channel_names[0])
        attributes["omeify"] = extension
    return attributes


def profile_errors(attributes: Mapping[str, Any]) -> tuple[str, ...]:
    """Local, packaged profile validation, distinct from whole-spec conformance."""
    from omeify.reports import report_validator
    try:
        encoded = json.dumps(attributes, allow_nan=False, ensure_ascii=False).encode("utf-8")
    except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError) as exc:
        return (f"Metadata must be finite JSON: {type(exc).__name__}",)
    if len(encoded) > MAX_METADATA_BYTES - 4096:
        return ("Generated node metadata exceeds its 8 MiB budget",)
    errors = report_validator("ome_zarr_profile.schema.json").iter_errors(attributes)
    findings = tuple(f"{'/'.join(map(str, e.path)) or '$'}: failed {e.validator}" for e in errors)
    if findings:
        return findings
    ome = attributes["ome"]
    ms = ome["multiscales"][0]
    result = []
    if [d["path"] for d in ms["datasets"]] != [str(i) for i in range(len(ms["datasets"]))]:
        result.append("ome/multiscales/0/datasets: paths must be consecutive from 0")
    channels = ome["omero"]["channels"]
    for i, channel in enumerate(channels):
        w = channel["window"]
        if not (w["min"] <= w["start"] <= w["end"] <= w["max"] and w["min"] < w["max"]):
            result.append(f"ome/omero/channels/{i}/window: invalid display range ordering")
    extension = attributes.get("omeify", {})
    if extension.get("image_type") == "label" and extension["channel_name"] != channels[0]["label"]:
        result.append("omeify/channel_name: must agree with the stored label channel")
    if extension.get("image_type") == "rgb" and "icc_profile_base64" not in extension:
        result.append("omeify/icc_profile_base64: canonical sRGB profile is required")
    if "icc_profile_base64" in extension:
        try:
            profile = base64.b64decode(extension["icc_profile_base64"], validate=True)
        except ValueError:
            result.append("omeify/icc_profile_base64: invalid base64 encoding")
        else:
            from ..rgb_color import SRGB_ICC

            if profile != SRGB_ICC:
                result.append("omeify/icc_profile_base64: not the canonical sRGB profile")
    return tuple(result)
