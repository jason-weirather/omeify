"""Bounded, lossless NGFF 0.5 production with verified, no-replace publication."""
from __future__ import annotations

import itertools
import json
import os
import logging
import math
import shutil
import tempfile
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Self

import numpy as np

from omeify.progress import ProgressLogger
from omeify.reports import complete_report

from .._writer.configuration import normalize_float32_mantissa_bits
from .._writer.precision import float32_significant_bits, round_float32_mantissa
from .._writer.pyramid import auto_level_shapes, mean_downsample_2x
from ..base import Image
from ..image_metadata import ImageLevel, integer
from ..image_planes import ImagePlaneSource
from ..ome_multi_series_writer import OMEImageSeries
from .metadata import PROFILE, WRITE_VERSION, output_attributes, profile_errors
from .reader import OMEZarrReader
from .store import preflight_directory, publish_directory, store_size

LOGGER = logging.getLogger(__name__)


def output_levels(image: Image, *, level: int, tile_size: int, pyramid_levels: int | None,
                  method: str) -> tuple[ImageLevel, ...]:
    base = image._metadata.level(level)
    if base.pixel_size is None:
        raise ValueError("OME-Zarr output requires calibrated pixels")
    size = base.pixel_size.converted_to("µm")
    origin = base.origin_yx or (0.0, 0.0)
    shapes = auto_level_shapes(base.shape, axes=base.axes, tile_size=tile_size,
                              pyramid_levels=pyramid_levels)
    result = []
    for i, shape in enumerate(shapes):
        factor = float(2 ** i)
        offset = (0.0, 0.0) if method == "mean" else ((1 - factor) / 2,) * 2
        shift = (factor - 1) / 2 if method == "mean" else 0.0
        result.append(ImageLevel(i, base.axes, shape, size.scaled(factor), (factor, factor),
                                 offset, (origin[0] + shift * size.y, origin[1] + shift * size.x)))
    return tuple(result)


def _stored_shape(level):
    if level.axes == "YXS":
        return (3, *level.spatial_shape)
    return level.shape


def _points(height, width):
    return tuple(dict.fromkeys(((0, 0), (height // 2, width // 2), (height - 1, width - 1))))


def _equal_bytes(left, right):
    a = np.ascontiguousarray(left).view(np.uint8)
    b = np.ascontiguousarray(right).view(np.uint8)
    return np.array_equal(a, b)


def _set_tile(array, selection, values):
    array[selection] = values


def _write_jobs(array, jobs, workers, progress):
    # Only immutable pixel buffers enter workers. Provider reads stay on the calling
    # thread because ImageSource does not promise thread safety.
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = set()
        for selection, values in jobs:
            while len(pending) >= 2 * workers:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                    progress.update(completed)
            pending.add(pool.submit(_set_tile, array, selection, values))
        for future in pending:
            future.result()
            completed += 1
            progress.update(completed)


def _tiles(height, width, size):
    for y in range(0, height, size):
        for x in range(0, width, size):
            yield y, min(height, y + size), x, min(width, x + size)


def _selection(ndim, channel, y0, y1, x0, x1):
    return ((channel,) if ndim == 3 else ()) + (slice(y0, y1), slice(x0, x1))


def _display_windows(array, *, rgb):
    count = array.shape[0] if array.ndim == 3 else 1
    result = []
    for c in range(count):
        if rgb:
            lo, hi = 0.0, 255.0
        else:
            h, w = array.shape[-2:]
            samples = []
            for y, x in _points(h, w):
                tile = np.asarray(array[_selection(array.ndim, c, y, min(h, y + 32), x, min(w, x + 32))])
                samples.append(tile[np.isfinite(tile)].reshape(-1))
            finite = np.concatenate(samples)
            lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
            if lo == hi:
                lo, hi = (0.0, hi) if hi > 0 else (lo, 1.0)
        result.append({"min": lo, "max": hi, "start": lo, "end": hi})
    return result


class OMEZarrWriter:
    """Write one Image to a NEW local NGFF 0.5 / Zarr v3 directory.

    Storage is explicitly little-endian, unsharded and lossless. `tile_size` is the
    spatial chunk edge; each stored scalar channel has its own chunks. Existing
    destinations are never replaced. No network publishing or arbitrary attributes.
    """
    def __init__(self, path: str | Path, *, compression: str | None = None,
                 tile_size: int | None = None, pyramid_levels: int | None = None,
                 downsample: str = "mean", max_workers: int | None = None,
                 float32_mantissa_bits: int | None = None):
        self.path = preflight_directory(path)
        if compression is not None and not isinstance(compression, str):
            raise TypeError("compression must be a string or None")
        self.compression = ("ZSTD" if compression is None else compression).strip().lower()
        if self.compression not in {"zstd", "gzip", "uncompressed"}:
            raise ValueError("OME-Zarr compression must be ZSTD, Gzip, or Uncompressed (not TIFF LZW/Deflate/JPEG)")
        self.tile_size = 512 if tile_size is None else integer(tile_size, "tile_size", minimum=16)
        if self.tile_size % 16:
            raise ValueError("Spatial chunk edge must be divisible by 16")
        self.pyramid_levels = (None if pyramid_levels is None else
                               integer(pyramid_levels, "pyramid_levels"))
        if self.pyramid_levels is not None and self.pyramid_levels > 32:
            raise ValueError("pyramid_levels must not exceed 32")
        if downsample not in {"mean", "nearest"}:
            raise ValueError("downsample must be mean or nearest")
        self.downsample = downsample
        self.max_workers = 4 if max_workers is None else integer(max_workers, "max_workers", minimum=1)
        if self.max_workers > 128:
            raise ValueError("max_workers must not exceed 128")
        self.float32_mantissa_bits = normalize_float32_mantissa_bits(float32_mantissa_bits)

    def _codecs(self, dtype):
        import zarr

        serializer = zarr.codecs.BytesCodec(endian="little" if np.dtype(dtype).itemsize > 1 else None)
        compressors = ([] if self.compression == "uncompressed" else
                       [zarr.codecs.ZstdCodec(level=3, checksum=False)] if self.compression == "zstd" else
                       [zarr.codecs.GzipCodec(level=6)])
        return serializer, compressors

    def _prepare(self, image, level, *, name=None):
        # Shared image/dtype/semantic-loss contract. No TIFF staging or TIFF encoding.
        spec = ImagePlaneSource(image, level=level).output_spec()
        preflight_directory(self.path, protected=image.backing_paths)
        if self.float32_mantissa_bits is not None and spec.dtype != np.dtype("float32"):
            raise TypeError("Mantissa trimming requires float32 pixels")
        method = "nearest" if image.image_type == "label" else self.downsample
        levels = output_levels(image, level=level, tile_size=self.tile_size,
                               pyramid_levels=self.pyramid_levels, method=method)
        metadata = replace(image._metadata, shape=levels[0].shape, pixel_size=levels[0].pixel_size,
                           levels=levels)
        dummy = [{"min": 0, "max": 1, "start": 0, "end": 1}] * metadata.sample_count
        attributes = output_attributes(metadata, levels, method=method, windows=dummy, name=name)
        errors = profile_errors(attributes)
        if errors:
            raise ValueError("OME-Zarr output metadata violates its profile: " + "; ".join(errors))
        return metadata, levels, method

    def _write_image(self, image, stage, *, level=0, name=None):
        import zarr

        metadata, levels, method = self._prepare(image, level, name=name)
        serializer, compressors = self._codecs(metadata.dtype)
        group = zarr.open_group(store=str(stage), mode="w", zarr_format=3)
        try:
            arrays = []
            source_planes = ImagePlaneSource(image, level=level).plane_readers()
            expected_chunks = 0
            for i, descriptor in enumerate(levels):
                shape = _stored_shape(descriptor)
                chunks = (1, self.tile_size, self.tile_size) if len(shape) == 3 else (self.tile_size,) * 2
                array = group.create_array(
                    str(i), shape=shape, dtype=metadata.dtype, chunks=chunks,
                    serializer=serializer, compressors=compressors, filters=None,
                    dimension_names=("c", "y", "x") if len(shape) == 3 else ("y", "x"),
                    chunk_key_encoding={"name": "default", "configuration": {"separator": "/"}},
                    fill_value=metadata.background_label if metadata.image_type == "label" else 0,
                    config={"write_empty_chunks": True, "order": "C"},
                )
                h, w = descriptor.spatial_shape
                count = shape[0] if len(shape) == 3 else 1
                total = count * ((h + self.tile_size - 1) // self.tile_size) * ((w + self.tile_size - 1) // self.tile_size)
                expected_chunks += total
                progress = ProgressLogger(LOGGER, f"Writing OME-Zarr level {i}", total, unit="chunks")

                def jobs(index=i, current=array, height=h, width=w, channel_count=count):
                    if index == 0:
                        for logical, plane in enumerate(source_planes):
                            for y0, y1, x0, x1 in _tiles(height, width, self.tile_size):
                                block = plane.read_region(y0, y1, x0, x1)
                                if self.float32_mantissa_bits is not None:
                                    block = round_float32_mantissa(block, self.float32_mantissa_bits)
                                if metadata.image_type == "rgb":
                                    for component in range(3):
                                        yield _selection(3, component, y0, y1, x0, x1), block[..., component]
                                else:
                                    yield _selection(current.ndim, logical, y0, y1, x0, x1), block
                    else:
                        previous = arrays[index - 1]
                        ph, pw = previous.shape[-2:]
                        for channel in range(channel_count):
                            for y0, y1, x0, x1 in _tiles(height, width, self.tile_size):
                                block = np.asarray(previous[_selection(previous.ndim, channel,
                                                   2*y0, min(ph, 2*y1), 2*x0, min(pw, 2*x1))])
                                reduced = (block[::2, ::2] if method == "nearest" else
                                           mean_downsample_2x(block, (y1 - y0, x1 - x0)))
                                if self.float32_mantissa_bits is not None:
                                    reduced = round_float32_mantissa(reduced, self.float32_mantissa_bits)
                                yield _selection(current.ndim, channel, y0, y1, x0, x1), reduced
                _write_jobs(array, jobs(), self.max_workers, progress)
                progress.finish()
                arrays.append(array)
                image.clear_cache()
            windows = _display_windows(arrays[-1], rgb=metadata.image_type == "rgb")
            attributes = output_attributes(metadata, levels, method=method, windows=windows, name=name)
            if self.float32_mantissa_bits is not None:
                attributes.setdefault("omeify", {})["float32_mantissa_bits"] = self.float32_mantissa_bits
            errors = profile_errors(attributes)
            if errors:
                raise ValueError("Generated NGFF metadata failed profile validation: " + "; ".join(errors))
            group.attrs.update(attributes)
        finally:
            group.store.close()
        verification = self._verify(image, stage, level, metadata, levels, method, attributes,
                                    serializer, compressors, expected_chunks)
        bits = self.float32_mantissa_bits
        report = {
            "ome": {"format": "OME-Zarr", "version": WRITE_VERSION, "zarr_format": 3,
                    "profile": PROFILE, "profile_valid": True, "attributes": attributes},
            "miti_header": None,
            "output_file": {"path": str(self.path), "size_bytes": store_size(stage),
                            "type_description": "Pyramidal OME-Zarr", "dtype": metadata.dtype.name,
                            "shape": list(metadata.shape), "axes": metadata.axes,
                            "byte_order": "little", "lossless_compression": True},
            "image": {"image_type": metadata.image_type, "channel_names": list(metadata.channel_names),
                      "size_c": metadata.sample_count, "logical_channel_count": metadata.logical_channel_count,
                      "samples_per_pixel": metadata.samples_per_pixel,
                      "interleaved": metadata.image_type == "rgb",  # logical regional read layout
                      "pixel_size": list(metadata.pixel_size.to_tuple()),
                      "significant_bits": metadata.dtype.itemsize * 8 if bits is None else float32_significant_bits(bits),
                      "float32_mantissa_bits": bits, "float_precision_bits": None if bits is None else bits + 1,
                      "icc_profile_present": metadata.icc_profile is not None},
            "pyramid": {"tile_size": self.tile_size, "axes": metadata.axes, "downsample_method": method,
                        "level_shapes": [list(d.shape) for d in levels], "subresolution_count": len(levels) - 1},
            "verification": verification,
            "options": {"metadata_minimization": True, "compression": self.compression_name,
                        "float32_mantissa_bits": bits, "max_workers": self.max_workers,
                        "output_format": "ome_zarr", "ngff_version": WRITE_VERSION},
        }
        return complete_report(report, "zarr_write_report.schema.json")

    @property
    def compression_name(self):
        return {"zstd": "ZSTD", "gzip": "Gzip", "uncompressed": "Uncompressed"}[self.compression]

    def _verify(self, source, stage, input_level, metadata, levels, method, attributes,
                serializer, compressors, expected_chunks):
        import zarr

        plain = zarr.open_group(str(stage), mode="r", use_consolidated=False)
        try:
            if plain.attrs.asdict() != attributes:
                raise ValueError("NGFF metadata readback differs from the generated allowlist")
            chunks_checked = base_points = pyramid_points = 0
            arrays = [plain[str(i)] for i in range(len(levels))]
            for i, (array, descriptor) in enumerate(zip(arrays, levels)):
                raw = array.metadata.to_dict()
                shape = _stored_shape(descriptor)
                chunk_shape = (1, self.tile_size, self.tile_size) if len(shape) == 3 else (self.tile_size,) * 2
                fill = metadata.background_label if metadata.image_type == "label" else 0
                if (raw.get("zarr_format") != 3 or raw.get("node_type") != "array"
                        or raw.get("fill_value") != fill or raw.get("storage_transformers", [])
                        or tuple(array.shape) != shape or np.dtype(array.dtype).newbyteorder("=") != metadata.dtype
                        or tuple(array.chunks) != chunk_shape or raw.get("attributes", {})
                        or tuple(raw.get("dimension_names") or ()) != tuple("cyx" if len(shape) == 3 else "yx")
                        or tuple(raw.get("codecs", ())) != (serializer.to_dict(), *(c.to_dict() for c in compressors))
                        or raw.get("chunk_key_encoding") != {"name": "default", "configuration": {"separator": "/"}}):
                    raise ValueError("Zarr array structure or codecs differ from the output profile")
                for coords in itertools.product(*(range((n + c - 1) // c) for n, c in zip(shape, chunk_shape))):
                    key = stage / str(i) / "c" / Path(*map(str, coords))
                    if not key.is_file() or key.is_symlink() or key.stat().st_size == 0:
                        raise ValueError("Generated store has a missing or invalid chunk")
                    chunks_checked += 1
            # Every file must be accounted for: no stale chunks or arbitrary metadata files.
            files = sum(len(names) for _, _, names in os.walk(stage))
            if chunks_checked != expected_chunks or files != expected_chunks + len(levels) + 1:
                raise ValueError("Generated store inventory differs from the planned dense image")
            with OMEZarrReader(stage) as readback:
                if (readback.channel_names != metadata.channel_names or readback.image_type != metadata.image_type
                        or readback._metadata.icc_profile != metadata.icc_profile):
                    raise ValueError("Image semantics failed NGFF readback")
                for i, descriptor in enumerate(levels):
                    actual = readback.levels[i]
                    if actual != descriptor:
                        # Scale/unit round trips allow floating arithmetic, but never shape changes.
                        if (actual.shape != descriptor.shape or not np.allclose(actual.origin_yx, descriptor.origin_yx,
                                rtol=1e-12, atol=1e-12) or not np.allclose(actual.offset_yx, descriptor.offset_yx,
                                rtol=1e-12, atol=1e-12) or actual.pixel_size != descriptor.pixel_size):
                            raise ValueError("Image geometry failed NGFF readback")
                    h, w = descriptor.spatial_shape
                    for channel in range(metadata.logical_channel_count):
                        for y, x in _points(h, w):
                            got = readback[channel].read_region(y, y+1, x, x+1, level=i)
                            if i == 0:
                                expected = source[channel].read_region(y, y+1, x, x+1, level=input_level)
                                base_points += 1
                            else:
                                ph, pw = levels[i - 1].spatial_shape
                                block = readback[channel].read_region(2*y, min(ph, 2*y+2), 2*x, min(pw, 2*x+2), level=i-1)
                                expected = block[::2, ::2] if method == "nearest" else mean_downsample_2x(block, (1, 1))
                                pyramid_points += 1
                            if self.float32_mantissa_bits is not None:
                                expected = round_float32_mantissa(expected, self.float32_mantissa_bits)
                            equal = (_equal_bytes(got, expected) if i == 0 or method == "nearest" else
                                     np.array_equal(got, expected, equal_nan=True))
                            if not equal:
                                raise ValueError("NGFF representative pixel verification failed")
            return {"ome_zarr_recognized": True, "metadata_matches_profile": True,
                    "geometry_matches": True, "storage_matches_requested": True,
                    "dense_chunk_inventory_verified": True, "chunks_checked": chunks_checked,
                    "output_pixels_decodable": True, "base_pixel_values_checked": True,
                    "base_pixel_values_match": True,
                    "pixel_verification": {"mode": "sampled", "all_pixels_checked": False,
                        "sampling": "top-left, center, bottom-right; deduplicated",
                        "base_points_compared": base_points, "pyramid_points_compared": pyramid_points,
                        "source_reader_independent": False, "base_comparison": "bitwise",
                        "pyramid_comparison": "bitwise" if method == "nearest" else "numeric_equal_nan"}}
        finally:
            plain.store.close()

    def write(self, image: Image, *, level: int = 0):
        self._prepare(image, level)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".omeify-zarr-", dir=self.path.parent))
        try:
            report = self._write_image(image, stage, level=level)
            preflight_directory(self.path, protected=image.backing_paths)
            publish_directory(stage, self.path)
            return report
        finally:
            if stage.exists():
                shutil.rmtree(stage)


class OMEZarrCollectionWriter(OMEZarrWriter):
    """Ordered heterogeneous images using NGFF's transitional collection layout."""
    def write(self, images: Sequence[OMEImageSeries], *, provenance=None):
        import zarr

        entries = tuple(images)
        if not 1 <= len(entries) <= 4096 or any(not isinstance(e, OMEImageSeries) for e in entries):
            raise TypeError("Collection writing requires OMEImageSeries entries")
        names = [e.name for e in entries]
        if len(set(names)) != len(names):
            raise ValueError("Collection image names must be unique")
        if provenance is not None:
            # Only caller-supplied, deliberate context. Never copy source attributes.
            encoded = json.dumps(provenance, allow_nan=False)
            if len(encoded.encode("utf-8")) > 1024 * 1024:
                raise ValueError("Collection provenance exceeds 1 MiB")
            provenance = json.loads(encoded)
        children = []
        for i, entry in enumerate(entries):
            entry._source.output_spec()  # Honor the entry's pinned source session.
            child = OMEZarrWriter(
                self.path / str(i), compression=entry.compression or self.compression_name,
                tile_size=self.tile_size, pyramid_levels=self.pyramid_levels,
                downsample=entry.downsample, max_workers=self.max_workers,
                float32_mantissa_bits=(self.float32_mantissa_bits
                                       if entry.image.dtype == np.dtype("float32") else None),
            )
            child._prepare(entry.image, entry.level, name=entry.name)
            children.append(child)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".omeify-zarr-", dir=self.path.parent))
        try:
            root = zarr.open_group(str(stage), mode="w", zarr_format=3)
            try:
                root.create_group("OME", attributes={"ome": {"version": WRITE_VERSION,
                                                              "series": [str(i) for i in range(len(entries))]}})
                reports = [children[i]._write_image(e.image, stage / str(i), level=e.level, name=e.name)
                           for i, e in enumerate(entries)]
                attrs = {"ome": {"version": WRITE_VERSION, "bioformats2raw.layout": 3}}
                if provenance is not None:
                    attrs["omeify"] = {"provenance": provenance}
                root.attrs.update(attrs)
            finally:
                root.store.close()
            check_root = zarr.open_group(str(stage), mode="r", use_consolidated=False)
            try:
                if check_root.attrs.asdict() != attrs:
                    raise ValueError("Collection metadata/provenance failed readback")
            finally:
                check_root.store.close()
            with OMEZarrReader(stage) as check:
                if check.series_names != tuple(names):
                    raise ValueError("Collection order/names failed readback")
            for i, report in enumerate(reports):
                report["output_file"]["path"] = str(self.path / str(i))
            report = complete_report({"output_file": {"path": str(self.path),
                                      "size_bytes": store_size(stage), "series_count": len(entries)},
                                      "series": reports, "metadata_minimization": True},
                                     "zarr_collection_report.schema.json")
            for entry in entries:
                preflight_directory(self.path, protected=entry.image.backing_paths)
            publish_directory(stage, self.path)
            return report
        finally:
            if stage.exists():
                shutil.rmtree(stage)


class TemporaryOMEZarrWriter:
    """Temporary-path ownership, delegating image construction to the ordinary writer."""
    def __init__(self, **settings):
        self.settings = settings
        self._temporary = None
        self.path = None

    def __enter__(self) -> Self:
        if self._temporary is not None:
            raise RuntimeError("Nested temporary writer contexts are not supported")
        self._temporary = tempfile.TemporaryDirectory(prefix="omeify-zarr-")
        self.path = Path(self._temporary.name) / "image.ome.zarr"
        return self

    def write(self, image, *, level=0):
        if self._temporary is None:
            raise RuntimeError("Use TemporaryOMEZarrWriter inside a context")
        return OMEZarrWriter(self.path, **self.settings).write(image, level=level)

    def __exit__(self, *_):
        self._temporary.cleanup()
        self._temporary = None
