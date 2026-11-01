"""Stream Z/T datasets and XY/XYZ pyramids into the canonical NGFF 0.5 storage.

No full volume is materialized. T/C are never reduced. The public 2D writer is
unchanged; this adapter shares codecs, publication, numeric reduction and checks.
"""
from __future__ import annotations

import base64
import itertools
import math
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from omeify.reports import complete_report
from .._writer.precision import round_float32_mantissa
from ..dataset import open_dataset
from ..dataset_writer import DatasetPlaneSource
from ..image_metadata import integer
from ..volume_pyramid import plan_volume_pyramid, reduce_zyx
from .metadata import profile_errors
from .store import preflight_directory, publish_directory, store_size
from .writer import _equal_bytes, _points, _tiles


def dataset_attributes(source, plan, *, method, windows, bits=None):
    """Fresh allowlisted metadata, including nominal XYZ voxel-center transforms."""
    spec = source.spec
    rgb = spec.is_rgb
    names = ("Red", "Green", "Blue") if rgb else spec.channel_names
    axes = [{"name": "t", "type": "time"}]
    if spec.time_increment_s is not None:
        axes[0]["unit"] = "second"
    axes += [{"name": "c", "type": "channel"}, {"name": "z", "type": "space"}]
    if spec.z_spacing_um is not None:
        axes[-1]["unit"] = "micrometer"
    axes += [{"name": a, "type": "space", "unit": "micrometer"} for a in "yx"]
    size = spec.pixel_size.converted_to("µm")
    scale = [spec.time_increment_s or 1., 1., spec.z_spacing_um or 1., size.y, size.x]
    origin = [spec.time_origin_s or 0., 0., spec.z_origin_um or 0., *(spec.origin_yx or (0., 0.))]
    datasets = []
    for index, shape in enumerate(plan.shapes):
        if index:
            factors = (1, 1, *plan.steps[index-1])
            if method == "mean":
                origin = [b + s*(f-1)/2 for b,s,f in zip(origin,scale,factors,strict=True)]
            scale = [s*f for s,f in zip(scale,factors,strict=True)]
        transforms = [{"type": "scale", "scale": list(scale)}]
        if any(origin):
            transforms.append({"type": "translation", "translation": list(origin)})
        datasets.append({"path": str(index), "coordinateTransformations": transforms})
    channels = [{"label": name, "color": ("FF0000", "00FF00", "0000FF")[i]
                 if rgb else "FFFFFF", "active": rgb or i == 0, "window": windows[i]}
                for i, name in enumerate(names)]
    attrs = {"ome": {"version": "0.5", "multiscales": [{"axes": axes,
        "datasets": datasets, "type": method, "metadata": {
            "method": f"omeify.{method}_zyx", "edge_policy": "available_samples",
            "factors_zyx": [list(f) for f in plan.steps]}}], "omero": {"channels": channels}}}
    extension = {}
    if rgb:
        extension = {"image_type": "rgb", "logical_channel_name": spec.channel_names[0],
                     "icc_profile_base64": base64.b64encode(spec.icc_profile).decode("ascii")}
    elif spec.image_type == "label":
        extension = {"image_type": "label", "background_label": source.background_label,
                     "channel_name": spec.channel_names[0]}
    if bits is not None:
        extension["float32_mantissa_bits"] = bits
    if extension:
        attrs["omeify"] = extension
    return attrs


def _windows(array, rgb):
    windows = []
    for c in range(array.shape[1]):
        if rgb:
            lo, hi = 0., 255.
        else:
            values = []
            for t in sorted({0, array.shape[0]//2, array.shape[0]-1}):
                for z in sorted({0, array.shape[2]//2, array.shape[2]-1}):
                    for y,x in _points(*array.shape[-2:]):
                        value = float(array[t,c,z,y,x])
                        if math.isfinite(value):
                            values.append(value)
            lo, hi = (min(values), max(values)) if values else (0., 1.)
            if lo == hi:
                lo, hi = (0., hi) if hi > 0 else (lo, 1.)
        windows.append({"min": lo, "max": hi, "start": lo, "end": hi})
    return windows


def _validate_work_budget(plan, *, chunk_z, tile_size, dtype, samples, max_bytes):
    for i, (z, y, x) in enumerate(plan.shapes):
        shape = (min(chunk_z,z), min(tile_size,y), min(tile_size,x))
        out = math.prod(shape) * samples
        if i == 0:
            needed = out * np.dtype(dtype).itemsize
        else:
            incoming = math.prod(tuple(min(n, s*f) for n,s,f in
                zip(plan.shapes[i-1],shape,plan.steps[i-1],strict=True))) * samples
            # Input, accumulation, output, counts and a conservative temporary
            # allowance. Upstream compressor/native-library scratch is separate.
            needed = incoming*(np.dtype(dtype).itemsize+8) + out*(25+np.dtype(dtype).itemsize)
        # Zarr may pad edge chunks to their configured full shape. This is a
        # known allocation, not compressor scratch, even for a tiny edge image.
        needed += chunk_z * tile_size * tile_size * samples * np.dtype(dtype).itemsize
        if needed > max_bytes:
            raise ValueError("Dataset writer working-block estimate exceeds max_block_bytes; "
                             "reduce tile_size or chunk_z")


def write_zarr_dataset(writer, dataset, *, pyramid_axes="xy", pyramid_factors=None,
                       chunk_z=1, max_block_bytes=64*1024*1024):
    import zarr

    source = DatasetPlaneSource(dataset)
    dataset.require_regular_export()
    spec = source.spec
    chunk_z = integer(chunk_z, "chunk_z", minimum=1)
    max_block_bytes = integer(max_block_bytes, "max_block_bytes", minimum=1)
    if writer.float32_mantissa_bits is not None and spec.dtype != np.dtype("float32"):
        raise TypeError("Mantissa trimming requires float32 pixels")
    method = "nearest" if spec.image_type == "label" else writer.downsample
    spacing = None if spec.z_spacing_um is None else (spec.z_spacing_um, spec.pixel_size.y, spec.pixel_size.x)
    plan = plan_volume_pyramid((spec.size_z,spec.size_y,spec.size_x),tile_size=writer.tile_size,
        chunk_z=chunk_z,levels=writer.pyramid_levels,axes=pyramid_axes,spacing=spacing,factors=pyramid_factors)
    _validate_work_budget(plan,chunk_z=chunk_z,tile_size=writer.tile_size,dtype=spec.dtype,
                          samples=spec.samples_per_pixel,max_bytes=max_block_bytes)
    components = spec.size_c
    dummy = [{"min":0.,"max":1.,"start":0.,"end":1.} for _ in range(components)]
    attrs = dataset_attributes(source,plan,method=method,windows=dummy,bits=writer.float32_mantissa_bits)
    errors = profile_errors(attrs)
    if errors:
        raise ValueError("Generated stack metadata violates its profile: " + "; ".join(errors))
    preflight_directory(writer.path,protected=dataset.backing_paths)
    writer.path.parent.mkdir(parents=True,exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".omeify-zarr-stack-",dir=writer.path.parent))
    serializer, compressors = writer._codecs(spec.dtype)
    readers = source.plane_readers()
    try:
        group = zarr.open_group(str(stage),mode="w",zarr_format=3)
        arrays = []
        try:
            for i, (depth,height,width) in enumerate(plan.shapes):
                array = group.create_array(str(i),shape=(spec.size_t,components,depth,height,width),
                    chunks=(1,1,chunk_z,writer.tile_size,writer.tile_size),dtype=spec.dtype,
                    serializer=serializer,compressors=compressors,filters=None,
                    dimension_names=tuple("tczyx"),
                    chunk_key_encoding={"name":"default","configuration":{"separator":"/"}},
                    fill_value=source.background_label if spec.image_type=="label" else 0,
                    config={"write_empty_chunks":True,"order":"C"})
                arrays.append(array)
                # One source/working block at a time. Upstream Zarr owns codec
                # and store concurrency; this path does not multiply it by an
                # additional plane/thread pool.
                for t in range(spec.size_t):
                    for c in range(spec.logical_channel_count if i==0 else components):
                        for z0 in range(0,depth,chunk_z):
                            z1=min(depth,z0+chunk_z)
                            for y0,y1,x0,x1 in _tiles(height,width,writer.tile_size):
                                if i==0:
                                    shape=(z1-z0,y1-y0,x1-x0)
                                    if spec.is_rgb:
                                        shape=(*shape,3)
                                    block=np.empty(shape,dtype=spec.dtype)
                                    for local,z in enumerate(range(z0,z1)):
                                        index=(t*spec.logical_channel_count+c)*spec.size_z+z
                                        block[local]=readers[index].read_region(y0,y1,x0,x1)
                                    reduced=block
                                else:
                                    fz,fy,fx=plan.steps[i-1]
                                    previous=arrays[i-1]
                                    block=np.asarray(previous[t,c,z0*fz:min(previous.shape[2],z1*fz),
                                        y0*fy:min(previous.shape[3],y1*fy),x0*fx:min(previous.shape[4],x1*fx)])
                                    reduced=reduce_zyx(block,(fz,fy,fx),method=method)
                                if writer.float32_mantissa_bits is not None:
                                    reduced=round_float32_mantissa(reduced,writer.float32_mantissa_bits)
                                if i==0 and spec.is_rgb:
                                    for sample in range(3):
                                        array[t,sample,z0:z1,y0:y1,x0:x1]=reduced[...,sample]
                                else:
                                    array[t,c,z0:z1,y0:y1,x0:x1]=reduced
                readers.clear_cache()
            attrs=dataset_attributes(source,plan,method=method,windows=_windows(arrays[-1],spec.is_rgb),
                                     bits=writer.float32_mantissa_bits)
            errors=profile_errors(attrs)
            if errors:
                raise ValueError("Generated stack metadata violates its profile: " + "; ".join(errors))
            group.attrs.update(attrs)
        finally:
            group.store.close()
            readers.clear_cache()
        verification=_verify(writer,source,stage,plan,method,attrs,serializer,compressors,chunk_z)
        semantic_shapes=[(spec.size_t,*((spec.logical_channel_count,) if spec.axes=="TCZYX" else ()),
                          *shape,*((3,) if spec.is_rgb else ())) for shape in plan.shapes]
        report=complete_report({
            "output_file":{"path":str(writer.path),"format":"ome_zarr","size_bytes":store_size(stage),
                "axes":spec.axes,"shape":list(spec.shape),"dtype":spec.dtype.name},
            "image":{"image_type":spec.image_type,"channel_names":list(spec.channel_names),
                "pixel_size":list(spec.pixel_size.to_tuple()),"z_spacing_um":spec.z_spacing_um,
                "time_increment_s":spec.time_increment_s,"z_origin_um":spec.z_origin_um,
                "time_origin_s":spec.time_origin_s},
            "pyramid":{"axes":pyramid_axes,"method":method,"level_shapes":[list(s) for s in semantic_shapes],
                       "factors_zyx":[list(f) for f in plan.steps]},
            "verification":verification,"metadata":{"ngff_version":"0.5","profile_valid":True},
            "options":{"compression":writer.compression_name,"tile_size":writer.tile_size,
                "lossless_compression":True,"float32_mantissa_bits":writer.float32_mantissa_bits,
                "color_management":source.color_management,"chunk_z":chunk_z,"max_block_bytes":max_block_bytes},
        },"dataset_write_report.schema.json")
        preflight_directory(writer.path,protected=dataset.backing_paths)
        publish_directory(stage,writer.path)
        return report
    finally:
        readers.clear_cache()
        if stage.exists():
            shutil.rmtree(stage)


def _verify(writer,source,stage,plan,method,attributes,serializer,compressors,chunk_z):
    import zarr
    spec=source.spec
    group=zarr.open_group(str(stage),mode="r",use_consolidated=False)
    readers=source.plane_readers()
    chunks_checked=base_points=pyramid_points=0
    try:
        if group.attrs.asdict()!=attributes:
            raise ValueError("Stack metadata readback differs from generated allowlist")
        arrays=[group[str(i)] for i in range(len(plan.shapes))]
        chunks=(1,1,chunk_z,writer.tile_size,writer.tile_size)
        for i,(array,shape) in enumerate(zip(arrays,plan.shapes,strict=True)):
            raw=array.metadata.to_dict()
            expected_shape=(spec.size_t,spec.size_c,*shape)
            if (tuple(array.shape)!=expected_shape or tuple(array.chunks)!=chunks or
                np.dtype(array.dtype).newbyteorder("=")!=spec.dtype or raw.get("zarr_format")!=3 or
                raw.get("node_type")!="array" or tuple(raw.get("dimension_names") or ())!=tuple("tczyx") or
                tuple(raw.get("codecs",()))!=(serializer.to_dict(),*(c.to_dict() for c in compressors)) or
                raw.get("chunk_key_encoding")!={"name":"default","configuration":{"separator":"/"}} or
                raw.get("attributes",{}) or raw.get("storage_transformers",[]) or
                raw.get("fill_value")!=(source.background_label if spec.image_type=="label" else 0)):
                raise ValueError("Stack array structure or storage differs from the plan")
            for coordinates in itertools.product(*(range((n+c-1)//c) for n,c in zip(expected_shape,chunks,strict=True))):
                path=stage/str(i)/"c"/Path(*map(str,coordinates))
                if path.is_symlink() or not path.is_file() or path.stat().st_size==0:
                    raise ValueError("Generated stack has a missing or invalid dense chunk")
                chunks_checked+=1
        if sum(len(names) for _,_,names in os.walk(stage))!=chunks_checked+len(arrays)+1:
            raise ValueError("Generated stack inventory differs from planned dense output")
        with open_dataset(stage,input_type="ome_zarr") as readback:
            if readback.image_type!=spec.image_type or readback.channel_names!=spec.channel_names:
                raise ValueError("Stack image semantics failed readback")
            for i,shape in enumerate(plan.shapes):
                depth,height,width=shape
                actual_level=readback.levels[i]
                if actual_level.shape!=(spec.size_t,spec.size_c,*shape):
                    raise ValueError("Stack logical shape failed readback")
                expected_transform=attributes['ome']['multiscales'][0]['datasets'][i]['coordinateTransformations']
                scale=expected_transform[0]['scale']
                origin=expected_transform[1]['translation'] if len(expected_transform)>1 else [0.]*5
                for name,s,b in zip("TCZYX",scale,origin,strict=True):
                    if name=='C':continue
                    axis=actual_level.axis(name)
                    if not (math.isclose(axis.scale,s,rel_tol=1e-12) and math.isclose(axis.origin,b,rel_tol=1e-12,abs_tol=1e-12)):
                        raise ValueError("Stack coordinate geometry failed readback")
                for t in range(spec.size_t):
                    for z in range(depth):
                        with readback.plane(t=t,z=z,native_level=i) as plane:
                            if plane.icc_profile!=spec.icc_profile:
                                raise ValueError("Stack ICC profile failed readback")
                            for c in range(spec.logical_channel_count):
                                for y,x in _points(height,width):
                                    got=plane[c].read_region(y,y+1,x,x+1)
                                    if i==0:
                                        expected=readers[(t*spec.logical_channel_count+c)*spec.size_z+z].read_region(y,y+1,x,x+1)
                                        base_points+=1
                                    else:
                                        fz,fy,fx=plan.steps[i-1]
                                        previous=arrays[i-1]
                                        components=range(3) if spec.is_rgb else (c,)
                                        values=[]
                                        for component in components:
                                            block=np.asarray(previous[t,component,z*fz:min(previous.shape[2],(z+1)*fz),
                                                y*fy:min(previous.shape[3],(y+1)*fy),x*fx:min(previous.shape[4],(x+1)*fx)])
                                            values.append(reduce_zyx(block,(fz,fy,fx),method=method)[0])
                                        expected=np.stack(values,axis=-1) if spec.is_rgb else values[0]
                                        pyramid_points+=1
                                    if writer.float32_mantissa_bits is not None:
                                        expected=round_float32_mantissa(expected,writer.float32_mantissa_bits)
                                    if not (_equal_bytes(got,expected) if i==0 or method=='nearest' else
                                            np.array_equal(got,expected,equal_nan=True)):
                                        raise ValueError("Stack representative pixel verification failed")
        return {"ome_zarr_recognized":True,"metadata_matches_profile":True,"geometry_matches":True,
            "storage_matches_requested":True,"dense_chunk_inventory_verified":True,"chunks_checked":chunks_checked,
            "output_pixels_decodable":True,"base_pixel_values_checked":True,"base_pixel_values_match":True,
            "pixel_verification":{"mode":"sampled","all_pixels_checked":False,
                "sampling":"top-left, center, bottom-right on every T/C/Z plane; deduplicated",
                "base_points_compared":base_points,"pyramid_points_compared":pyramid_points,
                "source_reader_independent":False,"base_comparison":"bitwise",
                "pyramid_comparison":"bitwise" if method=='nearest' else "numeric_equal_nan"}}
    finally:
        readers.clear_cache()
        group.store.close()
