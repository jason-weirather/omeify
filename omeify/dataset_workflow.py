"""Explicit stack workflows; ordinary 2D conversion/mutation stay on their old path."""
from __future__ import annotations

from contextlib import ExitStack

from .io.dataset import open_dataset
from .io.image_metadata import integer
from .io.output import image_writer
from .reports import complete_report
from .workflow import apply_channel_renames, validate_channel_rename_mapping, validate_image_paths


def stack_requested(*, all_planes=False, z_range=None, t_range=None):
    if not isinstance(all_planes, bool):
        raise TypeError("all_planes must be boolean")
    return all_planes or z_range is not None or t_range is not None


def stack_selection(dataset, *, all_planes=False, z=None, t=None, z_range=None, t_range=None):
    """Resolve explicit half-open scopes, rejecting every ambiguous non-singleton axis."""
    if all_planes and any(v is not None for v in (z,t,z_range,t_range)):
        raise ValueError("all_planes cannot be combined with Z/T indices or ranges")
    result, record = {}, {}
    for name, index, bounds in (("Z",z,z_range),("T",t,t_range)):
        info = dataset.axis_info(name)
        if index is not None and bounds is not None:
            raise ValueError(f"{name} index and range are mutually exclusive")
        value = index
        if bounds is not None:
            if len(bounds) != 2:
                raise ValueError(f"{name} range requires a half-open (start, stop) pair")
            lo,hi=(integer(v,name+" range") for v in bounds)
            value=slice(lo,hi)
        elif all_planes and info is not None:
            value=slice(None)
        indices=dataset._indices(name,value,0)
        # select(None) means all in the view factory; resolve absent/singleton here.
        result[name.lower()]=value if isinstance(value,slice) else indices[0]
        record[name.lower()]=None if info is None else [indices[0],indices[-1]+1]
    return result,record


def process_dataset(input_path, output_path, *, operation, input_type, output_format="ome_tiff",
        series=0, group="", z=None, t=None, all_planes=False, z_range=None, t_range=None,
        image_type=None, intrinsic_only=False, storage_options=None,
        rename_channels=None, rename_channels_by=None, pixel_size=None,
        z_spacing_um=None, time_increment_s=None, pyramid_axes="xy", pyramid_factors=None,
        chunk_z=1, max_block_bytes=64*1024*1024, dtype=None, float32_mantissa_bits=None,
        range_mode=None, sample_pixels_per_channel=262144, auto_max_normalized_rmse=.005,
        compression=None,tile_size=None,pyramid_levels=None,downsample="mean",max_workers=None,
        display_uuid=True,software=None,overwrite=True,cache_directory=None,
        jpeg_quality=90,jpeg_subsampling="422"):
    """Write an explicitly scoped regular stack, preserving fresh metadata policy."""
    if operation not in {"convert","mutate"}:
        raise ValueError("Unsupported dataset operation")
    if input_type not in {"ome_tiff","ome_zarr"}:
        raise ValueError("Stack workflows require OME-TIFF or OME-Zarr input")
    if output_format == "ome_tiff" and (pyramid_axes != "xy" or pyramid_factors is not None or
                                        chunk_z != 1 or max_block_bytes != 64*1024*1024):
        raise ValueError("XYZ pyramid options and chunk_z require OME-Zarr output")
    if operation == "convert" and (dtype is not None or float32_mantissa_bits is not None):
        raise ValueError("Numerical changes belong to mutate, not convert")
    if operation == "mutate":
        if (dtype is None) == (float32_mantissa_bits is None):
            raise ValueError("mutate requires exactly one of dtype or float32_mantissa_bits")
        if float32_mantissa_bits is not None:
            from .io._writer.configuration import normalize_float32_mantissa_bits
            if range_mode is not None:
                raise ValueError("range_mode requires dtype mutation")
            float32_mantissa_bits = normalize_float32_mantissa_bits(float32_mantissa_bits)
            if float32_mantissa_bits == 23:
                raise ValueError(
                    "float32_mantissa_bits=23 retains full float32 precision; use convert "
                    "when no pixel-value mutation is intended"
                )
    source,destination=validate_image_paths(input_path,output_path,input_type=input_type,
                                           output_format=output_format,overwrite=overwrite)
    renames,mode=validate_channel_rename_mapping(rename_channels,rename_channels_by)
    writer=image_writer(destination,output_format=output_format,compression=compression,
        tile_size=tile_size,pyramid_levels=pyramid_levels,downsample=downsample,max_workers=max_workers,
        float32_mantissa_bits=float32_mantissa_bits,overwrite=overwrite,cache_directory=cache_directory,
        display_uuid=display_uuid,software=software,jpeg_quality=jpeg_quality,jpeg_subsampling=jpeg_subsampling)
    with ExitStack() as stack:
        original=stack.enter_context(open_dataset(source,input_type=input_type,series=series,group=group,
            image_type=image_type,intrinsic_only=intrinsic_only,storage_options=storage_options))
        selection,coordinates=stack_selection(original,all_planes=all_planes,z=z,t=t,
                                               z_range=z_range,t_range=t_range)
        selected=stack.enter_context(original.select(**selection))
        names=apply_channel_renames(selected.channel_names,renames,mode)
        prepared=stack.enter_context(selected.with_metadata(channel_names=names,pixel_size=pixel_size,
                            z_spacing_um=z_spacing_um,time_increment_s=time_increment_s))
        mutation=None
        if operation == "mutate":
            if prepared.image_type != "multichannel" or prepared.dtype.kind != "f":
                raise TypeError("Stack mutation requires scalar floating-point intensity data")
            if dtype is not None:
                from .dataset_mutation import analyze_dataset_dtype_mutation, apply_dataset_dtype_mutation
                plans=analyze_dataset_dtype_mutation(prepared,dtype=dtype,range_mode=range_mode or "auto",
                    sample_pixels_per_channel=sample_pixels_per_channel,
                    auto_max_normalized_rmse=auto_max_normalized_rmse)
                mutation={"kind":"integer","source_dtype":prepared.dtype.name,
                    "target_dtype":plans[0].target_dtype.name,"range_mode":range_mode or "auto",
                    "scope":"one fixed mapping per channel across the selected T/Z extent",
                    "sample_layout":"deterministic row-concatenated T/Z raster; no physical interpretation",
                    "channels":[p.report for p in plans]}
                prepared=stack.enter_context(apply_dataset_dtype_mutation(prepared,plans))
            else:
                mutation={"kind":"float_precision","float32_mantissa_bits":float32_mantissa_bits}
        if output_format == "ome_zarr":
            report=writer.write_dataset(prepared,pyramid_axes=pyramid_axes,pyramid_factors=pyramid_factors,
                                        chunk_z=chunk_z,max_block_bytes=max_block_bytes)
        else:
            report=writer.write_dataset(prepared)
    return complete_report({"operation":operation,"input":{"path":str(source),"format":input_type,
            "series":series,"group":group,"z_range":coordinates["z"],"t_range":coordinates["t"]},
        "writer_report":report,"mutation":mutation},"dataset_operation_report.schema.json")
