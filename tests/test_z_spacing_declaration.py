"""Strict Z intent on conversion/mutation; real writer tests require normal deps."""
from __future__ import annotations

import inspect

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from lxml import etree

from omeify import AxisInfo, ImageDataset, PixelSize, convert, mutate, open_dataset
from omeify.cli import main
from omeify.dataset_workflow import resolve_z_spacing
from omeify.reports import validate_report


def source_tiff(tmp_path, *, dtype="uint16", z_size=4, physical_z=False):
    source = tmp_path / "source.ome.tif"
    values = np.arange(2*z_size*16*20, dtype=np.float32).reshape(2, z_size, 16, 20)
    values = values.astype(dtype)
    metadata = {"axes": "TZYX", "PhysicalSizeX": .5, "PhysicalSizeY": .6}
    if physical_z:
        metadata["PhysicalSizeZ"] = 2.5
    tifffile.imwrite(source, values, ome=True, metadata=metadata)
    return source, values


@pytest.mark.parametrize("operation", ("convert", "mutate"))
def test_missing_z_spacing_requires_explicit_intent_before_writing(tmp_path, operation):
    src, _ = source_tiff(tmp_path, dtype="float32" if operation == "mutate" else "uint16")
    target = tmp_path / "out.ome.tif"
    f = convert if operation == "convert" else mutate
    opts = {"float32_mantissa_bits": 10} if operation == "mutate" else {}
    with pytest.raises(ValueError, match="--z-spacing-ordinal.*--z-spacing-size"):
        f(src, target, input_type="ome_tiff", **opts)
    assert not target.exists()


@pytest.mark.parametrize("operation", ("convert", "mutate"))
def test_missing_z_spacing_cli_explains_both_ways_forward(tmp_path, operation):
    src, _ = source_tiff(tmp_path, dtype="float32" if operation == "mutate" else "uint16")
    cmd = [operation, str(src), "--type", "ome_tiff", "-o", str(tmp_path / "out.tif")]
    if operation == "mutate":
        cmd += ["--float32-mantissa-bits", "10"]
    result = CliRunner().invoke(main, cmd)
    assert result.exit_code != 0
    assert "--z-spacing-ordinal" in result.output
    assert "--z-spacing-size" in result.output
    assert "--z-spacing-unit" in result.output
    assert not (tmp_path / "out.tif").exists()


def test_units_use_same_physical_length_conversions_as_xy():
    pixels = np.zeros((2, 4, 6, 8), dtype=np.uint16)
    with ImageDataset.from_array(pixels, axes="TZYX", pixel_size=PixelSize(.5, .5, "µm")) as ds:
        assert resolve_z_spacing(ds, z_spacing_size=500, z_spacing_unit="nm") == (.5, "override")
        assert resolve_z_spacing(ds, z_spacing_size=.5) == (.5, "override")
        assert resolve_z_spacing(ds, z_spacing_ordinal=True) == (None, "ordinal")
        with pytest.raises(ValueError, match="--z-spacing-unit requires --z-spacing-size"):
            resolve_z_spacing(ds, z_spacing_unit="nm")
        with pytest.raises(ValueError, match="conflicts"):
            resolve_z_spacing(ds, z_spacing_ordinal=True, z_spacing_size=1)
        with pytest.raises(ValueError, match="Unsupported physical-size unit"):
            resolve_z_spacing(ds, z_spacing_size=1, z_spacing_unit="furlong")
        with pytest.raises(ValueError, match="positive finite"):
            resolve_z_spacing(ds, z_spacing_size=0)
        with pytest.raises(ValueError, match="positive finite"):
            resolve_z_spacing(ds, z_spacing_size=float("nan"))


def test_source_spacing_is_used_and_cannot_be_discarded_as_ordinal():
    vals=np.zeros((2,4,6,8), dtype=np.uint16)
    with ImageDataset.from_array(vals, axes="TZYX", pixel_size=PixelSize(.5,.5,"µm"),
                                 axis_info=(AxisInfo("Z",4,"space","µm",2.5),)) as ds:
        assert resolve_z_spacing(ds) == (None,"source")
        with pytest.raises(ValueError, match="cannot discard known physical spacing"):
            resolve_z_spacing(ds, z_spacing_ordinal=True)
        assert resolve_z_spacing(ds,z_spacing_size=3) == (3,"override")


def test_ordinal_requirement_only_applies_to_multi_z_output(tmp_path):
    src, _ = source_tiff(tmp_path, z_size=4)
    with open_dataset(src, input_type="ome_tiff") as ds:
        with ds.select(z=2) as selected:
            assert resolve_z_spacing(selected) == (None,"not_applicable")
            with pytest.raises(ValueError, match="multiple Z sections"):
                resolve_z_spacing(selected, z_spacing_ordinal=True)
    # Selecting Z=2 leaves T=2 in this source: a valid T-only stack without Z calibration.
    report = convert(src, tmp_path / "t-only.ome.tif", input_type="ome_tiff", z=2,
                     compression="Deflate", tile_size=16, pyramid_levels=0)
    assert report["input"]["z_spacing_policy"] == "not_applicable"
    # Two indices are the established 2D API path and never demand Z spacing.
    selected = convert(src, tmp_path / "plane.ome.tif", input_type="ome_tiff", z=2, t=1,
                       compression="Deflate", tile_size=16, pyramid_levels=0)
    assert selected["schema"] == "omeify.convert/2"


@pytest.mark.parametrize("format", ("ome_tiff", "ome_zarr"))
@pytest.mark.parametrize("args,mode,expected", (
    ({"z_spacing_ordinal": True}, "ordinal", None),
    ({"z_spacing_size": 500, "z_spacing_unit": "nm"}, "override", .5),
    ({"z_spacing_size": 1.25}, "override", 1.25),
))
def test_real_converter_writes_explicit_interpretation(tmp_path,format,args,mode,expected):
    src, values = source_tiff(tmp_path)
    dest = tmp_path / ("out.tif" if format=="ome_tiff" else "out.zarr")
    report=convert(src,dest,input_type="ome_tiff",output_format=format,
                   compression="Deflate" if format=="ome_tiff" else "ZSTD",
                   tile_size=16,pyramid_levels=0,**args)
    validate_report(report,"dataset_operation_report.schema.json")
    assert report["input"]["z_spacing_policy"] == mode
    if mode == "override":
        assert report["input"]["z_spacing_override"] == {
            "size": args["z_spacing_size"],
            "unit": args.get("z_spacing_unit", "µm"),
        }
    else:
        assert report["input"]["z_spacing_override"] is None
    assert report["writer_report"]["image"]["z_spacing_um"] == expected
    with open_dataset(dest,input_type=format) as ds:
        zinfo=ds.axis_info("Z")
        if expected is None:
            assert zinfo.unit is None  # No claim of physical Z calibration.
        else:
            assert (zinfo.unit,zinfo.scale)==("µm",expected)
        with ds.plane(t=1,z=3) as image:
            assert image.channel_count == 1
            actual = image.asarray()
            # NGFF stack metadata contains an explicit C=1 component, whereas
            # TIFF can expose the same singleton observation directly as YX.
            assert image.axes in ("YX", "CYX")
            np.testing.assert_array_equal(
                actual[0] if image.axes == "CYX" else actual, values[1,3]
            )
    if format=="ome_tiff":
        with tifffile.TiffFile(dest) as tif:
            ns="{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
            pixels=etree.fromstring(tif.ome_metadata.encode()).find(f"{ns}Image/{ns}Pixels")
            assert ("PhysicalSizeZ" in pixels.attrib)==(expected is not None)
            assert ("PhysicalSizeZUnit" in pixels.attrib)==(expected is not None)
            if expected is not None:
                assert float(pixels.get("PhysicalSizeZ"))==expected
                assert pixels.get("PhysicalSizeZUnit")=="µm"


def test_real_mutation_ordinal_and_override(tmp_path):
    src, values = source_tiff(tmp_path, dtype="float32")
    for params, expected in (({"z_spacing_ordinal":True},None),
                             ({"z_spacing_size":750,"z_spacing_unit":"nm"},.75)):
        dest=tmp_path / ("ordinal.tif" if expected is None else "physical.tif")
        report=mutate(src,dest,input_type="ome_tiff",float32_mantissa_bits=10,
                      compression="Deflate",tile_size=16,pyramid_levels=0,**params)
        validate_report(report,"dataset_operation_report.schema.json")
        assert report["writer_report"]["image"]["z_spacing_um"] == expected
        with open_dataset(dest,input_type="ome_tiff") as ds, ds.plane(t=1,z=3) as img:
            assert img.asarray().shape == values[1,3].shape


def test_removed_option_and_no_misleading_2d_override(tmp_path):
    assert "z_spacing_um" not in inspect.signature(convert).parameters
    assert "z_spacing_um" not in inspect.signature(mutate).parameters
    for command in ("convert", "mutate"):
        result=CliRunner().invoke(main,[command,"--help"])
        assert result.exit_code == 0
        for option in ("--z-spacing-size", "--z-spacing-unit", "--z-spacing-ordinal"):
            assert option in result.output
        assert "--z-spacing-um" not in result.output
        rejected=CliRunner().invoke(main,[command,"--z-spacing-um","1"])
        assert rejected.exit_code != 0
        assert "No such option" in rejected.output
    src,_=source_tiff(tmp_path,z_size=1)
    with pytest.raises(ValueError,match="multiple Z sections"):
        convert(src,tmp_path/"unexpected.tif",input_type="ome_tiff",z_spacing_ordinal=True)

@pytest.mark.parametrize("flag", ("--z-spacing-ordinal", "--z-spacing-size"))
def test_cli_can_convert_with_explicit_z_interpretation(tmp_path,flag):
    src, _ = source_tiff(tmp_path)
    dest = tmp_path / "cli.ome.tif"
    options = [flag] if flag == "--z-spacing-ordinal" else [flag, "500", "--z-spacing-unit", "nm"]
    result = CliRunner().invoke(main,["convert",str(src),"--type","ome_tiff", "-o", str(dest),
                                      "--compression","Deflate", "--tile-size","16",
                                      "--pyramid-levels","0", *options])
    assert result.exit_code == 0, (result.output, result.exception)
    with open_dataset(dest,input_type="ome_tiff") as ds:
        axis=ds.axis_info("Z")
        if flag == "--z-spacing-ordinal":
            assert axis.unit is None
        else:
            assert (axis.scale,axis.unit)==(.5,"µm")


def test_ome_zarr_input_requires_explicit_z_intent(tmp_path):
    from omeify import OMEZarrWriter
    data=np.arange(2*4*16*20,dtype=np.uint16).reshape(2,4,16,20)
    zarr_source=tmp_path/"unitless.zarr"
    with ImageDataset.from_array(data,axes="TZYX",pixel_size=PixelSize(.5,.6,"µm")) as ds:
        # Low-level writers are allowed to represent an uncalibrated ordinal dataset.
        OMEZarrWriter(zarr_source,tile_size=16,pyramid_levels=0).write_dataset(ds)
    with pytest.raises(ValueError,match="--z-spacing-ordinal"):
        convert(zarr_source,tmp_path/"refused.tif",input_type="ome_zarr")
    dest=tmp_path/"ordinal.tif"
    report=convert(zarr_source,dest,input_type="ome_zarr",z_spacing_ordinal=True,
                   compression="Deflate",tile_size=16,pyramid_levels=0)
    assert report["input"]["z_spacing_policy"]=="ordinal"
    with open_dataset(dest,input_type="ome_tiff") as ds:
        assert ds.axis_info("Z").unit is None
        with ds.plane(t=1,z=3) as img:
            np.testing.assert_array_equal(img.asarray(),data[1,3])


def test_dataset_operation_report_consistency_checks_z_policy():
    import json
    from omeify.schemas import __file__ as schema_package
    from pathlib import Path
    schema=json.loads((Path(schema_package).parent/"dataset_operation_report.schema.json").read_text())
    # Exercise the conditional schema independently of actual storage dependencies.
    allof=schema["allOf"]
    assert len(allof)==4
    from jsonschema import Draft202012Validator
    for policy,valid,invalid in (("ordinal",None,1.),("override",.5,None),("source",.5,None)):
        # The conditional branches must actually discriminate on Z policy.
        check=Draft202012Validator({"allOf":allof})
        entry={"input":{"z_spacing_policy":policy,
                         "z_spacing_override": ({"size": 500., "unit": "nm"}
                                                if policy == "override" else None)},
               "writer_report":{"image":{"z_spacing_um":valid}}}
        assert check.is_valid(entry)
        entry["writer_report"]["image"]["z_spacing_um"]=invalid
        assert not check.is_valid(entry)
    override = {"input": {"z_spacing_policy": "override", "z_spacing_override": None},
                "writer_report": {"image": {"z_spacing_um": .5}}}
    assert not check.is_valid(override)
