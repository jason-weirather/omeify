"""Real storage and workflow regressions. No fake Zarr implementation or codec."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from click.testing import CliRunner

zarr = pytest.importorskip("zarr", minversion="3.1.3")

from omeify import (ImageLevel, LabelImage, MultichannelImage, OMEImageSeries,
                   OMEZarrCollectionWriter, OMEZarrLabelReader, OMEZarrReader,
                   OMEZarrWriter, PixelSize, RGBImage, TemporaryOMEZarrWriter,
                   ZarrInspector, convert, crop, mutate)
from omeify.cli import main
from omeify.io.ome_zarr.metadata import ArrayInfo, parse_image
from omeify.io.ome_zarr.store import publish_directory
from omeify.reports import validate_report
from zarr_fixture import attributes, fixture


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_versioned_regional_reads_lifetime_and_origins(tmp_path, version):
    path = tmp_path / "source"
    data, arrays, _ = fixture(path, version=version)
    with OMEZarrReader(path) as image:
        assert image.ngff_version == version
        assert image.axes == "CYX"
        assert image.origin_yx == (11.25, -9.5)
        assert image.pixel_size == PixelSize(.5, .6, "µm")
        assert image.levels[1].offset_yx == (-.5, -.5)
        np.testing.assert_array_equal(image.read_region(13, 29, 17, 38, channels=[2, 0, 2]),
                                      data[[2, 0, 2], 13:29, 17:38])
        np.testing.assert_array_equal(image.asarray(level=1), arrays[1])
        with image.crop(3, 23, 5, 26) as selected:
            assert selected.origin_yx == (13.05, -7.0)
            np.testing.assert_array_equal(selected.asarray(), data[:, 3:23, 5:26])
        handle = image[0]
        snapshot = image.inspect()
    assert snapshot.report["ngff_version"] == version
    with pytest.raises(RuntimeError):
        handle.asarray()
    with image:
        with pytest.raises(RuntimeError):
            handle.asarray()


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_explicit_planes_and_spatial_axis_reordering(tmp_path, version):
    data = np.arange(2*3*2*35*31, dtype="uint16").reshape(2,3,2,35,31)
    path = tmp_path / "five-d"
    fixture(path, data=data, axes="tczxy", version=version)
    with pytest.raises(ValueError, match="explicit"):
        with OMEZarrReader(path):
            pass
    with OMEZarrReader(path, z=1, t=1) as im:
        np.testing.assert_array_equal(im.asarray(), data[1,:,1].transpose(0,2,1))
    with pytest.raises(IndexError):
        with OMEZarrReader(path, z=2, t=0):
            pass


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_normalize_all_versions_through_convert(tmp_path, version):
    source = tmp_path / "source"
    data, _, attrs = fixture(source, version=version)
    root = zarr.open_group(source, mode="a")
    root.attrs["secret_patient"] = "should-not-escape"
    root["0"].attrs["private"] = "should-not-escape"
    root.store.close()
    report = convert(source, tmp_path / "normalized", input_type="ome_zarr",
                     output_format="ome_zarr", tile_size=16, max_workers=2,
                     rename_channels={0: "DAPI"}, rename_channels_by="index")
    validate_report(report, "conversion_report.schema.json")
    assert report["input_file"]["size_bytes"] is None
    assert report["ome"]["version"] == "0.5"
    with OMEZarrReader(tmp_path / "normalized") as im:
        np.testing.assert_array_equal(im.asarray(), data)
        assert im.origin_yx == (11.25, -9.5)
        assert im.channel_names[0] == "DAPI"
    for p in (tmp_path / "normalized").rglob("*.json"):
        assert "should-not-escape" not in p.read_text()
    with pytest.raises(FileExistsError):
        convert(source, tmp_path / "normalized", input_type="ome_zarr", output_format="ome_zarr")


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "uint32", "int8", "int16", "int32", "float32", "float64"])
@pytest.mark.parametrize("compression", ["ZSTD", "Gzip", "Uncompressed"])
def test_writer_numeric_fidelity_and_pyramids(tmp_path, dtype, compression):
    data = (np.arange(2*33*47).reshape(2,33,47) % 101).astype(dtype)
    path = tmp_path / "out"
    with MultichannelImage.from_array(data, axes="CYX", pixel_size=PixelSize(.5,.5,"µm")) as im:
        report = OMEZarrWriter(path, compression=compression, tile_size=16, max_workers=2).write(im)
    validate_report(report, "zarr_write_report.schema.json")
    assert report["verification"]["dense_chunk_inventory_verified"]
    with OMEZarrReader(path) as im:
        assert im.dtype == data.dtype
        assert im.origin_yx == (0, 0)
        assert im.levels[1].origin_yx == (.25, .25)
        np.testing.assert_array_equal(im.asarray(), data)
        from omeify.io._writer.pyramid import mean_downsample_2x
        expected = np.stack([mean_downsample_2x(c, (17,24)) for c in data])
        np.testing.assert_array_equal(im.asarray(level=1), expected)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_float_payloads_and_signed_zero(tmp_path, dtype):
    bits = np.array([0, 0x80000000, 0x7fc01234, 0x7f800000, 0xff800000], dtype="uint32")
    if dtype == "float64":
        bits = np.array([0, 0x8000000000000000, 0x7ff8000000001234,
                         0x7ff0000000000000, 0xfff0000000000000], dtype="uint64")
    data = np.tile(bits.view(dtype), (19, 7))
    with MultichannelImage.from_array(data, axes="YX", pixel_size=PixelSize(1,1,"µm")) as im:
        OMEZarrWriter(tmp_path / "out", tile_size=16).write(im)
    with OMEZarrReader(tmp_path / "out") as im:
        assert im.asarray().tobytes() == data.tobytes()


def test_rgb_and_labels_have_explicit_meaning(tmp_path):
    rgb = (np.arange(33*47*3) % 255).astype("uint8").reshape(33,47,3)
    with RGBImage.from_array(rgb, pixel_size=PixelSize(.5,.5,"µm"), icc_profile=b"test-icc") as im:
        OMEZarrWriter(tmp_path / "rgb", tile_size=16).write(im)
    with OMEZarrReader(tmp_path / "rgb") as im:
        assert im.image_type == "rgb" and im.channel_count == 1
        assert im._metadata.icc_profile == b"test-icc"
        np.testing.assert_array_equal(im.asarray(), rgb)
    for dtype, background in [("int32", -1), ("uint32", 0)]:
        data = np.full((33,47), background, dtype=dtype)
        data[2:19,6:38] = 1234567
        with LabelImage.from_array(data, pixel_size=PixelSize(1,1,"µm"), background_label=background) as im:
            OMEZarrWriter(tmp_path / dtype, tile_size=16).write(im)
        with OMEZarrLabelReader(tmp_path / dtype) as im:
            np.testing.assert_array_equal(im.asarray(level=1), data[::2, ::2])
            assert im._metadata.background_label == background
            assert im.levels[1].origin_yx == (0,0)
            assert im.levels[1].offset_yx == (-.5,-.5)


def test_three_channels_are_not_guessed_rgb(tmp_path):
    data, _, _ = fixture(tmp_path / "a", data=np.zeros((3,20,20),dtype="uint8"))
    with OMEZarrReader(tmp_path / "a") as im:
        assert im.image_type == "multichannel"
    with OMEZarrReader(tmp_path / "a", image_type="rgb") as im:
        assert im.asarray().shape == (20,20,3)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_independent_community_model_reads_fixture(tmp_path, version):
    model = pytest.importorskip("ome_zarr_models.v" + version.replace(".", ""))
    path = tmp_path / "foreign"
    fixture(path, version=version)
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    try:
        model.Image.from_zarr(root)
    finally:
        root.store.close()


def test_independent_model_reads_canonical_output(tmp_path):
    model = pytest.importorskip("ome_zarr_models.v05")
    path = tmp_path / "out"
    with MultichannelImage.from_array(np.zeros((2,33,47),dtype="uint16"), axes="CYX",
                                     pixel_size=PixelSize(1,1,"µm")) as im:
        OMEZarrWriter(path,tile_size=16).write(im)
    root = zarr.open_group(path, mode="r", use_consolidated=False)
    try:
        found = model.Image.from_zarr(root)
        assert found.attributes.ome.version == "0.5"
    finally:
        root.store.close()


def test_extra_frames_cannot_be_laundered_by_views(tmp_path):
    p = tmp_path / "source"
    fixture(p, version="0.6")
    root = zarr.open_group(p,mode="a")
    attrs = root.attrs.asdict()
    ms = attrs["ome"]["multiscales"][0]
    ms["coordinateSystems"].append({"name":"registered", "axes":copy.deepcopy(ms["coordinateSystems"][0]["axes"])})
    ms["coordinateTransformations"] = [{"type":"translation", "translation":[0,2,3],
                                          "input":{"name":"physical"},"output":{"name":"registered"}}]
    root.attrs.update(attrs)
    root.store.close()
    with OMEZarrReader(p) as im:
        with im.crop(1,20,1,20) as view:
            with pytest.raises(ValueError, match="spatial relationships"):
                OMEZarrWriter(tmp_path / "crop").write(view)
        with MultichannelImage.from_channels([im[0]]) as view:
            with pytest.raises(ValueError, match="spatial relationships"):
                OMEZarrWriter(tmp_path / "channels").write(view)
    with OMEZarrReader(p, intrinsic_only=True) as im:
        OMEZarrWriter(tmp_path / "explicit").write(im)


def test_intensity_and_precision_mutations_reuse_math(tmp_path):
    p = tmp_path / "source"
    data = np.linspace(0, 4095, 2*33*47, dtype="float32").reshape(2,33,47)
    fixture(p, data=data)
    report = mutate(p, tmp_path / "int", input_type="ome_zarr", output_format="ome_zarr",
                    dtype="uint16", range_mode="preserve", tile_size=16)
    validate_report(report, "mutation_report.schema.json")
    with OMEZarrReader(tmp_path / "int") as im:
        np.testing.assert_array_equal(im.asarray(), np.rint(data).astype("uint16"))
    report = mutate(p, tmp_path / "float", input_type="ome_zarr", output_format="ome_zarr",
                    float32_mantissa_bits=11, tile_size=16)
    from omeify.io._writer.precision import round_float32_mantissa
    with OMEZarrReader(tmp_path / "float") as im:
        np.testing.assert_array_equal(im.asarray(), round_float32_mantissa(data,11))
    assert report["options"]["metadata_minimization"]


def test_collection_crop_shatter_and_temporary_writer(tmp_path):
    source = tmp_path / "source"
    data, _, _ = fixture(source)
    geo = {"type":"FeatureCollection","features":[
        {"type":"Feature","properties":{"name":n},"geometry":{"type":"Polygon", "coordinates":
           [[[x,2],[x+20,2],[x+20,23],[x,23],[x,2]]]}}
        for n,x in [("left",1),("right",27),("left",49)]]}
    out = tmp_path / "rois"
    r = crop(source,out,input_type="ome_zarr",output_format="ome_zarr",geojson=geo,tile_size=16)
    validate_report(r,"crop_report.schema.json")
    for i, x in enumerate((1,27,49)):
        with OMEZarrReader(out,series=i) as im:
            np.testing.assert_array_equal(im.asarray(), data[:,2:23,x:x+20])
            assert im.origin_yx == (12.45, -9.5+x*.5)
    r = crop(source,tmp_path / "split.ome.zarr",input_type="ome_zarr",output_format="ome_zarr",
             geojson=geo,shatter="by_name",tile_size=16)
    assert len(r["outputs"]) == 2
    with OMEZarrReader(source) as im, TemporaryOMEZarrWriter(tile_size=16) as writer:
        writer.write(im)
        path = writer.path
        assert path.exists()
    assert not path.exists()


@pytest.mark.parametrize("args", [
    ["convert", "--output-format", "ome_zarr"],
    ["mutate", "--output-format", "ome_zarr", "--dtype", "uint16", "--range-mode", "preserve"],
    ["crop", "--output-format", "ome_zarr", "--bounds", "1", "2", "25", "31"],
])
def test_cli_workflows(tmp_path, args):
    fixture(tmp_path / "source",data=np.ones((2,33,47),dtype="float32"))
    result = CliRunner().invoke(main, [*args,str(tmp_path / "source"),"--type","ome_zarr",
                                      "-o",str(tmp_path / "out"),"--tile-size","16"])
    assert result.exit_code == 0, result.output
    assert "Wrote" in result.output
    inspection = CliRunner().invoke(main,["inspect",str(tmp_path / "out"),"--json"])
    assert inspection.exit_code == 0, inspection.output
    assert json.loads(inspection.output)["ngff_version"] == "0.5"


@pytest.mark.parametrize("key,value",[("type","affine"),("scale",[1,-1,1])])
def test_invalid_intrinsic_transform_is_not_silently_ignored(tmp_path,key,value):
    p = tmp_path / "a";fixture(p,version="0.6")
    root=zarr.open_group(p,mode="a");a=root.attrs.asdict()
    outer=a["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    if key == "scale" and outer["type"] == "sequence":
        outer["transformations"][0][key] = value
    else:
        outer[key]=value
    root.attrs.update(a);root.store.close()
    with pytest.raises((ValueError,NotImplementedError)):
        with OMEZarrReader(p): pass
    assert ZarrInspector(p).report["diagnostics"]


def test_missing_metadata_and_array_dimensions_fail(tmp_path):
    p=tmp_path / "a";fixture(p)
    raw=json.loads((p/"0/zarr.json").read_text());raw["dimension_names"]=["y","x","c"]
    (p/"0/zarr.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError,match="dimension_names"):
        with OMEZarrReader(p):pass


def test_source_and_destination_tree_protection(tmp_path):
    p=tmp_path / "a";fixture(p)
    with pytest.raises(ValueError,match="overlap"):
        convert(p,p/"bad",input_type="ome_zarr",output_format="ome_zarr")
    result=CliRunner().invoke(main,["inspect",str(p),"-o",str(p/"zarr.json")])
    assert result.exit_code != 0
    (tmp_path/"alias").symlink_to(p,target_is_directory=True)
    with pytest.raises((ValueError,FileExistsError)):
        convert(p,tmp_path/"alias",input_type="ome_zarr",output_format="ome_zarr")


def test_publication_never_replaces_racing_destination(tmp_path):
    stage=tmp_path/"stage";stage.mkdir();(stage/"new").write_text("new")
    target=tmp_path/"target";target.mkdir();(target/"old").write_text("old")
    with pytest.raises(OSError): publish_directory(stage,target)
    assert (target/"old").read_text()=="old" and stage.exists()


@pytest.mark.parametrize("fault", ["missing", "corrupt", "extra"])
def test_writer_verification_failure_cleans_staging(tmp_path, monkeypatch, fault):
    from omeify.io.ome_zarr import writer as module
    original = module.OMEZarrWriter._verify
    def fail(self,stage,*args,**kwargs):
        chunk=next(p for p in (stage/"0/c").rglob("*") if p.is_file())
        if fault=="missing":chunk.unlink()
        elif fault=="corrupt":chunk.write_bytes(b"not a zstd frame")
        else:(stage/"secret").write_text("not allowed")
        return original(self,stage,*args,**kwargs)
    monkeypatch.setattr(module.OMEZarrWriter,"_verify",fail)
    with MultichannelImage.from_array(np.zeros((33,47),dtype="uint16"),axes="YX",
                                     pixel_size=PixelSize(1,1,"µm")) as im:
        with pytest.raises(Exception):OMEZarrWriter(tmp_path/"out",tile_size=16).write(im)
    assert not list(tmp_path.iterdir())


def test_nan_fill_metadata_can_be_inspected(tmp_path):
    fixture(tmp_path/"a",version="0.4",data=np.full((2,20,20),np.nan,dtype="float32"),fill_value=np.nan)
    report=ZarrInspector(tmp_path/"a").to_json()
    assert json.loads(report)["ngff_version"]=="0.4"


def test_06_array_dimension_names_are_not_05_axis_contract(tmp_path):
    p=tmp_path/"a";data,_,_=fixture(p,version="0.6")
    # 0.6 defines array-to-intrinsic geometry by ordered transforms. It no longer
    # carries 0.5's requirement to duplicate intrinsic axes in dimension_names.
    for path in [p/"0/zarr.json",p/"1/zarr.json"]:
        raw=json.loads(path.read_text());raw.pop("dimension_names",None)
        path.write_text(json.dumps(raw))
    with OMEZarrReader(p) as im:
        np.testing.assert_array_equal(im.asarray(),data)


def test_explicit_selected_level_is_new_base(tmp_path):
    data,arrays,_=fixture(tmp_path/"a",version="0.6")
    with OMEZarrReader(tmp_path/"a") as im:
        OMEZarrWriter(tmp_path/"out",tile_size=16).write(im,level=1)
    with OMEZarrReader(tmp_path/"out") as im:
        np.testing.assert_array_equal(im.asarray(),arrays[1])
        assert im.pixel_size==PixelSize(1,1.2,"µm")
        assert im.origin_yx==(11.25,-9.5)


def test_local_chunk_symlink_cannot_escape_store(tmp_path):
    fixture(tmp_path/"a",levels=1)
    chunk=tmp_path/"a/0/c/0/0/0"
    outside=tmp_path/"private";outside.write_bytes(chunk.read_bytes())
    chunk.unlink();chunk.symlink_to(outside)
    with OMEZarrReader(tmp_path/"a") as im:
        with pytest.raises(ValueError,match="outside"):
            im.read_region(0,1,0,1,channels=0)


@pytest.mark.parametrize("kwargs",[{"tile_size":0},{"max_workers":0},{"pyramid_levels":-1},
                                  {"compression":""},{"compression":"JPEG"}])
def test_bad_writer_options_fail_before_output(tmp_path,kwargs):
    with pytest.raises((ValueError,TypeError)):
        OMEZarrWriter(tmp_path/"out",**kwargs)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("version",["0.5","0.6"])
def test_sharded_zarr_v3_input(tmp_path,version):
    p=tmp_path/"a"
    data=np.arange(3*65*81,dtype="uint16").reshape(3,65,81)
    root=zarr.open_group(p,mode="w",zarr_format=3)
    a=root.create_array("0",shape=data.shape,dtype=data.dtype,chunks=(1,16,16),
                        shards=(1,64,64),dimension_names=("c","y","x"))
    a[:]=data
    root.attrs.update(attributes(version,"cyx",["0"],[[1,.6,.5]],[[0,10,20]]))
    root.store.close()
    with OMEZarrReader(p) as im:
        np.testing.assert_array_equal(im.read_region(12,39,13,41,channels=[2,0]),data[[2,0],12:39,13:41])


@pytest.mark.parametrize("field,value", [("fill_value", 7), ("attributes", {"private": "leak"})])
def test_array_metadata_corruption_prevents_publication(tmp_path, monkeypatch, field, value):
    path = tmp_path / "out"
    writer = OMEZarrWriter(path, tile_size=16, pyramid_levels=0)
    verify = writer._verify

    def tamper(source, stage, *args):
        node = stage / "0" / "zarr.json"
        metadata = json.loads(node.read_text())
        metadata[field] = value
        node.write_text(json.dumps(metadata))
        return verify(source, stage, *args)

    monkeypatch.setattr(writer, "_verify", tamper)
    with MultichannelImage.from_array(np.ones((32, 32), dtype="uint16"), axes="YX",
                                     pixel_size=PixelSize(.5, .5, "µm")) as image:
        with pytest.raises(ValueError, match="array structure"):
            writer.write(image)
    assert not path.exists()
    assert not list(tmp_path.glob(".omeify-zarr-*"))
