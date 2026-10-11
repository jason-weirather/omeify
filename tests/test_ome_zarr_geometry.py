"""Coordinate and metadata contract tests, runnable without optional Zarr I/O."""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
from lxml import etree

from omeify import ImageLevel, ImageMetadata, MultichannelImage, PixelSize
from omeify.io.ome_zarr.metadata import ArrayInfo, parse_image, relative_path, profile_errors
from omeify.io.ome_zarr.writer import output_levels
from omeify.io.spec import OMEImageSpec
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.ome_geometry import read_geometry, calibrated_levels
from zarr_fixture import attributes


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_scale_translation_composition_and_unit_conversion(version):
    axes="cyx"
    attrs=attributes(version,axes,["0","1"],[[1,600,500],[1,1200,1000]],
                     [[0,12000,13000],[0,12300,13250]])
    ome=attrs if version=="0.4" else attrs["ome"]
    ms=ome["multiscales"][0]
    descriptions=ms["axes"] if version!="0.6" else ms["coordinateSystems"][0]["axes"]
    for axis in descriptions:
        if axis["name"] in "yx":axis["unit"]="nanometer"
    fmt=2 if version=="0.4" else 3
    info=[ArrayInfo((3,33,47),np.dtype("uint16"),fmt,tuple(axes)),
          ArrayInfo((3,17,24),np.dtype("uint16"),fmt,tuple(axes))]
    parsed=parse_image(attrs,fmt,info)
    assert parsed.metadata.pixel_size==PixelSize(.5,.6,"µm")
    np.testing.assert_allclose(parsed.metadata.levels[0].origin_yx,(12,13))
    np.testing.assert_allclose(parsed.metadata.levels[1].offset_yx,(0,0))
    if version!="0.6":
        ms["coordinateTransformations"]=[{"type":"scale","scale":[1,2,3]},
                                         {"type":"translation","translation":[0,1000,-1000]}]
        parsed=parse_image(attrs,fmt,info)
        np.testing.assert_allclose(parsed.metadata.levels[0].origin_yx,(25,38))
        assert parsed.metadata.pixel_size==PixelSize(1.5,1.2,"µm")


def test_identity_06_intrinsic_transform():
    attrs=attributes("0.6","yx",["0"],[[1,1]],[[0,0]])
    trans=attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
    trans.pop("scale");trans["type"]="identity"
    parsed=parse_image(attrs,3,[ArrayInfo((33,47),np.dtype("uint16"),3,("y","x"))])
    assert parsed.metadata.pixel_size==PixelSize(1,1,"µm")


@pytest.mark.parametrize("path",["../a","/a","a//b","a/./b","a\\b","https://host/a","%2e%2e/a","a?b","a#b"])
def test_metadata_paths_are_not_network_or_filesystem_authority(path):
    with pytest.raises(ValueError):relative_path(path)


@pytest.mark.parametrize("fault",["missing_scale","bad_version","negative_scale","duplicate_axes",
                                  "duplicate_path","nonfinite","bad_dimensions"])
def test_bad_metadata_is_rejected(fault):
    attrs=attributes("0.5","cyx",["0","1"],[[1,.5,.5],[1,1,1]],[[0,0,0],[0,.25,.25]])
    ms=attrs["ome"]["multiscales"][0]
    info=[ArrayInfo((3,33,47),np.dtype("uint16"),3,("c","y","x")),
          ArrayInfo((3,17,24),np.dtype("uint16"),3,("c","y","x"))]
    if fault=="missing_scale":ms["datasets"][0]["coordinateTransformations"]=[]
    elif fault=="bad_version":attrs["ome"]["version"]="0.7"
    elif fault=="negative_scale":ms["datasets"][0]["coordinateTransformations"][0]["scale"][1]=-1
    elif fault=="duplicate_axes":ms["axes"][1]["name"]="x"
    elif fault=="duplicate_path":ms["datasets"][1]["path"]="0"
    elif fault=="nonfinite":ms["datasets"][0]["coordinateTransformations"][0]["scale"][1]=float("nan")
    else:info[1]=replace(info[1],dimension_names=("y","x","c"))
    with pytest.raises((ValueError,NotImplementedError)):
        parse_image(attrs,3,info)


@pytest.mark.parametrize("method",["mean","nearest"])
def test_tiff_annotation_records_same_geometry_without_claiming_stage_position(method):
    base=ImageLevel(0,"CYX",(2,33,47),PixelSize(.5,.6,"µm"),origin_yx=(12.0,-3.0))
    metadata=ImageMetadata("CYX",base.shape,np.dtype("uint16"),levels=(base,),pixel_size=base.pixel_size)
    data=np.zeros(base.shape,dtype="uint16")
    with MultichannelImage.from_array(data,axes="CYX",pixel_size=base.pixel_size,levels=(base,)) as im:
        levels=output_levels(im,level=0,tile_size=16,pyramid_levels=None,method=method)
    spec=OMEImageSpec.from_shape(image_type="multichannel",axes="CYX",shape=base.shape,
                                 dtype="uint16",channel_names=("a","b"),pixel_size=base.pixel_size,
                                 origin_yx=base.origin_yx)
    spec=replace(spec,pyramid_method=method)
    xml=generate_ome_xml(spec,level_shapes=[l.shape for l in levels])["xml_string"]
    root=etree.fromstring(xml.encode())
    image=root.find("{*}Image")
    record=read_geometry(root,image)
    assert record["origin_yx_um"]==[12,-3]
    assert "PositionX" not in xml and "PositionY" not in xml
    original=tuple(replace(l,origin_yx=None,offset_yx=(0,0)) for l in levels)
    assert calibrated_levels(original,record)==levels


def test_metadata_origin_overrides_validate_before_arithmetic():
    with MultichannelImage.from_array(np.zeros((17,19)),axes="YX",pixel_size=PixelSize(1,1,"µm")) as im:
        for bad in [(True,0),(0,float("nan")),(0,),"bad"]:
            with pytest.raises((ValueError,TypeError)):
                im.with_metadata(origin_yx=bad)
        with im.with_metadata(origin_yx=(9,11)) as located:
            with located.crop(1,10,2,11) as roi:
                assert roi.origin_yx==(10,13)
