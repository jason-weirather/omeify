"""Cross-format tests use real TIFF/Zarr storage; XSD-dependent tests are explicit."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from lxml import etree

from omeify import (LabelImage, MultichannelImage, OMETiffReader, OMETiffWriter,
                   OMEZarrReader, OMEZarrWriter, PixelSize, RGBImage, convert, mutate)
from omeify.io.spec import OMEImageSpec
from omeify.utils.generate_ome_xml import generate_ome_xml
from zarr_fixture import fixture


@pytest.mark.parametrize("dtype",["uint16","float32"])
def test_tiff_to_zarr_preserves_values_channels_and_calibration(tmp_path,dtype):
    data=(np.arange(2*33*47).reshape(2,33,47)%1023).astype(dtype)
    source=tmp_path/"source.ome.tif"
    tifffile.imwrite(source,data,ome=True,photometric="minisblack",tile=(16,16),metadata={
        "axes":"CYX","PhysicalSizeX":.5,"PhysicalSizeXUnit":"µm",
        "PhysicalSizeY":.6,"PhysicalSizeYUnit":"µm","Channel":{"Name":["DAPI","CD3"]},
        "Description":"PRIVATE-SOURCE"})
    report=convert(source,tmp_path/"output",input_type="ome_tiff",output_format="ome_zarr",tile_size=16)
    assert report["input_file"]["size_bytes"]==source.stat().st_size
    with OMEZarrReader(tmp_path/"output") as image:
        np.testing.assert_array_equal(image.asarray(),data)
        assert image.channel_names==("DAPI","CD3")
        assert image.pixel_size==PixelSize(.5,.6,"µm")
    assert "PRIVATE-SOURCE" not in (tmp_path/"output/zarr.json").read_text()


@pytest.mark.parametrize("method",["mean","nearest"])
def test_tiff_reader_recovers_linked_image_origin(tmp_path,method):
    data=np.arange(2*33*47,dtype="uint16").reshape(2,33,47)
    spec=OMEImageSpec.from_shape(image_type="multichannel",axes="CYX",shape=data.shape,
                                 dtype=data.dtype,channel_names=("a","b"),pixel_size=PixelSize(.5,.6,"µm"),
                                 origin_yx=(12,-3))
    from dataclasses import replace
    spec=replace(spec,pyramid_method=method)
    xml=generate_ome_xml(spec,[data.shape])["xml_string"]
    source=tmp_path/"source.tif"
    tifffile.imwrite(source,data,description=xml.encode("utf-8"),metadata=None,photometric="minisblack",tile=(16,16),
                      resolution=(20000,10000/.6),resolutionunit="CENTIMETER")
    with OMETiffReader(source) as image:
        assert image.origin_yx==(12,-3)
        np.testing.assert_array_equal(image.asarray(),data)
        OMEZarrWriter(tmp_path/"out",tile_size=16).write(image)
    with OMEZarrReader(tmp_path/"out") as image:
        assert image.origin_yx==(12,-3)


@pytest.mark.parametrize("version",["0.4","0.5","0.6"])
def test_zarr_tiff_zarr_full_round_trip_with_real_xsd(tmp_path,version):
    expected,_,_=fixture(tmp_path/"a",version=version)
    convert(tmp_path/"a",tmp_path/"b.ome.tif",input_type="ome_zarr",compression="Uncompressed",tile_size=16)
    with OMETiffReader(tmp_path/"b.ome.tif") as im:
        np.testing.assert_array_equal(im.asarray(),expected)
        assert im.origin_yx==(11.25,-9.5)
    convert(tmp_path/"b.ome.tif",tmp_path/"c",input_type="ome_tiff",output_format="ome_zarr",tile_size=16)
    with OMEZarrReader(tmp_path/"c") as im:
        np.testing.assert_array_equal(im.asarray(),expected)
        assert im.origin_yx==(11.25,-9.5)


def test_collection_precision_skips_non_float_entries(tmp_path):
    from omeify import OMEZarrCollectionWriter,OMEImageSeries
    with MultichannelImage.from_array(np.full((33,47),1.234,dtype="float32"),axes="YX",
                                     pixel_size=PixelSize(1,1,"µm")) as a,\
         RGBImage.from_array(np.zeros((33,47,3),dtype="uint8"),pixel_size=PixelSize(1,1,"µm")) as b:
        OMEZarrCollectionWriter(tmp_path/"out",float32_mantissa_bits=11,tile_size=16).write(
            [OMEImageSeries("float",a),OMEImageSeries("rgb",b)])
    with OMEZarrReader(tmp_path/"out",series=1) as image:
        assert image.dtype==np.dtype("uint8") and image.image_type=="rgb"
