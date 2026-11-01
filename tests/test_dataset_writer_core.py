"""Actual uncompressed TIFF engine/readback and independent metadata tests.

These exercise the shared encoder below its public XSD-validation boundary.
Public writer tests live in test_dataset_writers.py and require ome-schema/Zarr.
"""
import numpy as np
import pytest
import tifffile

from omeify import AxisInfo, ImageDataset, OMETiffWriter, PixelSize, open_dataset
from omeify.io._writer import WriterEngine, prepare_image
from omeify.io._writer.single_verification import verify_single_output
from omeify.io.dataset_writer import DatasetPlaneSource
from omeify.io.ome_zarr.dataset_writer import dataset_attributes
from omeify.io.ome_zarr.metadata import ArrayInfo, parse_image, profile_errors
from omeify.io.volume_pyramid import plan_volume_pyramid
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header


def fixture(kind='multichannel', dtype='uint16'):
    axes,shape={'multichannel':('TCZYX',(2,3,4,17,19)),
                'rgb':('TZYXS',(2,4,17,19,3)),
                'label':('TZYX',(2,4,17,19))}[kind]
    a=(np.arange(np.prod(shape))%173).reshape(shape).astype(dtype)
    ds=ImageDataset.from_array(a,axes=axes,image_type=kind,pixel_size=PixelSize(.5,.75,'µm'),
        axis_info=(AxisInfo('T',2,'time','s',4,8),AxisInfo('Z',4,'space','µm',2,10)))
    return ds,a,axes


@pytest.mark.parametrize('kind,dtype',[('multichannel','uint16'),('rgb','uint8'),('label','uint32')])
def test_stack_encoder_without_omexsd_substitute(tmp_path,kind,dtype):
    ds,a,axes=fixture(kind,dtype)
    path=tmp_path/'stack.ome.tif'
    writer=OMETiffWriter(path,compression='Uncompressed',tile_size=16,pyramid_levels=2)
    with ds:
        source=DatasetPlaneSource(ds)
        item=prepare_image(source,source.spec,name=None,downsample=writer._downsample,
            compression_name=writer._settings.compression_name,settings=writer._settings,lossy_policy='rgb-only')
        xml=generate_ome_xml(item.spec,item.level_shapes)['xml_string']
        minimum=validate_miti_ome_tiff_header(xml)
        assert minimum.is_valid,minimum.errors
        result=WriterEngine(writer._settings).write((item,),xml,verify=lambda p:
            verify_single_output(item,p,software=writer._settings.software))
        assert result.verification['planes_checked']==item.spec.plane_count
        assert result.verification['base_pixel_values_match']
    with tifffile.TiffFile(path) as independent:
        np.testing.assert_array_equal(independent.series[0].asarray(),a)
    with open_dataset(path,input_type='ome_tiff',image_type='label' if kind=='label' else None) as back:
        assert back.axis_info('T').origin==8
        assert back.axis_info('Z').origin==10
        assert back.axis_info('Z').scale==2
        for t in range(2):
            for z in range(4):
                with back.plane(t=t,z=z) as im:
                    assert len(im.levels)==3
                    expected=a[t,:,z] if kind=='multichannel' else a[t,z]
                    np.testing.assert_array_equal(im.asarray(),expected)


@pytest.mark.parametrize('kind,dtype',[('multichannel','uint16'),('rgb','uint8'),('label','uint32')])
@pytest.mark.parametrize('method',['mean','nearest'])
def test_stack_ngff_metadata_native_xyz_not_optical_slice(kind,dtype,method):
    ds,_,_=fixture(kind,dtype)
    with ds:
        source=DatasetPlaneSource(ds)
        plan=plan_volume_pyramid((4,17,19),axes='xyz',levels=2)
        if kind=='label':method='nearest'
        attrs=dataset_attributes(source,plan,method=method,
            windows=[dict(min=0,max=255,start=0,end=255)]*source.spec.size_c)
        assert not profile_errors(attrs)
        arrays=[ArrayInfo((2,source.spec.size_c,*shape),source.spec.dtype,3,tuple('tczyx'))
                for shape in plan.shapes]
        parsed=parse_image(attrs,3,arrays,z=1,t=1)
        assert parsed.level_indices==(0,)
        assert len(parsed.metadata.levels)==1
        assert len(parsed.dataset_levels)==3
        coarse=parsed.dataset_levels[1]
        assert coarse.axis('Z').scale==4
        assert coarse.axis('Z').origin==(11 if method=='mean' else 10)
        assert not coarse.plane_preserving
        assert coarse.plane_metadata.axes=={'multichannel':'CYX','rgb':'YXS','label':'YX'}[kind]


def test_dataset_adapter_is_lazy_and_pinned():
    ds,_,_=fixture()
    with ds:
        source=DatasetPlaneSource(ds)
        readers=source.plane_readers()
        assert len(readers)==24
        for reader in readers:
            reader.read_region(0,2,0,2)
            assert len(readers._readers)<=4
        assert ds.is_open
        readers.clear_cache()
    with pytest.raises(RuntimeError,match='closed|session'):
        source.plane_readers()
