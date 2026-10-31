"""Mandatory public stack writes with real validators, encoders and readback.

Missing ordinary Zarr/codecs/OME-XSD dependencies are failures, never skips.
Small full arrays provide independent numerical expectations at every level.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from omeify import (AxisInfo, ImageDataset, OMETiffWriter, OMEZarrWriter,
                    PixelSize, convert, mutate, open_dataset)
from omeify.io.volume_pyramid import plan_volume_pyramid
from omeify.reports import validate_report


def make_dataset(kind='multichannel',dtype='uint16'):
    axes={'multichannel':'TCZYX','rgb':'TZYXS','label':'TZYX'}[kind]
    shape={'multichannel':(2,2,5,19,21),'rgb':(2,5,19,21,3),'label':(2,5,19,21)}[kind]
    a=(np.arange(np.prod(shape),dtype=np.uint32).reshape(shape)%197).astype(dtype)
    ds=ImageDataset.from_array(a,axes=axes,image_type=kind,pixel_size=PixelSize(.5,.5,'µm'),
        axis_info=(AxisInfo('Z',5,'space','µm',2,7),AxisInfo('T',2,'time','s',3,0)))
    return ds,a,axes


def independent_mean(a,step):
    shape=tuple((n+f-1)//f for n,f in zip(a.shape[:3],step))
    out=np.empty((*shape,*a.shape[3:]),dtype=a.dtype)
    for z,y,x in np.ndindex(shape):
        block=a[z*step[0]:(z+1)*step[0],y*step[1]:(y+1)*step[1],x*step[2]:(x+1)*step[2]]
        value=block.mean(axis=(0,1,2),dtype=np.float64)
        out[z,y,x]=np.rint(value) if a.dtype.kind in 'iu' else value
    return out


@pytest.mark.parametrize('fmt',['ome_tiff','ome_zarr'])
@pytest.mark.parametrize('kind,dtype',[('multichannel','uint16'),('rgb','uint8'),('label','uint32')])
def test_public_stack_write_read_every_plane_and_xy_level(tmp_path,fmt,kind,dtype):
    ds,values,axes=make_dataset(kind,dtype)
    path=tmp_path/('out.ome.tif' if fmt=='ome_tiff' else 'out.ome.zarr')
    cls=OMETiffWriter if fmt=='ome_tiff' else OMEZarrWriter
    kwargs={'compression':'Deflate','overwrite':False} if fmt=='ome_tiff' else {'compression':'ZSTD'}
    with ds:
        report=cls(path,tile_size=16,pyramid_levels=2,**kwargs).write_dataset(ds)
        validate_report(report,'dataset_write_report.schema.json')
        assert ds.is_open
    expected=values
    with open_dataset(path,input_type=fmt,image_type='label' if kind=='label' else None) as back:
        assert back.axes == axes
        assert back.shape == values.shape
        if fmt == 'ome_zarr':
            assert back.native_axes == 'TCZYX'
            assert back.native_shape == ((2, 3 if kind == 'rgb' else
                                          1 if kind == 'label' else 2), 5, 19, 21)
        for level in range(3):
            for t in range(2):
                for z in range(5):
                    with back.plane(t=t,z=z,native_level=level) as image:
                        exp=expected[t,:,z] if kind=='multichannel' else expected[t,z]
                        np.testing.assert_array_equal(image.asarray(),exp)
            if level<2:
                if kind=='multichannel':
                    expected=np.stack([np.stack([independent_mean(v,(1,2,2)) for v in ts]) for ts in expected])
                elif kind=='rgb':expected=np.stack([independent_mean(v,(1,2,2)) for v in expected])
                else:expected=expected[:,:,::2,::2]
    if fmt=='ome_tiff':
        with tifffile.TiffFile(path) as tif:
            np.testing.assert_array_equal(tif.asarray(),values)


@pytest.mark.parametrize('axes_mode',['xyz','auto'])
@pytest.mark.parametrize('kind,dtype',[('multichannel','uint16'),('rgb','uint8'),('label','uint32')])
def test_public_xyz_and_anisotropic_pyramids(tmp_path,axes_mode,kind,dtype):
    ds,values,_=make_dataset(kind,dtype);path=tmp_path/'volume.zarr'
    method='nearest' if kind=='label' else 'mean'
    with ds:
        report=OMEZarrWriter(path,tile_size=16,pyramid_levels=2).write_dataset(ds,pyramid_axes=axes_mode)
    plan=plan_volume_pyramid((5,19,21),tile_size=16,levels=2,axes=axes_mode,spacing=(2,.5,.5))
    assert report['pyramid']['factors_zyx']==[list(f) for f in plan.steps]
    expected=values
    with open_dataset(path,input_type='ome_zarr') as back:
        for level,shape in enumerate(plan.shapes):
            for t in range(2):
                for z in range(shape[0]):
                    with back.plane(t=t,z=z,native_level=level) as p:
                        exp=expected[t,:,z] if kind=='multichannel' else expected[t,z]
                        np.testing.assert_array_equal(p.asarray(),exp)
            if level<len(plan.steps):
                step=plan.steps[level]
                if method=='nearest':expected=expected[:,::step[0],::step[1],::step[2]]
                elif kind=='multichannel':
                    expected=np.stack([np.stack([independent_mean(v,step) for v in ts]) for ts in expected])
                else:expected=np.stack([independent_mean(v,step) for v in expected])


def test_public_dataset_conversion_and_global_mutation(tmp_path):
    a=np.arange(2*3*17*19,dtype=np.float32).reshape(2,3,17,19)
    source=tmp_path/'source.ome.tif'
    # OME-XML 2016-06 defaults omitted physical-size unit attributes to µm.
    # Exercise that legitimate input through the public whole-stack workflow.
    tifffile.imwrite(source,a,ome=True,photometric='minisblack',metadata={
        'axes':'TZYX','PhysicalSizeX':.5,'PhysicalSizeY':.5,'PhysicalSizeZ':2})
    converted=tmp_path/'converted.zarr'
    report=convert(source,converted,input_type='ome_tiff',output_format='ome_zarr',
                   all_planes=True,tile_size=16,pyramid_levels=1,pyramid_axes='xyz')
    validate_report(report,'dataset_operation_report.schema.json')
    out=tmp_path/'mutated.ome.tif'
    report=mutate(converted,out,input_type='ome_zarr',all_planes=True,dtype='uint8',range_mode='full',
                  compression='Deflate',tile_size=16,pyramid_levels=1)
    validate_report(report,'dataset_operation_report.schema.json')
    mapping=report['mutation']['channels'][0]['mapping']
    with open_dataset(out,input_type='ome_tiff') as ds:
        for t in range(2):
            for z in range(3):
                with ds.plane(t=t,z=z) as p:
                    expected=np.rint((a[t,z].astype('float64')-mapping['offset_source_units'])/
                                     mapping['quantum_source_units_per_output_code']).astype('uint8')
                    actual=p.asarray()
                    np.testing.assert_array_equal(actual[0] if actual.ndim==3 else actual,expected)


def test_stack_write_failure_preserves_existing_destinations(tmp_path):
    ds,_,_=make_dataset();path=tmp_path/'existing.tif';path.write_bytes(b'keep')
    with ds:
        with pytest.raises(FileExistsError):
            OMETiffWriter(path,overwrite=False).write_dataset(ds)
        assert path.read_bytes()==b'keep'
        zarr_path=tmp_path/'existing.zarr';zarr_path.mkdir()
        with pytest.raises(FileExistsError):OMEZarrWriter(zarr_path).write_dataset(ds)
        assert list(zarr_path.iterdir())==[]


def test_xyz_work_budget_is_checked_before_publication(tmp_path):
    ds,_,_=make_dataset();path=tmp_path/'limited.zarr'
    with ds:
        with pytest.raises(ValueError,match='working-block'):
            OMEZarrWriter(path,tile_size=16,pyramid_levels=1).write_dataset(ds,
                                    pyramid_axes='xyz',max_block_bytes=100)
    assert not path.exists()
