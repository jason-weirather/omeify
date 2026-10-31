"""Source views, explicit scopes, and fixed whole-stack mutation without I/O mocks."""
from __future__ import annotations

import numpy as np
import pytest
from click.testing import CliRunner

from omeify import (ImageDataset, PixelSize, apply_dataset_dtype_mutation,
                    analyze_dataset_dtype_mutation, convert, mutate)
from omeify.cli import main
from omeify.dataset_workflow import stack_selection
from omeify.io.dataset_metadata import AxisInfo


def test_mutation_mapping_is_fixed_across_time_and_depth():
    pixels = np.arange(2*2*3*7*9, dtype=np.float32).reshape(2,2,3,7,9)
    pixels[:,1] *= 100
    with ImageDataset.from_array(pixels, axes="TCZYX", channel_names=("a","b")) as dataset:
        plans=analyze_dataset_dtype_mutation(dataset,dtype="uint8",range_mode="full",
                                             sample_pixels_per_channel=1024)
        assert len(plans)==2
        for channel,plan in enumerate(plans):
            assert plan.source_lower==pixels[:,channel].min()
            assert plan.source_upper==pixels[:,channel].max()
            assert plan.report['source']['total_pixels']==2*3*7*9
        with apply_dataset_dtype_mutation(dataset,plans) as result:
            for t in range(2):
                for z in range(3):
                    with result.plane(t=t,z=z) as plane:
                        expected=np.stack([p.quantize(pixels[t,c,z]) for c,p in enumerate(plans)])
                        np.testing.assert_array_equal(plane.asarray(),expected)
        with dataset.plane(t=0,z=0) as plane:
            np.testing.assert_array_equal(plane.asarray(),pixels[0,:,0])


@pytest.mark.parametrize('value',[np.nan,np.inf,-np.inf])
def test_mutation_finds_nonfinite_values_in_last_plane(value):
    pixels=np.ones((2,3,5,7),dtype=np.float64);pixels[-1,-1,-1,-1]=value
    with ImageDataset.from_array(pixels,axes="TZYX") as dataset:
        with pytest.raises(ValueError,match="cannot be represented"):
            analyze_dataset_dtype_mutation(dataset,dtype="uint16")


def test_stack_scopes_are_explicit():
    with ImageDataset.from_array(np.zeros((4,5,6,7)),axes="TZYX") as ds:
        for kwargs in [{},{"z_range":(1,3)},{"t":1}]:
            with pytest.raises(ValueError,match="requires an explicit"):
                stack_selection(ds,**kwargs)
        selection,record=stack_selection(ds,z_range=(1,3),t=2)
        assert selection==dict(z=slice(1,3),t=2)
        assert record==dict(z=[1,3],t=[2,3])
        with ds.select(**selection) as selected:
            assert selected.axis_info('Z').size==2
            assert selected.axis_info('T').size==1
        selection,record=stack_selection(ds,all_planes=True)
        assert record==dict(z=[0,5],t=[0,4])
        for kwargs in [{'all_planes':True,'z':1},{'z':1,'z_range':(1,2),'t':1},
                       {'z_range':(3,2),'t':0},{'z_range':(0,6),'t':0}]:
            with pytest.raises(ValueError):stack_selection(ds,**kwargs)


def test_calibration_override_preserves_pixels_and_known_origin():
    values=np.arange(2*3*5*7).reshape(2,3,5,7).astype('uint16')
    with ImageDataset.from_array(values,axes='TZYX') as ds:
        with ds.with_metadata(z_spacing_um=2,time_increment_s=4,z_origin_um=10,time_origin_s=20) as view:
            assert view.axis_info('Z').coordinate(2)==14
            assert view.axis_info('T').coordinate(1)==24
            with view.plane(t=1,z=2) as p:np.testing.assert_array_equal(p.asarray(),values[1,2])
        assert ds.axis_info('Z').scale is None


def test_unitless_xy_origin_is_not_pretended_to_be_micrometers():
    info=(AxisInfo('Y',3,'space',None,1,20),AxisInfo('X',4,'space',None,1,10))
    with ImageDataset.from_array(np.zeros((3,4)),axes='YX',axis_info=info) as ds:
        assert ds.axis_info('X').coordinate(1)==11
        with ds.plane() as p:
            assert p.origin_yx is None
            assert p.pixel_size is None


def test_coordinate_overflow_is_explicit():
    with pytest.raises(ValueError,match='overflow'):
        AxisInfo('Z',3,'space','µm',1e308,1e308).coordinate(2)


@pytest.mark.parametrize('command',['convert','mutate'])
def test_stack_help_and_old_plane_options(command):
    result=CliRunner().invoke(main,[command,'--help'])
    assert result.exit_code==0
    for option in ['--all-planes','--z-range','--t-range','--pyramid-axes','--chunk-z','--z-index']:
        assert option in result.output


@pytest.mark.parametrize('function',[convert,mutate])
def test_2d_workflows_do_not_ignore_stack_parameters(tmp_path,function):
    with pytest.raises(ValueError,match='require all_planes'):
        function(tmp_path/'missing.tif',tmp_path/'out.tif',input_type='ome_tiff',pyramid_axes='xyz')


def test_dataset_and_selection_share_demand_priority_gate():
    with ImageDataset.from_array(np.zeros((3,4,5)),axes='ZYX') as dataset:
        with dataset.select(z=slice(1,3)) as selected:
            assert selected._gate is dataset._gate


def test_stack_scan_retains_bounded_samples_for_narrow_images():
    a=np.arange(10_003,dtype='float32').reshape(1,10003,1,1)
    with ImageDataset.from_array(a,axes='TZYX') as ds:
        plan=analyze_dataset_dtype_mutation(ds,dtype='uint16',sample_pixels_per_channel=1024)[0]
        assert plan.report['source']['sample_pixels']<=1024
        assert plan.report['source']['total_pixels']==10003


def test_selected_images_retain_plane_identity(tmp_path):
    from test_dataset_reading import write_tiff
    from omeify import open_dataset
    p=tmp_path/'identity.ome.tif';write_tiff(p)
    with open_dataset(p,input_type='ome_tiff') as ds, ds.plane(t=1,z=2) as plane:
        with plane.crop(0,4,0,5) as crop, crop.with_metadata(channel_names=('a','b','c')) as named:
            assert named.plane_selection==plane.plane_selection


def test_prefetch_budget_includes_padded_zarr_chunk_payload():
    from omeify.io.ome_zarr.dataset_writer import _validate_work_budget
    from omeify.io.volume_pyramid import plan_volume_pyramid
    plan=plan_volume_pyramid((1,1,1),levels=0)
    with pytest.raises(ValueError,match='working-block'):
        _validate_work_budget(plan,chunk_z=1024,tile_size=512,dtype=np.dtype('uint16'),
                              samples=1,max_bytes=64*1024*1024)


def test_array_axis_units_are_canonical_without_losing_origin():
    values = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    with ImageDataset.from_array(values, axes="TZYX", axis_info=(
            AxisInfo("T", 2, "time", "millisecond", 250, 1000),
            AxisInfo("Z", 3, "space", "nm", 1500, -3000),
            AxisInfo("Y", 4, "space", "nm", 500, 1000),
            AxisInfo("X", 5, "space", "nm", 500, 2000),
    )) as ds:
        assert ds.axis_info("T").unit == "s"
        assert ds.axis_info("T").coordinate(1) == 1.25
        assert ds.axis_info("Z").coordinate(2) == 0
        with ds.plane(t=1, z=2) as plane:
            assert plane.pixel_size == PixelSize(.5, .5, "µm")
            assert plane.origin_yx == (1., 2.)


def test_time_units_preserve_case_and_extreme_si_prefixes():
    from omeify.io.dataset_metadata import time_unit_seconds
    assert time_unit_seconds("Ms") == 1e6
    assert time_unit_seconds("ms") == 1e-3
    assert time_unit_seconds("yoctosecond") == 1e-24
    assert time_unit_seconds("yottasecond") == 1e24
    assert time_unit_seconds("hs") == 100
    with pytest.raises(NotImplementedError):
        time_unit_seconds("furlong")


def test_dataset_writer_accepts_equivalent_xy_units_in_metadata_view():
    from omeify.io.dataset_writer import DatasetPlaneSource
    a = np.arange(2*5*7, dtype=np.uint16).reshape(2, 5, 7)
    with ImageDataset.from_array(a, axes="TYX") as dataset:
        with dataset.with_metadata(pixel_size=PixelSize(500, 500, "nm")) as calibrated:
            source = DatasetPlaneSource(calibrated)
            readers = source.plane_readers()
            try:
                assert source.spec.pixel_size == PixelSize(.5, .5, "µm")
                np.testing.assert_array_equal(readers[1].read_region(0, 5, 0, 7), a[1])
            finally:
                readers.clear_cache()


def test_metadata_override_does_not_relabel_unitless_origin_as_physical():
    a = np.zeros((2, 3, 4), dtype=np.uint16)
    with ImageDataset.from_array(a, axes="TYX", axis_info=(
            AxisInfo("T", 2, "time", None, 1, 5),
            AxisInfo("Y", 3, "space", None, 1, 10),
            AxisInfo("X", 4, "space", None, 1, 20),
    )) as dataset:
        with pytest.raises(ValueError, match="explicit origin"):
            dataset.with_metadata(time_increment_s=10)
        with pytest.raises(ValueError, match="explicit sampling"):
            dataset.with_metadata(time_origin_s=10)
        with dataset.with_metadata(pixel_size=PixelSize(.5, .5, "µm"),
                                   time_increment_s=10, time_origin_s=0) as calibrated:
            assert calibrated.axis_info("T").coordinate(1) == 10
            assert calibrated.axis_info("Y").origin is None
            with calibrated.plane(t=0) as plane:
                assert plane.origin_yx is None
