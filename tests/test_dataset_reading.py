"""Small independent source fixtures: every T/C/Z plane has distinguishable pixels."""
from __future__ import annotations

import itertools
from concurrent.futures import CancelledError, ThreadPoolExecutor
from threading import Event

import numpy as np
import pytest
import tifffile
from lxml import etree

from omeify import (AxisInfo, Image, ImageDataset, OMETiffReader, PixelSize,
                    PlaneRequest, PrefetchPolicy, open_dataset)
from omeify.io.read_cache import ReadCache

SIZE = PixelSize(.5, .75, "µm")


def values():
    t, c, z, y, x = np.indices((2, 3, 4, 17, 19))
    return (t*20000 + c*4000 + z*500 + y*19 + x).astype(np.uint16)


def write_tiff(path, axes="TCZYX", data=None):
    a = values() if data is None else data
    if data is None:
        a = a.transpose(["TCZYX".index(n) for n in axes])
    tifffile.imwrite(path, a, ome=True, metadata={"axes": axes,
        "PhysicalSizeX": .5, "PhysicalSizeY": .75,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
        "PhysicalSizeZ": 2., "PhysicalSizeZUnit": "µm", "TimeIncrement": 4.},
        photometric="minisblack", tile=(16, 16))
    return a


@pytest.mark.parametrize("axis", ["".join(p)+"YX" for p in itertools.permutations("TCZ")])
def test_tiff_all_dimension_orders(tmp_path, axis):
    p = tmp_path/"planes.ome.tif"
    write_tiff(p, axis)
    reference = values()
    with open_dataset(p, input_type="ome_tiff") as ds:
        assert ds.axes == "TCZYX"
        assert ds.shape == reference.shape
        assert ds.native_axes == axis
        assert ds.native_shape == tuple(reference.transpose(
            ["TCZYX".index(n) for n in axis]).shape)
        assert ds.axis_info("Z").size == 4
        assert ds.axis_info("T").scale == 4
        assert ds.axis_info("Z").origin is None
        for ti, zi in itertools.product(range(2), range(4)):
            with ds.plane(z=zi, t=ti) as plane:
                assert isinstance(plane, Image)
                assert plane.axes == "CYX"
                np.testing.assert_array_equal(plane.read_region(1,16,3,18,channels=[2,0,2]),
                                              reference[ti,[2,0,2],zi,1:16,3:18])
        result = ds.read_block(0,17,0,19,z=slice(1,3),t=slice(None))
        np.testing.assert_array_equal(result,reference[:,:,1:3])


def test_tiff_requires_plane_choice_and_reader_navigation_shares_cache(tmp_path, monkeypatch):
    p=tmp_path/"a.ome.tif";write_tiff(p)
    with pytest.raises(ValueError,match="explicit"):
        OMETiffReader(p).open()
    with OMETiffReader(p,z=0,t=0,cache_bytes=40000) as reader:
        handle=reader._source._tiff
        a=reader.read_region(0,16,0,16)
        loaded=reader.cache_info()["loads"]
        with reader.plane(z=1,t=0) as other:
            other.read_region(0,16,0,16)
        with reader.plane(z=0,t=0) as again:
            np.testing.assert_array_equal(again.read_region(0,16,0,16),a)
        assert reader._source._tiff is handle
        assert reader.cache_info()["loads"] == loaded*2
        assert reader.cache_info()["hits"] >= 3
        assert reader.cache_info()["resident_bytes"] <= 40000


def test_views_do_not_reconnect_or_close_parent(tmp_path):
    p=tmp_path/"a.ome.tif";write_tiff(p)
    ds=open_dataset(p,input_type="ome_tiff").open()
    view=ds.plane(z=0,t=0).open()
    patch=view.read_region(0,2,0,2)
    ds.close();ds.open()
    with pytest.raises(RuntimeError,match="closed|session"):
        view.read_region(0,2,0,2)
    view.close()
    assert ds.is_open
    ds.close()
    assert patch.flags.owndata


def test_array_selection_and_calibration():
    a=values()
    with ImageDataset.from_array(a,axes="TCZYX",pixel_size=SIZE,
          axis_info=(AxisInfo("Z",4,"space","µm",2,10),AxisInfo("T",2,"time","s",4,0))) as ds:
        with ds.select(z=slice(1,3),channels=[2,0,2],bounds=(2,8,3,12)) as selected:
            assert selected.axis_info("Z").origin == 12
            actual=selected.read_block(0,6,0,9,z=slice(None),t=slice(None))
            expected=a[:,[2,0,2],1:3,2:8,3:12]
            np.testing.assert_array_equal(actual,expected)
        with pytest.raises(ValueError,match="max_bytes"):
            ds.read_block(0,17,0,19,z=slice(None),t=slice(None),max_bytes=100)
        with pytest.raises(ValueError,match="explicit"):
            ds.read_block(0,2,0,2)


@pytest.mark.parametrize("kind,axes,shape",[("label","TZYX",(2,3,5,7)),
                                           ("rgb","TZYXS",(2,3,5,7,3)),
                                           ("multichannel","TCZYX",(2,3,4,5,7))])
def test_array_semantics(kind,axes,shape):
    a=np.arange(np.prod(shape),dtype=np.uint8).reshape(shape)
    with ImageDataset.from_array(a,axes=axes,image_type=kind,pixel_size=SIZE) as ds:
        with ds.plane(z=1,t=1) as plane:
            expect=a[1,:,1] if kind=="multichannel" else a[1,1]
            np.testing.assert_array_equal(plane.asarray(),expect)
            assert plane.axes == {"label":"YX","rgb":"YXS","multichannel":"CYX"}[kind]


def test_tiff_diagnostic_scan_limit_not_operational_limit(tmp_path,monkeypatch):
    import omeify._calibration as calibration
    p=tmp_path/"many.ome.tif"
    write_tiff(p,"ZYX",np.arange(4097,dtype=np.uint16).reshape(4097,1,1))
    monkeypatch.setattr(calibration,"_MAX_PAGES",1)
    with open_dataset(p,input_type="ome_tiff") as ds:
        assert ds.axis_info("Z").size==4097
        with ds.plane(z=4096) as im:
            assert im.read_region(0,1,0,1).item()==4096


def test_tiff_reversed_time_mapping(tmp_path):
    p=tmp_path/"reversed.ome.tif"
    a=np.stack([np.full((5,7),i,np.uint16) for i in range(3)])
    xml=tifffile.OmeXml()
    xml.addimage(a.dtype,a.shape,(3,1,1,5,7,1),axes="TYX")
    root=etree.fromstring(xml.tostring().encode())
    ns={"o":"http://www.openmicroscopy.org/Schemas/OME/2016-06"}
    px=root.find("o:Image/o:Pixels",ns)
    for node in px.findall("o:TiffData",ns):px.remove(node)
    for t in range(3):
        etree.SubElement(px,"{"+ns['o']+"}TiffData",IFD=str(2-t),FirstT=str(t),PlaneCount="1")
    tifffile.imwrite(p,a,description=etree.tostring(root),metadata=None,photometric="minisblack")
    with open_dataset(p,input_type="ome_tiff") as ds:
        for t in range(3):
            with ds.plane(t=t) as plane:
                assert plane.asarray()[0,0]==2-t


def test_prefetch_opt_in_and_consumed_cache(tmp_path):
    p=tmp_path/"a.ome.tif";write_tiff(p)
    with open_dataset(p,input_type="ome_tiff",cache_bytes=40000) as ds:
        assert ds.prefetch_info()=={"enabled":False}
        futures=ds.hint_navigation(PlaneRequest((0,16,0,16),z=0,t=0),axis="Z")
        for future in futures:future.result(timeout=10)
        loaded=ds.cache_info()["loads"]
        with ds.plane(z=1,t=0) as plane:plane.read_region(0,16,0,16)
        assert ds.cache_info()["loads"]==loaded
        assert ds.cache_info()["prefetched_hits"]==3
        assert ds.prefetch_info()["completed"]==1
        with pytest.raises(ValueError,match="max_bytes"):
            ds.prefetch([PlaneRequest((0,17,0,19),z=0,t=0)],
                        policy=PrefetchPolicy(max_bytes=1))


def test_cache_budgets_dedup_and_clear_during_load():
    cache=ReadCache(128)
    began,release=Event(),Event()
    def load():
        began.set();assert release.wait(5)
        return np.arange(16,dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first=pool.submit(cache.load,"x",load)
        assert began.wait(5)
        second=pool.submit(cache.load,"x",load)
        release.set()
        np.testing.assert_array_equal(first.result(),second.result())
    assert cache.info()["loads"]==1
    assert cache.info()["resident_bytes"]==16
    began.clear();release.clear();cache.clear()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future=pool.submit(cache.load,"x",load)
        assert began.wait(5);cache.clear();release.set();future.result()
    assert cache.info()["resident_bytes"]==0
    for i in range(30):cache.load(i,lambda:np.zeros(32,np.uint8))
    assert cache.info()["resident_bytes"]<=128
    cache.load("too-big",lambda:np.zeros(200,np.uint8))
    assert "too-big" not in cache


def test_speculation_cannot_evict_demand():
    cache=ReadCache(128)
    cache.load("demand",lambda:np.zeros(128,np.uint8))
    cache.load("guess",lambda:np.zeros(16,np.uint8),speculative=True)
    assert "demand" in cache and "guess" not in cache


def test_array_return_is_owned_and_byte_order_preserved():
    a=np.arange(2*5*7,dtype='>u2').reshape(2,5,7)
    with ImageDataset.from_array(a,axes="ZYX") as ds:
        with ds.plane(z=1) as plane:
            out=plane.asarray();assert out.flags.owndata and out.dtype.isnative
            out[:]=0
            np.testing.assert_array_equal(plane.asarray(),a[1])


def test_explicit_reader_and_borrowed_plane_share_observation_identity(tmp_path):
    p=tmp_path/'identity.ome.tif';write_tiff(p)
    with OMETiffReader(p,z=2,t=1) as reader, reader.plane(z=2,t=1) as plane:
        assert reader.plane_selection==plane.plane_selection
        with reader.crop(0,2,0,3) as cropped:
            assert cropped.plane_selection==reader.plane_selection


def test_cache_failure_remains_an_error_and_is_not_retained():
    cache = ReadCache(128)
    def fail():
        raise OSError("synthetic read failure")
    with pytest.raises(OSError, match="synthetic"):
        cache.load("bad", fail, speculative=True)
    assert "bad" not in cache
    with pytest.raises(OSError, match="synthetic"):
        cache.load("bad", fail)
    assert cache.info()["load_errors"] == 2
    good = cache.load("good", lambda: np.arange(5, dtype=np.uint8))
    np.testing.assert_array_equal(good, np.arange(5, dtype=np.uint8))


def test_prefetch_queue_supersedes_pending_and_exposes_errors():
    from omeify.io.prefetch import PrefetchQueue
    started, release = Event(), Event()
    seen = []
    def execute(request, cancelled):
        if request.z == 0:
            started.set()
            assert release.wait(5)
        if cancelled():
            raise CancelledError()
        seen.append(request.z)
        if request.z == 3:
            raise OSError("synthetic speculative failure")
    queue = PrefetchQueue(execute, lambda request: 1, PrefetchPolicy())
    requests = lambda *zs: [PlaneRequest((0, 1, 0, 1), z=z) for z in zs]
    try:
        first = queue.submit(requests(0, 1))
        assert started.wait(5)
        latest = queue.submit(requests(2, 3))
        assert first[1].cancelled()
        release.set()
        with pytest.raises(CancelledError):
            first[0].result(timeout=5)
        latest[0].result(timeout=5)
        with pytest.raises(OSError, match="speculative"):
            latest[1].result(timeout=5)
        assert seen == [2, 3]
        assert queue.info()["failed"] == 1
        assert queue.info()["cancelled"] == 2
        assert queue.info()["pending"] == 0
    finally:
        release.set()
        queue.close()


def test_acquisition_measurements_are_separate_and_selection_aware(tmp_path):
    path = tmp_path / "measured.ome.tif"
    a = np.arange(2*3*5*7, dtype=np.uint16).reshape(2, 3, 5, 7)
    tifffile.imwrite(path, a, ome=True, photometric="minisblack", metadata={
        "axes": "TZYX", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
        "PhysicalSizeZ": 2, "TimeIncrement": 10,
        "Plane": {"DeltaT": [0, 1, 2, 10, 11, 12],
                  "PositionZ": [20, 22, 24, 20, 22, 24]},
    })
    with open_dataset(path, input_type="ome_tiff") as dataset:
        assert dataset.axis_info("Z").origin is None
        assert dataset.axis_info("T").scale == 10
        record = dataset.acquisition_plane(z=2, t=1)
        assert record["DeltaT"] == (12., "s")
        assert record["PositionZ"] == (24., "reference frame")
        with dataset.select(z=slice(1, 3), t=1) as selected:
            assert selected.acquisition_plane(z=1, t=0) == record
        # Only DeltaT is curated by export; stage positions no longer block conversion.
        dataset.require_regular_export()
        from omeify.io.plane_times import plane_delta_t
        assert plane_delta_t(dataset)[-1] == (1, 0, 2, 12.)
        with dataset.plane(z=2, t=1) as plane:
            np.testing.assert_array_equal(plane.asarray(), a[1, 2])


def test_source_clear_cancels_borrowed_predictors_and_reader_close_joins_them(tmp_path):
    path = tmp_path / "shared.ome.tif"
    write_tiff(path)
    with OMETiffReader(path, z=0, t=0) as reader:
        dataset = reader.dataset
        with dataset.select() as selected:
            futures = selected.prefetch([PlaneRequest((0, 16, 0, 16), z=1, t=0)])
            for future in futures:
                future.result(timeout=5)
            assert selected.cache_info()["resident_bytes"] > 0
            reader.clear_cache()
            assert selected.cache_info()["resident_bytes"] == 0
            # Closing the physical reader joins workers owned by its borrowed
            # dataset descendants, not merely its own facade's worker.
            reader.close()
            assert selected.prefetch_info() == {"enabled": False}
            assert not selected.is_open
            with pytest.raises(RuntimeError, match="closed|session"):
                selected.plane(z=1, t=0)
