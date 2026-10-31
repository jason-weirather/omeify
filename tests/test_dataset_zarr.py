"""Mandatory real Zarr 0.4/0.5/0.6 I/O tests, not optional integration skips."""
from __future__ import annotations

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import zarr

from omeify import OMEZarrReader, PlaneRequest, PrefetchPolicy, open_dataset
from zarr_fixture import attributes, fixture


def data():
    t,c,z,y,x=np.indices((3,2,5,19,21))
    return (t*10000+c*1000+z*100+y*3+x).astype('uint16')


@pytest.mark.parametrize('version',['0.4','0.5','0.6'])
@pytest.mark.parametrize('axes',['tczyx','tcxyz'])
def test_zt_planes_and_regions_are_source_axis_independent(tmp_path,version,axes):
    expected=data()
    values=expected.transpose(['tczyx'.index(a) for a in axes])
    path=tmp_path/'input.zarr'
    fixture(path,version=version,data=values,axes=axes)
    with open_dataset(path,input_type='ome_zarr') as ds:
        assert ds.axes==axes.upper()
        assert ds.axis_info('Z').size==5
        assert ds.axis_info('T').size==3
        for t in range(3):
            for z in range(5):
                with ds.plane(t=t,z=z) as image:
                    assert image.axes=='CYX'
                    np.testing.assert_array_equal(image.read_region(2,18,3,20,channels=[1,0,1]),
                                                   expected[t,[1,0,1],z,2:18,3:20])
                    np.testing.assert_array_equal(image.asarray(level=1),expected[t,:,z,::2,::2])
        store=ds._source.store
        with ds.plane(t=0,z=0) as a: a.read_region(0,5,0,5,channels=0)
        before=ds.cache_info()['loads']
        with ds.plane(t=0,z=0) as b: b.read_region(0,5,0,5,channels=0)
        assert ds.cache_info()['loads']==before
        assert ds._source.store is store


def wide_chunk_fixture(path, *, sharded=False):
    expected=data()
    root=zarr.open_group(str(path),mode='w',zarr_format=3)
    kwargs={'shards':(3,2,4,32,32)} if sharded else {}
    arr=root.create_array('0',shape=expected.shape,chunks=(3,2,2,16,16),dtype='uint16',
        serializer=zarr.codecs.BytesCodec(endian='little'),compressors=[],
        dimension_names=tuple('tczyx'),**kwargs)
    arr[:]=expected
    root.attrs.update(attributes('0.5','tczyx',['0'],[[1,1,2,.5,.5]],[[0]*5],['a','b']))
    root.store.close()
    return expected


@pytest.mark.parametrize('sharded',[False,True])
def test_native_chunk_cache_shares_t_z_channels(tmp_path,sharded):
    path=tmp_path/'chunks.zarr';expected=wide_chunk_fixture(path,sharded=sharded)
    with open_dataset(path,input_type='ome_zarr',cache_bytes=128*1024) as ds:
        with ds.plane(t=0,z=0) as p:
            a=p.read_region(0,8,0,8,channels=0)
        loaded=ds.cache_info()['loads']
        with ds.plane(t=2,z=1) as p:
            b=p.read_region(0,8,0,8,channels=1)
        np.testing.assert_array_equal(a,expected[0,0:1,0,:8,:8])
        np.testing.assert_array_equal(b,expected[2,1:2,1,:8,:8])
        assert loaded==1
        assert ds.cache_info()['loads']==loaded
        assert ds.cache_info()['resident_bytes']<=128*1024
        b[:]=0
        with ds.plane(t=2,z=1) as p:
            np.testing.assert_array_equal(p.read_region(0,8,0,8,channels=1),expected[2,1:2,1,:8,:8])


@pytest.mark.parametrize('version',['0.4','0.5','0.6'])
def test_xyz_levels_are_explicit_not_false_optical_sections(tmp_path,version):
    expected=data();path=tmp_path/'xyz.zarr';fmt=2 if version=='0.4' else 3
    root=zarr.open_group(str(path),mode='w',zarr_format=fmt)
    for i in range(2):
        values=expected[:,:,::2**i,::2**i,::2**i]
        kwargs={'compressor':None} if fmt==2 else dict(compressors=[],dimension_names=tuple('tczyx'))
        arr=root.create_array(str(i),shape=values.shape,dtype='uint16',chunks=(1,1,1,16,16),**kwargs)
        arr[:]=values
    root.attrs.update(attributes(version,'tczyx',['0','1'],[[1,1,2,.5,.5],[1,1,4,1,1]],
                                 [[0]*5,[0]*5],['a','b']))
    root.store.close()
    with OMEZarrReader(path,z=3,t=1) as plane:
        assert len(plane.levels)==1
        np.testing.assert_array_equal(plane.asarray(),expected[1,:,3])
        assert len(plane.source_levels)==2
    with open_dataset(path,input_type='ome_zarr') as ds:
        with ds.plane(t=1,z=1,level=1) as plane:
            np.testing.assert_array_equal(plane.asarray(),expected[1,:,2,::2,::2])
        mapping=ds.locate_plane(t=1,z=2,level=1)
        assert mapping['z']==1
        assert mapping['z_displacement']==0
        assert not mapping['plane_preserving']


def test_http_plane_changes_do_not_repeat_discovery_or_shared_chunk_fetches(tmp_path):
    expected=wide_chunk_fixture(tmp_path/'source.zarr')
    requests=[]
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            super().do_GET()
        def log_message(self,*args):pass
    server=ThreadingHTTPServer(('127.0.0.1',0),functools.partial(Handler,directory=str(tmp_path)))
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    def chunks():
        return [p for p in requests if p.rsplit('/',1)[-1] not in
                {'zarr.json','.zarray','.zattrs','.zgroup','.zmetadata'}]
    try:
        url=f'http://127.0.0.1:{server.server_port}/source.zarr'
        with open_dataset(url,input_type='ome_zarr') as ds:
            assert not chunks()
            requests.clear()
            with ds.plane(t=0,z=0) as plane:
                plane.read_region(0,8,0,8,channels=0)
            assert len(chunks())==1
            requests.clear()
            with ds.plane(t=2,z=1) as plane:
                np.testing.assert_array_equal(plane.read_region(0,8,0,8,channels=1),
                                               expected[2,1:2,1,:8,:8])
            assert not requests
            futures=ds.prefetch([PlaneRequest((0,8,0,8),z=2,t=0,channels=(0,))],
                                 policy=PrefetchPolicy(max_bytes=128*1024))
            for future in futures:future.result(timeout=10)
            requests.clear()
            with ds.plane(t=2,z=3) as plane:plane.read_region(0,8,0,8,channels=1)
            assert not requests
    finally:
        server.shutdown();server.server_close();worker.join(timeout=5)
