"""Static HTTP transport tests with real requests, chunk codecs and caches."""
from __future__ import annotations

import functools
import json
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

zarr = pytest.importorskip("zarr", minversion="3.1.3")
from omeify import OMEZarrReader, ZarrInspector, convert
from zarr_fixture import fixture


@pytest.fixture
def server(tmp_path):
    requests = []
    denied = set()
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path in denied:
                self.send_error(403)
                return
            super().do_GET()
        def log_message(self, *args):
            pass
    handler = functools.partial(Handler, directory=str(tmp_path))
    httpd = ThreadingHTTPServer(("127.0.0.1",0),handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}", requests, denied
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=5)


def is_chunk(path):
    return path.rsplit("/",1)[-1] not in {"zarr.json",".zarray",".zattrs",".zgroup",".zmetadata"}


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
def test_metadata_only_open_and_bounded_http_region(tmp_path, server, version):
    data,_,_=fixture(tmp_path/"a",version=version)
    base,requests,_=server
    with OMEZarrReader(base+"/a") as im:
        assert not any(is_chunk(p) for p in requests), requests
        requests.clear()
        np.testing.assert_array_equal(im.read_region(1,8,2,9,channels=1),data[1:2,1:8,2:9])
        chunks=[p for p in requests if is_chunk(p)]
        assert len(chunks)==1, requests
        requests.clear()
        im.read_region(1,8,2,9,channels=1)
        assert not requests, requests
        session=im._source.store._store.fs._session
    assert session.closed
    requests.clear()
    inspector=ZarrInspector(base+"/a")
    assert not inspector.validation_errors()
    inspector.to_json()
    assert not any(is_chunk(p) for p in requests)


def test_http_sparse_missing_is_fill_but_forbidden_is_an_error(tmp_path,server):
    fixture(tmp_path/"a",levels=1)
    base,requests,denied=server
    chunk=tmp_path/"a/0/c/0/0/0"
    chunk.unlink()
    with OMEZarrReader(base+"/a") as im:
        assert not im.read_region(0,8,0,8,channels=0).any()
    denied.add("/a/0/c/0/0/0")
    with OMEZarrReader(base+"/a") as im:
        with pytest.raises(Exception):
            im.read_region(0,8,0,8,channels=0)


def test_close_does_not_close_another_readers_session(tmp_path,server):
    fixture(tmp_path/"a")
    base,_,_=server
    a=OMEZarrReader(base+"/a").open()
    with OMEZarrReader(base+"/a") as b:
        a.close()
        assert b.read_region(0,2,0,2).shape==(3,2,2)


def test_cache_limit_eviction_and_clear(tmp_path,server):
    fixture(tmp_path/"a",levels=1)
    base,requests,_=server
    with OMEZarrReader(base+"/a",cache_bytes=600) as im:
        for c in range(3):
            im.read_region(0,8,0,8,channels=c)
            assert im._source.store.cache_size <= 600
        requests.clear()
        im.clear_cache()
        im.read_region(0,8,0,8,channels=2)
        assert any(is_chunk(p) for p in requests)


def test_remote_conversion_does_not_require_directory_listing(tmp_path,server):
    expected,_,_=fixture(tmp_path/"a",version="0.6")
    base,requests,_=server
    report=convert(base+"/a",tmp_path/"b",input_type="ome_zarr",output_format="ome_zarr",tile_size=16)
    assert report["input_file"]["size_bytes"] is None
    assert not any(p.endswith("/") for p in requests),requests
    with OMEZarrReader(tmp_path/"b") as im:
        np.testing.assert_array_equal(im.asarray(),expected)


def test_metadata_path_cannot_escape_over_http(tmp_path,server):
    fixture(tmp_path/"a")
    path=tmp_path/"a/zarr.json";meta=json.loads(path.read_text())
    meta["attributes"]["ome"]["multiscales"][0]["datasets"][0]["path"]="../outside"
    path.write_text(json.dumps(meta))
    base,requests,_=server
    with pytest.raises(ValueError,match="relative"):
        with OMEZarrReader(base+"/a"):pass
    assert not any("outside" in p for p in requests)
