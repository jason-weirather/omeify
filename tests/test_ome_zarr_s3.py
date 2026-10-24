"""Optional S3 reader transport contract; no account or network required."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from omeify.io.ome_zarr.store import location, read_store


@pytest.mark.parametrize("path", [
    "s3://bucket/data.ome.zarr", "s3://bucket/path/to/store",
    "https://images.example.org/slide.ome.zarr",
])
def test_store_location(path):
    assert location(path) == path


@pytest.mark.parametrize("bad", [
    "ftp://example.org/a.zarr", "s3://bucket/data.ome.zarr?key=abc",
    "s3://username:secret@bucket/a.zarr", "s3://bucket/../outside",
])
def test_bad_store_root(bad):
    with pytest.raises(ValueError):
        location(bad)


def test_s3_store_routes_to_upstream_fsspec(monkeypatch):
    from omeify.io.ome_zarr import store as module

    seen = []

    class FakeBackend:
        fs = SimpleNamespace()
        def close(self):
            seen.append("closed")

    class FakeFsspecStore:
        @classmethod
        def from_url(cls, path, *, storage_options, read_only):
            seen.append((path, storage_options, read_only))
            return FakeBackend()

    class FakeWrapper:
        def __init__(self, backend):
            self._store = backend

    fake_zarr = SimpleNamespace(storage=SimpleNamespace(
        FsspecStore=FakeFsspecStore, WrapperStore=FakeWrapper,
    ))
    monkeypatch.setitem(sys.modules, "zarr", fake_zarr)
    monkeypatch.setitem(sys.modules, "s3fs", SimpleNamespace())
    out = read_store("s3://bucket/a.ome.zarr", storage_options={
        "client_kwargs": {"endpoint_url": "http://127.0.0.1:9000"}}, cache_bytes=0)
    assert seen == [("s3://bucket/a.ome.zarr", {
        "client_kwargs": {"endpoint_url": "http://127.0.0.1:9000"},
        "skip_instance_cache": True}, True)]
    out.close()
    assert seen[-1] == "closed"
