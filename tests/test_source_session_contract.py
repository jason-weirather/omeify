"""The source lifecycle is common; 2D and dataset metadata contracts are not."""
from __future__ import annotations

import numpy as np
import pytest
import tifffile

from omeify import ImageDataset, ImageMetadata, ImageSource, OMETiffReader
from omeify.io.dataset_source import DatasetSourceBase, check_dataset_source
from omeify.io.dataset_metadata import AxisInfo, DatasetLevel


class _Tiny2D(ImageSource):
    def __init__(self, *, publish=True):
        super().__init__()
        self.publish = publish
        self.open_calls = 0
        self.close_calls = 0

    def _open(self):
        self.open_calls += 1
        if self.publish:
            self._metadata = ImageMetadata("YX", (3, 4), np.dtype("uint16"))

    def _close(self):
        self.close_calls += 1
        self._metadata = None

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        return np.full((y1 - y0, x1 - x0), 42, dtype="uint16")


class _TinyDataset(DatasetSourceBase):
    def __init__(self, *, publish=True):
        super().__init__()
        self.publish = publish
        self.open_calls = self.close_calls = 0
        self._levels = ()

    def _open(self):
        self.open_calls += 1
        if self.publish:
            meta = ImageMetadata("YX", (3, 4), np.dtype("uint16"))
            self._levels = (DatasetLevel(0, (AxisInfo("Y", 3, "space"),
                                             AxisInfo("X", 4, "space")), meta),)

    def _close(self):
        self.close_calls += 1
        self._levels = ()

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._levels

    def plane_metadata(self, *, z=None, t=None, level=0):
        self._ensure_open()
        if z is not None or t is not None or level != 0:
            raise ValueError("No Z/T or other native levels")
        return self.dataset_levels[0].plane_metadata

    def read_plane_region(self, y0, y1, x0, x1, *, z, t, level, channels,
                          speculative=False):
        self.plane_metadata(z=z, t=t, level=level)
        return np.full((y1 - y0, x1 - x0), 42, dtype="uint16")


def test_shared_context_generation_and_validation_are_metadata_neutral():
    image_source, dataset_source = _Tiny2D(), _TinyDataset()
    check_dataset_source(dataset_source)
    assert not isinstance(dataset_source, ImageSource)
    assert not hasattr(dataset_source, "metadata")
    assert not hasattr(dataset_source, "_metadata")
    for source in (image_source, dataset_source):
        assert source.generation == 0
        with source as opened:
            assert opened is source
            assert source.is_open and source.generation == 1
            assert source.open() is source  # same session, not a second handle
            with pytest.raises(RuntimeError, match="Nested contexts"):
                with source:
                    pass
        assert not source.is_open and source.generation == 2
        source.close()
        assert source.generation == 2  # close is idempotent
        with source:
            assert source.generation == 3
        assert source.generation == 4
        assert source.open_calls == source.close_calls == 2


def test_open_validation_rolls_back_both_contracts():
    for source in (_Tiny2D(publish=False), _TinyDataset(publish=False)):
        with pytest.raises(TypeError):
            source.open()
        assert not source.is_open
        assert source.generation == 0
        assert source.close_calls == 1  # acquired resources are rolled back
        source.publish = True
        with source:
            assert source.is_open and source.generation == 1
        assert source.close_calls == 2


def test_native_dataset_level_is_authoritative_not_a_second_2d_descriptor():
    values = np.arange(2 * 3 * 4 * 5, dtype="uint16").reshape(2, 3, 4, 5)
    with ImageDataset.from_array(values, axes="TZYX") as dataset:
        source = dataset._source
        assert isinstance(source, DatasetSourceBase)
        assert not hasattr(source, "metadata")
        assert not hasattr(source, "_metadata")
        assert len(source.dataset_levels) == 1
        with dataset.plane(t=1, z=2) as plane:
            assert isinstance(plane._source, ImageSource)
            np.testing.assert_array_equal(plane.asarray(), values[1, 2])
            retained = plane
        with dataset.with_metadata() as viewed:
            assert not hasattr(viewed._source, "_metadata")
            with viewed.plane(t=1, z=2) as plane:
                np.testing.assert_array_equal(plane.asarray(), values[1, 2])
        with dataset.select(z=slice(1, 3), t=slice(0, 2)) as selected:
            assert not hasattr(selected._source, "_metadata")
            with selected.plane(z=1, t=1) as plane:
                np.testing.assert_array_equal(plane.asarray(), values[1, 2])
    with pytest.raises(RuntimeError):
        retained.read_region(0, 1, 0, 1)


def test_existing_tiff_reader_is_one_backend_with_borrowed_dataset(tmp_path):
    data = np.arange(2 * 3 * 5 * 7, dtype="uint16").reshape(2, 3, 5, 7)
    path = tmp_path / "single-backend.ome.tif"
    tifffile.imwrite(path, data, ome=True, metadata={"axes": "TZYX"})
    reader = OMETiffReader(path, z=1, t=0)
    with reader:
        assert reader.dataset._source is reader._source
        assert reader.dataset._owns_source is False
        with reader.dataset.plane(z=2, t=1) as alternate:
            np.testing.assert_array_equal(alternate.asarray(), data[1, 2])
        np.testing.assert_array_equal(reader.asarray(), data[0, 1])
        borrowed = reader.dataset
        token = reader._source.generation
        borrowed.close()
        assert reader.is_open and reader._source.generation == token
        # The reader's dataset property recreates a borrowed navigation scope.
        assert reader.dataset._source is reader._source
        np.testing.assert_array_equal(reader.asarray(), data[0, 1])
    assert not reader._source.is_open


def test_reopening_parent_does_not_resurrect_old_borrowed_views():
    array = np.arange(2 * 3 * 4 * 5, dtype="uint16").reshape(2, 3, 4, 5)
    parent = ImageDataset.from_array(array, axes="TZYX")
    parent.open()
    view = parent.with_metadata()
    view.open()
    plane = view.plane(t=1, z=2)
    plane.open()
    try:
        np.testing.assert_array_equal(plane.asarray(), array[1, 2])
        parent.close()
        parent.open()
        assert not view.is_open and not plane.is_open
        with pytest.raises(RuntimeError):
            plane.read_region(0, 1, 0, 1)
        with pytest.raises(RuntimeError):
            view.plane(t=1, z=2)
        with parent.plane(t=1, z=2) as fresh:
            np.testing.assert_array_equal(fresh.asarray(), array[1, 2])
    finally:
        plane.close()
        view.close()
        parent.close()


def test_failed_validation_preserves_acquisition_error_if_cleanup_also_fails():
    class BadCleanup(_Tiny2D):
        def _close(self):
            super()._close()
            raise OSError("untrusted path hidden in note")

    source = BadCleanup(publish=False)
    with pytest.raises(TypeError) as failure:
        source.open()
    assert not source.is_open
    assert source.generation == 0
    assert any("Source cleanup also failed with OSError" in note
               for note in failure.value.__notes__)
    assert all("untrusted path" not in note for note in failure.value.__notes__)
