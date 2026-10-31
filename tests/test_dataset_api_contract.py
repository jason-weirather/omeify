"""Logical axes, native-level meaning and independent dataset-source lifetimes."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from omeify import AxisInfo, ImageDataset, PixelSize, PlaneRequest
from omeify.io.dataset_metadata import DatasetLevel, selected_index
from omeify.io.dataset_source import DatasetSourceBase, check_dataset_source
from omeify.io.image_metadata import ImageLevel, ImageMetadata
from omeify.io.image_source import ImageSource


def _coordinate_volume():
    t, c, z, y, x = np.indices((2, 3, 4, 5, 7))
    return (t * 10_000 + c * 1_000 + z * 100 + y * 7 + x).astype("uint16")


@pytest.mark.parametrize("input_axes", ["TCZYX", "XYZCT", "ZTCXY", "YXZCT"])
def test_scalar_blocks_have_canonical_logical_axes_in_any_input_order(input_axes):
    original = _coordinate_volume()
    values = original.transpose(["TCZYX".index(a) for a in input_axes])
    with ImageDataset.from_array(values, axes=input_axes) as dataset:
        assert dataset.axes == "TCZYX"
        assert dataset.shape == original.shape
        assert dataset.native_axes == input_axes
        assert dataset.native_shape == values.shape
        block = dataset.read_block(1, 4, 2, 6, t=slice(None), z=slice(1, 4),
                                   channels=[2, 0, 2])
        assert block.shape == (2, 3, 3, 3, 4)
        np.testing.assert_array_equal(block, original[:, [2, 0, 2], 1:4, 1:4, 2:6])


@pytest.mark.parametrize("kind,input_axes,canonical", [
    ("multichannel", "YX", "TZYX"),
    ("rgb", "SZYXT", "TZYXS"),
    ("label", "ZYXT", "TZYX"),
])
def test_non_channel_blocks_are_canonical_without_guessing(kind, input_axes, canonical):
    base = (np.arange(2 * 3 * 5 * 7) % 247).reshape(2, 3, 5, 7).astype("uint8")
    if kind == "multichannel":
        values = base[0, 0]
        tsel = zsel = None
        expected = values[None, None]
    elif kind == "rgb":
        components = np.stack((base, base + 1, base + 2), axis=-1)
        values = components.transpose(["TZYXS".index(a) for a in input_axes])
        tsel = zsel = slice(None)
        expected = components
    else:
        values = base.transpose(["TZYX".index(a) for a in input_axes])
        tsel = zsel = slice(None)
        expected = base
    with ImageDataset.from_array(values, axes=input_axes, image_type=kind) as dataset:
        assert dataset.axes == canonical
        assert dataset.native_axes == input_axes
        assert dataset.shape == expected.shape
        observed = dataset.read_block(0, 5 if kind != "multichannel" else 5,
                                      0, 7, t=tsel, z=zsel)
        np.testing.assert_array_equal(observed, expected)


class _ThreeLevelSource(DatasetSourceBase):
    """Tiny independent fake volume with a mixed XYZ/XY native pyramid."""

    def __init__(self):
        self.arrays = []
        base = np.arange(2 * 2 * 4 * 9 * 11, dtype="uint16").reshape(2, 2, 4, 9, 11)
        self.arrays.append(base)
        self.arrays.append(base[:, :, ::2, ::2, ::2].copy())
        self.arrays.append(base[:, :, :, ::4, ::4].copy())
        # XY base, one Z-reduced coarse level, then a Z-preserving XY level.
        native_levels = []
        specs = ((4, 9, 11, 2., 10., .5, 0.),
                 (2, 5, 6, 4., 11., 1., .25),
                 (4, 3, 3, 2., 10., 2., .75))
        for i, (z, y, x, z_spacing, z_origin, xy_spacing, xy_origin) in enumerate(specs):
            axes = (
                AxisInfo("T", 2, "time", "s", 5., 0.),
                AxisInfo("C", 2, "channel"),
                AxisInfo("Z", z, "space", "µm", z_spacing, z_origin),
                AxisInfo("Y", y, "space", "µm", xy_spacing, xy_origin),
                AxisInfo("X", x, "space", "µm", xy_spacing, xy_origin),
            )
            size = PixelSize(xy_spacing, xy_spacing, "µm")
            level = ImageLevel(0, "CYX", (2, y, x), size,
                               origin_yx=(xy_origin, xy_origin))
            meta = ImageMetadata("CYX", (2, y, x), np.dtype("uint16"),
                                 channel_names=("A", "B"), pixel_size=size,
                                 levels=(level,))
            native_levels.append(DatasetLevel(i, axes, meta, i != 1))
        base_level = native_levels[0].plane_metadata.levels[0]
        aligned = ImageLevel(1, "CYX", (2, 3, 3), PixelSize(2., 2., "µm"),
                             downsample_yx=(4., 4.), offset_yx=(0., 0.),
                             origin_yx=(.75, .75))
        self._complete_base = replace(native_levels[0].plane_metadata,
                                      levels=(base_level, aligned))
        super().__init__()
        self._native_levels = tuple(native_levels)
        self.plane_level_indices = (0, 2)

    @property
    def dataset_levels(self):
        self._ensure_open()
        return self._native_levels

    def plane_metadata(self, *, z=None, t=None, level=0):
        self._ensure_open()
        if level not in range(3):
            raise IndexError("No native level")
        desc = self._native_levels[level]
        selected_index(desc.axis("Z"), z, name="Z")
        selected_index(desc.axis("T"), t, name="T")
        return self._complete_base if level == 0 else desc.plane_metadata

    def read_plane_region(self, y0, y1, x0, x1, *, z, t, level, channels,
                          speculative=False):
        self.plane_metadata(z=z, t=t, level=level)
        a = self.arrays[level]
        return np.stack([a[t, c, z, y0:y1, x0:x1] for c in channels])


def test_metadata_override_preserves_native_levels_and_safe_2d_pyramid():
    source = _ThreeLevelSource()
    with ImageDataset(source) as dataset:
        with dataset.with_metadata(pixel_size=PixelSize(1., 1., "µm"),
                                   z_spacing_um=3., z_origin_um=20.,
                                   channel_names=("DNA", "CK")) as adjusted:
            assert adjusted.axes == "TCZYX"
            assert adjusted.native_axes == "TCZYX"
            assert len(adjusted.levels) == 3
            assert adjusted._plane_level_map(0) == (0, 2)
            assert adjusted.axis_info("Z").scale == 3.
            assert adjusted.axis_info("Z").origin == 20.
            assert adjusted.axis_info("Z", native_level=1).scale == 6.
            assert adjusted.axis_info("Z", native_level=1).origin == 21.5
            assert adjusted.axis_info("X", native_level=1).scale == 2.
            assert adjusted.axis_info("X", native_level=1).origin == .5
            assert adjusted.axis_info("X", native_level=2).scale == 4.
            assert adjusted.axis_info("X", native_level=2).origin == 1.5
            with adjusted.plane(t=1, z=3) as plane:
                assert len(plane.levels) == 2
                assert plane.channel_names == ("DNA", "CK")
                assert plane.levels[1].pixel_size == PixelSize(4., 4., "µm")
                np.testing.assert_array_equal(plane.asarray(level=1),
                                              source.arrays[2][1, :, 3])
            with adjusted.plane(t=1, z=1, native_level=1) as coarse:
                assert len(coarse.levels) == 1
                assert coarse.pixel_size == PixelSize(2., 2., "µm")
                np.testing.assert_array_equal(coarse.asarray(), source.arrays[1][1, :, 1])
            with adjusted.with_metadata(channel_names=("C", "D")) as renamed:
                assert len(renamed.levels) == 3
                assert renamed._plane_level_map(0) == (0, 2)
                with renamed.plane(t=0, z=0) as plane:
                    assert plane.channel_names == ("C", "D")
                    assert len(plane.levels) == 2
    assert not source.is_open


def test_explicit_native_resolution_keyword_and_source_contract():
    with ImageDataset(_ThreeLevelSource()) as dataset:
        check_dataset_source(dataset._source)
        assert isinstance(dataset._source, DatasetSourceBase)
        assert not isinstance(dataset._source, ImageSource)
        assert not hasattr(dataset._source, "read_region")
        with dataset.plane(t=0, z=0) as view:
            assert isinstance(view._source, ImageSource)
            assert view.plane_selection.native_level == 0
        with pytest.raises(TypeError, match="level"):
            dataset.plane(t=0, z=0, level=1)
        with pytest.raises(TypeError, match="level"):
            dataset.axis_info("Z", level=1)
        with dataset.plane(t=0, z=1, native_level=1) as view:
            assert view.plane_selection.native_level == 1
        request = PlaneRequest((0, 2, 0, 2), z=1, t=0, native_level=1)
        assert request.native_level == 1
        mapped = dataset.locate_plane(z=2, t=0, native_level=1)
        assert mapped["native_level"] == 1
        assert mapped["z"] == 1
