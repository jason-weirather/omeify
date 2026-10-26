"""Conservative recovery of missing OME SubIFD sampling metadata."""

import numpy as np
import pytest
import tifffile

from omeify import OMETiffReader, PixelSize


def _write_pyramid(
    path, shapes, *, level_resolutions=None, physical_size=True, channels=2,
    reduced_flags=True,
):
    """Make linked physical SubIFDs without reduced-level TIFF calibration by default."""
    level_resolutions = level_resolutions or {}
    metadata = {"axes": "CYX", "Channel": {"Name": [f"C{i}" for i in range(channels)]}}
    if physical_size:
        metadata.update(PhysicalSizeX=0.5, PhysicalSizeY=0.5,
                        PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm")
    with tifffile.TiffWriter(path, ome=True) as writer:
        for index, (height, width) in enumerate(shapes):
            # Values are distinct at each level so reading a selected level is verifiable.
            pixels = np.full((channels, height, width), index + 5, dtype=np.uint16)
            options = {}
            if index in level_resolutions:
                options["resolution"] = level_resolutions[index]
                options["resolutionunit"] = "CENTIMETER"
            writer.write(
                pixels,
                tile=(16, 16),
                subifds=len(shapes) - 1 if index == 0 else None,
                subfiletype=int(reduced_flags) if index else 0,
                metadata=metadata if index == 0 else None,
                **options,
            )


@pytest.mark.parametrize("step", [2, 3, 4])
def test_exact_linked_ome_pyramid_recovers_scales_without_reading_pixels(tmp_path, step):
    shapes = [(288 // step**i, 576 // step**i) for i in range(3)]
    path = tmp_path / "legacy.ome.tif"
    _write_pyramid(path, shapes)

    with OMETiffReader(path) as image:
        assert len(image.levels) == len(shapes)
        assert image.pixel_size == PixelSize(0.5, 0.5, "µm")
        for index, level in enumerate(image.levels):
            assert level.spatial_shape == shapes[index]
            assert level.downsample_yx == (float(step**index),) * 2
            assert level.pixel_size == image.pixel_size.scaled(step**index)
            if index:
                assert level.offset_yx is None  # Neither scale nor shape proves phase.
        np.testing.assert_array_equal(
            image.read_region(1, 4, 2, 6, level=2, channels=[1]),
            np.full((1, 3, 4), 7, dtype=np.uint16),
        )


@pytest.mark.parametrize("shapes", [
    [(192, 384), (96, 193)],  # Noninteger X ratio.
    [(192, 384), (96, 192), (47, 96)],  # Rounded, rather than exact, division.
    [(192, 384), (96, 192), (24, 48)],  # Changing reduction step.
    [(192, 384), (64, 192)],  # Anisotropic reduction.
])
def test_dimensions_that_do_not_prove_a_regular_pyramid_remain_unknown(tmp_path, shapes):
    path = tmp_path / "ambiguous.ome.tif"
    _write_pyramid(path, shapes)
    with OMETiffReader(path) as image:
        assert image.levels[0].downsample_yx == (1.0, 1.0)
        assert all(level.downsample_yx is None for level in image.levels[1:])
        assert all(level.pixel_size is None for level in image.levels[1:])


def test_conflicting_independent_tiff_calibration_blocks_the_entire_inference(tmp_path):
    path = tmp_path / "conflicting.ome.tif"
    _write_pyramid(path, [(192, 384), (96, 192), (48, 96)],
                   level_resolutions={1: (20000, 20000)})
    with OMETiffReader(path) as image:
        # Level one declares 0.5 µm again, contradicting its 2x raster dimensions.
        assert image.levels[1].pixel_size == PixelSize(0.5, 0.5, "µm")
        assert image.levels[1].downsample_yx == (1.0, 1.0)
        assert image.levels[2].downsample_yx is None


def test_consistent_independent_tiff_calibration_is_preserved(tmp_path):
    path = tmp_path / "partly-calibrated.ome.tif"
    _write_pyramid(path, [(192, 384), (96, 192), (48, 96)],
                   level_resolutions={1: (10000, 10000)})
    with OMETiffReader(path) as image:
        assert image.levels[1].downsample_yx == (2.0, 2.0)
        assert image.levels[1].pixel_size == PixelSize(1.0, 1.0, "µm")
        assert image.levels[2].downsample_yx == (4.0, 4.0)
        assert image.levels[2].pixel_size == PixelSize(2.0, 2.0, "µm")
        assert image.levels[2].offset_yx is None


def test_missing_base_physical_size_does_not_create_absolute_calibration(tmp_path):
    path = tmp_path / "no-calibration.ome.tif"
    _write_pyramid(path, [(192, 384), (96, 192)], physical_size=False)
    with OMETiffReader(path) as image:
        assert image.pixel_size is None
        assert image.levels[1].downsample_yx == (2.0, 2.0)
        assert image.levels[1].pixel_size is None
        assert image.levels[1].offset_yx is None


def test_missing_reduced_image_flags_cannot_authorize_inference(tmp_path):
    path = tmp_path / "no-reduced-flag.ome.tif"
    _write_pyramid(path, [(192, 384), (96, 192)], reduced_flags=False)
    with OMETiffReader(path) as image:
        assert len(image.levels) == 2  # tifffile still groups the SubIFD as a level.
        assert image.levels[1].downsample_yx is None


def test_explicit_ome_geometry_record_is_never_repaired_by_inference(tmp_path, monkeypatch):
    path = tmp_path / "broken-record.ome.tif"
    _write_pyramid(path, [(192, 384), (96, 192)])
    monkeypatch.setattr(
        "omeify.io._tiff_sources.ome.read_geometry",
        lambda *_: {"schema": "omeify.geometry/1", "origin_yx_um": [0, 0],
                   "pyramid_method": "mean"},
    )
    with pytest.raises(ValueError, match="requires calibrated pyramid levels"):
        with OMETiffReader(path):
            pass


def test_thirteen_channel_seven_level_ome_uses_existing_subifds(tmp_path):
    # Exercise the multiplex layout and all per-plane SubIFD offsets, not an RGB shortcut.
    path = tmp_path / "multiplex.ome.tif"
    base_height, base_width = 512, 768
    shapes = [(base_height // 2**i, base_width // 2**i) for i in range(8)]
    _write_pyramid(path, shapes, channels=13)
    with OMETiffReader(path) as image:
        assert image.channel_count == 13
        assert len(image.levels) == 8
        assert image.levels[-1].spatial_shape == (4, 6)
        assert image.levels[-1].downsample_yx == (128.0, 128.0)
        assert image.levels[-1].offset_yx is None
        np.testing.assert_array_equal(
            image.read_region(0, 4, 0, 6, level=7, channels=[9, 10, 11, 12]),
            np.full((4, 4, 6), 12, dtype=np.uint16),
        )
