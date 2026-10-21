"""Missing optional TIFF pyramid calibration must not warn during ordinary reads."""

import logging

import numpy as np
import pytest
import tifffile

from omeify import AperioSVSReader, OMETiffReader, PixelSize


@pytest.mark.parametrize("kind", ["svs", "ome_tiff"])
def test_missing_reduced_level_calibration_is_debug_only(tmp_path, caplog, kind):
    base = (np.arange(128 * 160 * 3, dtype=np.uint32) % 251).astype(np.uint8)
    base = base.reshape(128, 160, 3)
    levels = (base, base[::2, ::2], base[::4, ::4])
    path = tmp_path / ("slide.svs" if kind == "svs" else "slide.ome.tif")

    if kind == "svs":
        description = (
            "Aperio Image Library v10.0.51 160x128 "
            "[0,0 160x128] (16x16) Deflate/RGB|AppMag = 20|MPP = 0.5000"
        )
        with tifffile.TiffWriter(path) as writer:
            writer.write(
                levels[0], photometric="rgb", tile=(16, 16), compression="deflate",
                description=description, metadata=None,
                resolution=(20000, 20000), resolutionunit="CENTIMETER",
            )
            # Aperio keeps its associated thumbnail as a separate TIFF series.
            writer.write(levels[2], photometric="rgb", metadata=None)
            # Ordinary reduced IFDs lack their own resolution/physical-size tags.
            for pixels in levels[1:]:
                writer.write(
                    pixels, photometric="rgb", tile=(16, 16),
                    compression="deflate", metadata=None,
                )
        reader_class = AperioSVSReader
        expected_levels = 3
    else:
        with tifffile.TiffWriter(path, ome=True) as writer:
            writer.write(
                levels[0], photometric="rgb", tile=(16, 16), compression="deflate",
                subifds=1,
                metadata={"axes": "YXS", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
                          "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"},
                resolution=(20000, 20000), resolutionunit="CENTIMETER",
            )
            writer.write(
                levels[1], photometric="rgb", tile=(16, 16), compression="deflate",
                subfiletype=1, metadata=None,
            )
        reader_class = OMETiffReader
        expected_levels = 2

    logger_name = "omeify.io._tiff_sources.base"
    with caplog.at_level(logging.WARNING, logger=logger_name):
        with reader_class(path) as image:
            assert image.pixel_size == PixelSize(0.5, 0.5, "µm")
            assert len(image.levels) == expected_levels
            for level in image.levels[1:]:
                assert level.pixel_size is None
                assert level.downsample_yx is None
            np.testing.assert_array_equal(
                image.read_region(0, 8, 0, 9, level=1), levels[1][:8, :9],
            )
    assert not [record for record in caplog.records if record.name == logger_name
                and "Pyramid level" in record.getMessage()]

    caplog.clear()
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        with reader_class(path) as image:
            assert len(image.levels) == expected_levels
    diagnostics = [record for record in caplog.records if record.name == logger_name
                   and "Pyramid level" in record.getMessage()]
    assert len(diagnostics) == expected_levels - 1
    assert all(record.levelno == logging.DEBUG for record in diagnostics)
    assert all("scale will not be inferred" in record.getMessage() for record in diagnostics)
