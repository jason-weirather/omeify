"""Run with python -I from a fresh wheel environment outside the source checkout.

Unlike unit fixtures, this requires real codecs and the real OME XSD. It does not
use a scanner image, call a model, or certify any downstream scientific pipeline.
"""
from __future__ import annotations

import json
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path
from tempfile import TemporaryDirectory

import imagecodecs
import numpy as np
import tifffile

import omeify
from omeify import (
    LabelImage, MultichannelImage, OMEImageSeries, OMEMultiSeriesWriter,
    OMETiffLabelReader, OMETiffReader, OMETiffWriter, PixelSize, RGBImage,
)
from omeify.reports import report_validator, validate_report
from omeify.utils.ome_schema_validator import OMESchemaValidator


def main() -> None:
    checkout = Path(__file__).resolve().parents[1]
    assert checkout not in Path(omeify.__file__).resolve().parents, "Source checkout was imported"
    assert omeify.__version__ == version("omeify")
    assert version("omeify") != "0+unknown"
    OMESchemaValidator()
    assert imagecodecs.jpeg_decode and imagecodecs.lzw_decode
    schemas = [p.name for p in files("omeify.schemas").iterdir()
               if p.name.endswith(".schema.json")]
    for name in schemas:
        report_validator(name)
    assert "write_report.schema.json" in schemas
    assert "crop_provenance.schema.json" in schemas
    size = PixelSize(.5, .7, "µm")
    with TemporaryDirectory() as directory:
        root = Path(directory)
        values = np.arange(2 * 33 * 49, dtype=np.uint16).reshape(2, 33, 49)
        with MultichannelImage.from_array(values, axes="CYX", pixel_size=size) as image:
            report = OMETiffWriter(root / "signal.ome.tif", tile_size=16,
                                   pyramid_levels=2).write(image)
        validate_report(report, "write_report.schema.json")
        assert report["options"]["compression"] == "LZW"
        json.dumps(report, allow_nan=False)
        with OMETiffReader(root / "signal.ome.tif") as image:
            np.testing.assert_array_equal(image.asarray(), values)
            assert image.pixel_size == size and len(image.levels) == 3

        rgb = np.full((64, 96, 3), (230, 91, 170), np.uint8)
        labels = np.arange(64 * 96, dtype=np.uint32).reshape(64, 96)
        with (RGBImage.from_array(rgb, pixel_size=size) as image,
              LabelImage.from_array(labels, pixel_size=size) as label_image):
            report = OMETiffWriter(root / "rgb.ome.tif", tile_size=32,
                                   pyramid_levels=1).write(image)
            assert report["options"]["jpeg_subsampling"] == "422"
            with OMETiffReader(root / "rgb.ome.tif") as decoded:
                for level in range(2):
                    np.testing.assert_allclose(decoded.asarray(level=level)[8, 8],
                                               rgb[8, 8], atol=4, rtol=0)
            report = OMEMultiSeriesWriter(root / "pair.ome.tif", tile_size=32,
                                          pyramid_levels=1).write([
                OMEImageSeries("RGB", image), OMEImageSeries("Labels", label_image),
            ])
            validate_report(report, "multi_series_report.schema.json")
            json.dumps(report, allow_nan=False)
        with OMETiffLabelReader(root / "pair.ome.tif", series=1) as image:
            np.testing.assert_array_equal(image.asarray(), labels)
        with tifffile.TiffFile(root / "rgb.ome.tif") as tiff:
            assert int(tiff.pages[0].photometric) == 6
            assert tuple(tiff.pages[0].tags["YCbCrSubSampling"].value) == (2, 1)
    print(f"Installed omeify {omeify.__version__}: {len(schemas)} schemas; real LZW/JPEG/label checks passed")


if __name__ == "__main__":
    main()
