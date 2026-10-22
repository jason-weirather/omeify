"""Leica SCN XML/IFD selection, brightfield JPEG color, and normalized output."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import LeicaSCNHEReader, OMETiffReader, convert
from omeify.cli import main
from omeify.io.source_reader import source_reader

_NS = "http://www.leica-microsystems.com/scn/2010/10/01"


def _rgb(height: int, width: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    image[16:48, 16:48] = (255, 255, 255)
    image[16:48, 48:80] = (92, 58, 148)
    image[48:80, 16:48] = (219, 177, 198)
    return image


def _description(
    *, namespace: str = _NS, illumination: str = "brightfield",
    bad_ifd: bool = False, duplicate_ifd: bool = False, bad_view: bool = False,
) -> str:
    # Match a real Leica Versa-style document: two independent images, 2x
    # pyramid levels as top-level IFDs, plus an interleaved supplemental label.
    detail_ifd = 1 if duplicate_ifd else 3
    if bad_ifd:
        detail_ifd = 2  # IFD 2 is the grayscale slide label, not the detailed scan.
    detail_view = 999000 if bad_view else 112000
    return f'''<?xml version="1.0" encoding="utf-8"?>
<scn xmlns="{namespace}">
  <collection name="" sizeX="25000000" sizeY="75000000">
    <barcode>PRIVATE_BARCODE_SENTINEL</barcode>
    <image name="PRIVATE_OVERVIEW_NAME">
      <device model="Versa" version="1.0" />
      <pixels sizeX="96" sizeY="128">
        <dimension sizeX="96" sizeY="128" r="0" z="0" ifd="0" />
        <dimension sizeX="48" sizeY="64" r="1" z="0" ifd="1" />
      </pixels>
      <view sizeX="384000" sizeY="640000" offsetX="0" offsetY="1000000" />
      <scanSettings><illuminationSettings>
        <illuminationSource>brightfield</illuminationSource>
      </illuminationSettings></scanSettings>
    </image>
    <image name="PRIVATE_DETAIL_NAME">
      <device model="Versa" version="1.0" />
      <pixels sizeX="224" sizeY="192">
        <dimension sizeX="224" sizeY="192" r="0" z="0" ifd="{detail_ifd}" />
        <dimension sizeX="112" sizeY="96" r="1" z="0" ifd="4" />
      </pixels>
      <view sizeX="{detail_view}" sizeY="115200" offsetX="6300" offsetY="12500" />
      <scanSettings><illuminationSettings>
        <illuminationSource>{illumination}</illuminationSource>
      </illuminationSettings></scanSettings>
    </image>
    <supplementalImage type="label" ifd="2" />
  </collection>
</scn>'''


def _source(
    path: Path, *, jpeg: bool, namespace: str = _NS,
    illumination: str = "brightfield", bad_ifd: bool = False,
    duplicate_ifd: bool = False, bad_view: bool = False,
    view_only_calibration: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    overview = _rgb(128, 96, (239, 205, 224))
    detailed = _rgb(192, 224, (231, 173, 204))
    label = np.full((32, 64, 3), 50, np.uint8)[..., 0]
    compression = "jpeg" if jpeg else None

    def save(writer, pixels, *, sx, sy, description=None):
        kwargs = dict(
            photometric="rgb", tile=(32, 32), metadata=None,
            description=description,
            compression=compression,
            resolution=(10000 / sx, 10000 / sy),
            resolutionunit="NONE" if view_only_calibration else "CENTIMETER",
        )
        if jpeg:
            kwargs.update(compressionargs={"level": 95, "outcolorspace": "YCBCR"},
                          subsampling=(1, 1))
        writer.write(pixels, **kwargs)

    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        save(writer, overview, sx=4, sy=5, description=_description(
            namespace=namespace, illumination=illumination, bad_ifd=bad_ifd,
            duplicate_ifd=duplicate_ifd, bad_view=bad_view,
        ))
        save(writer, overview[::2, ::2], sx=8, sy=10)
        writer.write(label, photometric="minisblack", tile=(32, 32), metadata=None)
        save(writer, detailed, sx=0.5, sy=0.6)
        save(writer, detailed[::2, ::2], sx=1, sy=1.2)
    return overview, detailed


def test_scn_xml_mapping_and_slide_origin_without_jpeg(tmp_path):
    path = tmp_path / "versa.scn"
    overview, detail = _source(path, jpeg=False)
    with tifffile.TiffFile(path) as tiff:
        assert tiff.is_scn and len(tiff.pages) == 5
    with LeicaSCNHEReader(path, series=0) as image:
        assert image.image_type == "rgb"
        assert image.series_names == ("PRIVATE_OVERVIEW_NAME", "PRIVATE_DETAIL_NAME")
        assert image.shape == overview.shape and len(image.levels) == 2
        assert image.pixel_size.x == 4 and image.pixel_size.y == 5
        assert image.origin_yx == (1002.5, 2.0)
        np.testing.assert_array_equal(image.read_region(14, 50, 21, 60), overview[14:50, 21:60])
    with LeicaSCNHEReader(path, series=1) as image:
        assert image.shape == detail.shape
        assert image.pixel_size.x == .5 and image.pixel_size.y == .6
        assert image.origin_yx == (12.8, 6.55)
        assert image.levels[1].downsample_yx == (2.0, 2.0)
        np.testing.assert_array_equal(image.asarray(level=0), detail)
        np.testing.assert_array_equal(image.asarray(level=1), detail[::2, ::2])
    with pytest.raises(IndexError, match="2 images"):
        with LeicaSCNHEReader(path, series=2):
            pass


@pytest.mark.parametrize("namespace", sorted((
    "http://www.leica-microsystems.com/scn/2010/03/10",
    "http://www.leica-microsystems.com/scn/2010/10/01",
)))
def test_both_leica_namespaces_and_view_calibration_fallback(tmp_path, namespace):
    path = tmp_path / "from-view.scn"
    _source(path, jpeg=False, namespace=namespace, view_only_calibration=True)
    with LeicaSCNHEReader(path, series=1) as image:
        assert image.pixel_size.x == .5
        assert image.pixel_size.y == .6
        assert image.origin_yx == (12.8, 6.55)


@pytest.mark.parametrize("error,kwargs", [
    ("recognized Leica", {"namespace": "http://other-vendor.invalid/scn"}),
    ("brightfield", {"illumination": "fluorescence"}),
    ("IFD 1 is declared more than once", {"duplicate_ifd": True}),
    ("dimensions disagree", {"bad_ifd": True}),
    ("view extent", {"bad_view": True}),
])
def test_scn_profile_rejects_bad_metadata_without_writing(tmp_path, error, kwargs):
    source = tmp_path / "bad.scn"
    _source(source, jpeg=False, **kwargs)
    dest = tmp_path / "untouched.ome.tif"
    dest.write_bytes(b"KEEP")
    with pytest.raises(ValueError, match=error):
        with source_reader(source, input_type="scn_he", series=1):
            pass
    assert dest.read_bytes() == b"KEEP"


def test_wrong_vendor_profile_does_not_accept_scn(tmp_path):
    source = tmp_path / "versa.scn"
    _source(source, jpeg=False)
    with pytest.raises(ValueError, match="Aperio"):
        with source_reader(source, input_type="svs"):
            pass


def test_scn_jpeg_ycbcr_is_decoded_as_rgb_once(tmp_path):
    path = tmp_path / "leica-444.scn"
    _, detail = _source(path, jpeg=True)
    with tifffile.TiffFile(path) as tiff:
        for level in (3, 4):
            page = tiff.pages[level]
            assert int(page.photometric) == 6
            assert tuple(page.tags["YCbCrSubSampling"].value) == (1, 1)
            assert int(page.compression) == 7
        expected = tiff.pages[3].asarray()
    with LeicaSCNHEReader(path, series=1) as image:
        actual = image.asarray()
        np.testing.assert_array_equal(actual, expected)
        # Guard against a double YCbCr->RGB conversion (the historical magenta glass bug).
        np.testing.assert_allclose(actual[20, 20], (255, 255, 255), atol=5, rtol=0)
        np.testing.assert_allclose(actual[20, 60], (92, 58, 148), atol=8, rtol=0)
        assert actual.shape == detail.shape
        assert image.image_type == "rgb" and image.channel_count == 1


def test_scn_convert_default_jpeg_and_explicit_lossless(tmp_path):
    path = tmp_path / "leica.scn"
    _source(path, jpeg=True)
    jpeg_out = tmp_path / "normalized.ome.tif"
    report = convert(path, jpeg_out, input_type="scn_he", series=1)
    assert report["options"]["input_type"] == "scn_he"
    assert report["options"]["compression"] == "JPEG"
    assert report["options"]["jpeg_quality"] == 90
    assert report["options"]["jpeg_subsampling"] == "422"
    assert report["pyramid"]["tile_size"] == 512
    assert report["output_file"]["axes"] == "YXS"
    with tifffile.TiffFile(jpeg_out) as tiff:
        for level in tiff.series[0].levels:
            page = level.pages[0].aspage()
            assert int(page.photometric) == 6
            assert int(page.compression) == 7
            assert tuple(page.tags["YCbCrSubSampling"].value) == (2, 1)
            assert page.tilewidth == page.tilelength == 512
        assert b"PRIVATE_BARCODE_SENTINEL" not in jpeg_out.read_bytes()
        assert b"PRIVATE_DETAIL_NAME" not in jpeg_out.read_bytes()
    lossless_out = tmp_path / "lossless.ome.tif"
    convert(path, lossless_out, input_type="scn_he", series=1, compression="Deflate")
    with LeicaSCNHEReader(path, series=1) as input_image, OMETiffReader(lossless_out) as output:
        np.testing.assert_array_equal(output.asarray(), input_image.asarray())
        assert output.origin_yx == input_image.origin_yx
        assert output.pixel_size == input_image.pixel_size


def test_scn_cli_default_and_crop_routing(tmp_path):
    source = tmp_path / "leica.scn"
    _source(source, jpeg=True)
    target = tmp_path / "overview.ome.tif"
    result = CliRunner().invoke(main, [
        "convert", str(source), "--type", "scn_he", "--series", "0",
        "--output", str(target), "--no-overwrite",
    ])
    assert result.exit_code == 0, result.output
    with OMETiffReader(target) as image:
        assert image.shape == (128, 96, 3)
        assert image.pixel_size.x == 4
    cropped = tmp_path / "cropped.ome.tif"
    result = CliRunner().invoke(main, [
        "crop", str(source), "--type", "scn_he", "--series", "1",
        "--bounds", "16", "16", "96", "112", "--output", str(cropped),
    ])
    assert result.exit_code == 0, result.output
    with OMETiffReader(cropped) as image:
        assert image.shape == (96, 80, 3)


def test_scn_rejects_unsupported_photometric_in_reduced_level(tmp_path):
    source = tmp_path / "bad-photometric.scn"
    _source(source, jpeg=False)
    with tifffile.TiffFile(source, mode="r+") as tiff:
        # YCbCr tags WITHOUT JPEG are not three decoded RGB samples.
        tiff.pages[4].tags["PhotometricInterpretation"].overwrite(6)
    with pytest.raises(ValueError, match="Unsupported RGB photometric.*level 1"):
        with LeicaSCNHEReader(source, series=1):
            pass
