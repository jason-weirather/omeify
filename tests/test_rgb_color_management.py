"""RGB output isolates source ICC metadata and maps colors before encoding.

Fixtures are synthetic and deterministic, so no vendor scanner or private slide
is required to exercise the color/metadata boundary. The full image tests use
real output codecs and the packaged OME XSD in a normal Omeify test environment.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
import tifffile
from PIL import ImageCms
from click.testing import CliRunner

from omeify import (
    OMEImageSeries, OMEMultiSeriesWriter, OMETiffReader, OMETiffWriter,
    OMEZarrCollectionWriter, OMEZarrReader, OMEZarrWriter, PixelSize, RGBImage,
    concat, convert, crop,
)
from omeify.cli import main
from omeify.io.ome_zarr.metadata import output_attributes, profile_errors
from omeify.io.rgb_color import (
    SRGB_ICC, SRGB_SHA256, canonical_rgb, color_management_report,
)

SIZE = PixelSize(0.5, 0.5, "µm")
SECRET = "PATIENT_CODE_IN_ICC"


def foreign_rgb_profile() -> bytes:
    """Valid RGB ICC with a changed transfer curve and a text canary."""
    profile = bytearray(SRGB_ICC)
    # Standard LittleCMS sRGB fixture stores its shared parametric transfer
    # curve at offset 520; update the s15Fixed16 gamma to 1.0.
    assert profile[520:524] == b"para"
    profile[532:536] = (1 << 16).to_bytes(4, "big")
    # The copyright tag uses UTF-16BE and has a fixed 24-character payload.
    original = "No copyright, use freely".encode("utf-16-be")
    replacement = SECRET.ljust(24).encode("utf-16-be")
    assert len(original) == len(replacement) == 48
    assert original in profile
    profile[:] = profile.replace(original, replacement)
    assert bytes(profile) != SRGB_ICC
    assert ImageCms.ImageCmsProfile(io.BytesIO(profile)).profile.xcolor_space.strip() == "RGB"
    return bytes(profile)


def pixels():
    y, x = np.indices((33, 47))
    return np.stack(((x * 5 + y) % 255, (x * 3 + y * 2) % 255,
                     (x + y * 7) % 255), axis=-1).astype(np.uint8)


def expected_srgb(source: np.ndarray, profile: bytes) -> np.ndarray:
    from PIL import Image

    transform = ImageCms.buildTransformFromOpenProfiles(
        ImageCms.ImageCmsProfile(io.BytesIO(profile)),
        ImageCms.ImageCmsProfile(io.BytesIO(SRGB_ICC)), "RGB", "RGB",
        renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    )
    return np.array(ImageCms.applyTransform(Image.fromarray(source), transform))


def check_color_report(item, *, converted):
    assert item["output_space"] == "sRGB"
    assert item["output_profile_sha256"] == SRGB_SHA256
    assert item["pixels_transformed"] is converted
    assert item["source_profile_present"] is converted
    assert item["source_profile"] == ("converted_icc" if converted else "assumed_srgb")


@pytest.mark.parametrize("profile", [None, SRGB_ICC, "foreign"])
def test_regional_color_managed_view_has_no_source_metadata(profile):
    source = pixels()
    icc = foreign_rgb_profile() if profile == "foreign" else profile
    with RGBImage.from_array(source, pixel_size=SIZE, icc_profile=icc) as image:
        with canonical_rgb(image) as managed:
            expected = expected_srgb(source, icc) if icc not in (None, SRGB_ICC) else source
            np.testing.assert_array_equal(managed.read_region(7, 19, 3, 26), expected[7:19, 3:26])
            assert managed.icc_profile == SRGB_ICC
            if icc != SRGB_ICC:
                assert managed.icc_profile != icc
            status = color_management_report(managed)
            assert status["pixels_transformed"] == (profile == "foreign")
            assert status["source_profile"] == (
                "converted_icc" if profile == "foreign" else
                "canonical_srgb" if icc is not None else "assumed_srgb"
            )
        np.testing.assert_array_equal(image.asarray(), source)
        assert image.icc_profile == icc


@pytest.mark.parametrize("bad", [
    b"not a profile", bytes(128), SRGB_ICC[:120],
    SRGB_ICC + b"EXTRA", b"X" * (1024 * 1024 + 1),
])
def test_bad_profiles_fail_closed_before_writer_creates_a_destination(tmp_path, bad):
    with RGBImage.from_array(np.zeros((16, 16, 3), dtype=np.uint8),
                             pixel_size=SIZE, icc_profile=bad) as image:
        for writer, path, collection in (
            (OMETiffWriter(tmp_path / "bad.ome.tif"), tmp_path / "bad.ome.tif", False),
            (OMEZarrWriter(tmp_path / "bad.ome.zarr"), tmp_path / "bad.ome.zarr", False),
            (OMEMultiSeriesWriter(tmp_path / "bad-series.ome.tif"),
             tmp_path / "bad-series.ome.tif", True),
            (OMEZarrCollectionWriter(tmp_path / "bad-series.ome.zarr"),
             tmp_path / "bad-series.ome.zarr", True),
        ):
            with pytest.raises(ValueError, match="source RGB ICC"):
                writer.write([OMEImageSeries("bad", image)]) if collection else writer.write(image)
            assert not path.exists()


def test_generated_zarr_attributes_require_only_the_fixed_profile():
    from dataclasses import replace

    with RGBImage.from_array(pixels(), pixel_size=SIZE) as image:
        with canonical_rgb(image) as normalized:
            w = [dict(min=0, max=255, start=0, end=255)] * 3
            attrs = output_attributes(normalized._metadata, normalized.levels,
                                      method="mean", windows=w)
            assert profile_errors(attrs) == ()
            assert base64.b64decode(attrs["omeify"]["icc_profile_base64"]) == SRGB_ICC
            alt = foreign_rgb_profile()
            attrs["omeify"]["icc_profile_base64"] = base64.b64encode(alt).decode()
            assert profile_errors(attrs)
        with pytest.raises(ValueError, match="canonical sRGB"):
            output_attributes(replace(image._metadata, icc_profile=foreign_rgb_profile()),
                              image.levels, method="mean", windows=w)


@pytest.mark.parametrize("backend", ["tiff", "zarr"])
@pytest.mark.parametrize("with_source_icc", [False, True])
def test_public_writers_emit_only_known_srgb_and_record_color_change(tmp_path, backend, with_source_icc):
    raw = pixels()
    source_profile = foreign_rgb_profile() if with_source_icc else None
    expected = expected_srgb(raw, source_profile) if with_source_icc else raw
    destination = tmp_path / ("output.ome.tif" if backend == "tiff" else "output.ome.zarr")
    writer = (OMETiffWriter(destination, compression="Uncompressed", tile_size=16, pyramid_levels=1)
              if backend == "tiff" else OMEZarrWriter(destination, tile_size=16, pyramid_levels=1))
    with RGBImage.from_array(raw, icc_profile=source_profile, pixel_size=SIZE) as image:
        result = writer.write(image)
    check_color_report(result["image"]["color_management"], converted=with_source_icc)
    assert result["image"]["icc_profile_present"]
    if backend == "tiff":
        with tifffile.TiffFile(destination) as tiff:
            base = tiff.pages[0]
            assert bytes(base.iccprofile) == SRGB_ICC
            for reduced in base.pages:
                assert bytes(reduced.iccprofile) == SRGB_ICC
            np.testing.assert_array_equal(tiff.series[0].asarray(), expected)
        assert result["verification"]["icc_profile_is_canonical_srgb"] is True
        assert SECRET.encode("utf-16-be") not in destination.read_bytes()
    else:
        with OMEZarrReader(destination) as out:
            assert out.icc_profile == SRGB_ICC
            np.testing.assert_array_equal(out.asarray(), expected)
        assert SECRET not in (destination / "zarr.json").read_text()


@pytest.mark.parametrize("backend", ["tiff", "zarr"])
def test_heterogeneous_collection_does_not_copy_source_icc(tmp_path, backend):
    raw = pixels()
    src = foreign_rgb_profile()
    path = tmp_path / ("collection.ome.tif" if backend == "tiff" else "collection.ome.zarr")
    writer = (OMEMultiSeriesWriter(path, compression="Uncompressed", tile_size=16, pyramid_levels=0)
              if backend == "tiff" else OMEZarrCollectionWriter(path, tile_size=16, pyramid_levels=0))
    with RGBImage.from_array(raw, icc_profile=src, pixel_size=SIZE) as image:
        report = writer.write([OMEImageSeries("H&E", image)])
    if backend == "tiff":
        check_color_report(report["series"][0]["color_management"], converted=True)
        with tifffile.TiffFile(path) as tiff:
            assert bytes(tiff.pages[0].iccprofile) == SRGB_ICC
            np.testing.assert_array_equal(tiff.series[0].asarray(), expected_srgb(raw, src))
    else:
        check_color_report(report["series"][0]["image"]["color_management"], converted=True)
        with OMEZarrReader(path) as image:
            assert image.icc_profile == SRGB_ICC
            np.testing.assert_array_equal(image.asarray(), expected_srgb(raw, src))


def test_convert_crop_concat_cli_remain_color_managed(tmp_path):
    src = tmp_path / "input.ome.tif"
    raw = pixels()
    foreign = foreign_rgb_profile()
    tifffile.imwrite(src, raw, ome=True, metadata={
        "axes": "YXS", "PhysicalSizeX": 0.5, "PhysicalSizeY": 0.5,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    }, iccprofile=foreign)
    output = tmp_path / "convert.ome.tif"
    cli = CliRunner().invoke(main, ["convert", str(src), "-o", str(output),
                                    "--type", "ome_tiff", "--compression", "Uncompressed"])
    assert cli.exit_code == 0, cli.output
    target = tmp_path / "crop.ome.tif"
    crop(src, target, bounds=(2, 3, 20, 23), compression="Uncompressed")
    joined = tmp_path / "concat.ome.tif"
    concat([__import__('omeify').ConcatSeries(src, 0)], joined, compression="Uncompressed")
    expected = expected_srgb(raw, foreign)
    for path, wanted in ((output, expected), (target, expected[3:23, 2:20]),
                         (joined, expected)):
        with tifffile.TiffFile(path) as tiff:
            assert bytes(tiff.pages[0].iccprofile) == SRGB_ICC
            np.testing.assert_array_equal(tiff.series[0].asarray(), wanted)
        assert SECRET.encode("utf-16-be") not in path.read_bytes()


def test_foreign_zarr_icc_is_normalized_on_tiff_export(tmp_path):
    import zarr

    source, target = tmp_path / "foreign.ome.zarr", tmp_path / "normalized.ome.tif"
    raw = pixels()
    foreign = foreign_rgb_profile()
    with RGBImage.from_array(raw, pixel_size=SIZE) as image:
        OMEZarrWriter(source, tile_size=16, pyramid_levels=0).write(image)
    root = zarr.open_group(source, mode="a")
    attrs = root.attrs.asdict()
    attrs["omeify"]["icc_profile_base64"] = base64.b64encode(foreign).decode("ascii")
    root.attrs.update(attrs)
    root.store.close()
    report = convert(source, target, input_type="ome_zarr", compression="Uncompressed",
                     tile_size=16, pyramid_levels=0)
    check_color_report(report["image"]["color_management"], converted=True)
    with tifffile.TiffFile(target) as output:
        assert bytes(output.pages[0].iccprofile) == SRGB_ICC
        np.testing.assert_array_equal(output.series[0].asarray(), expected_srgb(raw, foreign))
    assert SECRET.encode("utf-16-be") not in target.read_bytes()
