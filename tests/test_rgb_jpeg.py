"""RGB JPEG encoding policy, public entry points, and optional independent readback."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import (
    LabelImage,
    MultichannelImage,
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffReader,
    OMETiffWriter,
    PixelSize,
    RGBImage,
    TemporaryOMETiffWriter,
    convert,
)
from omeify.cli import main
from omeify.io._writer.configuration import compression_settings
from omeify.io._writer.model import PreparedImage
from omeify.io._writer.verification import verify_page_layout, verify_storage
from omeify.io._writer.writing import common_write_options
from omeify.io.image_planes import ImagePlaneSource

_SAMPLING = {"444": (1, 1), "422": (2, 1), "420": (2, 2), "411": (4, 1)}
_SIZE = PixelSize(0.5048, 0.5048, "µm")


def _swatches() -> np.ndarray:
    """Known colors, including white glass, primaries, and H&E-like pastels."""
    colors = np.array([
        (255, 255, 255), (12, 12, 12), (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (242, 199, 220), (94, 65, 139), (126, 172, 218), (128, 128, 128), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (185, 122, 162), (210, 175, 205), (245, 245, 245),
    ], dtype=np.uint8).reshape(3, 5, 3)
    # Odd edges exercise incomplete tiles and rounded pyramid shapes.
    return np.pad(colors.repeat(64, axis=0).repeat(64, axis=1),
                  ((0, 1), (0, 1), (0, 0)), mode="edge")


def _assert_colors(actual: np.ndarray, level: int) -> None:
    base = _swatches()
    factor = 2**level
    assert actual.shape == ((base.shape[0] + factor - 1) // factor,
                            (base.shape[1] + factor - 1) // factor, 3)
    assert actual.dtype == np.uint8
    y = np.arange(32, 192, 64)
    x = np.arange(32, 320, 64)
    # Sample swatch interiors, away from deliberately lossy chroma edges.
    np.testing.assert_allclose(actual[(y // factor)[:, None], x // factor],
                               base[y[:, None], x], atol=12, rtol=0)


@pytest.mark.parametrize("sampling", _SAMPLING)
def test_jpeg_color_space_is_explicit_and_preserves_sampling(sampling: str) -> None:
    settings = compression_settings("JPEG", np.dtype("uint8"), is_rgb=True,
                                    jpeg_quality=90, jpeg_subsampling=sampling)
    assert settings.compression_args == {
        "level": 90, "outcolorspace": "RGB" if sampling == "444" else "YCBCR",
    }
    assert settings.subsampling == _SAMPLING[sampling]
    assert not settings.lossless
    gray = compression_settings("JPEG", np.dtype("uint8"), is_rgb=False,
                                jpeg_quality=90, jpeg_subsampling=sampling)
    assert gray.compression_args == {"level": 90}
    assert gray.subsampling is None


def test_encoder_options_do_not_mutate_prepared_policy(image_factory) -> None:
    image = image_factory(_swatches(), kind="rgb", pixel_size=_SIZE)
    source = ImagePlaneSource(image)
    compression = compression_settings("JPEG", image.dtype, is_rgb=True,
                                       jpeg_quality=90, jpeg_subsampling="444")
    prepared = PreparedImage(source, source.output_spec(), "mean", compression, (image.shape,))
    first = common_write_options(prepared, tile_size=64, max_workers=1)
    first["compressionargs"]["outcolorspace"] = "YCBCR"
    second = common_write_options(prepared, tile_size=64, max_workers=1)
    assert second["compressionargs"] == {"level": 90, "outcolorspace": "RGB"}
    assert second["subsampling"] == (1, 1)
    assert second["photometric"] == "rgb"
    assert second["planarconfig"] == "contig"


@pytest.mark.parametrize("sampling", _SAMPLING)
@pytest.mark.parametrize("reduced", [False, True])
def test_storage_requires_matching_photometric(image_factory, sampling: str, reduced: bool) -> None:
    image = image_factory(_swatches(), kind="rgb", pixel_size=_SIZE)
    spec = ImagePlaneSource(image).output_spec()
    expected = 2 if sampling == "444" else 6
    page = SimpleNamespace(is_tiled=True, compression=7, samplesperpixel=3,
                           photometric=expected, planarconfig=1, subfiletype=int(reduced),
                           tags={"YCbCrSubSampling": SimpleNamespace(value=_SAMPLING[sampling])})
    kwargs = dict(spec=spec, expected_compression=7, expected_subsampling=_SAMPLING[sampling],
                  reduced=reduced, context="RGB test")
    verify_page_layout(page, **kwargs)
    page.photometric = 6 if expected == 2 else 2
    with pytest.raises(ValueError, match="PhotometricInterpretation"):
        verify_page_layout(page, **kwargs)


def test_storage_checks_subifd_color_policy_not_only_base(image_factory) -> None:
    image = image_factory(_swatches(), kind="rgb", pixel_size=_SIZE)
    source = ImagePlaneSource(image)
    compression = compression_settings("JPEG", image.dtype, is_rgb=True,
                                       jpeg_quality=90, jpeg_subsampling="444")
    prepared = PreparedImage(source, source.output_spec(), "mean", compression,
                             (image.shape, (97, 161, 3)))
    sub = SimpleNamespace(is_tiled=True, compression=7, samplesperpixel=3, photometric=6,
                          planarconfig=1, subfiletype=1,
                          tags={"YCbCrSubSampling": SimpleNamespace(value=(1, 1))})
    sub.aspage = lambda: sub
    base = SimpleNamespace(**{k: v for k, v in vars(sub).items() if k != "aspage"})
    base.photometric, base.subfiletype, base.subifds, base.pages = 2, 0, (512,), [sub]
    base.aspage = lambda: base
    with pytest.raises(ValueError, match="level 1 PhotometricInterpretation"):
        verify_storage([base], prepared)


@pytest.fixture(params=["444", "422", "420", "411"])
def sampling(request) -> str:
    return request.param


@pytest.fixture(params=["single", "temporary", "multi", "convert", "cli"])
def rgb_output(request, tmp_path: Path, sampling: str):
    # Do not substitute codecs or schema validators in these end-to-end tests.
    pytest.importorskip("imagecodecs")
    pytest.importorskip("omeschema")
    pixels = _swatches()
    output = tmp_path / "colors.ome.tif"
    settings = dict(tile_size=64, pyramid_levels=2, max_workers=1)
    # Exercise actual public defaults at 4:4:4, not an explicitly supplied value.
    if sampling != "444":
        settings["jpeg_subsampling"] = sampling
    series = 0
    with RGBImage.from_array(pixels, pixel_size=_SIZE) as image:
        if request.param == "single":
            OMETiffWriter(output, compression="JPEG", **settings).write(image)
        elif request.param == "temporary":
            with TemporaryOMETiffWriter(compression="JPEG", **settings) as writer:
                writer.write(image)
                shutil.copyfile(writer.path, output)
        elif request.param == "multi":
            with (
                MultichannelImage.from_array(pixels[..., 1], axes="YX", pixel_size=_SIZE) as gray,
                LabelImage.from_array(pixels[..., 0].astype(np.uint32), pixel_size=_SIZE) as labels,
            ):
                OMEMultiSeriesWriter(output, compression="Uncompressed", **settings).write((
                    OMEImageSeries("Scalar JPEG preview", gray, compression="JPEG"),
                    OMEImageSeries("RGB", image, compression="JPEG"),
                    OMEImageSeries("Labels", labels),
                ))
            series = 1
        else:
            source = tmp_path / "source.svs"
            # A small independent Aperio-profile fixture, not an Omeify output.
            tifffile.imwrite(source, pixels, tile=(64, 64), photometric="rgb", metadata=None,
                             description="Aperio Image Library v10.0.51|MPP = 0.5048|AppMag = 20")
            if request.param == "convert":
                convert(source, output, input_type="svs", **settings)
            else:
                args = ["convert", str(source), "--output", str(output), "--type", "svs",
                        "--tile-size", "64", "--pyramid-levels", "2", "--workers", "1"]
                if sampling != "444":
                    args += ["--jpeg-subsampling", sampling]
                result = CliRunner().invoke(main, args)
                assert result.exit_code == 0, result.output
    return output, series


def test_rgb_jpeg_public_paths_and_all_levels(rgb_output, sampling: str) -> None:
    import imagecodecs

    path, series_index = rgb_output
    with tifffile.TiffFile(path) as tiff, OMETiffReader(path, series=series_index) as image:
        assert image.image_type == "rgb"
        assert image.channel_count == 1
        assert image.sample_count == 3
        assert image.pixel_size == _SIZE
        levels = tiff.series[series_index].levels
        assert len(levels) == 3
        for level_index, level in enumerate(levels):
            page = level.pages[0].aspage()
            assert int(page.photometric) == (2 if sampling == "444" else 6)
            assert int(page.planarconfig) == 1
            assert int(page.compression) == 7
            assert page.samplesperpixel == 3
            assert tuple(page.tags["YCbCrSubSampling"].value) == _SAMPLING[sampling]
            _assert_colors(level.asarray(), level_index)
            _assert_colors(image.asarray(level=level_index), level_index)
            # Decode a complete JPEG tile independently of TIFF's photometric
            # overrides. A header-only relabel of YCbCr as RGB must not pass.
            tiff.filehandle.seek(page.dataoffsets[0])
            encoded = tiff.filehandle.read(page.databytecounts[0])
            decoded = imagecodecs.jpeg_decode(encoded)
            np.testing.assert_allclose(decoded[8, 8], (255, 255, 255), atol=3, rtol=0)
            if sampling == "444":
                forced = imagecodecs.jpeg_decode(encoded, colorspace="RGB", outcolorspace="RGB")
                np.testing.assert_array_equal(forced, decoded)
        if series_index == 1:
            assert int(tiff.series[0].pages[0].photometric) == 1
            assert int(tiff.series[0].pages[0].compression) == 7
            np.testing.assert_array_equal(tiff.series[2].asarray(),
                                          _swatches()[..., 0].astype(np.uint32))


@pytest.fixture
def bioformats_jar() -> Path:
    value = os.environ.get("OMEIFY_BIOFORMATS_JAR")
    if not value:
        pytest.skip("Set OMEIFY_BIOFORMATS_JAR for independent Bio-Formats readback")
    path = Path(value).expanduser().resolve()
    assert path.is_file(), f"Bio-Formats JAR not found: {path}"
    assert shutil.which("java"), "Bio-Formats readback requires Java 11+"
    return path


def test_bioformats_rgb_color_readback(bioformats_jar: Path, rgb_output, tmp_path: Path) -> None:
    """Optional real Java reader: no reimplementation of Bio-Formats' decoder."""
    path, series_index = rgb_output
    destination = tmp_path / "bioformats"
    destination.mkdir()
    helper = Path(__file__).parent / "java" / "RGBReadback.java"
    result = subprocess.run(
        ["java", "-Djava.awt.headless=true", "--class-path", str(bioformats_jar),
         str(helper), str(path), str(series_index), str(destination)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    files = sorted(destination.glob("*.rgb"))
    assert len(files) == 3, result.stdout + result.stderr
    for path in files:
        level, height, width = (int(value) for value in path.stem.split("-"))
        actual = np.fromfile(path, dtype=np.uint8).reshape(height, width, 3)
        _assert_colors(actual, level)
