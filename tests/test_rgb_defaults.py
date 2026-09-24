"""Automatic RGB storage is shared by CLI, conversion, and all public writers."""
from __future__ import annotations

import inspect
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import (
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffWriter,
    PixelSize,
    RGBImage,
    TemporaryOMETiffWriter,
    convert,
)
from omeify.cli import main
from omeify.io._writer.preparation import prepare_image
from omeify.io._writer.verification import verify_storage
from omeify.io.image_planes import ImagePlaneSource

_SIZE = PixelSize(0.5, 0.5, "µm")


def _plan(image, writer, *, compression=None):
    source = ImagePlaneSource(image)
    return prepare_image(
        source, source.output_spec(), name=None, downsample=None,
        compression_name=(
            writer._settings.compression_name if compression is None else compression
        ),
        settings=writer._settings, lossy_policy="non-label",
    )


@pytest.mark.parametrize("writer_type", [OMETiffWriter, OMEMultiSeriesWriter])
@pytest.mark.parametrize("kind", ["rgb", "scalar", "three-channel", "label"])
def test_defaults_follow_image_meaning(image_factory, tmp_path: Path, writer_type, kind) -> None:
    shape = {
        "rgb": (769, 1025, 3), "three-channel": (3, 769, 1025),
        "scalar": (769, 1025), "label": (769, 1025),
    }[kind]
    image = image_factory(
        np.zeros(shape, dtype=np.uint8),
        kind=kind if kind in {"rgb", "label"} else "multichannel", pixel_size=_SIZE,
    )
    writer = writer_type(tmp_path / "default.ome.tif")
    prepared = _plan(image, writer)
    rgb = kind == "rgb"
    assert prepared.compression.name == ("JPEG" if rgb else "LZW")
    assert prepared.compression.lossless is not rgb
    assert prepared.tile_size == (256 if rgb else 1024)
    assert prepared.downsample == ("nearest" if kind == "label" else "mean")
    assert len(prepared.level_shapes) == (4 if rgb else 2)
    if rgb:
        assert prepared.level_shapes[-1] == (97, 129, 3)
        assert prepared.compression.compression_args == {"level": 90, "outcolorspace": "YCBCR"}
        assert prepared.compression.subsampling == (2, 1)
    # Preparing one image must not lock a reusable writer to that image's defaults.
    assert writer._settings.compression_name is None
    assert writer._settings.tile_size is None


@pytest.mark.parametrize("writer_type", [OMETiffWriter, OMEMultiSeriesWriter])
@pytest.mark.parametrize("codec", ["LZW", "Deflate", "ZSTD", "Uncompressed", "JPEG"])
def test_explicit_storage_overrides_win(image_factory, tmp_path, writer_type, codec) -> None:
    image = image_factory(np.zeros((17, 33, 3), np.uint8), kind="rgb", pixel_size=_SIZE)
    prepared = _plan(image, writer_type(
        tmp_path / "explicit.ome.tif", compression=codec, tile_size=512,
        jpeg_quality=83, jpeg_subsampling="444", pyramid_levels=0,
    ))
    assert prepared.tile_size == 512
    assert prepared.compression.name == codec
    assert len(prepared.level_shapes) == 1
    if codec == "JPEG":
        assert prepared.compression.compression_args == {"level": 83, "outcolorspace": "RGB"}
        assert prepared.compression.subsampling == (1, 1)
    else:
        assert prepared.compression.lossless


def test_conversion_and_cli_share_public_defaults(tmp_path, monkeypatch) -> None:
    signature = inspect.signature(convert)
    assert signature.parameters["compression"].default is None
    assert signature.parameters["tile_size"].default is None
    assert signature.parameters["jpeg_quality"].default == 90
    assert signature.parameters["jpeg_subsampling"].default == "422"
    source = tmp_path / "source.svs"
    source.touch()  # The CLI only validates existence; conversion is the boundary under test.
    seen = {}

    def capture(*args, **kwargs):
        seen.update(kwargs)
        return {"status": "captured"}

    monkeypatch.setattr("omeify.cli.convert", capture)
    result = CliRunner().invoke(main, ["convert", str(source), "--type", "svs",
                                       "-o", str(tmp_path / "out.ome.tif")])
    assert result.exit_code == 0, result.output
    assert seen["compression"] is None and seen["tile_size"] is None
    assert seen["jpeg_quality"] == 90 and seen["jpeg_subsampling"] == "422"
    help_result = CliRunner().invoke(main, ["convert", "--help"])
    assert "256 for RGB; 1024 otherwise" in help_result.output
    assert "RGB OME-TIFF" in " ".join(help_result.output.split())


def test_temporary_writer_inherits_rgb_defaults(image_factory) -> None:
    image = image_factory(np.zeros((17, 33, 3), np.uint8), kind="rgb", pixel_size=_SIZE)
    with TemporaryOMETiffWriter() as temporary:
        prepared = _plan(image, temporary._writer)
        assert prepared.tile_size == 256
        assert prepared.compression.compression_args == {"level": 90, "outcolorspace": "YCBCR"}
        assert prepared.compression.subsampling == (2, 1)


@pytest.mark.parametrize("bad_size", [0, 17, True, 256.0])
def test_invalid_explicit_tile_sizes_are_not_treated_as_auto(tmp_path, bad_size) -> None:
    with pytest.raises((ValueError, TypeError)):
        OMETiffWriter(tmp_path / "bad.ome.tif", tile_size=bad_size)


@pytest.mark.parametrize("bad_compression", ["", " ", False])
def test_invalid_compression_is_not_treated_as_auto(tmp_path, bad_compression) -> None:
    with pytest.raises(ValueError, match="compression"):
        OMETiffWriter(tmp_path / "bad.ome.tif", compression=bad_compression)


@pytest.mark.parametrize("wrong_level", [0, 1])
def test_verification_enforces_per_image_tile_size(image_factory, tmp_path, wrong_level) -> None:
    image = image_factory(np.zeros((513, 513, 3), np.uint8), kind="rgb", pixel_size=_SIZE)
    prepared = _plan(image, OMETiffWriter(tmp_path / "rgb.ome.tif", pyramid_levels=1))

    def page(reduced):
        value = SimpleNamespace(
            is_tiled=True, tilewidth=256, tilelength=256, compression=7,
            samplesperpixel=3, photometric=6, planarconfig=1, subfiletype=int(reduced),
            tags={"YCbCrSubSampling": SimpleNamespace(value=(2, 1))},
        )
        value.aspage = lambda: value
        return value

    base, reduced = page(False), page(True)
    base.pages = [reduced]
    verify_storage([base], prepared)
    [base, reduced][wrong_level].tilewidth = 1024
    with pytest.raises(ValueError, match="requested 256-pixel tiles"):
        verify_storage([base], prepared)


@pytest.fixture
def real_codecs_and_schema():
    # Integration means real codecs and real schema validation, never substitutes.
    pytest.importorskip("imagecodecs")
    pytest.importorskip("omeschema")


def _write_source(path: Path, pixels: np.ndarray, profile: str) -> None:
    if profile == "svs":
        tifffile.imwrite(
            path, pixels, photometric="rgb", metadata=None, tile=(256, 256),
            description="Aperio Image Library v10.0.51|MPP = 0.5|AppMag = 20",
        )
    else:
        tifffile.imwrite(path, pixels, photometric="rgb", ome=True, tile=(256, 256), metadata={
            "axes": "YXS", "PhysicalSizeX": 0.5, "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5, "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["RGB"]},
        })


@pytest.mark.usefixtures("real_codecs_and_schema")
@pytest.mark.parametrize("route", ["single", "temporary", "multi", "convert-svs", "cli-svs",
                                   "convert-ome", "cli-ome"])
def test_all_public_routes_write_rgb_with_no_storage_options(tmp_path, route) -> None:
    pixels = np.full((513, 769, 3), (242, 199, 220), dtype=np.uint8)
    output = tmp_path / "default.ome.tif"
    temporary = TemporaryOMETiffWriter() if route == "temporary" else None
    with nullcontext() if temporary is None else temporary:
        with RGBImage.from_array(pixels, pixel_size=_SIZE) as image:
            if route == "single":
                report = OMETiffWriter(output).write(image)
            elif route == "temporary":
                report = temporary.write(image)
                output = temporary.path
            elif route == "multi":
                report = OMEMultiSeriesWriter(output).write([OMEImageSeries("H&E", image)])
            else:
                profile = "svs" if route.endswith("svs") else "ome_tiff"
                source = tmp_path / ("source.svs" if profile == "svs" else "source.ome.tif")
                _write_source(source, pixels, profile)
                if route.startswith("convert"):
                    report = convert(source, output, input_type=profile)
                else:
                    result = CliRunner().invoke(main, ["convert", str(source), "--type", profile,
                                                       "-o", str(output)])
                    assert result.exit_code == 0, result.output
                    report = json.loads(result.output)
        assert report["options"]["jpeg_quality"] == 90
        assert report["options"]["jpeg_subsampling"] == "422"
        plan = report["series"][0] if route == "multi" else report["pyramid"]
        assert plan["tile_size"] == 256
        assert len(plan["level_shapes"]) == 3
        with tifffile.TiffFile(output) as tiff:
            for index, level in enumerate(tiff.series[0].levels):
                page = level.pages[0].aspage()
                assert (page.tilewidth, page.tilelength) == (256, 256)
                assert int(page.compression) == 7 and int(page.photometric) == 6
                assert tuple(page.tags["YCbCrSubSampling"].value) == (2, 1)
                factor = 2**index
                assert level.shape == (
                    (513 + factor - 1) // factor, (769 + factor - 1) // factor, 3,
                )
                # Check the interior, including an interior tile seam, away from padded edges.
                values = level.asarray()
                np.testing.assert_allclose(
                    values[10:30, 10:30], np.broadcast_to(pixels[0, 0], (20, 20, 3)),
                    atol=4, rtol=0,
                )
                if index == 0:
                    np.testing.assert_allclose(
                        values[250:262, 250:262], np.broadcast_to(pixels[0, 0], (12, 12, 3)),
                        atol=4, rtol=0,
                    )


@pytest.mark.usefixtures("real_codecs_and_schema")
def test_mixed_series_defaults_and_compression_precedence(image_factory, tmp_path) -> None:
    rgb = image_factory(np.full((257, 513, 3), 128, np.uint8), kind="rgb", pixel_size=_SIZE)
    labels_data = np.arange(257 * 513, dtype=np.uint32).reshape(257, 513)
    labels = image_factory(labels_data, kind="label", pixel_size=_SIZE)
    planar = image_factory(np.full((3, 257, 513), 120, np.uint8), pixel_size=_SIZE)
    entries = [OMEImageSeries("RGB", rgb), OMEImageSeries("Labels", labels),
               OMEImageSeries("Three scalar channels", planar)]
    output = tmp_path / "mixed.ome.tif"
    report = OMEMultiSeriesWriter(output).write(entries)
    assert [item["compression"] for item in report["series"]] == ["JPEG", "LZW", "LZW"]
    assert [item["tile_size"] for item in report["series"]] == [256, 1024, 1024]
    assert [item["subresolution_count"] for item in report["series"]] == [2, 0, 0]
    with tifffile.TiffFile(output) as tiff:
        for index, expected_tile in enumerate([256, 1024, 1024]):
            for level in tiff.series[index].levels:
                for frame in level.pages:
                    page = frame.aspage()
                    assert page.tilewidth == page.tilelength == expected_tile
        np.testing.assert_array_equal(tiff.series[1].asarray(), labels_data)
        np.testing.assert_array_equal(tiff.series[2].asarray(), planar.asarray())
    # Per-series compression > writer compression > automatic per-image policy.
    entries[0] = OMEImageSeries("RGB", rgb, compression="Uncompressed")
    report = OMEMultiSeriesWriter(output, compression="Deflate", tile_size=512).write(entries)
    assert [item["compression"] for item in report["series"]] == [
        "Uncompressed", "Deflate", "Deflate",
    ]
    assert [item["tile_size"] for item in report["series"]] == [512, 512, 512]
    with tifffile.TiffFile(output) as tiff:
        np.testing.assert_array_equal(tiff.series[0].asarray(), rgb.asarray())


def test_multi_series_prepares_independent_defaults_and_overrides(
    image_factory, tmp_path, monkeypatch,
) -> None:
    """Check the public multi-series preparation boundary without encoding files."""
    from importlib import import_module

    module = import_module("omeify.io.ome_multi_series_writer.ome_multi_series_writer")
    image = image_factory(np.zeros((17, 33, 3), np.uint8), kind="rgb", pixel_size=_SIZE)
    labels = image_factory(np.zeros((17, 33), np.uint16), kind="label", pixel_size=_SIZE)
    entries = [OMEImageSeries("RGB", image), OMEImageSeries("Labels", labels)]
    plans = []

    class PlansCaptured(Exception):
        pass

    def capture(*args, **kwargs):
        prepared = prepare_image(*args, **kwargs)
        plans.append(prepared)
        return prepared

    def stop_before_xml(*args, **kwargs):
        raise PlansCaptured

    monkeypatch.setattr(module, "prepare_image", capture)
    monkeypatch.setattr(module, "generate_multi_series_ome_xml", stop_before_xml)
    writer = OMEMultiSeriesWriter(tmp_path / "not-written.ome.tif")
    with pytest.raises(PlansCaptured):
        writer.write(entries)
    assert [(p.compression.name, p.tile_size) for p in plans] == [("JPEG", 256), ("LZW", 1024)]
    plans.clear()
    writer = OMEMultiSeriesWriter(writer.path, compression="Deflate", tile_size=512)
    entries[0] = OMEImageSeries("RGB", image, compression="JPEG")
    with pytest.raises(PlansCaptured):
        writer.write(entries)
    assert [(p.compression.name, p.tile_size) for p in plans] == [("JPEG", 512), ("Deflate", 512)]
    assert not writer.path.exists()


def test_mixed_tile_sizes_reach_real_staging_and_final_ifds(image_factory, tmp_path):
    """Exercise actual TIFF/pyramid machinery without JPEG or the public XML validator.

    This is an uncompressed engine test, not an end-to-end JPEG/schema test.
    """
    from omeify.io._writer.engine import WriterEngine
    from omeify.io.ome_multi_series_writer.verification import verify_output
    from omeify.utils.generate_ome_xml import generate_multi_series_ome_xml

    rgb_data = np.arange(769 * 1025 * 3, dtype=np.uint32).reshape(769, 1025, 3).astype(np.uint8)
    label_data = np.arange(769 * 1025, dtype=np.uint32).reshape(769, 1025)
    images = [image_factory(rgb_data, kind="rgb", pixel_size=_SIZE),
              image_factory(label_data, kind="label", pixel_size=_SIZE)]
    output = tmp_path / "engine.ome.tif"
    settings = OMEMultiSeriesWriter(output, compression="Uncompressed")._settings
    prepared = []
    for index, image in enumerate(images):
        source = ImagePlaneSource(image)
        prepared.append(prepare_image(
            source, source.output_spec(), name=f"Image {index}", downsample=None,
            compression_name=settings.compression_name, settings=settings, lossy_policy="non-label",
        ))
    xml = generate_multi_series_ome_xml(tuple((p.name, p.spec, p.level_shapes) for p in prepared))
    result = WriterEngine(settings).write(
        prepared, str(xml["xml_string"]),
        verify=lambda path: verify_output(
            prepared, path, provenance_json=None, software=settings.software,
        ),
    )
    assert result.verification["storage_matches_requested"]
    with tifffile.TiffFile(output) as tiff:
        for index, (tile, depth) in enumerate([(256, 4), (1024, 2)]):
            assert len(tiff.series[index].levels) == depth
            for level in tiff.series[index].levels:
                page = level.pages[0].aspage()
                assert page.tilewidth == page.tilelength == tile
        np.testing.assert_array_equal(tiff.series[0].asarray(), rgb_data)
        np.testing.assert_array_equal(tiff.series[1].asarray(), label_data)
        np.testing.assert_array_equal(tiff.series[1].levels[1].asarray(), label_data[::2, ::2])
