"""Three-plane RGB recovery: deterministic pixels, regional ownership and reporting."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import MultichannelImage, OMETiffReader, PixelSize, RGBImage, extract_rgb, probe_rgb
from omeify.cli import main
from omeify.rgb_extraction import _open_rgb_source, decode_rgb
from omeify.reports import complete_report, validate_report


SIZE = PixelSize(.32619, .32619, "µm")


def _rgb_sources(width=36, height=29):
    y, x = np.indices((height, width))
    rgb = np.stack((
        ((7 * x + 3 * y) % 255).astype(np.uint8),
        ((11 * x + y) % 255).astype(np.uint8),
        ((5 * x + 13 * y) % 255).astype(np.uint8),
    ))
    rgb[:, 0:3, 0:3] = 0
    m = np.zeros((6, height, width), dtype=np.uint16)
    m[1] = rgb[2].astype(np.uint16) * 257  # deliberately reorder the RGB channels
    m[3] = rgb[0].astype(np.uint16) * 257
    m[5] = rgb[1].astype(np.uint16) * 257
    m[0] = 19
    return m, rgb


@pytest.fixture
def source_file(tmp_path):
    m, rgb = _rgb_sources()
    path = tmp_path / "source.ome.tiff"
    tifffile.imwrite(path, m, ome=True, photometric="minisblack", metadata={
        "axes": "CYX", "Channel": {"Name": [
            "DAPI", "BLUE", "CD8", "RED", "CK", "GREEN",
        ]},
        "PhysicalSizeX": SIZE.x, "PhysicalSizeY": SIZE.y,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    return path, m, rgb


@pytest.mark.parametrize("factor", [256, 257])
def test_high_byte_recovers_expanded_eight_bit_samples(factor):
    values = np.array([
        [[0, 128, 255], [20, 45, 230]],
        [[0, 40, 201], [70, 89, 180]],
        [[0, 160, 175], [240, 19, 23]],
    ], dtype=np.uint16) * factor
    expected = np.moveaxis(values // factor, 0, -1).astype(np.uint8)
    np.testing.assert_array_equal(decode_rgb(values, encoding="high-byte", background="keep"), expected)
    expected[0, 0] = 255
    np.testing.assert_array_equal(decode_rgb(values, encoding="high-byte"), expected)


def test_white_background_uses_original_zero_triplets_only():
    values = np.array([[[0, 1, 256]], [[0, 0, 0]], [[0, 1, 0]]], dtype=np.uint16)
    actual = decode_rgb(values, encoding="high-byte")
    np.testing.assert_array_equal(actual[0], [[255, 255, 255], [0, 0, 0], [1, 0, 0]])
    np.testing.assert_array_equal(
        decode_rgb(values, encoding="high-byte", background="keep")[0, 0], [0, 0, 0]
    )


def test_rounding_and_low_byte_are_deliberate_not_modulo():
    continuous = np.array([[[0, 65535]], [[128, 32768]], [[257, 65000]]], dtype=np.uint16)
    expected = ((continuous.astype(np.uint32) * 255 + 32767) // 65535).astype(np.uint8)
    np.testing.assert_array_equal(decode_rgb(continuous, encoding="rounded", background="keep"),
                                  np.moveaxis(expected, 0, -1))
    with pytest.raises(ValueError, match="<= 255"):
        decode_rgb(continuous, encoding="low-byte")
    safe = np.array([[[255]], [[18]], [[10]]], dtype=np.uint16)
    np.testing.assert_array_equal(decode_rgb(safe, encoding="low-byte"), [[[255, 18, 10]]])
    plain = safe.astype(np.uint8)
    np.testing.assert_array_equal(decode_rgb(plain, encoding="native", background="keep"),
                                  [[[255, 18, 10]]])
    with pytest.raises(ValueError):
        decode_rgb(safe, encoding="native")


def test_lazy_recovery_keeps_parent_open_and_preserves_calibration():
    m, rgb = _rgb_sources()
    with MultichannelImage.from_array(
        m, axes="CYX", channel_names=("DAPI", "BLUE", "CD8", "RED", "CK", "GREEN"),
        pixel_size=SIZE,
    ) as parent:
        out, indices, encoding = _open_rgb_source(
            parent, red="RED", green="GREEN", blue="BLUE", encoding="auto",
            background="white", name="H&E", pixel_size=None,
        )
        assert indices == (3, 5, 1) and encoding == "high-byte"
        with out:
            assert isinstance(out, RGBImage)
            assert out.image_type == "rgb"
            assert out.axes == "YXS" and out.dtype == np.dtype("uint8")
            assert out.shape == (29, 36, 3) and out.channel_names == ("H&E",)
            assert out.pixel_size == SIZE and len(out.levels) == 1
            recovered = out.read_region(1, 15, 2, 22)
            expected = np.moveaxis(rgb[:, 1:15, 2:22], 0, -1).copy()
            expected[:2, :1] = 255
            np.testing.assert_array_equal(recovered, expected)
        assert parent.is_open
        stale = out  # old RGB source must not reconnect after parent changes
        parent.close()
        parent.open()
        with pytest.raises(RuntimeError):
            stale.open()


@pytest.mark.parametrize("selectors", [
    ("NOT_FOUND", "GREEN", "BLUE"),
    ("RED", "RED", "BLUE"),
    (-1, "GREEN", "BLUE"),
    (999, "GREEN", "BLUE"),
    (True, "GREEN", "BLUE"),
])
def test_bad_channel_selection_fails_without_writing(selectors):
    m, _ = _rgb_sources()
    with MultichannelImage.from_array(
        m, axes="CYX", pixel_size=SIZE,
        channel_names=("DAPI", "BLUE", "CD8", "RED", "CK", "GREEN"),
    ) as parent:
        with pytest.raises((ValueError, TypeError, KeyError, IndexError)):
            _open_rgb_source(parent, red=selectors[0], green=selectors[1], blue=selectors[2],
                             encoding="auto", background="white", name="RGB", pixel_size=None)


def test_probe_reports_byte_repeat_not_conclusive(source_file):
    path, _, _ = source_file
    report = probe_rgb(path)
    assert report["source_dtype"] == "uint16" and report["sample_windows"] == 5
    assert report["samples_examined"] > 0
    assert [c["index"] for c in report["channels"]] == [3, 5, 1]
    assert all(c["byte_repeated_fraction"] == 1. for c in report["channels"])


def _single_writer_fixture(path: Path, rgb: RGBImage) -> dict:
    """Synthetic schema-valid successful writer result for orchestration tests."""
    from omeify.io.rgb_color import SRGB_SHA256
    p = PixelSize(.32619, .32619, "µm")
    return complete_report({
        "ome": {"xml_string": "<synthetic/>", "schema_location": "fixture", "xml_is_valid": True,
                "uuid": None},
        "miti_header": {"profile": "omeify.miti_ome_tiff_header", "source": "fixture",
                        "schema_resource": "omeify.schemas/miti_ome_tiff_header.schema.json",
                        "status": "pass", "is_valid": True, "errors": [], "warnings": [],
                        "missing_fields": [], "has_extra_metadata": False, "extra_metadata": []},
        "output_file": {"path": str(path), "size_bytes": 1, "type_description": "Pyramidal OME-TIFF",
                        "dtype": "uint8", "shape": list(rgb.shape), "axes": "YXS",
                        "byte_order": "little", "lossless_compression": True},
        "image": {"image_type": "rgb", "channel_names": list(rgb.channel_names), "size_c": 3,
                  "logical_channel_count": 1, "samples_per_pixel": 3, "interleaved": True,
                  "pixel_size": list(p.to_tuple()), "significant_bits": 8,
                  "float32_mantissa_bits": None, "float_precision_bits": None,
                  "icc_profile_present": True,
                  "color_management": {"output_space": "sRGB", "output_profile_sha256": SRGB_SHA256,
                                       "source_profile": "assumed_srgb", "source_profile_present": False,
                                       "pixels_transformed": False, "rendering_intent": None}},
        "pyramid": {"tile_size": 512, "axes": "YXS", "downsample_method": "mean",
                    "level_shapes": [list(rgb.shape)], "subresolution_count": 0},
        "verification": {**dict.fromkeys((
            "ome_tiff_recognized", "bigtiff", "software_tag_matches",
            "tiff_data_mapping_matches", "pyramid_annotation_linked",
            "top_level_ifd_count_matches", "output_pixels_decodable",
            "base_pixel_values_checked", "base_pixel_values_match", "ome_physical_sizes_match_specs",
            "tiff_calibration_matches_specs", "pyramid_calibration_matches_specs",
            "byte_order_metadata_matches_tiff", "significant_bits_matches_spec",
            "channel_sample_layout_matches", "dtype_matches_source", "axes_match",
            "pyramid_level_shapes_match", "samples_per_pixel_match", "photometric_matches",
            "compression_matches_requested", "subifd_layout_matches", "all_levels_tiled",
        ), True), "jpeg_subsampling_matches_requested": None,
            "icc_profile_is_canonical_srgb": True,
            "output_byte_order": "little", "planes_checked": 1, "points_per_plane": 1,
            "calibration_ifds_checked": 1, "calibration_relative_tolerance": 1e-5,
            "pixel_verification": {
                "schema_version": "1.0", "mode": "sampled",
                "sampling": "top_left_center_bottom_right_per_plane_per_level",
                "all_pixels_checked": False, "plane_levels_checked": 1,
                "base_points_decoded": 1, "lossless_base_points_compared": 1,
                "pyramid_points_decoded": 0, "lossless_pyramid_points_compared": 0,
                "base_comparison": "bitwise_after_byte_order_normalization",
                "pyramid_comparison": "nearest_bitwise_or_mean_numeric_equal_nan",
                "pyramid_reference": "preceding_decoded_output_level_for_lossless_series",
                "source_reader_independent": False,
            }},
        "options": {"metadata_minimization": True, "compression": "Deflate",
                    "jpeg_quality": None, "jpeg_subsampling": None, "display_uuid": True,
                    "software": "omeify test", "float32_mantissa_bits": None, "max_workers": 1},
    }, "write_report.schema.json")


def test_library_extract_rgb_uses_single_writer_and_validates_report(source_file, monkeypatch):
    path, _, rgb = source_file
    destination = path.with_name("HE.ome.tiff")
    written = []

    class SpyWriter:
        def __init__(self, path):
            self.path = path

        def write(self, image):
            written.append((image.asarray(), image.channel_names, image.pixel_size))
            return _single_writer_fixture(self.path, image)

    def fake_writer(path, **kwargs):
        assert kwargs["compression"] == "Deflate"
        assert kwargs["pyramid_levels"] is None
        assert kwargs["tile_size"] is None
        assert kwargs["downsample"] == "mean"
        return SpyWriter(path)

    monkeypatch.setattr("omeify.rgb_extraction.image_writer", fake_writer)
    report = extract_rgb(path, destination, name="H&E")
    validate_report(report, "rgb_extraction_report.schema.json")
    assert report["schema"] == "omeify.extract_rgb/1"
    assert report["requested_encoding"] == "auto" and report["encoding"] == "high-byte"
    assert report["background"] == "white" and report["output_name"] == "H&E"
    assert [c["source_index"] for c in report["channels"]] == [3, 5, 1]
    assert written[0][1:] == (("H&E",), SIZE)
    expected = np.moveaxis(rgb, 0, -1).copy()
    expected[:3, :3] = 255
    np.testing.assert_array_equal(written[0][0], expected)
    json.dumps(report, allow_nan=False)


def test_cli_probe_and_options(source_file, monkeypatch):
    input_path, _, _ = source_file
    runner = CliRunner()
    result = runner.invoke(main, ["extract-rgb", str(input_path), "--probe"])
    assert result.exit_code == 0, result.output
    assert "byte-repeated=100.0%" in result.output
    result = runner.invoke(main, ["extract-rgb", str(input_path)])
    assert result.exit_code == 2 and "--output" in result.output
    result = runner.invoke(main, ["extract-rgb", str(input_path), "--probe", "-o", "x.tif"])
    assert result.exit_code == 2 and "--probe" in result.output
    result = runner.invoke(main, ["extract-rgb", str(input_path), "--help"])
    assert result.exit_code == 0 and "--pyramid-levels" in result.output

    out = input_path.parent / "result.ome.tiff"
    report = {"schema": "omeify.extract_rgb/1", "writer_report": {
        "pyramid": {"subresolution_count": 6}, "options": {"compression": "Deflate"}}}
    operation = Mock(return_value=report)
    monkeypatch.setattr("omeify.cli.extract_rgb", operation)
    result = runner.invoke(main, ["extract-rgb", str(input_path), "-o", str(out),
                                  "--red", "3", "--green", "5", "--blue", "name:BLUE",
                                  "--name", "H&E", "--no-overwrite"])
    assert result.exit_code == 0, result.output
    assert "6 reduced levels" in result.output
    assert operation.call_args.kwargs["red"] == 3
    assert operation.call_args.kwargs["green"] == 5
    assert operation.call_args.kwargs["blue"] == "BLUE"
    assert operation.call_args.kwargs["name"] == "H&E"


def test_cli_preflight_rejects_source_overwrite_and_report_collisions(source_file):
    source, _, _ = source_file
    runner = CliRunner()
    for extra in [["-o", str(source)],
                  ["-o", str(source.with_name("out.tif")), "--output-json", str(source)]]:
        result = runner.invoke(main, ["extract-rgb", str(source), *extra])
        assert result.exit_code != 0, result.output
    assert source.is_file()


def test_real_tiff_writer_roundtrip_pyramid_and_values(source_file, tmp_path):
    """Full integration test (requires the normal ome-schema/imagecodecs dependencies)."""
    path, _, rgb = source_file
    out = tmp_path / "recovered.ome.tiff"
    report = extract_rgb(path, out, name="H&E", tile_size=16, pyramid_levels=2,
                         overwrite=False)
    assert report["writer_report"]["pyramid"]["subresolution_count"] == 2
    with OMETiffReader(out) as image:
        assert image.image_type == "rgb" and image.axes == "YXS"
        assert image.dtype == np.dtype("uint8") and image.channel_names == ("H&E",)
        assert image.pixel_size == SIZE
        assert len(image.levels) == 3
        expected = np.moveaxis(rgb, 0, -1).copy()
        expected[:3, :3] = 255
        np.testing.assert_array_equal(image.asarray(), expected)
    with tifffile.TiffFile(out) as tiff:
        assert len(tiff.pages) == 1
        assert len(tiff.pages[0].pages) == 2
        assert tiff.pages[0].compression.name == "ADOBE_DEFLATE"
        np.testing.assert_array_equal(tiff.asarray(), expected)



def test_auto_pyramidal_output_is_default(source_file, tmp_path):
    """Avoid recreating the old extractor's flat-TIFF default."""
    path, _, _ = source_file
    output = tmp_path / "auto.ome.tiff"
    report = extract_rgb(path, output, tile_size=16, overwrite=False)
    # Input 29 x 36 -> 15 x 18 -> 8 x 9, so two reductions fit tile 16.
    assert report["writer_report"]["pyramid"]["subresolution_count"] == 2
    with tifffile.TiffFile(output) as tiff:
        assert len(tiff.pages[0].pages) == 2


def test_uint8_native_roundtrip(source_file, tmp_path):
    _, _, rgb = source_file
    source = tmp_path / "rgb_uint8.ome.tiff"
    mixed = np.stack((rgb[0], rgb[1], rgb[2], rgb[0]))
    tifffile.imwrite(source, mixed, ome=True, metadata={
        "axes": "CYX", "Channel": {"Name": ["RED", "GREEN", "BLUE", "EXTRA"]},
        "PhysicalSizeX": .5, "PhysicalSizeY": .5,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    output = tmp_path / "eight.ome.tiff"
    result = extract_rgb(source, output, background="keep", compression="LZW", tile_size=16,
                         pyramid_levels=1)
    assert result["encoding"] == "native"
    np.testing.assert_array_equal(tifffile.imread(output), np.moveaxis(rgb, 0, -1))
