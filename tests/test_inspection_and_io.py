from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import (
    LabelImage,
    MultichannelImage,
    OMETiffLabelReader,
    OMETiffReader,
    OMETiffWriter,
    PixelSize,
    RGBImage,
    TiffInspector,
)
from omeify.cli import main
from omeify.io._writer.configuration import compression_settings
from omeify.io._writer.precision import round_float32_mantissa
from omeify.io.tiff import TiffPlaneReader
from omeify.utils.ome_schema_validator import OMESchemaValidator


def _write_generic_tiff(path: Path) -> np.ndarray:
    data = np.arange(24 * 32, dtype=np.uint16).reshape(24, 32)
    tifffile.imwrite(
        path,
        data,
        tile=(16, 16),
        description=(
            '<?xml version="1.0" encoding="utf-16"?>'
            "<Vendor><Name>Example</Name><Secret>hidden</Secret></Vendor>"
        ),
        metadata=None,
    )
    return data


def _write_resolution_tiff(path: Path) -> None:
    tifffile.imwrite(
        path,
        np.zeros((24, 32), dtype=np.uint16),
        tile=(16, 16),
        resolution=(20056.913009774024, 20056.913009774024),
        resolutionunit="CENTIMETER",
        metadata=None,
    )


def _write_ome_tiff(path: Path) -> np.ndarray:
    data = np.arange(3 * 24 * 32, dtype=np.uint16).reshape(3, 24, 32)
    tifffile.imwrite(
        path,
        data,
        ome=True,
        tile=(16, 16),
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.6,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["DAPI", "PanCK", "CD3"]},
        },
    )
    return data

def _write_nonconforming_ome_tiff(path: Path) -> None:
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Instrument ID="Instrument:0"/>'
        '<Image ID="Image:0" Name="potentially-identifying-name">'
        '<Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" '
        'SizeX="8" SizeY="8" SizeZ="1" SizeC="1" SizeT="1" '
        'PhysicalSizeX="0.5" PhysicalSizeXUnit="um" '
        'PhysicalSizeY="0.5" PhysicalSizeYUnit="um">'
        '<Channel ID="Channel:0:0" Name="DAPI" SamplesPerPixel="1"/>'
        '<TiffData IFD="0" PlaneCount="1"/>'
        '</Pixels></Image></OME>'
    )
    tifffile.imwrite(
        path,
        np.zeros((8, 8), dtype=np.uint16),
        description=xml,
        metadata=None,
    )


def test_inspector_handles_non_ome_tiff_and_validates_schema(tmp_path: Path) -> None:
    source = tmp_path / "generic.tif"
    _write_generic_tiff(source)

    inspector = TiffInspector(source, detail=1)
    report = inspector.report

    assert report["file"]["is_ome"] is False
    assert report["file"]["format"] == "TIFF"
    assert report["series"][0]["axes"] == "YX"
    assert report["series"][0]["levels"][0]["shape"] == [24, 32]
    assert report["ome"] is None
    assert report["series"][0]["tiff_resolution_pixel_size"] is None
    assert report["warnings"] == []
    assert inspector.validation_errors() == ()
    assert "Series 0" in inspector.render_text()
    assert "TIFF resolution pixel size: N/A" in inspector.render_text()
    assert "Level 0" in inspector.render_text()


def test_detail_three_includes_tags_and_parsed_xml_description(tmp_path: Path) -> None:
    source = tmp_path / "generic.tif"
    _write_generic_tiff(source)

    report = TiffInspector(source, detail=3, max_text_length=80).report
    page = report["series"][0]["levels"][0]["pages"][0]

    assert page["description"]["format"] == "xml"
    assert page["description"]["xml_root"] == "Vendor"
    assert page["description"]["parsed_xml"]["children"][0]["tag"] == "Name"
    assert any(tag["name"] == "ImageDescription" for tag in page["tags"])
    assert TiffInspector(source, detail=3).validation_errors() == ()


def test_inspector_reports_pixel_size_calculated_from_tiff_resolution_tags(
    tmp_path: Path,
) -> None:
    source = tmp_path / "resolution.tif"
    _write_resolution_tiff(source)

    inspector = TiffInspector(source)
    report = inspector.report
    tiff_pixel_size = report["series"][0]["tiff_resolution_pixel_size"]

    assert tiff_pixel_size == {
        "x": {"value": 0.498581, "unit": "µm"},
        "y": {"value": 0.498581, "unit": "µm"},
        "z": None,
    }
    assert (
        "TIFF resolution pixel size: X=0.498581 µm, Y=0.498581 µm"
        in inspector.render_text()
    )
    assert inspector.validation_errors() == ()


def test_ome_inspection_uses_header_for_channels_and_physical_size(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.tif"
    _write_ome_tiff(source)

    inspector = TiffInspector(source)
    report = inspector.report
    ome = report["ome"]

    assert ome is not None
    assert ome["version"] == "2016-06"
    assert ome["images"][0]["size"] == {"x": 32, "y": 24, "z": 1, "c": 3, "t": 1}
    assert [item["name"] for item in ome["images"][0]["channels"]] == [
        "DAPI",
        "PanCK",
        "CD3",
    ]
    assert report["series"][0]["channel_names"] == ["DAPI", "PanCK", "CD3"]
    assert report["series"][0]["physical_size"]["x"]["value"] == pytest.approx(0.5)
    assert ome["miti"]["status"] in {"pass", "fail"}
    assert "MITI header profile:" in inspector.render_text()
    assert "OME channels (3): DAPI, PanCK, CD3" in inspector.render_text()
    assert inspector.validation_errors() == ()


def test_ome_inspection_reports_miti_missing_fields_and_additional_metadata(
    tmp_path: Path,
) -> None:
    source = tmp_path / "nonconforming.ome.tif"
    _write_nonconforming_ome_tiff(source)

    inspector = TiffInspector(source)
    report = inspector.report
    assessment = report["ome"]["miti"]

    assert report["file"]["is_ome"] is True
    assert assessment["status"] == "fail"
    assert "Image[0]/Pixels/@BigEndian" in assessment["missing_fields"]
    assert "Image[0]/Pixels/@Interleaved" in assessment["missing_fields"]
    assert "Image[0]/Pixels/@SignificantBits" in assessment["missing_fields"]
    assert "OME/Image[0]/@Name" in assessment["extra_metadata"]
    assert "OME/Instrument[0]" in assessment["extra_metadata"]
    assert "MITI header profile: FAIL" in inspector.render_text()
    assert inspector.validation_errors() == ()


def test_ome_tiff_reader_holds_file_open_reads_regions_and_matches_inspect(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.ome.tif"
    data = _write_ome_tiff(source)

    reader = OMETiffReader(source)
    assert issubclass(OMETiffReader, MultichannelImage)
    assert not reader.is_open

    with reader as ome:
        assert ome.is_open
        assert ome.axes == "CYX"
        assert ome.shape == data.shape
        assert ome.dtype == np.dtype("uint16")
        assert ome.channel_names == ("DAPI", "PanCK", "CD3")
        assert ome.size_c == 3
        np.testing.assert_array_equal(ome.read_region(2, 7, 3, 9), data[:, 2:7, 3:9])
        np.testing.assert_array_equal(
            ome.read_region(2, 7, 3, 9, channels=[2, 0]),
            data[[2, 0], 2:7, 3:9],
        )
        expected = TiffInspector(source).render_text()
        assert str(ome) == expected
        cli_result = CliRunner().invoke(main, ["inspect", str(source)])
        assert cli_result.exit_code == 0, cli_result.output
        assert cli_result.output.rstrip("\n") == expected

    assert not reader.is_open
    with pytest.raises(RuntimeError, match="must be opened"):
        _ = reader.axes


def test_ome_tiff_reader_reuses_decoded_segments_across_region_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "cached-regions.ome.tif"
    data = _write_ome_tiff(source)
    cache_misses = 0
    original_decode_segment = TiffPlaneReader._decode_segment

    def count_cache_misses(
        plane_reader: TiffPlaneReader,
        index: int,
    ) -> tuple[np.ndarray, int, int]:
        nonlocal cache_misses
        if index not in plane_reader._cache:
            cache_misses += 1
        return original_decode_segment(plane_reader, index)

    monkeypatch.setattr(TiffPlaneReader, "_decode_segment", count_cache_misses)

    reader = OMETiffReader(source)
    with reader as ome:
        np.testing.assert_array_equal(
            ome.read_region(2, 7, 3, 9, channels=[0]),
            data[[0], 2:7, 3:9],
        )
        np.testing.assert_array_equal(
            ome[0].read_region(7, 12, 8, 14),
            data[0, 7:12, 8:14],
        )
        assert cache_misses == 1

    # Closing the OME reader releases decoded segments. Reopening starts with
    # a new plane reader rather than retaining data backed by the old TIFF.
    with reader as ome:
        np.testing.assert_array_equal(
            ome.read_region(2, 7, 3, 9, channels=[0]),
            data[[0], 2:7, 3:9],
        )
        assert cache_misses == 2


def test_reader_rejects_non_ome_tiff(tmp_path: Path) -> None:
    source = tmp_path / "generic.tif"
    _write_generic_tiff(source)

    with pytest.raises(ValueError, match="does not contain recognized OME metadata"):
        with OMETiffReader(source):
            pass


def test_generic_rgb_and_label_interfaces_are_available() -> None:
    assert RGBImage.samples_per_pixel == 3
    assert RGBImage.channel_names == ("RGB",)
    assert RGBImage.sample_names == ("Red", "Green", "Blue")
    assert LabelImage.background_label == 0


def test_cli_inspect_text_json_and_output_file(tmp_path: Path) -> None:
    source = tmp_path / "source.ome.tif"
    _write_ome_tiff(source)
    runner = CliRunner()

    text_result = runner.invoke(main, ["inspect", str(source)])
    assert text_result.exit_code == 0, text_result.output
    assert "OME channels (3): DAPI, PanCK, CD3" in text_result.output

    json_result = runner.invoke(main, ["inspect", str(source), "--json", "--detail", "2"])
    assert json_result.exit_code == 0, json_result.output
    report = json.loads(json_result.output)
    assert report["detail"] == 2
    assert report["file"]["is_ome"] is True
    assert report["series"][0]["levels"][0]["pages"]

    output = tmp_path / "inspection.json"
    output_result = runner.invoke(
        main,
        ["inspect", str(source), "--json", "--output", str(output)],
    )
    assert output_result.exit_code == 0, output_result.output
    assert output_result.output == ""
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.2"


def test_ome_tiff_reader_reads_interleaved_rgb_regions(tmp_path: Path) -> None:
    source = tmp_path / "rgb.ome.tif"
    data = np.arange(24 * 32 * 3, dtype=np.uint8).reshape(24, 32, 3)
    tifffile.imwrite(
        source,
        data,
        ome=True,
        tile=(16, 16),
        photometric="rgb",
        metadata={"axes": "YXS", "Channel": {"Name": ["RGB"]}},
    )

    with OMETiffReader(source) as ome:
        assert ome.axes == "YXS"
        assert ome.size_c == 3
        assert ome.channel_names == ("RGB",)
        assert ome.logical_channel_count == 1
        assert ome.sample_count == 3
        assert ome.sample_names == ("Red", "Green", "Blue")
        np.testing.assert_array_equal(ome.read_region(1, 6, 2, 9), data[1:6, 2:9, :])
        np.testing.assert_array_equal(
            ome.read_region(1, 6, 2, 9, channels=[2, 0]),
            data[1:6, 2:9, :][..., [2, 0]],
        )


def test_public_ome_tiff_writer_emits_miti_profiled_multichannel_output(
    tmp_path: Path,
) -> None:
    output = tmp_path / "writer.ome.tif"
    data = np.arange(3 * 25 * 33, dtype=np.uint16).reshape(3, 25, 33)

    report = OMETiffWriter(
        output,
        image_type="multichannel",
        channel_names=["DAPI", "PanCK", "CD3"],
        pixel_size=PixelSize(0.5, 0.6, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
    ).write(data)

    assert report["image"]["image_type"] == "multichannel"
    assert report["miti_header"]["is_valid"] is True
    assert report["verification"]["base_pixel_values_match"] is True
    assessment = TiffInspector(output).report["ome"]["miti"]
    assert assessment["status"] == "pass"
    assert assessment["missing_fields"] == []
    assert assessment["extra_metadata"] == []

    with OMETiffReader(output) as image:
        assert image.logical_channel_count == 3
        assert image.sample_count == 3
        np.testing.assert_array_equal(
            image.read_region(2, 8, 3, 10),
            data[:, 2:8, 3:10],
        )


def test_writer_software_tag_defaults_to_omeify_and_accepts_override(
    tmp_path: Path,
) -> None:
    data = np.zeros((1, 16, 16), dtype=np.uint16)
    default_output = tmp_path / "default-software.ome.tif"
    custom_output = tmp_path / "custom-software.ome.tif"

    default_report = OMETiffWriter(
        default_output,
        image_type="multichannel",
        channel_names=["DAPI"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
    ).write(data)
    custom_report = OMETiffWriter(
        custom_output,
        image_type="multichannel",
        channel_names=["DAPI"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        software="downstream-app 2.4",
    ).write(data)

    with tifffile.TiffFile(default_output) as tiff:
        assert str(tiff.pages[0].tags["Software"].value).startswith("omeify ")
    with tifffile.TiffFile(custom_output) as tiff:
        assert tiff.pages[0].tags["Software"].value == "downstream-app 2.4"
    assert default_report["verification"]["software_tag_matches"] is True
    assert custom_report["verification"]["software_tag_matches"] is True
    assert custom_report["options"]["software"] == "downstream-app 2.4"

    with pytest.raises(ValueError, match="non-empty string"):
        OMETiffWriter(
            tmp_path / "bad-software.ome.tif",
            channel_names=["DAPI"],
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            software="   ",
        )


def test_float32_writer_can_trim_mantissa_and_declare_reduced_significant_bits(
    tmp_path: Path,
) -> None:
    output = tmp_path / "trimmed-float32.ome.tif"
    data = np.array(
        [
            [0.0, 1.0003, 1.0006, 38.126743],
            [-2.1254, 150.1234, 1024.75, 65504.125],
        ],
        dtype=np.float32,
    )
    expected = round_float32_mantissa(data, 10)

    report = OMETiffWriter(
        output,
        image_type="multichannel",
        channel_names=["DAPI"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        float32_mantissa_bits=10,
    ).write(data)

    assert report["image"]["significant_bits"] == 19
    assert report["image"]["float32_mantissa_bits"] == 10
    assert report["image"]["float_precision_bits"] == 11
    assert report["verification"]["significant_bits_matches_spec"] is True
    assert report["verification"]["base_pixel_values_match"] is True
    with tifffile.TiffFile(output) as tiff:
        pixels = TiffInspector.from_tiff(
            tiff,
            file_path=output,
            detail=1,
        ).report["ome"]["images"][0]
        assert pixels["pixel_type"] == "float"
        assert pixels["significant_bits"] == 19
        np.testing.assert_array_equal(tiff.series[0].asarray(), expected)


def test_float32_mantissa_rounding_uses_nearest_ties_to_even() -> None:
    step = np.float32(2.0**-10)
    half = np.float32(2.0**-11)
    values = np.array(
        [
            np.float32(1.0) + half,
            np.float32(1.0) + step + half,
            -(np.float32(1.0) + half),
            -(np.float32(1.0) + step + half),
        ],
        dtype=np.float32,
    )

    rounded = round_float32_mantissa(values, 10)

    np.testing.assert_array_equal(
        rounded,
        np.array(
            [
                1.0,
                np.float32(1.0) + np.float32(2.0) * step,
                -1.0,
                -(np.float32(1.0) + np.float32(2.0) * step),
            ],
            dtype=np.float32,
        ),
    )


def test_float32_precision_trimming_is_applied_to_pyramid_values(tmp_path: Path) -> None:
    output = tmp_path / "trimmed-pyramid.ome.tif"
    data = np.linspace(0.0, 200.0, 32 * 32, dtype=np.float32).reshape(32, 32)

    OMETiffWriter(
        output,
        image_type="multichannel",
        channel_names=["DAPI"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
        float32_mantissa_bits=10,
    ).write(data)

    with tifffile.TiffFile(output) as tiff:
        for level in tiff.series[0].levels:
            values = np.asarray(level.asarray(), dtype=np.float32)
            raw = values.view(np.uint32)
            finite = np.isfinite(values)
            assert np.all((raw[finite] & np.uint32((1 << 13) - 1)) == 0)


def test_compression_settings_expose_float_predictor_three() -> None:
    floating = compression_settings(
        "LZW",
        np.dtype("float32"),
        is_rgb=False,
        jpeg_quality=90,
        jpeg_subsampling="444",
        predictor="floatingpoint",
    )
    integer_auto = compression_settings(
        "LZW",
        np.dtype("uint16"),
        is_rgb=False,
        jpeg_quality=90,
        jpeg_subsampling="444",
        predictor="auto",
    )

    assert floating.predictor == 3
    assert floating.predictor_name == "floatingpoint"
    assert integer_auto.predictor == 2
    assert integer_auto.predictor_name == "horizontal"

    with pytest.raises(ValueError, match="floatingpoint TIFF prediction requires"):
        compression_settings(
            "LZW",
            np.dtype("uint16"),
            is_rgb=False,
            jpeg_quality=90,
            jpeg_subsampling="444",
            predictor="floatingpoint",
        )
    with pytest.raises(ValueError, match="requires LZW, Deflate, or ZSTD"):
        compression_settings(
            "Uncompressed",
            np.dtype("float32"),
            is_rgb=False,
            jpeg_quality=90,
            jpeg_subsampling="444",
            predictor="floatingpoint",
        )


def test_generic_rgb_writer_defaults_to_lossless_lzw(tmp_path: Path) -> None:
    writer = OMETiffWriter(
        tmp_path / "rgb-default.ome.tif",
        image_type="rgb",
        pixel_size=PixelSize(0.5, 0.5, "µm"),
    )

    assert writer.compression_name == "LZW"


def test_planar_writer_rejects_lossy_compression(tmp_path: Path) -> None:
    output = tmp_path / "lossy-planar.ome.tif"
    data = np.zeros((1, 16, 16), dtype=np.uint8)

    with pytest.raises(ValueError, match="Lossy compression is restricted to RGB"):
        OMETiffWriter(
            output,
            image_type="multichannel",
            channel_names=["DAPI"],
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            compression="JPEG",
            tile_size=16,
            pyramid_levels=0,
        ).write(data)

    assert not output.exists()


def test_writer_fails_closed_when_ome_schema_validation_is_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "schema-unavailable.ome.tif"
    data = np.zeros((1, 16, 16), dtype=np.uint16)
    monkeypatch.setattr(OMESchemaValidator, "validate", lambda self, xml: None)

    with pytest.raises(RuntimeError, match="schema validation could not be performed"):
        OMETiffWriter(
            output,
            image_type="multichannel",
            channel_names=["DAPI"],
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
        ).write(data)

    assert not output.exists()


def test_label_writer_and_reader_stay_at_the_virtual_io_boundary(tmp_path: Path) -> None:
    output = tmp_path / "labels.ome.tif"
    labels = np.array(
        [
            [0, 1, 1, 0, 7],
            [0, 1, 1, 0, 7],
            [9, 0, 0, 9, 7],
            [9, 9, 0, 0, 0],
        ],
        dtype=np.uint16,
    )

    report = OMETiffWriter(
        output,
        image_type="label",
        channel_names=["Cells"],
        pixel_size=PixelSize(0.5, 0.5, "µm"),
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=1,
    ).write(labels)

    assert report["image"]["image_type"] == "label"
    assert report["pyramid"]["downsample_method"] == "nearest"
    assert report["miti_header"]["is_valid"] is True

    with OMETiffLabelReader(output) as image:
        assert isinstance(image, LabelImage)
        assert image.background_label == 0
        assert not hasattr(image, "label_count")
        np.testing.assert_array_equal(image.asarray(), labels)
        np.testing.assert_array_equal(image.read_region(1, 4, 1, 5), labels[1:4, 1:5])

    with pytest.raises(ValueError, match="lossless compression"):
        OMETiffWriter(
            tmp_path / "lossy-label.ome.tif",
            image_type="label",
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            compression="JPEG",
            tile_size=16,
            pyramid_levels=0,
        ).write(labels.astype(np.uint8))


def test_runtime_dependencies_exclude_scikit_image() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as handle:
        dependencies = tomllib.load(handle)["project"]["dependencies"]
    normalized = " ".join(str(item).lower() for item in dependencies)
    assert "scikit-image" not in normalized
    assert "skimage" not in normalized
