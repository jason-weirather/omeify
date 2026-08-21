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
    OMETiffReader,
    RGBImage,
    TiffInspector,
)
from omeify.cli import main


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
    assert inspector.validation_errors() == ()
    assert "Series 0" in inspector.render_text()
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
    assert "OME channels (3): DAPI, PanCK, CD3" in inspector.render_text()
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


def test_reader_rejects_non_ome_tiff(tmp_path: Path) -> None:
    source = tmp_path / "generic.tif"
    _write_generic_tiff(source)

    with pytest.raises(ValueError, match="does not contain recognized OME metadata"):
        with OMETiffReader(source):
            pass


def test_generic_rgb_and_label_interfaces_are_available() -> None:
    assert RGBImage.samples_per_pixel == 3
    assert RGBImage.channel_names == ("Red", "Green", "Blue")
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
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == "1.0"


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
        np.testing.assert_array_equal(ome.read_region(1, 6, 2, 9), data[1:6, 2:9, :])
        np.testing.assert_array_equal(
            ome.read_region(1, 6, 2, 9, channels=[2, 0]),
            data[1:6, 2:9, :][..., [2, 0]],
        )
