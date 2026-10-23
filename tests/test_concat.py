"""Ordered CLI selectors, real input-reader semantics, and real TIFF round trips."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import (
    ConcatSeries,
    MultichannelImage,
    OMEImageSeries,
    OMEMultiSeriesWriter,
    OMETiffReader,
    PixelSize,
    concat,
)
from omeify.cli import main


def _input(tmp_path: Path) -> tuple[Path, np.ndarray, np.ndarray, np.ndarray]:
    """Three independent real OME images, with distinct physical resolutions."""
    path = tmp_path / "source.ome.tiff"
    y, x = np.indices((32, 48))
    labels = (1001 + (x + 10 * y) % 7).astype(np.uint32)
    field = np.stack((x[:16, :24], y[:16, :24]), axis=0).astype(np.float32)
    rgb = np.stack((x % 255, y % 255, (x + y) % 255), axis=-1).astype(np.uint8)
    with tifffile.TiffWriter(path, bigtiff=True, ome=True) as writer:
        for name, values, axes, pixel_size, rgb_mode in (
            ("Nuclear segmentation", labels, "YX", 0.5, False),
            ("Embedding field", field, "CYX", 32.0, False),
            ("Native RGB", rgb, "YXS", 0.5, True),
        ):
            writer.write(
                values, tile=(16, 16), photometric="rgb" if rgb_mode else "minisblack",
                metadata={
                    "Name": name, "axes": axes,
                    "PhysicalSizeX": pixel_size, "PhysicalSizeY": pixel_size,
                    "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
                },
            )
    return path, labels, field, rgb


def test_concat_cli_interleaved_selectors_keep_exact_order(tmp_path: Path) -> None:
    path = tmp_path / "input.ome.tiff"
    output = tmp_path / "combined.ome.tiff"
    report = {
        "output_file": {"series_count": 4},
        "series": [{"name": name} for name in ("A", "B", "C", "D")],
        "verification": {"output_pixels_decodable": True},
    }
    with patch("omeify.cli.concat", return_value=report) as called:
        result = CliRunner().invoke(main, [
            "concat", "--series", str(path), "1",
            "--named-series", "B", str(path), "2",
            "--named-series", "C", str(path), "0",
            "--series", str(path), "3",
            "-o", str(output), "--no-overwrite",
        ])
    assert result.exit_code == 0, result.output
    selections = called.call_args.args[0]
    assert [(s.series, s.name) for s in selections] == [
        (1, None), (2, "B"), (0, "C"), (3, None),
    ]
    assert called.call_args.kwargs["overwrite"] is False
    assert "4 series" in result.output


@pytest.mark.parametrize("arguments, expected", [
    (["concat", "-o", "output.ome.tif"], "Provide at least one"),
    (["concat", "--series", "input.ome.tif", "-1", "-o", "output.ome.tif"], "is not in the range"),
    (["concat", "--named-series", "name", "input.ome.tif", "NaN",
      "-o", "output.ome.tif"], "not a valid integer"),
    (["concat", "--series", "input.ome.tif", "0"], "Missing option"),
    (["concat", "--series", "input.ome.tif", "0", "-o", "output.ome.tif",
      "--compression", "JPEG"], "Invalid value"),
    (["concat", "--named-series", " ", "input.ome.tif", "0", "-o", "output.ome.tif"],
     "must not be empty"),
])
def test_concat_cli_rejects_bad_selection(arguments: list[str], expected: str) -> None:
    result = CliRunner().invoke(main, arguments)
    assert result.exit_code != 0
    assert expected in result.output, result.output


def test_concat_reads_heterogeneous_ome_series_without_materializing(
    tmp_path: Path, monkeypatch,
) -> None:
    source, labels, field, rgb = _input(tmp_path)
    captured = []

    class Writer:
        def __init__(self, path, **kwargs):
            assert kwargs["compression"] == "LZW"
            assert str(path) == str(tmp_path / "all.ome.tiff")

        def write(self, entries):
            assert all(entry.image.is_open for entry in entries)
            for entry in entries:
                captured.append((
                    entry.name, entry.image.shape, entry.image.dtype,
                    entry.image.pixel_size, entry.downsample,
                    entry.significant_bits, entry.image.read_region(0, 2, 0, 2),
                ))
            return {"series": [{"name": e.name} for e in entries]}

    monkeypatch.setattr("omeify.concatenation.OMEMultiSeriesWriter", Writer)
    result = concat([
        ConcatSeries(source, 2, "H&E"),
        ConcatSeries(source, 0),
        ConcatSeries(source, 1, "Whole-window embeddings"),
    ], tmp_path / "all.ome.tiff")
    assert [record["name"] for record in result["series"]] == [
        "H&E", "Nuclear segmentation", "Whole-window embeddings",
    ]
    assert [(r[1], r[2].name, r[4]) for r in captured] == [
        (rgb.shape, "uint8", "mean"),
        (labels.shape, "uint32", "nearest"),
        (field.shape, "float32", "mean"),
    ]
    np.testing.assert_array_equal(captured[0][-1], rgb[:2, :2])
    np.testing.assert_array_equal(captured[1][-1], labels[:2, :2])
    np.testing.assert_array_equal(captured[2][-1], field[:, :2, :2])
    assert captured[1][3].x == 0.5
    assert captured[2][3].x == 32.0


def test_unnamed_series_fallback_does_not_expose_source_filename(tmp_path: Path, monkeypatch):
    source, *_ = _input(tmp_path)
    monkeypatch.setattr(OMETiffReader, "series_name", property(lambda self: None))

    class Writer:
        def __init__(self, path, **kwargs):
            pass

        def write(self, entries):
            assert [item.name for item in entries] == ["Series 0"]
            assert source.name not in entries[0].name
            return {}

    monkeypatch.setattr("omeify.concatenation.OMEMultiSeriesWriter", Writer)
    concat([ConcatSeries(source, 0)], tmp_path / "result.ome.tif")


def test_concat_rejects_duplicate_names_without_rewriting_or_silent_disambiguation(
    tmp_path: Path,
) -> None:
    source, *_ = _input(tmp_path)
    destination = tmp_path / "combined.ome.tif"
    with pytest.raises(ValueError, match="series names must be unique"):
        concat([ConcatSeries(source, 0), ConcatSeries(source, 0)], destination)
    assert not destination.exists()


def test_concat_protects_input_and_validates_entries(tmp_path: Path) -> None:
    source, *_ = _input(tmp_path)
    with pytest.raises(ValueError, match="replace a protected input"):
        concat([ConcatSeries(source, 0)], source)
    with pytest.raises(ValueError, match="at least one"):
        concat([], tmp_path / "empty.ome.tif")
    with pytest.raises(ValueError, match="lossless"):
        concat([ConcatSeries(source, 0)], tmp_path / "bad.ome.tif", compression="JPEG")
    with pytest.raises(ValueError, match="non-empty"):
        ConcatSeries(source, 0, " ")
    with pytest.raises(ValueError, match="at least"):
        ConcatSeries(source, -1)
    dest = tmp_path / "preexisting.ome.tif"
    dest.write_bytes(b"old result")
    with pytest.raises(FileExistsError):
        concat([ConcatSeries(source, 0)], dest, overwrite=False)
    assert dest.read_bytes() == b"old result"


def test_concat_preserves_significant_bits_as_metadata_without_rounding(
    tmp_path: Path, monkeypatch,
):
    source, *_ = _input(tmp_path)
    real = __import__("omeify.concatenation", fromlist=["_source_significant_bits"])
    monkeypatch.setattr(
        real, "_source_significant_bits", lambda reader: 19 if reader.series_index == 1 else None,
    )

    class Writer:
        def __init__(self, path, **kwargs):
            pass

        def write(self, entries):
            assert entries[0].significant_bits == 19
            assert entries[0].image.dtype == np.dtype("float32")
            return {}

    monkeypatch.setattr(real, "OMEMultiSeriesWriter", Writer)
    concat([ConcatSeries(source, 1)], tmp_path / "result.ome.tif")


def test_concat_full_writer_round_trip_of_independent_tiff_series(tmp_path: Path) -> None:
    """Requires the ordinary installed Omeify imagecodecs and OME XSD."""
    source, labels, field, rgb = _input(tmp_path)
    destination = tmp_path / "combined.ome.tiff"
    report = concat([
        ConcatSeries(source, 0),
        ConcatSeries(source, 2, "H&E review"),
        ConcatSeries(source, 1),
    ], destination, tile_size=16, pyramid_levels=1, overwrite=False)
    assert report["verification"]["base_pixel_values_match"] is True
    assert [item["name"] for item in report["series"]] == [
        "Nuclear segmentation", "H&E review", "Embedding field",
    ]
    assert [item["downsample_method"] for item in report["series"]] == [
        "nearest", "mean", "mean",
    ]
    with tifffile.TiffFile(destination) as tiff:
        assert [item.name for item in tiff.series] == [
            "Nuclear segmentation", "H&E review", "Embedding field",
        ]
        np.testing.assert_array_equal(tiff.series[0].asarray(), labels)
        np.testing.assert_array_equal(tiff.series[1].asarray(), rgb)
        np.testing.assert_array_equal(tiff.series[2].asarray(), field)
        np.testing.assert_array_equal(tiff.series[0].levels[1].asarray(), labels[::2, ::2])
    with OMETiffReader(destination, series=2) as output:
        assert output.pixel_size.x == 32.0
        assert output.dtype == np.dtype("float32")
        assert output.channel_count == 2


def test_multi_writer_accepts_metadata_only_significant_bits_override(tmp_path: Path) -> None:
    """An exported m10 field must not acquire a misleading 32-bit precision claim."""
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "concat.ome.tif"
    pixels = (np.arange(16 * 16, dtype=np.float32) / 8).reshape(16, 16)
    with MultichannelImage.from_array(
        pixels, axes="YX", channel_names=("Embedding 0 (float32 m10)",),
        pixel_size=PixelSize(32.0, 32.0, "µm"),
    ) as prepared:
        with prepared.with_metadata(origin_yx=(20.0, 40.0)) as located:
            OMEMultiSeriesWriter(source, compression="LZW", tile_size=16,
                                 pyramid_levels=1).write([
                OMEImageSeries("Embedding field", located, downsample="nearest",
                               significant_bits=19),
            ])
    report = concat([ConcatSeries(source, 0)], output, pyramid_levels=1)
    assert report["series"][0]["downsample_method"] == "nearest"
    with tifffile.TiffFile(output) as tiff:
        assert 'SignificantBits="19"' in tiff.ome_metadata
        np.testing.assert_array_equal(tiff.series[0].asarray(), pixels)
        np.testing.assert_array_equal(tiff.series[0].levels[1].asarray(), pixels[::2, ::2])
    with OMETiffReader(output) as restored:
        assert restored.origin_yx == (20.0, 40.0)
