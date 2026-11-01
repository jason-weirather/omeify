"""Crop geometry, lazy reads, batch preflight, and real TIFF readback."""
from __future__ import annotations

import json
import os
from pathlib import Path

import click
import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from report_fixture import multi_report_fixture

from omeify import ImageLevel, MultichannelImage, OMETiffReader, PixelSize, crop
from omeify.cli import main
from omeify.regions import group_outputs, load_geojson, parse_regions, pixel_bounds, rectangle_feature


def collection(*features):
    return {"type": "FeatureCollection", "features": list(features)}


@pytest.fixture
def input_file(tmp_path):
    path = tmp_path / "source.ome.tiff"
    values = np.arange(2 * 48 * 64, dtype=np.uint16).reshape(2, 48, 64)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack", tile=(16, 16), metadata={
        "axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]},
        "PhysicalSizeX": .5, "PhysicalSizeY": .7,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    return path, values


@pytest.fixture
def writer_spy(monkeypatch):
    """Replace only export; real source open, ROI planning and regional reads remain."""
    calls = []

    class Writer:
        def __init__(self, path, **options):
            self.path, self.options = path, options

        def write(self, entries, *, provenance):
            calls.append({"path": self.path, "options": self.options, "provenance": provenance,
                          "pixels": [s.image.asarray() for s in entries],
                          "names": [s.name for s in entries],
                          "channels": [s.image.channel_names for s in entries],
                          "types": [s.image.image_type for s in entries],
                          "sizes": [s.image.pixel_size for s in entries]})
            return multi_report_fixture(self.path, entries, provenance)

    monkeypatch.setattr("omeify.io.output.OMEMultiSeriesWriter", Writer)
    return calls


@pytest.mark.parametrize("kind", ["multichannel", "rgb", "label"])
def test_lazy_crop_preserves_values_metadata_and_parent_session(image_factory, kind):
    shape = (2, 19, 27) if kind == "multichannel" else (19, 27, 3) if kind == "rgb" else (19, 27)
    a = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    if kind == "rgb":
        a = a.astype(np.uint8)
    parent = image_factory(a, kind=kind, pixel_size=PixelSize(.3, .4, "µm"))
    with parent.crop(3, 15, 5, 22) as view:
        assert view.pixel_size == parent.pixel_size
        assert view.channel_names == parent.channel_names and view.dtype == parent.dtype
        assert view.image_type == kind and len(view.levels) == 1
        expected = a[:, 3:15, 5:22] if kind == "multichannel" else a[3:15, 5:22]
        np.testing.assert_array_equal(view.asarray(), expected)
        with view.crop(1, 3, 2, 4) as nested:
            np.testing.assert_array_equal(nested.asarray(), expected[:, 1:3, 2:4]
                                          if kind == "multichannel" else expected[1:3, 2:4])
    assert parent.is_open
    stale = parent.crop(1, 4, 1, 5)
    parent.close()
    parent.open()
    with pytest.raises(RuntimeError):
        stale.open()


def test_crop_is_read_lazy_and_drops_source_pyramids(monkeypatch):
    array = np.zeros((19, 27), dtype=np.float32)
    size = PixelSize(.5, .5, "µm")
    levels = [ImageLevel(0, "YX", array.shape, size),
              ImageLevel(1, "YX", (10, 14), size.scaled(2), (2, 2))]
    with MultichannelImage.from_array(array, axes="YX", pixel_size=size,
                                     level_arrays=[array[::2, ::2]], levels=levels) as image:
        monkeypatch.setattr(image, "read_region", lambda *a, **kw: pytest.fail("Must remain lazy"))
        with image.crop(0, 10, 1, 11) as view:
            assert view.shape == (10, 10) and len(view.levels) == 1
        for args in [(0, 0, 0, 1), (0, 20, 0, 1), (False, 1, 0, 1), (.5, 1, 0, 1)]:
            with pytest.raises((ValueError, TypeError)):
                image.crop(*args)


def test_geometry_types_rounding_names_and_padding(tmp_path):
    a = rectangle_feature([1.2, 2.8, 9.1, 10.4], name="right tissue")
    b = rectangle_feature([20, 21, 30, 35], name="right tissue")
    c = rectangle_feature([0, 0, 5, 7], name="")
    regions = parse_regions(collection(a, b, c))
    destination = tmp_path / "slide.ome.tiff"
    assert group_outputs(regions, destination) == [(destination, regions)]
    groups = group_outputs(regions, destination, shatter="by_name")
    assert [p.name for p, _ in groups] == ["slide-right-tissue.ome.tiff", "slide-03.ome.tiff"]
    assert [len(r) for _, r in groups] == [2, 1]
    assert pixel_bounds(regions[0].bounds, 100, 100) == (1, 2, 10, 11)
    assert [p.name for p, _ in group_outputs(regions, destination, shatter="by_index")] == [
        f"slide-{i:02d}.ome.tiff" for i in range(1, 4)]
    many = parse_regions(collection(*[c] * 101))
    for mode in ("by_index", "by_name"):
        outputs = group_outputs(many, "slide", shatter=mode)
        assert outputs[0][0].name == "slide-001.ome.tiff"
        assert outputs[-1][0].name == "slide-101.ome.tiff"
    multi = {"type": "MultiPolygon", "coordinates": [a["geometry"]["coordinates"],
                                                   b["geometry"]["coordinates"]]}
    assert len(parse_regions(multi)) == 1
    assert parse_regions(multi)[0].bounds == (1.2, 2.8, 30., 35.)
    assert len(parse_regions({"type": "GeometryCollection", "geometries": [multi, multi]})) == 2


@pytest.mark.parametrize("bad", [
    '{"type":"Feature","type":"Polygon"}', '{"type":"Polygon","coordinates":NaN}',
    'null', '[]', '{"type":"Point","coordinates":[1,2]}',
    '{"type":"MultiPolygon","coordinates":null}',
    '{"type":"Feature","properties":false,"geometry":null}',
    '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,1]]]}',
    '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1e400],[0,0]]]}',
    '{"type":"Polygon","coordinates":[[[false,0],[1,0],[1,1],[false,0]]]}',
])
def test_malformed_geojson_fails(bad):
    with pytest.raises(ValueError):
        parse_regions(load_geojson(bad))


def test_clipping_and_name_collisions():
    assert pixel_bounds([-2, 3, 20, 30], 10, 12, clip=True) == (0, 3, 10, 12)
    with pytest.raises(ValueError):
        pixel_bounds([-2, 3, 20, 30], 10, 12)
    with pytest.raises(ValueError):
        pixel_bounds([20, 3, 30, 4], 10, 12, clip=True)
    for names in [("a/b", "a b"), ("A", "a"), ("../../", "safe"), ("02", ""), ("é" * 61, "safe")]:
        regions = parse_regions(collection(*[rectangle_feature([0, 0, 1, 1], name=n) for n in names]))
        # Filename restrictions apply only when annotation names become filenames.
        assert len(group_outputs(regions, "out")) == 1
        assert len(group_outputs(regions, "out", shatter="by_index")) == 2
        with pytest.raises(ValueError, match="by_index"):
            group_outputs(regions, "out", shatter="by_name")


@pytest.mark.parametrize("shatter, groups", [
    (None, [[0, 1, 2, 3]]), ("by_index", [[0], [1], [2], [3]]), ("by_name", [[0, 2], [1], [3]]),
])
def test_grouped_exports_use_real_lazy_crops(input_file, tmp_path, writer_spy, shatter, groups):
    path, pixels = input_file
    boxes = [(2, 3, 14, 19), (25, 10, 45, 25), (3, 5, 14, 12), (0, 0, 8, 8)]
    names = ["zeta", "alpha", "zeta", ""]
    features = [rectangle_feature(b, name=n) for b, n in zip(boxes, names)]
    for i, feature in enumerate(features):
        feature["id"] = 100 - i  # IDs and arbitrary properties are not the ordering key.
        feature["properties"]["index"] = 10 - i
    output = tmp_path / "out.he.ome.tiff"
    report = crop(path, output_path=output, geojson=collection(*features), shatter=shatter)
    assert report["schema"] == "omeify.crop/6" and report["shatter"] == shatter
    assert report["output_path"] == str(output) and "naming" not in report
    assert len(writer_spy) == len(report["outputs"]) == len(groups)
    if shatter is None:
        assert writer_spy[0]["path"] == output
    for call, result, indices in zip(writer_spy, report["outputs"], groups):
        assert call["names"] == [f"{i + 1:02d} - {names[i] or 'ROI'}" for i in indices]
        assert call["sizes"] == [PixelSize(.5, .7, "µm")] * len(indices)
        assert call["options"]["compression"] is None and call["options"]["tile_size"] is None
        assert call["provenance"]["schema"] == "omeify.crop_provenance/3"
        assert call["provenance"]["shatter"] == shatter
        assert call["provenance"]["regions"] == result["regions"]
        for series, i in enumerate(indices):
            x0, y0, x1, y1 = boxes[i]
            np.testing.assert_array_equal(call["pixels"][series], pixels[:, y0:y1, x0:x1])
            record = result["regions"][series]
            assert record["source_offset_xy"] == [x0, y0]
            assert record["input_index"] == i + 1 and record["output_series"] == series
            assert record["name"] == (names[i] or None)
            assert record["output_name"] == call["names"][series]


def test_selected_channels_are_ordered_renamed_and_audited(input_file, tmp_path, writer_spy):
    path, pixels = input_file
    doc = collection(rectangle_feature([2, 3, 14, 19], name="left"),
                     rectangle_feature([11, 7, 25, 20], name="right"))
    report = crop(path, tmp_path / "selected.ome.tif", geojson=doc,
                  channels=[("CD3", "CD3e"), 0], shatter="by_index")
    mapping = [
        {"source_index": 1, "source_name": "CD3", "output_index": 0,
         "output_name": "CD3e"},
        {"source_index": 0, "source_name": "DAPI", "output_index": 1,
         "output_name": "DAPI"},
    ]
    assert report["schema"] == "omeify.crop/6"
    assert report["channel_mapping"] == mapping
    assert len(writer_spy) == 2
    for call, output, bounds in zip(writer_spy, report["outputs"],
                                    ((2, 3, 14, 19), (11, 7, 25, 20))):
        assert call["channels"] == [("CD3e", "DAPI")]
        assert call["types"] == ["multichannel"]
        assert call["provenance"]["schema"] == "omeify.crop_provenance/3"
        assert call["provenance"]["channel_mapping"] == mapping
        assert output["write_report"]["provenance"]["value"]["channel_mapping"] == mapping
        x0, y0, x1, y1 = bounds
        np.testing.assert_array_equal(call["pixels"][0], pixels[[1, 0], y0:y1, x0:x1])


def test_rename_all_or_subset_without_writing_intermediates(input_file, tmp_path, writer_spy):
    path, pixels = input_file
    output = tmp_path / "renamed.ome.tif"
    report = crop(path, output, bounds=(2, 3, 15, 17),
                  rename_channels={"DAPI": "DNA"}, rename_channels_by="name")
    assert writer_spy[0]["channels"] == [("DNA", "CD3")]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[:, 3:17, 2:15])
    assert [c["source_index"] for c in report["channel_mapping"]] == [0, 1]
    writer_spy.clear()
    report = crop(path, output, bounds=(2, 3, 15, 17), channels=[1],
                  rename_channels={1: "CD3e"}, rename_channels_by="index")
    assert writer_spy[0]["channels"] == [("CD3e",)]
    assert report["channel_mapping"][0]["source_index"] == 1
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0][0], pixels[1, 3:17, 2:15])


def test_duplicate_channel_selection_is_deliberate_and_ordered(input_file, tmp_path, writer_spy):
    path, pixels = input_file
    report = crop(path, tmp_path / "two.ome.tif", bounds=(0, 0, 8, 9),
                  channels=[("DAPI", "DNA 1"), (0, "DNA 2")])
    assert writer_spy[0]["channels"] == [("DNA 1", "DNA 2")]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[[0, 0], :9, :8])
    assert [c["source_index"] for c in report["channel_mapping"]] == [0, 0]


@pytest.mark.parametrize("channels, error", [
    ([], "must not be empty"),
    ([99], "outside"),
    (["NotAChannel"], "Unknown channel"),
    ([True], "integer indices"),
    ([-1], "at least 0"),
    ([0.5], "integer indices"),
    ([0, (1, "  ")], "non-empty"),
    ("DAPI", "must be a sequence"),
])
def test_bad_channel_selections_fail_before_any_product(input_file, tmp_path, writer_spy,
                                                        channels, error):
    path, _ = input_file
    with pytest.raises((TypeError, ValueError, IndexError), match=error):
        crop(path, tmp_path / "output.ome.tif", bounds=(0, 0, 8, 9), channels=channels)
    assert not writer_spy


def test_ambiguous_duplicate_names_and_rename_conflicts(input_file, tmp_path, writer_spy):
    path, _ = input_file
    with pytest.raises(ValueError, match="either named channel selections or a rename mapping"):
        crop(path, tmp_path / "out.ome.tif", bounds=(0, 0, 8, 9),
             channels=[(0, "DNA")], rename_channels={0: "New"}, rename_channels_by="index")
    with pytest.raises(ValueError, match="unknown source names"):
        crop(path, tmp_path / "out.ome.tif", bounds=(0, 0, 8, 9),
             rename_channels={"missing": "New"}, rename_channels_by="name")
    source = tmp_path / "duplicates.ome.tif"
    data = np.zeros((2, 16, 16), dtype=np.uint16)
    tifffile.imwrite(source, data, ome=True, metadata={
        "axes": "CYX", "Channel": {"Name": ["dup", "dup"]},
        "PhysicalSizeX": .5, "PhysicalSizeY": .7,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    with pytest.raises(ValueError, match="ambiguous"):
        crop(source, tmp_path / "out.ome.tif", bounds=(0, 0, 8, 9), channels=["dup"])
    report = crop(source, tmp_path / "out.ome.tif", bounds=(0, 0, 8, 9),
                  channels=[(1, "second")])
    assert report["channel_mapping"][0]["source_index"] == 1
    assert not any(c["source_index"] == 0 for c in report["channel_mapping"])


def test_cli_channel_order_across_both_options(input_file, tmp_path, writer_spy):
    path, pixels = input_file
    args = ["crop", str(path), "-o", str(tmp_path / "out.ome.tif"),
            "--bounds", "0", "0", "10", "11", "--channel", "index:1",
            "--named-channel", "DNA", "DAPI", "--channel", "1", "--output-json", "-"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert [c["source_index"] for c in report["channel_mapping"]] == [1, 0, 1]
    assert [c["output_name"] for c in report["channel_mapping"]] == ["CD3", "DNA", "CD3"]
    assert writer_spy[0]["channels"] == [("CD3", "DNA", "CD3")]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[[1, 0, 1], :11, :10])


def test_cli_numeric_source_name_and_json_renames(input_file, tmp_path, writer_spy):
    path, _ = input_file
    rename_file = tmp_path / "rename.json"
    rename_file.write_text('{"0": "DNA", "1": "CD3e"}')
    base = ["crop", str(path), "-o", str(tmp_path / "out.ome.tif"),
            "--bounds", "0", "0", "10", "11", "--output-json", "-"]
    result = CliRunner().invoke(main, base + ["--channel", "1", "--channel", "0",
                                              "--rename-channels-json", str(rename_file),
                                              "--rename-channels-by", "index"])
    assert result.exit_code == 0, result.output
    assert writer_spy[0]["channels"] == [("CD3e", "DNA")]
    writer_spy.clear()
    result = CliRunner().invoke(main, base + ["--rename-channels-json", str(rename_file),
                                              "--rename-channels-by", "index"])
    assert result.exit_code == 0, result.output
    assert writer_spy[0]["channels"] == [("DNA", "CD3e")]
    writer_spy.clear()
    result = CliRunner().invoke(main, base + ["--named-channel", "DNA", "DAPI",
                                              "--rename-channels-json", str(rename_file),
                                              "--rename-channels-by", "index"])
    assert result.exit_code == 2 and not writer_spy
    assert "either --named-channel or --rename-channels-json" in result.output
    result = CliRunner().invoke(main, base + ["--channel", "index:01"])
    assert result.exit_code == 2 and "canonical zero-based integer" in result.output
    assert not writer_spy


def test_cli_renames_file_cannot_be_image_or_report_output(input_file, tmp_path, writer_spy):
    path, _ = input_file
    renames = tmp_path / "renames.json"
    renames.write_text('{"DAPI":"DNA"}')
    args = ["crop", str(path), "-o", str(renames), "--bounds", "0", "0", "8", "9",
            "--rename-channels-json", str(renames), "--rename-channels-by", "name"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code != 0 and not writer_spy
    assert renames.read_text() == '{"DAPI":"DNA"}'
    args[3] = str(tmp_path / "output.ome.tif")
    result = CliRunner().invoke(main, args + ["--output-json", str(renames)])
    assert result.exit_code != 0 and not writer_spy
    assert renames.read_text() == '{"DAPI":"DNA"}'


def test_cli_explicit_name_selects_numeric_name(tmp_path, writer_spy):
    path = tmp_path / "numeric.ome.tif"
    values = np.arange(2 * 16 * 16, dtype=np.uint16).reshape(2, 16, 16)
    tifffile.imwrite(path, values, ome=True, metadata={
        "axes": "CYX", "Channel": {"Name": ["0", "DAPI"]},
        "PhysicalSizeX": .5, "PhysicalSizeY": .5,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    result = CliRunner().invoke(main, ["crop", str(path), "-o", str(tmp_path / "out.ome.tif"),
                                       "--bounds", "0", "0", "8", "9",
                                       "--channel", "name:0", "--named-channel", "DNA", "index:1"])
    assert result.exit_code == 0, result.output
    assert writer_spy[0]["channels"] == [("0", "DNA")]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], values[:, :9, :8])


@pytest.mark.parametrize("kind", ["rgb", "label"])
def test_rgb_and_label_remain_indivisible_logical_channels(tmp_path, writer_spy, kind):
    path = tmp_path / f"{kind}.ome.tif"
    values = (np.arange(16 * 16 * 3, dtype=np.uint8).reshape(16, 16, 3)
              if kind == "rgb" else np.arange(16 * 16, dtype=np.uint16).reshape(16, 16))
    tifffile.imwrite(path, values, ome=True,
                     photometric="rgb" if kind == "rgb" else "minisblack",
                     metadata={"axes": "YXS" if kind == "rgb" else "YX",
                               "PhysicalSizeX": .5, "PhysicalSizeY": .5,
                               "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})
    kw = {"labels": kind == "label"}
    r = crop(path, tmp_path / "out.ome.tif", bounds=(0, 0, 9, 10),
             channels=[(0, "Renamed")], **kw)
    assert writer_spy[0]["channels"] == [("Renamed",)]
    assert writer_spy[0]["types"] == [kind]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], values[:10, :9])
    assert len(r["channel_mapping"]) == 1
    writer_spy.clear()
    with pytest.raises(ValueError, match="indivisible"):
        crop(path, tmp_path / "out.ome.tif", bounds=(0, 0, 9, 10),
             channels=[0, 0], **kw)
    assert not writer_spy


@pytest.mark.parametrize("shatter, suffix", [(None, ""), ("by_index", "-02"), ("by_name", "-b")])
def test_whole_batch_preflight_and_inputs_are_protected(input_file, tmp_path, writer_spy,
                                                        shatter, suffix):
    path, _ = input_file
    doc = collection(rectangle_feature([1, 1, 10, 10], name="a"),
                     rectangle_feature([2, 2, 15, 15], name="b"))
    output = tmp_path / "out.ome.tiff"
    later = tmp_path / f"out{suffix}.ome.tiff"
    later.write_text("keep")
    with pytest.raises(FileExistsError):
        crop(path, output, geojson=doc, shatter=shatter, overwrite=False)
    assert not writer_spy and later.read_text() == "keep"
    later.unlink()
    later.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        crop(path, output, geojson=doc, shatter=shatter, overwrite=False)
    later.unlink()
    os.link(path, later)
    with pytest.raises(ValueError, match="replace"):
        crop(path, output, geojson=doc, shatter=shatter)
    later.unlink()
    later.mkdir()
    with pytest.raises(IsADirectoryError):
        crop(path, output, geojson=doc, shatter=shatter)
    later.rmdir()
    with pytest.raises(ValueError):
        crop(path, output, bounds=(10, 10, 3, 3), shatter=shatter)
    with pytest.raises(ValueError, match="outside"):
        crop(path, output, shatter=shatter, geojson=collection(
            doc["features"][0], rectangle_feature([0, 0, 100, 100], name="b")))
    assert not writer_spy


@pytest.mark.parametrize("shatter, filenames", [
    (None, ["out.ome.tiff"]),
    ("by_index", ["out-01.ome.tiff", "out-02.ome.tiff", "out-03.ome.tiff"]),
    ("by_name", ["out-right.ome.tiff", "out-left.ome.tiff"]),
])
@pytest.mark.filterwarnings("error:.*get_text_stream.*:DeprecationWarning")
def test_cli_stdin_and_coordinate_guard(input_file, tmp_path, writer_spy, shatter, filenames):
    path, _ = input_file
    doc = collection(*[rectangle_feature([1, 2, 10, 11], name=n) for n in ("right", "left", "right")])
    args = ["crop", str(path), "-o", str(tmp_path / "out.ome.tiff"), "--geojson", "-", "--output-json", "-"]
    if shatter is not None:
        args += ["--shatter", shatter]
    result = CliRunner().invoke(main, args, input=json.dumps(doc))
    assert result.exit_code == 0, result.output
    report = json.loads(result.stdout)
    assert [Path(o["path"]).name for o in report["outputs"]] == filenames
    assert report["shatter"] == shatter
    doc["omeify"] = {"coordinate_system": "level0_pixels", "image_size": [128, 48], "series": 0}
    with pytest.raises(ValueError, match="dimensions or series"):
        crop(path, tmp_path / "out.ome.tiff", geojson=doc, shatter=shatter)
    assert len(writer_spy) == len(filenames)


def test_cli_stdin_utf8_does_not_require_deprecated_click_stream(input_file, tmp_path,
                                                                  writer_spy, monkeypatch):
    # Click 8.5 deprecates get_text_stream; Click 9 will remove it. Enforce
    # the migration even when testing against older supported Click versions.
    def deprecated_stream(*args, **kwargs):
        pytest.fail("CLI must not depend on deprecated click.get_text_stream")

    monkeypatch.setattr(click, "get_text_stream", deprecated_stream)
    path, pixels = input_file
    name = "Café µm"
    document = collection(rectangle_feature([1, 2, 10, 11], name=name))
    result = CliRunner().invoke(
        main,
        ["crop", str(path), "-o", str(tmp_path / "unicode.ome.tiff"),
         "--geojson", "-", "--output-json", "-"],
        input=json.dumps(document, ensure_ascii=False),
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["outputs"][0]["regions"][0]["name"] == name
    assert writer_spy[0]["names"] == [f"01 - {name}"]
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[:, 2:11, 1:10])


def test_real_crop_tiff_readback(input_file, tmp_path):
    path, a = input_file
    report = crop(path, tmp_path / "real", bounds=(2, 3, 40, 30), compression="Deflate", tile_size=16)
    output = Path(report["outputs"][0]["path"])
    assert output == tmp_path / "real"  # No implicit extension or index, even for bounds.
    with OMETiffReader(output) as actual:
        np.testing.assert_array_equal(actual.asarray(), a[:, 3:30, 2:40])
        assert actual.pixel_size == PixelSize(.5, .7, "µm")
        assert len(actual.levels) > 1


def test_real_selected_channel_crop_tiff_roundtrip(input_file, tmp_path):
    path, pixels = input_file
    target = tmp_path / "selected.ome.tif"
    report = crop(path, target, bounds=(3, 4, 27, 21),
                  channels=[("CD3", "CD3e"), "DAPI"],
                  compression="Uncompressed", tile_size=16)
    assert report["outputs"][0]["path"] == str(target)
    assert [entry["output_name"] for entry in report["channel_mapping"]] == ["CD3e", "DAPI"]
    with OMETiffReader(target) as image:
        assert image.channel_names == ("CD3e", "DAPI")
        assert image.pixel_size == PixelSize(.5, .7, "µm")
        np.testing.assert_array_equal(image.asarray(), pixels[[1, 0], 4:21, 3:27])
        assert image.series_name == "01 - ROI"
    assert report["outputs"][0]["write_report"]["provenance"]["value"]["channel_mapping"] == (
        report["channel_mapping"]
    )


@pytest.mark.parametrize("output, shattered", [
    ("out", "out-01.ome.tiff"), ("out.he", "out.he-01.ome.tiff"),
    ("out.he.ome.tiff", "out.he-01.ome.tiff"), ("out.ome.tif", "out-01.ome.tif"),
    ("out.tif", "out-01.tif"), ("out.tiff", "out-01.tiff"),
    ("out.OME.TIFF", "out-01.OME.TIFF"),
])
def test_output_path_is_literal_unless_shattered(output, shattered):
    regions = parse_regions(rectangle_feature([0, 0, 1, 1], name="arbitrary name"))
    assert group_outputs(regions, output)[0][0] == Path(output)
    assert group_outputs(regions, output, shatter="by_index")[0][0] == Path(shattered)


def test_invalid_output_modes_and_destinations(input_file, tmp_path, writer_spy):
    path, _ = input_file
    doc = rectangle_feature([0, 0, 10, 10], name="x")
    for mode in (True, "index", "name", "unexpected"):
        with pytest.raises(ValueError, match="shatter"):
            crop(path, tmp_path / "out", geojson=doc, shatter=mode)
    for output in ("", ".", "..", "/", tmp_path):
        with pytest.raises((ValueError, IsADirectoryError)):
            group_outputs(parse_regions(doc), output)
    with pytest.raises(ValueError, match="prefix"):
        group_outputs(parse_regions(doc), ".ome.tiff", shatter="by_index")
    args = ["crop", str(path), "-o", str(tmp_path / "out"), "--bounds", "0", "0", "10", "10"]
    for flags in (["--shatter"], ["--shatter", "index"], ["--naming", "index"]):
        assert CliRunner().invoke(main, args + flags).exit_code == 2
    assert not writer_spy
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 0, result.output
    assert writer_spy[0]["path"] == tmp_path / "out"
    assert writer_spy[0]["names"] == ["01 - ROI"]


def test_input_and_geojson_aliases_cannot_be_output(input_file, tmp_path, writer_spy):
    path, _ = input_file
    doc = rectangle_feature([0, 0, 10, 10], name="x")
    annotations = tmp_path / "regions.json"
    annotations.write_text(json.dumps(doc))
    for output in (path, annotations):
        with pytest.raises(ValueError, match="replace"):
            crop(path, output, geojson=annotations)
    alias = tmp_path / "alias.ome.tiff"
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="replace"):
        crop(path, alias, geojson=doc)
    first, second = tmp_path / "out-01.ome.tiff", tmp_path / "out-02.ome.tiff"
    first.write_text("prior")
    os.link(first, second)
    with pytest.raises(ValueError, match="alias each other"):
        crop(path, tmp_path / "out.ome.tiff", geojson=collection(doc, doc), shatter="by_index")
    assert not writer_spy and first.read_text() == second.read_text() == "prior"


@pytest.mark.parametrize("shatter", [None, "by_index", "by_name"])
@pytest.mark.parametrize("kind", ["multichannel", "rgb", "label"])
def test_real_crop_collection_roundtrip(tmp_path, kind, shatter):
    from omeify import OMETiffLabelReader

    shape = (2, 48, 64) if kind == "multichannel" else (48, 64, 3) if kind == "rgb" else (48, 64)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    if kind == "rgb":
        data = (data % 256).astype(np.uint8)
    axes = "CYX" if kind == "multichannel" else "YXS" if kind == "rgb" else "YX"
    source = tmp_path / "source.ome.tiff"
    tifffile.imwrite(source, data, ome=True, photometric="rgb" if kind == "rgb" else "minisblack",
                     metadata={"axes": axes, "PhysicalSizeX": .5, "PhysicalSizeY": .7,
                               "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm"})
    boxes = [(2, 3, 40, 30), (25, 10, 45, 25), (12, 13, 50, 40)]
    doc = collection(*[rectangle_feature(b, name=n) for b, n in zip(boxes, ("zeta", "alpha", "zeta"))])
    destination = tmp_path / "regions.ome.tiff"
    report = crop(source, destination, geojson=doc, shatter=shatter, labels=kind == "label",
                  compression="Deflate", tile_size=16)
    expected_groups = {
        None: [[1, 2, 3]], "by_index": [[1], [2], [3]], "by_name": [[1, 3], [2]],
    }[shatter]
    assert [[r["input_index"] for r in o["regions"]] for o in report["outputs"]] == expected_groups
    assert destination.exists() == (shatter is None)
    reader = OMETiffLabelReader if kind == "label" else OMETiffReader
    for output in report["outputs"]:
        write_report = output["write_report"]
        assert write_report["ome"]["xml_is_valid"]
        assert write_report["provenance"]["value"]["regions"] == output["regions"]
        for record in output["regions"]:
            with reader(output["path"], series=record["output_series"]) as image:
                x0, y0, x1, y1 = boxes[record["input_index"] - 1]
                expected = data[:, y0:y1, x0:x1] if kind == "multichannel" else data[y0:y1, x0:x1]
                np.testing.assert_array_equal(image.asarray(), expected)
                assert image.series_name == record["output_name"]
                assert len(image.series_names) == len(output["regions"])
                assert image.pixel_size == PixelSize(.5, .7, "µm") and image.dtype == data.dtype
                assert image.image_type == kind


def _polygon_feature(rings, name="ROI"):
    return {"type": "Feature", "properties": {"name": name},
            "geometry": {"type": "Polygon", "coordinates": rings}}


def _rectangle_ring(x0, y0, x1, y1):
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]]


def test_polygon_mask_respects_holes_and_preserves_channel_selection(
    input_file, tmp_path, writer_spy,
):
    source, pixels = input_file
    feature = _polygon_feature([
        _rectangle_ring(2, 3, 18, 17),
        _rectangle_ring(7, 7, 12, 12),
    ], name="with hole")
    report = crop(source, tmp_path / "masked.ome.tiff", geojson=collection(feature),
                  channels=[("CD3", "CD3e"), "DAPI"])
    result = writer_spy[0]["pixels"][0]
    expected = pixels[[1, 0], 3:17, 2:18].copy()
    expected[:, 4:9, 5:10] = 0
    np.testing.assert_array_equal(result, expected)
    assert report["crop_mode"] == "polygon_mask"
    assert report["fill_value"] == 0
    assert report["outputs"][0]["regions"][0]["mask_applied"] is True
    assert report["outputs"][0]["regions"][0]["geometry_sha256"] == parse_regions(
        collection(feature))[0].geometry_sha256
    assert report["outputs"][0]["write_report"]["provenance"]["value"]["crop_mode"] == "polygon_mask"


def test_multipolygon_union_keeps_disjoint_regions_and_leaves_gap_empty(
    input_file, tmp_path, writer_spy,
):
    source, pixels = input_file
    polygon = {"type": "MultiPolygon", "coordinates": [
        [_rectangle_ring(2, 4, 8, 10)],
        [_rectangle_ring(15, 6, 20, 12)],
    ]}
    crop(source, tmp_path / "islands.ome.tiff", geojson=polygon, shatter="by_index")
    observed = writer_spy[0]["pixels"][0]
    expected = np.zeros((2, 8, 18), dtype=pixels.dtype)
    expected[:, 0:6, 0:6] = pixels[:, 4:10, 2:8]
    expected[:, 2:8, 13:18] = pixels[:, 6:12, 15:20]
    np.testing.assert_array_equal(observed, expected)


def test_mask_keep_retains_old_bounding_rectangle(input_file, tmp_path, writer_spy):
    source, pixels = input_file
    geometry = _polygon_feature([[[2, 3], [19, 3], [2, 19], [2, 3]]])
    report = crop(source, tmp_path / "keep.ome.tiff", geojson=geometry,
                  mask_background="keep")
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[:, 3:19, 2:19])
    assert report["crop_mode"] == "bounding_rectangle"
    assert report["outputs"][0]["regions"][0]["mask_applied"] is False


def test_padding_adds_background_margin_inside_source(input_file, tmp_path, writer_spy):
    source, pixels = input_file
    report = crop(source, tmp_path / "margin.ome.tiff",
                  geojson=_polygon_feature([_rectangle_ring(6, 6, 10, 10)]), padding=2)
    actual = writer_spy[0]["pixels"][0]
    expected = np.zeros((2, 8, 8), dtype=pixels.dtype)
    expected[:, 2:6, 2:6] = pixels[:, 6:10, 6:10]
    np.testing.assert_array_equal(actual, expected)
    assert report["outputs"][0]["regions"][0]["bounds"] == [4, 4, 12, 12]
    assert report["padding"] == 2


def test_padding_beyond_image_is_explicit_and_preserves_source_offset(
    input_file, tmp_path, writer_spy,
):
    source, pixels = input_file
    geometry = _polygon_feature([_rectangle_ring(1, 1, 5, 5)])
    with pytest.raises(ValueError, match="outside the image"):
        crop(source, tmp_path / "reject.ome.tiff", geojson=geometry, padding=3)
    assert not writer_spy
    report = crop(source, tmp_path / "pad.ome.tiff", geojson=geometry,
                  padding=3, pad_outside=True)
    actual = writer_spy[0]["pixels"][0]
    expected = np.zeros((2, 10, 10), dtype=pixels.dtype)
    expected[:, 3:7, 3:7] = pixels[:, 1:5, 1:5]
    np.testing.assert_array_equal(actual, expected)
    region = report["outputs"][0]["regions"][0]
    assert region["bounds"] == [-2, -2, 8, 8]
    assert region["source_offset_xy"] == [-2, -2]
    assert region["clipped"] is False
    assert report["pad_outside"] is True


def test_clip_trims_polygon_extent_but_not_shape(input_file, tmp_path, writer_spy):
    source, pixels = input_file
    triangle = _polygon_feature([[[ -3, 3], [6, 3], [6, 12], [-3, 3]]])
    with pytest.raises(ValueError, match="outside the image"):
        crop(source, tmp_path / "reject.ome.tiff", geojson=triangle)
    report = crop(source, tmp_path / "clip.ome.tiff", geojson=triangle, clip=True)
    assert report["outputs"][0]["regions"][0]["bounds"] == [0, 3, 6, 12]
    assert report["outputs"][0]["regions"][0]["clipped"]
    assert (writer_spy[0]["pixels"][0][:, -1, 0] == 0).all()
    with pytest.raises(ValueError, match="mutually exclusive"):
        crop(source, tmp_path / "fail.ome.tiff", geojson=triangle,
             clip=True, pad_outside=True)


def test_white_outside_is_dtype_max_for_integer_channels(input_file, tmp_path, writer_spy):
    source, pixels = input_file
    feature = _polygon_feature([[[2, 3], [18, 3], [2, 19], [2, 3]]])
    report = crop(source, tmp_path / "white.ome.tiff", geojson=feature,
                  mask_background="white")
    assert report["fill_value"] == np.iinfo(np.uint16).max
    observed = writer_spy[0]["pixels"][0]
    assert observed.dtype == np.uint16
    assert (observed[:, -1, -1] == 65535).all()
    assert (observed[:, 0, 0] == pixels[:, 3, 2]).all()


def test_rgb_auto_white_and_lossless_default_for_masked_regions(
    tmp_path, writer_spy,
):
    path = tmp_path / "rgb.ome.tif"
    pixels = np.zeros((32, 40, 3), dtype=np.uint8)
    pixels[:, :, 0] = 80
    pixels[:, :, 1] = 125
    pixels[:, :, 2] = 200
    tifffile.imwrite(path, pixels, ome=True, photometric="rgb", metadata={
        "axes": "YXS", "PhysicalSizeX": .5, "PhysicalSizeY": .7,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    triangle = _polygon_feature([[[4, 3], [20, 3], [4, 19], [4, 3]]])
    report = crop(path, tmp_path / "rgb-masked.ome.tif", geojson=triangle)
    result = writer_spy[0]["pixels"][0]
    assert result.shape == (16, 16, 3)
    np.testing.assert_array_equal(result[0, 0], pixels[3, 4])
    np.testing.assert_array_equal(result[-1, -1], np.array([255, 255, 255], dtype=np.uint8))
    assert report["fill_value"] == 255
    assert writer_spy[0]["options"]["compression"] == "Deflate"
    writer_spy.clear()
    crop(path, tmp_path / "rgb-explicit.ome.tif", geojson=triangle, compression="JPEG")
    assert writer_spy[0]["options"]["compression"] == "JPEG"


def test_label_mask_uses_background_id_without_inventing_class(image_factory):
    from omeify.roi_raster import geometry_crop, mask_fill_value
    values = np.full((12, 12), 19, dtype=np.uint32)
    image = image_factory(values, kind="label", pixel_size=PixelSize(.3, .3, "µm"))
    polygon = parse_regions(_polygon_feature([[[2, 2], [10, 2], [2, 10], [2, 2]]]))[0]
    with geometry_crop(image, (2, 2, 10, 10), polygons=polygon.polygons,
                       mask=True, fill_value=mask_fill_value(image, "auto")) as view:
        out = view.asarray()
        assert out[0, 0] == 19 and out[-1, -1] == 0
    with pytest.raises(ValueError, match="invent a label ID"):
        mask_fill_value(image, "white")
    image.close()


def test_chunked_polygon_reads_are_stable_and_match_whole_region(input_file):
    from omeify.roi_raster import geometry_crop
    path, _ = input_file
    geometry = parse_regions(_polygon_feature([[[2, 2], [23, 2], [9, 29], [2, 2]]]))[0]
    with OMETiffReader(path) as image:
        with geometry_crop(image, (2, 2, 23, 29), polygons=geometry.polygons,
                           mask=True, fill_value=0) as roi:
            full = roi.asarray()
            tiles = np.concatenate([
                roi.read_region(0, 8, 0, 21),
                roi.read_region(8, 27, 0, 21),
            ], axis=1)
            np.testing.assert_array_equal(tiles, full)
            np.testing.assert_array_equal(roi.read_region(2, 8, 4, 10), full[:, 2:8, 4:10])


def test_cli_polygon_mask_padding_and_conflicts(input_file, tmp_path, writer_spy):
    source, _ = input_file
    feature = _polygon_feature([[[1, 1], [10, 1], [1, 10], [1, 1]]])
    geojson_path = tmp_path / "geometry.json"
    geojson_path.write_text(json.dumps(feature))
    result = CliRunner().invoke(main, ["crop", str(source), "-o", str(tmp_path / "cli.ome.tif"),
                                      "--geojson", str(geojson_path), "--padding", "3",
                                      "--pad-outside", "--mask-background", "white",
                                      "--shatter", "by_index"])
    assert result.exit_code == 0, result.output
    assert len(writer_spy) == 1
    assert writer_spy[0]["pixels"][0].shape == (2, 15, 15)
    assert writer_spy[0]["pixels"][0][0, 0, 0] == 65535
    bad = CliRunner().invoke(main, ["crop", str(source), "-o", str(tmp_path / "bad.ome.tif"),
                                   "--geojson", str(geojson_path), "--clip", "--pad-outside"])
    assert bad.exit_code != 0 and "mutually exclusive" in bad.output


def test_mask_respects_nonzero_label_background_and_known_physical_origin():
    from omeify import LabelImage
    from omeify.roi_raster import geometry_crop, mask_fill_value
    label = np.full((12, 12), 19, dtype=np.uint16)
    region = parse_regions(_polygon_feature([[[2, 2], [8, 2], [2, 8], [2, 2]]]))[0]
    with LabelImage.from_array(label, pixel_size=PixelSize(.5, .7, "µm"),
                               background_label=7) as image:
        with image.with_metadata(origin_yx=(100.0, 200.0)) as anchored:
            with geometry_crop(anchored, (-2, -3, 10, 9), polygons=region.polygons,
                               mask=True, fill_value=mask_fill_value(anchored, "auto")) as roi:
                assert roi.origin_yx == (100 - 3 * .7, 200 - 2 * .5)
                pixels = roi.asarray()
                assert pixels[0, 0] == 7
                assert pixels[5, 5] == 19
                assert pixels[-1, -1] == 7


def test_exact_rectangle_skips_mask_and_preserves_rgb_jpeg_policy(tmp_path, writer_spy):
    path = tmp_path / "rgb.ome.tif"
    pixels = np.full((24, 24, 3), 120, dtype=np.uint8)
    tifffile.imwrite(path, pixels, photometric="rgb", ome=True, metadata={
        "axes": "YXS", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    report = crop(path, tmp_path / "rectangle.ome.tif",
                  geojson=_polygon_feature([_rectangle_ring(2, 3, 18, 19)]))
    assert writer_spy[0]["options"]["compression"] is None
    assert report["outputs"][0]["regions"][0]["mask_applied"] is False
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[3:19, 2:18])
    assert parse_regions(_polygon_feature([[[2, 3], [18, 19], [18, 3],
                                                [2, 19], [2, 3]]]))[0].is_integer_rectangle is False
