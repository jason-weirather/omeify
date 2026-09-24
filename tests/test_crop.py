"""Crop geometry, lazy reads, batch preflight, and real optional TIFF readback."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

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
                          "sizes": [s.image.pixel_size for s in entries]})
            return {"test_double": True}

    monkeypatch.setattr("omeify.cropping.OMEMultiSeriesWriter", Writer)
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
    assert report["schema"] == "omeify.crop/2" and report["shatter"] == shatter
    assert report["output_path"] == str(output) and "naming" not in report
    assert len(writer_spy) == len(report["outputs"]) == len(groups)
    if shatter is None:
        assert writer_spy[0]["path"] == output
    for call, result, indices in zip(writer_spy, report["outputs"], groups):
        assert call["names"] == [f"{i + 1:02d} - {names[i] or 'ROI'}" for i in indices]
        assert call["sizes"] == [PixelSize(.5, .7, "µm")] * len(indices)
        assert call["options"]["compression"] is None and call["options"]["tile_size"] is None
        assert call["provenance"]["schema"] == "omeify.crop/2"
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
def test_cli_stdin_and_coordinate_guard(input_file, tmp_path, writer_spy, shatter, filenames):
    path, _ = input_file
    doc = collection(*[rectangle_feature([1, 2, 10, 11], name=n) for n in ("right", "left", "right")])
    args = ["crop", str(path), "-o", str(tmp_path / "out.ome.tiff"), "--geojson", "-"]
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


def test_real_crop_tiff_readback(input_file, tmp_path):
    pytest.importorskip("omeschema", reason="Real OME schema validation dependency is required")
    path, a = input_file
    report = crop(path, tmp_path / "real", bounds=(2, 3, 40, 30), compression="Deflate", tile_size=16)
    output = Path(report["outputs"][0]["path"])
    assert output == tmp_path / "real"  # No implicit extension or index, even for bounds.
    with OMETiffReader(output) as actual:
        np.testing.assert_array_equal(actual.asarray(), a[:, 3:30, 2:40])
        assert actual.pixel_size == PixelSize(.5, .7, "µm")
        assert len(actual.levels) > 1


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
    pytest.importorskip("omeschema", reason="Real OME schema validation dependency is required")
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
