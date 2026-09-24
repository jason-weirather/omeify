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
    doc = collection(a, b, c)
    regions = parse_regions(doc)
    groups = group_outputs(regions, tmp_path / "slide")
    assert [p.name for p, _ in groups] == ["slide-right-tissue.ome.tiff", "slide-03.ome.tiff"]
    assert [len(r) for _, r in groups] == [2, 1]
    assert pixel_bounds(regions[0].bounds, 100, 100) == (1, 2, 10, 11)
    assert [p.name for p, _ in group_outputs(regions, "slide.ome.tiff", naming="index")] == [
        f"slide-{i:02d}.ome.tiff" for i in range(1, 4)]
    many = parse_regions(collection(*[c] * 101))
    assert group_outputs(many, "slide")[0][0].name == "slide-001.ome.tiff"
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
    for names in [("a/b", "a b"), ("A", "a"), ("../../", "safe"), ("01", "")]:
        regions = parse_regions(collection(*[rectangle_feature([0, 0, 1, 1], name=n) for n in names]))
        # The unnamed member is 02, so use 02 to exercise the name/index namespace collision.
        if names == ("01", ""):
            regions = parse_regions(collection(rectangle_feature([0, 0, 1, 1], name="02"),
                                                rectangle_feature([0, 0, 1, 1], name="")))
        with pytest.raises(ValueError):
            group_outputs(regions, "out")
        assert len(group_outputs(regions, "out", naming="index")) == 2


def test_grouped_exports_use_real_lazy_crops(input_file, tmp_path, writer_spy):
    path, pixels = input_file
    doc = collection(rectangle_feature([2, 3, 14, 19], name="right"),
                     rectangle_feature([25, 10, 45, 25], name="right"))
    report = crop(path, tmp_path / "out", geojson=doc)
    assert len(writer_spy) == 1 and len(report["outputs"]) == 1
    np.testing.assert_array_equal(writer_spy[0]["pixels"][0], pixels[:, 3:19, 2:14])
    np.testing.assert_array_equal(writer_spy[0]["pixels"][1], pixels[:, 10:25, 25:45])
    assert writer_spy[0]["sizes"] == [PixelSize(.5, .7, "µm")] * 2
    assert writer_spy[0]["provenance"]["regions"][1]["source_offset_xy"] == [25, 10]
    assert writer_spy[0]["options"]["compression"] is None


def test_whole_batch_preflight_and_inputs_are_protected(input_file, tmp_path, writer_spy):
    path, _ = input_file
    doc = collection(rectangle_feature([1, 1, 10, 10], name="a"),
                     rectangle_feature([2, 2, 15, 15], name="b"))
    later = tmp_path / "out-b.ome.tiff"
    later.write_text("keep")
    with pytest.raises(FileExistsError):
        crop(path, tmp_path / "out", geojson=doc, overwrite=False)
    assert not writer_spy and later.read_text() == "keep"
    later.unlink()
    os.link(path, later)
    with pytest.raises(ValueError, match="replace"):
        crop(path, tmp_path / "out", geojson=doc)
    assert not writer_spy
    with pytest.raises(ValueError):
        crop(path, tmp_path / "out", bounds=(10, 10, 3, 3))
    with pytest.raises(ValueError):
        crop(path, tmp_path / "out", geojson=collection(
            doc["features"][0], rectangle_feature([0, 0, 100, 100], name="b")))
    assert not writer_spy


def test_cli_stdin_and_coordinate_guard(input_file, tmp_path, writer_spy):
    path, _ = input_file
    doc = collection(rectangle_feature([1, 2, 10, 11], name="right"))
    result = CliRunner().invoke(main, ["crop", str(path), "-o", str(tmp_path / "out"),
                                     "--geojson", "-"], input=json.dumps(doc))
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.stdout)["outputs"]) == 1
    doc["omeify"] = {"coordinate_system": "level0_pixels", "image_size": [128, 48], "series": 0}
    with pytest.raises(ValueError, match="dimensions or series"):
        crop(path, tmp_path / "out", geojson=doc)
    assert len(writer_spy) == 1


def test_real_crop_tiff_readback(input_file, tmp_path):
    pytest.importorskip("omeschema", reason="Real OME schema validation dependency is required")
    path, a = input_file
    report = crop(path, tmp_path / "real", bounds=(2, 3, 40, 30), compression="Deflate", tile_size=16)
    output = Path(report["outputs"][0]["path"])
    with OMETiffReader(output) as actual:
        np.testing.assert_array_equal(actual.asarray(), a[:, 3:30, 2:40])
        assert actual.pixel_size == PixelSize(.5, .7, "µm")
        assert len(actual.levels) > 1
