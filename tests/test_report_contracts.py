"""Current report contracts, resource resolution, and real installed-stack workflows."""
from __future__ import annotations

import ast
import json
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest
import tifffile
from jsonschema import Draft202012Validator

from omeify import (
    LabelImage, MultichannelImage, OMEImageSeries, OMEMultiSeriesWriter,
    OMETiffReader, OMETiffWriter, PixelSize, RGBImage, TiffInspector, convert, crop, mutate,
)
from omeify import _version
from omeify.preview import build_preview
from omeify.regions import rectangle_feature
from omeify.reports import _registry, complete_report, load_schema, validate_report
from omeify.utils.ome_schema_validator import OMESchemaValidator

SCHEMAS = tuple(sorted(p.name for p in files("omeify.schemas").iterdir()
                       if p.name.endswith(".schema.json")))
SIZE = PixelSize(.5, .7, "µm")


def walk(value):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


@pytest.mark.parametrize("name", SCHEMAS)
def test_all_packaged_schemas_and_references_resolve_locally(name):
    schema = load_schema(name)
    Draft202012Validator.check_schema(schema)
    resolver = _registry().resolver(schema["$id"])
    for node in walk(schema):
        if "$ref" in node:
            assert isinstance(resolver.lookup(node["$ref"]).contents, (dict, bool))


def test_schema_edits_do_not_change_cached_authority():
    schema = load_schema("write_report.schema.json")
    schema["properties"]["schema"]["const"] = "not-the-current-contract"
    assert load_schema("write_report.schema.json")["properties"]["schema"]["const"] == "omeify.write/1"


@pytest.fixture
def provenance():
    return {
        "coordinate_system": "level0_pixels", "shatter": None, "source_series": 0,
        "source_size": [64, 48], "crop_mode": "bounding_rectangle", "regions": [{
            "input_index": 1, "name": None, "output_name": "01 - ROI",
            "output_series": 0, "requested_bounds": [0., 0., 16., 16.],
            "bounds": [0, 0, 16, 16], "clipped": False, "source_offset_xy": [0, 0],
        }],
    }


def test_report_identifier_is_current_and_input_record_is_not_mutated(provenance):
    original = deepcopy(provenance)
    result = complete_report(provenance, "crop_provenance.schema.json")
    assert result["schema"] == "omeify.crop_provenance/1" and provenance == original
    result["schema"] = "omeify.crop/2"
    with pytest.raises(ValueError, match="const"):
        validate_report(result, "crop_provenance.schema.json")


@pytest.mark.parametrize("key", ["deidentify_ome", "significant_bits_matches_dtype",
                                "compression_ratio", "level_shapes_cyx", "shape_cyx",
                                "channels_checked", "points_per_channel"])
def test_removed_wire_fields_are_not_preserved_in_schema_properties(key):
    for name in SCHEMAS:
        for node in walk(load_schema(name)):
            assert key not in node.get("properties", {})


def test_inspection_and_preview_emit_current_json_without_writer_dependencies(tmp_path):
    data = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    source = tmp_path / "source.ome.tif"
    tifffile.imwrite(source, data, ome=True, photometric="minisblack", metadata={
        "axes": "YX", "PhysicalSizeX": .5, "PhysicalSizeY": .7,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    report = TiffInspector(source, detail=3).report
    validate_report(report, "tiff_inspection.schema.json")
    assert report["schema_version"] == "2.0"
    json.dumps(report, allow_nan=False)
    with OMETiffReader(source) as image:
        pixels, context = build_preview(image, max_size=128)
        assert pixels.dtype == np.uint8
        validate_report(context, "preview.schema.json")
        json.dumps(context, allow_nan=False)


@pytest.mark.parametrize("name", ["elsewhere", "OMEIFY", ""])
def test_version_rejects_a_neighboring_unrelated_project(tmp_path, monkeypatch, name):
    root = tmp_path / "tree"
    package = root / "omeify"
    package.mkdir(parents=True)
    monkeypatch.setattr(_version, "__file__", str(package / "_version.py"))
    (root / "pyproject.toml").write_text(f'[project]\nname = "{name}"\nversion = "99.0"\n')
    assert _version._source_tree_version() is None


def test_version_reads_matching_source_or_returns_no_source_value(tmp_path, monkeypatch):
    package = tmp_path / "omeify"
    package.mkdir()
    monkeypatch.setattr(_version, "__file__", str(package / "_version.py"))
    assert _version._source_tree_version() is None
    path = tmp_path / "pyproject.toml"
    path.write_text('[project]\nname = "omeify"\nversion = "8.2.1"\n')
    assert _version._source_tree_version() == "8.2.1"
    path.write_text("not TOML")
    assert _version._source_tree_version() is None


def test_supported_python_minimum_and_syntax_are_preserved():
    # Syntax parsing cannot replace tests running on Python 3.10 itself.
    root = Path(__file__).resolve().parents[1]
    project = _version.tomllib.loads((root / "pyproject.toml").read_text())
    assert project["project"]["requires-python"] == ">=3.10"
    assert project["tool"]["ruff"]["target-version"] == "py310"
    dependencies = project["project"]["dependencies"]
    assert any(d.startswith("tomli") for d in dependencies)
    for directory in ("omeify", "tests", "examples"):
        for path in (root / directory).rglob("*.py"):
            ast.parse(path.read_text(), filename=str(path), feature_version=(3, 10))


def test_missing_xsd_is_an_error_not_a_skipped_validation(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "omeschema", None)
    with pytest.raises(RuntimeError, match="requires the ome-schema dependency"):
        OMESchemaValidator()


@pytest.fixture
def real_stack():
    pytest.importorskip("imagecodecs", reason="Real installed codecs required for export integration")
    pytest.importorskip("omeschema", reason="Real local OME XSD required for export integration")
    return OMESchemaValidator()


@pytest.mark.parametrize("kind", ["multichannel", "rgb", "label"])
@pytest.mark.parametrize("compression", ["Uncompressed", "Deflate", "LZW"])
def test_real_single_and_multi_writer_reports_and_full_small_rasters(tmp_path, real_stack,
                                                                    kind, compression):
    shape = (2, 33, 49) if kind == "multichannel" else (33, 49, 3) if kind == "rgb" else (33, 49)
    values = np.arange(np.prod(shape)).reshape(shape).astype(np.uint8 if kind == "rgb" else np.uint16)
    factory = {"multichannel": MultichannelImage, "rgb": RGBImage, "label": LabelImage}[kind]
    kwargs = {"axes": "CYX"} if kind == "multichannel" else {}
    with factory.from_array(values, pixel_size=SIZE, **kwargs) as image:
        output = tmp_path / "single.ome.tif"
        report = OMETiffWriter(output, compression=compression, tile_size=16,
                               pyramid_levels=2).write(image)
        validate_report(report, "write_report.schema.json")
        assert report["options"]["metadata_minimization"] is True
        json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(tifffile.imread(output), values)
        multi = tmp_path / "multi.ome.tif"
        report = OMEMultiSeriesWriter(multi, compression=compression, tile_size=16,
                                      pyramid_levels=2).write([OMEImageSeries("Sample", image)])
        validate_report(report, "multi_series_report.schema.json")
        json.dumps(report, allow_nan=False)
        np.testing.assert_array_equal(tifffile.imread(multi), values)


@pytest.mark.parametrize("operation", ["convert", "integer", "precision", "crop", "shatter"])
def test_real_workflow_reports_and_pixels(tmp_path, real_stack, operation):
    source, output = tmp_path / "source.ome.tif", tmp_path / "output.ome.tif"
    values = np.arange(33 * 49, dtype=np.float32).reshape(33, 49) / 4
    tifffile.imwrite(source, values, ome=True, photometric="minisblack", metadata={
        "axes": "YX", "PhysicalSizeX": .5, "PhysicalSizeY": .7,
        "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
    })
    settings = {"tile_size": 16, "pyramid_levels": 1, "compression": "Deflate"}
    if operation == "convert":
        report = convert(source, output, input_type="ome_tiff", **settings)
        schema = "conversion_report.schema.json"
        np.testing.assert_array_equal(tifffile.imread(output), values)
    elif operation in {"integer", "precision"}:
        options = {"dtype": "uint16", "range_mode": "preserve"} if operation == "integer" else {
            "float32_mantissa_bits": 11,
        }
        report = mutate(source, output, input_type="ome_tiff", **settings, **options)
        schema = "mutation_report.schema.json"
        if operation == "integer":
            np.testing.assert_array_equal(tifffile.imread(output), np.rint(values).astype(np.uint16))
        else:
            # All these quarter-integers are exactly representable with 11 stored fraction bits.
            np.testing.assert_array_equal(tifffile.imread(output), values)
    else:
        report = crop(source, output, geojson={"type": "FeatureCollection", "features": [
            rectangle_feature([1, 2, 17, 18], name="µm region"),
            rectangle_feature([20, 5, 36, 21], name="second"),
        ]}, shatter="by_index" if operation == "shatter" else None, **settings)
        schema = "crop_report.schema.json"
        assert report["schema"] == "omeify.crop/3"
        for product in report["outputs"]:
            for region in product["regions"]:
                x0, y0, x1, y1 = region["bounds"]
                with OMETiffReader(product["path"], series=region["output_series"]) as image:
                    np.testing.assert_array_equal(image.asarray(), values[y0:y1, x0:x1])
    validate_report(report, schema)
    json.dumps(report, allow_nan=False)
    np.testing.assert_array_equal(tifffile.imread(source), values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("target", ["uint8", "uint16"])
@pytest.mark.parametrize("case,mode", [("zero", "auto"), ("constant", "preserve"),
                                      ("fractional", "auto"), ("negative", "full"),
                                      ("wide", "full")])
def test_real_mutation_analysis_records_match_the_packaged_contract(dtype, target, case, mode):
    from omeify.dtype_mutation import analyze_dtype_mutation

    values = np.arange(1024, dtype=dtype).reshape(32, 32)
    if case == "zero":
        values[:] = 0
    elif case == "constant":
        values[:] = 7.25
    elif case == "fractional":
        values /= 37
    elif case == "negative":
        values -= 700
    elif case == "wide":
        values *= 4096
    with MultichannelImage.from_array(values, axes="YX", pixel_size=SIZE) as image:
        plans = analyze_dtype_mutation(image, dtype=target, range_mode=mode,
                                       sample_pixels_per_channel=1024)
    schema = load_schema("mutation_report.schema.json")
    validator = Draft202012Validator({"$ref": schema["$id"] +
        "#/$defs/dtype_mutation/properties/channels/items"}, registry=_registry())
    for plan in plans:
        validator.validate(plan.report)
        json.dumps(plan.report, allow_nan=False)


def test_explicit_local_xsd_path_still_works(tmp_path):
    # A tiny custom XSD tests the explicit-path API, not OME conformance.
    path = tmp_path / "custom.xsd"
    path.write_text('<xs:schema xmlns:xs="http://www.w3.org/2001/XMLSchema">'
                    '<xs:element name="sample" type="xs:string"/></xs:schema>')
    validator = OMESchemaValidator(schema_location=path)
    assert validator.validate("<sample>µm</sample>") is True
    assert validator.validate("<unknown/>") is False
