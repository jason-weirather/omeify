"""Stored metadata contracts and inspection scope, using real small image stores."""
from __future__ import annotations

import copy
import json
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import zarr
from click.testing import CliRunner
from lxml import etree

from omeify import (
    MultichannelImage, OMEImageSeries, OMEZarrCollectionWriter, OMEZarrReader,
    PixelSize, ZarrInspector,
)
from omeify.cli import main
from omeify.io.ome_zarr.metadata import output_attributes, profile_errors
from omeify.io.ome_zarr.store import location
from omeify.reports import report_validator, validate_report
from omeify.utils import ome_geometry
from zarr_fixture import fixture

SIZE = PixelSize(.5, .7, "µm")
NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


@pytest.fixture
def canonical_attributes():
    with MultichannelImage.from_array(
        np.zeros((2, 16, 32), dtype=np.uint16), axes="CYX", channel_names=("A", "B"),
        pixel_size=SIZE,
    ) as base, base.with_metadata(origin_yx=(5, 7)) as image:
        return output_attributes(image._metadata, image.levels, method="mean",
                                 windows=[dict(min=0, max=65535, start=0, end=65535)] * 2)


@pytest.mark.parametrize("fault", [
    "duplicate_axes", "wrong_order", "missing_c", "translation_only", "reversed_transforms",
    "duplicate_scale", "rank_mismatch", "channel_scale", "channel_shift", "extra_root",
    "extra_dataset", "extra_channel", "method_disagreement", "mixed_extension", "invalid_version",
])
def test_canonical_image_schema_rejects_unsupported_structures(canonical_attributes, fault):
    a = canonical_attributes
    ms = a["ome"]["multiscales"][0]
    dataset = ms["datasets"][0]
    transforms = dataset["coordinateTransformations"]
    if fault == "duplicate_axes":
        ms["axes"][1]["name"] = "x"
    elif fault == "wrong_order":
        ms["axes"].reverse()
    elif fault == "missing_c":
        ms["axes"].pop(0)
    elif fault == "translation_only":
        transforms.pop(0)
    elif fault == "reversed_transforms":
        transforms.reverse()
    elif fault == "duplicate_scale":
        transforms[1] = copy.deepcopy(transforms[0])
    elif fault == "rank_mismatch":
        transforms[0]["scale"].pop()
    elif fault == "channel_scale":
        transforms[0]["scale"][0] = 2
    elif fault == "channel_shift":
        transforms[1]["translation"][0] = 1
    elif fault == "extra_root":
        a["patient"] = "FORBIDDEN_TEST_VALUE"
    elif fault == "extra_dataset":
        dataset["patient"] = "FORBIDDEN_TEST_VALUE"
    elif fault == "extra_channel":
        a["ome"]["omero"]["channels"][0]["patient"] = "FORBIDDEN_TEST_VALUE"
    elif fault == "method_disagreement":
        ms["metadata"]["method"] = "omeify.nearest_2x"
    elif fault == "mixed_extension":
        a["omeify"] = {"image_type": "label", "background_label": 0,
                       "channel_name": "A", "float32_mantissa_bits": 11}
    else:
        a["ome"]["version"] = "0.6"
    assert not report_validator("ome_zarr_profile.schema.json").is_valid(a)
    findings = profile_errors(a)
    assert findings
    assert "FORBIDDEN_TEST_VALUE" not in " ".join(findings)


@pytest.mark.parametrize("fault", ["paths", "windows", "nan", "infinity"])
def test_profile_checks_value_relationships_after_schema(canonical_attributes, fault):
    a = canonical_attributes
    if fault == "paths":
        a["ome"]["multiscales"][0]["datasets"][0]["path"] = "7"
    elif fault == "windows":
        a["ome"]["omero"]["channels"][0]["window"]["end"] = -1
    else:
        a["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][1] = (
            float("nan") if fault == "nan" else float("inf")
        )
    assert profile_errors(a)


def test_valid_attributes_and_geometry_use_packaged_contracts(canonical_attributes):
    assert profile_errors(canonical_attributes) == ()
    root = etree.Element(f"{{{NS}}}OME")
    image = etree.SubElement(root, f"{{{NS}}}Image", ID="Image:0")
    spec = SimpleNamespace(origin_yx=(7., -8.), pyramid_method="nearest")
    ome_geometry.attach_geometry(root, [image], [spec])
    record = ome_geometry.read_geometry(root, image)
    validate_report(record, "ome_geometry.schema.json")
    assert record == {"schema": "omeify.geometry/1", "origin_yx_um": [7, -8],
                      "pyramid_method": "nearest"}
    assert "PositionX" not in etree.tostring(root).decode()


@pytest.mark.parametrize("payload", [
    '{"schema":"omeify.geometry/1","origin_yx_um":[0,0],"pyramid_method":"mean","extra":"private"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[true,0],"pyramid_method":"mean"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[NaN,0],"pyramid_method":"mean"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[1e999,0],"pyramid_method":"mean"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[0,0],"pyramid_method":"mean","pyramid_method":"nearest"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[0],"pyramid_method":"mean"}',
    '{"schema":"omeify.geometry/1","origin_yx_um":[0,0],"pyramid_method":"affine"}',
])
def test_linked_geometry_rejects_ambiguous_or_out_of_contract_json(payload):
    root = etree.Element(f"{{{NS}}}OME")
    image = etree.SubElement(root, f"{{{NS}}}Image", ID="Image:0")
    ome_geometry.attach_geometry(root, [image], [SimpleNamespace(origin_yx=(0., 0.), pyramid_method="mean")])
    root.find("{*}StructuredAnnotations/{*}MapAnnotation/{*}Value/{*}M").text = payload
    with pytest.raises(ValueError):
        ome_geometry.read_geometry(root, image)


def test_geometry_checks_generation_and_encoded_byte_limit():
    with pytest.raises(ValueError):
        ome_geometry.record_for_spec(SimpleNamespace(origin_yx=(0, 0), pyramid_method="private"))
    root = etree.Element(f"{{{NS}}}OME")
    image = etree.SubElement(root, f"{{{NS}}}Image", ID="Image:0")
    ome_geometry.attach_geometry(root, [image], [SimpleNamespace(origin_yx=(0, 0), pyramid_method="mean")])
    root.find("{*}StructuredAnnotations/{*}MapAnnotation/{*}Value/{*}M").text = "µ" * 2100
    with pytest.raises(ValueError, match="bound"):
        ome_geometry.read_geometry(root, image)


@pytest.mark.parametrize("target", ["root", "manifest"])
def test_stored_collection_contracts_validate_before_publication(tmp_path, monkeypatch, target):
    real_open = zarr.open_group
    injected = False

    def corrupted_open(*args, **kwargs):
        nonlocal injected
        path = Path(str(args[0])) if args else None
        if (not injected and path is not None and kwargs.get("mode") == "r"
                and path.parent == tmp_path and path.name.startswith(".omeify-zarr-")):
            # Corrupt real stored metadata after construction, before final readback.
            metadata_path = path / ("OME/zarr.json" if target == "manifest" else "zarr.json")
            metadata = json.loads(metadata_path.read_text())
            metadata["attributes"]["private"] = "not-permitted"
            metadata_path.write_text(json.dumps(metadata))
            injected = True
        return real_open(*args, **kwargs)

    monkeypatch.setattr(zarr, "open_group", corrupted_open)
    output = tmp_path / "collection"
    with MultichannelImage.from_array(np.zeros((16, 16), dtype="uint16"), axes="YX", pixel_size=SIZE) as image:
        with pytest.raises(ValueError, match="schema.json"):
            OMEZarrCollectionWriter(output, tile_size=16).write([OMEImageSeries("A", image)])
        assert image.is_open
    assert injected
    assert not output.exists()
    assert not list(tmp_path.glob(".omeify-zarr-*"))


@pytest.mark.parametrize("paths", [["0", "0"], ["../private"], ["01"], ["0", 1], []])
def test_manifest_profile_rejects_invalid_keys(paths):
    with pytest.raises(ValueError):
        validate_report({"ome": {"version": "0.5", "series": paths}}, "ome_zarr_manifest.schema.json")


def make_collection(path):
    with MultichannelImage.from_array(np.zeros((16, 16), dtype="uint16"), axes="YX", pixel_size=SIZE) as image:
        OMEZarrCollectionWriter(path, tile_size=16).write(
            [OMEImageSeries("A", image), OMEImageSeries("B", image)],
            provenance={"explicit_note": "REVIEW_COLLECTION_ROOT"},
        )
    manifest = zarr.open_group(path / "OME", mode="a")
    manifest.attrs["foreign_note"] = "REVIEW_MANIFEST"
    manifest.store.close()


def test_inspection_exposes_consulted_root_and_manifest_without_pixel_reads(tmp_path, monkeypatch):
    path = tmp_path / "collection"
    make_collection(path)

    def no_pixels(*args, **kwargs):
        raise AssertionError("Metadata-only inspection decoded pixels")

    monkeypatch.setattr(zarr.Array, "__getitem__", no_pixels)
    inspector = ZarrInspector(path, detail=3, max_text_length=None)
    assert not inspector.validation_errors()
    report = inspector.report
    assert report["schema"] == "omeify.zarr_inspect/2"
    assert {n["group"] for n in report["metadata_nodes"]} == {"", "OME", "0", "1"}
    assert "REVIEW_COLLECTION_ROOT" in inspector.to_json()
    assert "REVIEW_MANIFEST" in inspector.render_text()
    assert report["profile"]["whole_store_audited"] is False
    assert report["raster_pixels_examined"] is False
    assert "Additional discovery metadata" in str(ZarrInspector(path))
    with OMEZarrReader(path, series=1) as image:
        snapshot = image.inspect()
    assert "REVIEW_COLLECTION_ROOT" in snapshot.to_json()
    assert not snapshot.validation_errors()
    cli = CliRunner().invoke(main, ["inspect", str(path), "--json"])
    assert cli.exit_code == 0, cli.output
    assert json.loads(cli.output)["metadata_nodes"] == report["metadata_nodes"]
    # Explicit subgroup inspection is not advertised as a review of its ancestors.
    child = ZarrInspector(path, group="1").report
    assert [n["group"] for n in child["metadata_nodes"]] == ["1"]


def test_strict_normalized_records_do_not_restrict_raw_foreign_metadata(tmp_path):
    fixture(tmp_path / "source", levels=1)
    report = ZarrInspector(tmp_path / "source").report
    schema = report_validator("zarr_inspection.schema.json")
    assert schema.is_valid(report)
    raw = copy.deepcopy(report)
    raw["attributes"]["arbitrary"] = {"patient": "REVIEW_RAW"}
    raw["arrays"][0]["unfamiliar_codec_metadata"] = {"value": True}
    assert schema.is_valid(raw)
    for section, key, value in [("image", "dtype", "nonsense"), ("image", "shape", [1]),
                                ("image", "unexpected", "private")]:
        invalid = copy.deepcopy(report)
        invalid[section][key] = value
        assert not schema.is_valid(invalid)
    invalid = copy.deepcopy(report)
    invalid["series"][0]["name"] = {"patient": "not-a-name"}
    assert not schema.is_valid(invalid)


def test_malformed_source_name_is_diagnostic_not_blindly_stringified(tmp_path):
    path = tmp_path / "source"
    fixture(path, levels=1)
    root = zarr.open_group(path, mode="a")
    ome = root.attrs["ome"]
    ome["multiscales"][0]["name"] = {"patient": "REVIEW_RAW"}
    root.attrs["ome"] = ome
    root.store.close()
    with pytest.raises(ValueError, match="name must be"):
        OMEZarrReader(path).open()
    report = ZarrInspector(path).report
    assert report["image"] is None and report["diagnostics"]
    assert "REVIEW_RAW" in json.dumps(report["metadata_nodes"])
    validate_report(report, "zarr_inspection.schema.json")


@pytest.mark.parametrize("url", ["https://name:secret@example.org/store",
                                  "https://example.org/store?token=secret"])
def test_unsupported_authentication_is_not_silently_rewritten(url):
    with pytest.raises(ValueError):
        location(url)


def test_s3_store_root_is_supported_without_embedding_credentials():
    # Native S3 input is supported in 0.22, unlike authenticated URL strings.
    assert location("s3://bucket/path/") == "s3://bucket/path"
    with pytest.raises(ValueError):
        location("s3://bucket/../other")


def test_schema_inventory_is_documented():
    root = Path(__file__).resolve().parents[1]
    inventory = (root / "docs/reports.md").read_text(encoding="utf-8")
    for resource in files("omeify.schemas").iterdir():
        if resource.name.endswith(".schema.json"):
            assert f"`{resource.name}`" in inventory
