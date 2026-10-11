"""Adversarial source text must not become output metadata by accident.

These are synthetic I/O fixtures, not claims of vendor-file certification or PHI
classification. Retained channel/ROI names and explicit provenance remain visible.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import get_args

import numpy as np
import pytest
import tifffile
import zarr
from click.testing import CliRunner

from omeify import (
    AkoyaFusionQPTiffReader, AkoyaMIFQPTiffReader,
    OMEZarrReader, OMEZarrWriter,
    PixelSize, RGBImage, ZarrInspector, convert, crop, mutate,
)
from omeify.cli import main
from omeify.io.akoya_qptiff import parse_akoya_qpi_channel_metadata
from omeify.io.image_planes import ImagePlaneSource
from omeify.io.ome_zarr.metadata import output_attributes, profile_errors
from omeify.io.source_reader import INPUT_TYPES, PLANAR_INPUT_TYPES, InputType, source_reader
from omeify.provenance import canonical_provenance
from omeify.utils.generate_ome_xml import generate_ome_xml
from zarr_fixture import fixture

PRIVATE = "REVIEW_FORBIDDEN_METADATA_63820"
RETAINED = "REVIEW_RETAINED_CHANNEL_52719"
SIZE = PixelSize(.5, .5, "µm")


def qpi_xml(body: str, *, namespace: bool = False) -> str:
    ns = ' xmlns="urn:synthetic-qpi"' if namespace else ''
    return (
        f'<PerkinElmer-QPI-ImageDescription{ns}>{body}'
        '<ScanProfile><root><ScanResolution><PixelSizeMicrons>0.5</PixelSizeMicrons>'
        '</ScanResolution></root></ScanProfile></PerkinElmer-QPI-ImageDescription>'
    )


def write_qpi(path: Path, body: str) -> np.ndarray:
    pixels = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    tifffile.imwrite(path, pixels, description=qpi_xml(body), tile=(16, 16),
                     software="PerkinElmer-QPI synthetic", metadata=None)
    return pixels


def metadata_text(path: Path) -> str:
    if path.is_dir():
        return "\n".join(p.read_text() for p in sorted(path.rglob("zarr.json")))
    with tifffile.TiffFile(path) as tif:
        values = []
        for series in tif.series:
            for level in series.levels:
                for page in level.pages:
                    values.extend(str(tag.value) for tag in page.aspage().tags.values())
        return "\n".join(values)


@pytest.mark.parametrize("module", ["omeify", "omeify.io", "omeify.io.vendor_tiff_readers"])
def test_component_reader_is_removed_without_alias(module):
    assert not hasattr(importlib.import_module(module), "AkoyaComponentTiffReader")
    assert "component" not in INPUT_TYPES
    assert "component" not in PLANAR_INPUT_TYPES
    assert "component" not in get_args(InputType)
    with pytest.raises(ValueError, match="Unsupported input_type"):
        source_reader("not-opened.tif", input_type="component")


@pytest.mark.parametrize("command", ["convert", "mutate", "crop"])
def test_component_cli_profile_is_rejected_before_io(tmp_path, command):
    source = tmp_path / "retired.tif"
    source.touch()
    args = [command, str(source), "-o", str(tmp_path / "out"), "--type", "component"]
    if command == "crop":
        args += ["--bounds", "0", "0", "16", "16"]
    result = CliRunner().invoke(main, args)
    assert result.exit_code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("reader", [AkoyaMIFQPTiffReader, AkoyaFusionQPTiffReader])
@pytest.mark.parametrize("direct", [False, True])
def test_qpi_reader_does_not_promote_nested_private_names(tmp_path, reader, direct):
    body = (f'<Name>{RETAINED}</Name>' if direct else '')
    body += f'<Patient><Name>{PRIVATE}</Name><Biomarker>{PRIVATE}</Biomarker></Patient>'
    body += f'<ext:Name xmlns:ext="urn:foreign">{PRIVATE}</ext:Name>'
    source = tmp_path / "source.qptiff"
    expected = write_qpi(source, body)
    with reader(source) as image:
        assert image.channel_names == ((RETAINED if direct else "Channel 1"),)
        xml = generate_ome_xml(ImagePlaneSource(image).output_spec(), [image.shape])["xml_string"]
        assert PRIVATE not in xml
        assert "Patient" not in xml
        target = tmp_path / "normalized.ome.zarr"
        OMEZarrWriter(target, tile_size=16, pyramid_levels=0).write(image)
    assert PRIVATE not in metadata_text(target)
    with OMEZarrReader(target) as image:
        np.testing.assert_array_equal(image[0].asarray(), expected)


@pytest.mark.parametrize("namespace", [False, True])
def test_qpi_scalar_fields_require_exact_paths_and_vocabulary(namespace):
    body = (f'<Patient><Name>{PRIVATE}</Name><PixelSizeMicrons>99</PixelSizeMicrons></Patient>'
            f'<ext:Name xmlns:ext="urn:foreign">{PRIVATE}</ext:Name>')
    parsed = parse_akoya_qpi_channel_metadata(qpi_xml(body, namespace=namespace))
    assert parsed.name is None and parsed.biomarker is None
    assert parsed.pixel_size_microns == .5


@pytest.mark.parametrize("body", [
    "<Name>A</Name><Name>B</Name>",
    "<Biomarker>A</Biomarker><Biomarker>B</Biomarker>",
    f"<Name>{RETAINED}<Patient>{PRIVATE}</Patient></Name>",
])
def test_ambiguous_or_structured_qpi_fields_fail(body):
    with pytest.raises(ValueError):
        parse_akoya_qpi_channel_metadata(qpi_xml(body))


def test_foreign_root_does_not_supply_qpi_names():
    with pytest.raises(ValueError, match="PerkinElmer-QPI-ImageDescription"):
        parse_akoya_qpi_channel_metadata(f"<Component><Name>{PRIVATE}</Name></Component>")


@pytest.mark.parametrize("output_format", ["ome_tiff", "ome_zarr"])
@pytest.mark.parametrize("input_type", ["qptiff_mif", "qptiff_fusion", "ome_tiff"])
def test_vendor_and_ome_conversion_retention_boundary(tmp_path, input_type, output_format):
    source = tmp_path / f"{PRIVATE}.tif"
    if input_type.startswith("qptiff"):
        data = write_qpi(source, f'<Name>{RETAINED}</Name><Patient><Name>{PRIVATE}</Name></Patient>')
    else:
        data = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
        tifffile.imwrite(source, data, ome=True, tile=(16, 16), software=PRIVATE,
                         metadata={"axes": "YX", "Name": PRIVATE, "Description": PRIVATE,
                                   "PhysicalSizeX": .5, "PhysicalSizeY": .5,
                                   "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
                                   "Channel": {"Name": [RETAINED]}})
    target = tmp_path / ("out.ome.zarr" if output_format == "ome_zarr" else "out.ome.tif")
    options = {"compression": "Uncompressed"} if output_format == "ome_tiff" else {}
    report = convert(source, target, input_type=input_type, output_format=output_format,
                     tile_size=16, pyramid_levels=0, **options)
    assert PRIVATE not in metadata_text(target)
    assert RETAINED in metadata_text(target)
    # Diagnostic provenance is deliberately NOT the minimized image product.
    assert PRIVATE in json.dumps(report)


@pytest.mark.parametrize("version", ["0.4", "0.5", "0.6"])
@pytest.mark.parametrize("output_format", ["ome_tiff", "ome_zarr"])
@pytest.mark.parametrize("operation", ["convert", "mutate"])
def test_foreign_ngff_metadata_is_not_copied_across_workflows(
    tmp_path, version, output_format, operation,
):
    source = tmp_path / f"source-{PRIVATE}"
    data = np.arange(3 * 33 * 47, dtype=np.float32).reshape(3, 33, 47)
    fixture(source, data=data, version=version, levels=1)
    root = zarr.open_group(source, mode="a")
    attrs = root.attrs.asdict()
    ome = attrs if version == "0.4" else attrs["ome"]
    ome["multiscales"][0]["name"] = PRIVATE
    ome["omero"]["channels"][0].update(label=RETAINED, patient_name=PRIVATE)
    attrs.update(patient_name=PRIVATE, omeify={"unrecognized_note": PRIVATE})
    root.attrs.update(attrs)
    root["0"].attrs["scanner_free_text"] = PRIVATE
    root.create_group("unrelated", attributes={"patient": PRIVATE})
    root.store.close()
    (source / "notes.txt").write_text(PRIVATE)
    target = tmp_path / ("out.ome.zarr" if output_format == "ome_zarr" else "out.ome.tif")
    options = {"compression": "Uncompressed"} if output_format == "ome_tiff" else {}
    if operation == "mutate":
        options.update(dtype="uint16", range_mode="preserve")
    report = {"convert": convert, "mutate": mutate}[operation](
        source, target, input_type="ome_zarr", output_format=output_format,
        tile_size=16, pyramid_levels=0, **options,
    )
    text = metadata_text(target)
    assert PRIVATE not in text
    assert RETAINED in text
    assert PRIVATE in json.dumps(report)
    if output_format == "ome_zarr":
        assert not (target / "unrelated").exists()
        assert not (target / "notes.txt").exists()
        with OMEZarrReader(target) as image:
            np.testing.assert_array_equal(image.asarray(), data)
            assert image.dtype == np.dtype("uint16" if operation == "mutate" else "float32")


def test_crop_keeps_only_deliberate_roi_context(tmp_path):
    source = tmp_path / "source"
    fixture(source, levels=1)
    name = "REVIEW_EXPLICIT_ROI_NAME"
    geojson = {"type": "Feature", "properties": {"name": name, "patient": PRIVATE},
               "geometry": {"type": "Polygon", "coordinates": [[[0, 0], [20, 0],
                            [20, 18], [0, 18], [0, 0]]]}}
    target = tmp_path / "rois"
    crop(source, target, input_type="ome_zarr", output_format="ome_zarr",
         geojson=geojson, tile_size=16, pyramid_levels=0)
    text = metadata_text(target)
    assert PRIVATE not in text and name in text
    report = ZarrInspector(target).report
    root = next(n["attributes"] for n in report["metadata_nodes"] if n["group"] == "")
    assert root["omeify"]["provenance"]["regions"][0]["name"] == name
    assert report["profile"]["whole_store_audited"] is False


@pytest.mark.parametrize("bad", [
    {"nested": {1: "not-a-string-key"}}, {"bad": float("nan")},
    {"bad": float("inf")}, {"bad": object()},
])
def test_explicit_provenance_rejects_coercion_and_non_json_values(bad):
    with pytest.raises((TypeError, ValueError)):
        canonical_provenance(bad)


def test_explicit_provenance_is_bounded_and_snapshotted():
    original = {"reviewed_name": [RETAINED]}
    encoded, value = canonical_provenance(original)
    original["reviewed_name"].append(PRIVATE)
    assert PRIVATE not in encoded and PRIVATE not in str(value)
    cycle = {}; cycle["self"] = cycle
    for bad in (cycle, {"text": "x" * (1024 * 1024)}):
        with pytest.raises(ValueError):
            canonical_provenance(bad)
    with pytest.raises(ValueError, match="crop_provenance.schema.json"):
        canonical_provenance({"schema": "omeify.crop_provenance/1", "patient": PRIVATE})


def test_icc_is_an_explicit_opaque_retention_exception_not_a_privacy_check():
    # This intentionally is NOT a valid profile. There is no implied ICC certification.
    payload = b"EXPLICIT_OPAQUE_PROFILE_WITH_REVIEW_TEXT"
    with RGBImage.from_array(np.zeros((16, 16, 3), dtype=np.uint8),
                             pixel_size=SIZE, icc_profile=payload) as image:
        assert ImagePlaneSource(image).output_spec().icc_profile == payload
        attrs = output_attributes(image._metadata, image.levels, method="mean",
                                  windows=[dict(min=0, max=255, start=0, end=255)] * 3)
    import base64
    assert base64.b64decode(attrs["omeify"]["icc_profile_base64"]) == payload
    assert profile_errors(attrs) == ()


@pytest.mark.parametrize("reader", [AkoyaMIFQPTiffReader, AkoyaFusionQPTiffReader])
def test_qpi_calibration_is_checked_across_all_channel_pages(tmp_path, reader):
    source = tmp_path / "conflicting.qptiff"
    with tifffile.TiffWriter(source, bigtiff=True) as writer:
        for index, size in enumerate((.5, .7)):
            description = qpi_xml(f"<Name>Channel {index}</Name>").replace(
                "<PixelSizeMicrons>0.5</PixelSizeMicrons>",
                f"<PixelSizeMicrons>{size}</PixelSizeMicrons>",
            )
            writer.write(np.zeros((32, 48), dtype="uint16"), tile=(16, 16), metadata=None,
                         description=description,
                         software="PerkinElmer-QPI synthetic" if index == 0 else False)
    with pytest.raises(ValueError, match="inconsistent PixelSizeMicrons"):
        reader(source).open()
