"""Real TIFF metadata tests. Inference tests use controlled responses, not a live model."""

from __future__ import annotations

import json
import struct
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from lxml import etree

from omeify import OMETiffReader, OMETiffWriter, PixelSize, TiffInspector
from omeify import _calibration as calibration
from omeify import intelligence as ai
from omeify.cli import main
from omeify.io._writer import ArraySource, WriterEngine, prepare_image
from omeify.io._writer.single_verification import verify_single_output
from omeify.io.ome_multi_series_writer.verification import verify_output
from omeify.io.spec import OMEImageSpec
from omeify.utils.generate_ome_xml import generate_multi_series_ome_xml, generate_ome_xml

OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def write_ome(path, *, x=0.5, y=0.6, x_unit="µm", y_unit="µm", resolution=None, unit=3):
    metadata = {"axes": "CYX", "Channel": {"Name": ["DAPI", "CD3"]}}
    for axis, value, units in (("X", x, x_unit), ("Y", y, y_unit)):
        if value is not None:
            metadata["PhysicalSize" + axis] = value
        if units is not None:
            metadata["PhysicalSize" + axis + "Unit"] = units
    tifffile.imwrite(
        path, np.zeros((2, 32, 48), dtype=np.uint16), photometric="minisblack", ome=True,
        resolution=resolution or (20000, 10000 / 0.6), resolutionunit=unit,
        metadata=metadata,
    )


def overwrite(path, name, value, *, page=0):
    with tifffile.TiffFile(path, mode="r+") as tiff:
        tiff.pages[page].aspage().tags[name].overwrite(value)


def remove_tag(path, name, *, page=0):
    # Rename a physical tag to a private unused ID without changing directory size.
    with tifffile.TiffFile(path) as tiff:
        offset = tiff.pages[page].aspage().tags[name].offset
        endian = tiff.byteorder
    with path.open("r+b") as handle:
        handle.seek(offset)
        handle.write(struct.pack(endian + "H", 65001))


def check(path, *, detail=1):
    inspector = TiffInspector(path, detail=detail)
    assert inspector.validation_errors() == ()
    json.dumps(inspector.report["calibration"], allow_nan=False)
    return inspector.report["calibration"]["series"][0]


@pytest.mark.parametrize("detail", [0, 1, 2, 3])
@pytest.mark.parametrize("units,x,y,unit,res", [
    ("µm", 0.5, 0.6, 3, (20000, 10000 / 0.6)),
    ("nm", 500, 600, 3, (20000, 10000 / 0.6)),
    ("mm", 0.0005, 0.0006, 2, (50800, 25400 / 0.6)),
    ("µm", 0.5, 0.6, 2, (50800, 25400 / 0.6)),
])
def test_equivalent_lengths_at_every_detail(tmp_path, detail, units, x, y, unit, res):
    path = tmp_path / "equivalent.ome.tif"
    write_ome(path, x=x, y=y, x_unit=units, y_unit=units, resolution=res, unit=unit)
    result = check(path, detail=detail)
    assert result["status"] == "consistent"
    assert result["pages_checked"] == 2
    assert result["ome"]["x"]["pixel_size_um"] == pytest.approx(0.5)
    assert "CONSISTENT" in TiffInspector(path, detail=detail).render_text()


def test_mixed_ome_axis_units(tmp_path):
    path = tmp_path / "mixed.ome.tif"
    write_ome(path, y=600, y_unit="nm")
    assert check(path)["status"] == "consistent"


def test_nonfirst_physical_frame_mismatch_is_not_hidden_by_keyframe(tmp_path):
    path = tmp_path / "bad-frame.ome.tif"
    write_ome(path)
    overwrite(path, "XResolution", (10000, 1), page=1)
    result = check(path)
    assert result["status"] == "mismatch"
    assert [p["axes"]["x"]["status"] for p in result["checks"]] == ["consistent", "mismatch"]
    assert result["checks"][1]["axes"]["y"]["status"] == "consistent"
    text = TiffInspector(path).render_text()
    assert "TIFF IFD 1" in text and "relative difference 0.5" in text


@pytest.mark.parametrize("name,value", [
    ("XResolution", (0, 1)), ("XResolution", (1, 0)),
    ("ResolutionUnit", 1), ("ResolutionUnit", 99),
])
def test_unusable_tiff_is_not_a_pass_or_fake_mismatch(tmp_path, name, value):
    path = tmp_path / "unusable.ome.tif"
    write_ome(path)
    for page in (0, 1):
        overwrite(path, name, value, page=page)
    result = check(path)
    assert result["status"] in ("not_comparable", "partial")
    assert result["checks"][0]["axes"]["x"]["status"] == "not_comparable"


def test_missing_one_ome_axis_is_partial(tmp_path):
    path = tmp_path / "partial.ome.tif"
    write_ome(path, y=None)
    result = check(path)
    assert result["status"] == "partial"
    assert result["ome"]["y"]["issue"] == "missing_value"


@pytest.mark.parametrize("unit", ["pixel", "reference frame", "nonsense"])
def test_nonphysical_or_unsupported_ome_units_are_not_guessed(tmp_path, unit):
    path = tmp_path / "unknown-units.ome.tif"
    write_ome(path, x_unit=unit, y_unit=unit)
    assert check(path)["status"] == "not_comparable"


def test_missing_ome_units_use_schema_default_without_fabricating_attributes(tmp_path):
    path = tmp_path / "default-units.ome.tif"
    write_ome(path, x_unit=None, y_unit=None)
    inspector = TiffInspector(path)
    result = check(path)
    assert result["status"] == "consistent"
    assert result["ome"]["x"]["unit"] is None
    assert result["ome"]["x"]["unit_defaulted"] is True
    assert inspector.report["ome"]["images"][0]["physical_size"]["x"]["unit"] is None
    assert any(
        "PhysicalSizeXUnit" in key for key in inspector.report["ome"]["miti"]["missing_fields"]
    )


def test_missing_tiff_unit_uses_inch_default_not_cm(tmp_path):
    path = tmp_path / "tiff-default.ome.tif"
    write_ome(path, unit=2, resolution=(50800, 25400 / 0.6))
    for page in (0, 1):
        remove_tag(path, "ResolutionUnit", page=page)
    result = check(path)
    assert result["status"] == "consistent"
    assert all(page["tiff"]["unit_defaulted"] for page in result["checks"])
    assert result["checks"][0]["tiff"]["resolution_unit"] is None


def test_missing_page_calibration_is_explicit_not_silently_inherited(tmp_path):
    path = tmp_path / "partial-tags.ome.tif"
    write_ome(path)
    remove_tag(path, "XResolution", page=1)
    result = check(path)
    assert result["status"] == "partial"
    assert result["checks"][1]["tiff"]["x"]["issue"] == "missing_tag"
    assert result["checks"][1]["tiff"]["x"]["pixel_size_um"] is None


def test_raw_rational_is_not_the_six_digit_reader_value(tmp_path):
    path = tmp_path / "rational.ome.tif"
    value = 10000 / 20056.913009774024
    write_ome(path, x=value, y=value, resolution=(20056.913009774024,) * 2)
    inspector = TiffInspector(path)
    page = inspector.report["calibration"]["series"][0]["checks"][0]
    n, d = page["tiff"]["x"]["rational"]
    assert page["tiff"]["x"]["pixel_size_um"] == 10000 / (n / d)
    assert page["tiff"]["x"]["pixel_size_um"] != inspector.report["series"][0][
        "tiff_resolution_pixel_size"]["x"]["value"]
    assert page["status"] == "consistent"


@pytest.mark.parametrize("offset,expected", [(5e-6, "consistent"), (2e-5, "mismatch")])
def test_documented_tolerance(offset, expected):
    assert calibration.compare_axis(0.5 * (1 + offset), 0.5)["status"] == expected
    # There is no absolute tolerance that can turn tiny, different lengths into a pass.
    assert calibration.compare_axis(2e-20, 1e-20)["status"] == "mismatch"


def multi_xml(*, external=False, overlap=False, default=False):
    root = etree.Element(f"{{{OME}}}OME", nsmap={None: OME}, UUID="urn:uuid:self")
    for i in range(2):
        image = etree.SubElement(root, f"{{{OME}}}Image", ID=f"Image:{i}", Name=f"Image {i}")
        pixels = etree.SubElement(image, f"{{{OME}}}Pixels", ID=f"Pixels:{i}",
                                  SizeX="48", SizeY="32", SizeZ="1", SizeC="1", SizeT="1",
                                  DimensionOrder="XYZCT", Type="uint16", PhysicalSizeX=str(i + 0.5),
                                  PhysicalSizeY=str(i + 0.5),
                                  PhysicalSizeXUnit="µm", PhysicalSizeYUnit="µm")
        etree.SubElement(
            pixels, f"{{{OME}}}Channel", ID=f"Channel:{i}:0", SamplesPerPixel="1", Name=f"C{i}"
        )
        td = etree.SubElement(pixels, f"{{{OME}}}TiffData")
        if not default:
            td.set("IFD", str(0 if overlap else i))
            td.set("PlaneCount", "1")
        if external:
            u = etree.SubElement(td, f"{{{OME}}}UUID", FileName="DO_NOT_OPEN.ome.tif")
            u.text = "urn:uuid:external"
    return etree.tostring(root)


def write_two(path, xml):
    with tifffile.TiffWriter(path, ome=False) as writer:
        for i in range(2):
            writer.write(np.zeros((32, 48), dtype=np.uint16), metadata=None,
                         description=xml if i == 0 else None,
                         resolution=(10000 / (i + 0.5),) * 2, resolutionunit=3)


def test_mapping_does_not_use_series_ordinal(tmp_path, monkeypatch):
    path = tmp_path / "two.ome.tif"
    write_two(path, multi_xml())
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        reversed_series = list(reversed(tiff.series))
        result = calibration.inspect_calibration(tiff, reversed_series)
        assert [s["ome_image_index"] for s in result["series"]] == [1, 0]
        assert all(s["status"] == "consistent" for s in result["series"])
        monkeypatch.setattr(tifffile.TiffFile, "series", property(lambda self: reversed_series))
        inspector = TiffInspector.from_tiff(tiff, file_path=path)
        assert [s["channel_names"] for s in inspector.report["series"]] == [["C1"], ["C0"]]
        assert inspector.validation_errors() == ()


def test_overlapping_tiff_data_does_not_manufacture_agreement(tmp_path):
    path = tmp_path / "overlap.ome.tif"
    write_two(path, multi_xml(overlap=True))
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        s = tifffile.TiffPageSeries([tiff.pages[0]], (32, 48), np.uint16, "YX")
        result = calibration.inspect_calibration(tiff, [s])["series"][0]
    assert result["mapping_reason"] == "ambiguous_tiff_data"
    assert result["status"] == "not_comparable"


def test_companion_references_never_resolve_by_filename_or_ifd(tmp_path):
    path = tmp_path / "external.ome.tif"
    write_two(path, multi_xml(external=True))
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        s = tifffile.TiffPageSeries([tiff.pages[0]], (32, 48), np.uint16, "YX")
        result = calibration.inspect_calibration(tiff, [s])["series"][0]
    assert result["ome_image_index"] is None
    assert result["status"] == "not_comparable"
    assert not (tmp_path / "DO_NOT_OPEN.ome.tif").exists()


def test_no_ome_is_not_uncalibrated(tmp_path):
    path = tmp_path / "ordinary.tif"
    tifffile.imwrite(path, np.zeros((16, 16), np.uint8), metadata=None,
                     resolution=(20000, 20000), resolutionunit=3)
    result = check(path)
    assert result["mapping_reason"] == "no_ome"
    assert result["checks"][0]["tiff"]["x"]["pixel_size_um"] == 0.5
    assert result["status"] == "not_comparable"


def test_page_budget_does_not_return_false_pass(tmp_path, monkeypatch):
    path = tmp_path / "budget.ome.tif"
    write_ome(path)
    monkeypatch.setattr(calibration, "_MAX_PAGES", 1)
    result = check(path)
    assert result["pages_total"] == 2 and result["pages_checked"] == 1
    assert result["scan_limited"]
    assert result["status"] != "consistent"


def test_metadata_check_does_not_decode_pixels(tmp_path, monkeypatch):
    path = tmp_path / "no-pixels.ome.tif"
    write_ome(path)
    def forbidden(*args, **kwargs):
        pytest.fail("Pixel decoding attempted")
    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    monkeypatch.setattr(tifffile.TiffFile, "asarray", forbidden)
    assert check(path)["status"] == "consistent"


def test_cli_no_intelligence_dependency_needed(tmp_path):
    path = tmp_path / "cli.ome.tif"
    write_ome(path)
    result = CliRunner().invoke(main, ["inspect", str(path)])
    assert result.exit_code == 0, result.output
    assert "CONSISTENT" in result.output
    assert "Summarizing" not in result.output
    result = CliRunner().invoke(main, ["inspect", str(path), "--json"])
    assert json.loads(result.output)["calibration"]["series"][0]["status"] == "consistent"


def test_inference_gets_the_same_computed_verdict_and_context(tmp_path):
    path = tmp_path / "CURRENT_PRIVATE_NAME.ome.tif"
    write_ome(path)
    calls = []
    def prompt(text, **options):
        packet = json.loads(text)
        calls.append((packet, options))
        computed = next(r for r in packet["records"] if r.get("origin") == "computed")
        response = {"overview": {"text": "The computed base-plane calibrations agree.",
                                 "record_ids": [computed["id"]]},
                    "findings": [{"record_id": computed["id"], "category": "calibration",
                                  "label": "Computed calibration",
                                  "interpretation": "Equivalent scales."}],
                    "cautions": []}
        return SimpleNamespace(text=lambda: json.dumps(response))
    @contextmanager
    def connect(**kwargs):
        assert kwargs["requires"] == {"system_prompt", "json_schema"}
        yield SimpleNamespace(prompt=prompt)
    source = SimpleNamespace(name="test", default_model="model", scope="institutional",
                             organization=None, connect=connect)
    def select(name, **kwargs):
        assert kwargs["allowed_scopes"] == ("institutional", "local")
        return source
    inspector = TiffInspector(path, detail=0, max_text_length=1)
    result = inspector.summarize_metadata(registry=SimpleNamespace(source=select))
    assert len(calls) == 1
    packet, options = calls[0]
    serialized = json.dumps(packet, ensure_ascii=False)
    assert path.name not in serialized and str(tmp_path) not in serialized
    assert "full-resolution X/Y only: consistent" in serialized
    assert "20000 pixels/cm and OME 0.5 µm/pixel agree exactly" in options["system"]
    assert "partial/not_comparable" in options["system"]
    assert "NOT a mismatch" in options["system"]
    assert "no micrometer resolution" in options["system"]
    computed = next(r for r in packet["records"] if r.get("origin") == "computed")
    assert result["summary"]["findings"][0]["value"] == computed["value"]
    assert "Computed evidence" in inspector.render_text()
    assert inspector.validation_errors() == ()
    assert result["coverage"]["computed_records_included"] == 1
    assert len(calls) == 1


def test_context_budget_and_mismatch_priority(tmp_path):
    path = tmp_path / "budget.ome.tif"
    write_ome(path)
    inspector = TiffInspector(path)
    checks = deepcopy(inspector.report["calibration"])
    checks["series"] = [deepcopy(checks["series"][0]) for _ in range(20)]
    for index, item in enumerate(checks["series"]):
        item["series_index"] = index
    checks["series"][-1]["status"] = "mismatch"
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        packet = ai.collect_metadata(tiff, max_chars=4096, calibration=checks)
    computed = [r for r in packet["records"] if r.get("origin") == "computed"]
    assert computed and "series 19" in computed[0]["value"]
    assert len(computed) < 20
    assert packet["coverage"]["computed_records_available"] == 20
    assert packet["coverage"]["computed_records_included"] == len(computed)
    assert len(json.dumps(packet["records"], ensure_ascii=False, separators=(",", ":"))) <= 4096
    assert packet["coverage"]["metadata_chars"] <= 4096


def prepared_image(path, *, name=None, size=PixelSize(0.5, 0.6, "µm")):
    data = np.zeros((2, 33, 49), dtype=np.uint16)
    writer = OMETiffWriter(path, pixel_size=size, compression="Uncompressed", tile_size=16,
                          pyramid_levels=2, channel_names=("DAPI", "CD3"))
    spec = OMEImageSpec.from_shape(image_type="multichannel", axes="CYX", shape=data.shape,
                                   dtype=data.dtype, channel_names=("DAPI", "CD3"), pixel_size=size)
    prepared = prepare_image(ArraySource(data, spec), spec, name=name, downsample="mean",
                             compression_name="Uncompressed", settings=writer._settings,
                             lossy_policy="rgb-only")
    return prepared, writer._settings


def write_engine(path):
    # Exercise the real engine, TIFF writing, pyramid building, verification,
    # and atomic install without requiring the optional local XSD fixture here.
    item, settings = prepared_image(path)
    xml = generate_ome_xml(item.spec, item.level_shapes)["xml_string"]
    result = WriterEngine(settings).write(
        [item], xml, verify=lambda p: verify_single_output(item, p, software=settings.software),
    )
    return item, settings, result


def test_writer_checks_all_ifds_and_odd_size_pyramid_scales(tmp_path):
    path = tmp_path / "writer.ome.tif"
    item, settings, report = write_engine(path)
    assert report.verification["calibration_ifds_checked"] == 6
    assert report.verification["tiff_calibration_matches_specs"]
    assert report.verification["ome_physical_sizes_match_specs"]
    assert report.verification["pyramid_calibration_matches_specs"]
    with tifffile.TiffFile(path) as tiff:
        sizes = [calibration.tiff_calibration(level.pages[0].aspage())["x"]["pixel_size_um"]
                 for level in tiff.series[0].levels]
    assert sizes == pytest.approx([0.5, 1., 2.])
    result = check(path)
    assert result["status"] == "consistent"
    assert len(result["checks"]) == 2  # SubIFDs did not enter the base comparison.


@pytest.mark.parametrize("target", ["ome", "base", "pyramid"])
def test_writer_catches_corrupted_emitted_calibration(tmp_path, target):
    path = tmp_path / "corrupt.ome.tif"
    item, settings, _ = write_engine(path)
    if target == "ome":
        with tifffile.TiffFile(path) as tiff:
            xml = tiff.ome_metadata.replace('PhysicalSizeX="0.5"', 'PhysicalSizeX="0.9"')
        overwrite(path, "ImageDescription", xml.encode())
    elif target == "base":
        overwrite(path, "XResolution", (10000, 1), page=1)
    else:
        with tifffile.TiffFile(path, mode="r+") as tiff:
            tiff.pages[1].aspage().pages[1].aspage().tags["YResolution"].overwrite((10000, 1))
    with pytest.raises(ValueError, match="calibration"):
        verify_single_output(item, path, software=settings.software)


def test_both_metadata_layers_can_agree_but_fail_the_intended_writer_spec(tmp_path):
    path = tmp_path / "coherent-but-wrong.ome.tif"
    item, settings, _ = write_engine(path)
    with tifffile.TiffFile(path) as tiff:
        xml = tiff.ome_metadata.replace('PhysicalSizeX="0.5"', 'PhysicalSizeX="1.0"')
    overwrite(path, "ImageDescription", xml.encode())
    for page in (0, 1):
        overwrite(path, "XResolution", (10000, 1), page=page)
    assert check(path)["status"] == "consistent"
    with pytest.raises(ValueError, match="writer specification"):
        verify_single_output(item, path, software=settings.software)


def test_multi_series_writer_uses_the_same_calibration_verification(tmp_path):
    path = tmp_path / "multi.ome.tif"
    one, settings = prepared_image(path, name="Signal")
    two, _ = prepared_image(path, name="Other", size=PixelSize(250, 400, "nm"))
    items = [one, two]
    xml = generate_multi_series_ome_xml(
        [(i.name, i.spec, i.level_shapes) for i in items]
    )["xml_string"]
    result = WriterEngine(settings).write(
        items, xml, verify=lambda p: verify_output(
            items, p, provenance_json=None, software=settings.software,
        ),
    )
    assert result.verification["calibration_ifds_checked"] == 12
    inspector = TiffInspector(path)
    assert all(s["status"] == "consistent" for s in inspector.report["calibration"]["series"])
    assert inspector.validation_errors() == ()
    with OMETiffReader(path, series=1) as reader:
        assert reader.pixel_size == PixelSize(0.25, 0.4, "µm")


def test_calibration_failure_preserves_destination_atomically(tmp_path, monkeypatch):
    from omeify.io._writer import engine
    path = tmp_path / "existing.ome.tif"
    path.write_bytes(b"existing destination")
    item, settings = prepared_image(path)
    original = engine.write_output
    def corrupted(*args, **kwargs):
        original(*args, **kwargs)
        overwrite(args[2], "XResolution", (10000, 1))
    monkeypatch.setattr(engine, "write_output", corrupted)
    xml = generate_ome_xml(item.spec, item.level_shapes)["xml_string"]
    with pytest.raises(ValueError, match="calibration"):
        WriterEngine(settings).write(
            [item], xml, verify=lambda p: verify_single_output(item, p, software=settings.software),
        )
    assert path.read_bytes() == b"existing destination"
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("x", [0, -1, float("nan"), float("inf")])
def test_invalid_ome_numbers_never_compare_successfully(tmp_path, x):
    path = tmp_path / "invalid-number.ome.tif"
    write_ome(path, x=x)
    inspector = TiffInspector(path)
    result = check(path)
    assert result["status"] == "partial"
    assert result["ome"]["x"]["issue"] == "invalid_value"
    json.dumps(inspector.report, allow_nan=False)


def test_bare_tiff_data_does_not_claim_extra_tail_ifds(tmp_path):
    path = tmp_path / "bare.ome.tif"
    root = etree.fromstring(multi_xml(default=True))
    root.remove(root[-1])  # One logical image with one plane, plus an unrelated TIFF page.
    write_two(path, etree.tostring(root))
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        base = tifffile.TiffPageSeries([tiff.pages[0]], (32, 48), np.uint16, "YX")
        extra = tifffile.TiffPageSeries([tiff.pages[1]], (32, 48), np.uint16, "YX")
        result = calibration.inspect_calibration(tiff, [base, extra])
    assert result["series"][0]["status"] == "consistent"
    assert result["series"][1]["ome_image_index"] is None


def test_explicit_ifd_defaults_to_one_plane_not_all_pages(tmp_path):
    path = tmp_path / "one-ifd.ome.tif"
    root = etree.fromstring(multi_xml())
    root.remove(root[-1])
    pixels = root[0][0]
    pixels.set("SizeC", "2")
    etree.SubElement(pixels, f"{{{OME}}}Channel", ID="Channel:0:1", SamplesPerPixel="1")
    td = pixels.find(f"{{{OME}}}TiffData")
    td.attrib.pop("PlaneCount")
    write_two(path, etree.tostring(root))
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        both = tifffile.TiffPageSeries(list(tiff.pages), (2, 32, 48), np.uint16, "CYX")
        result = calibration.inspect_calibration(tiff, [both])["series"][0]
    assert result["status"] == "not_comparable"
    assert result["mapping_reason"] == "unmapped_ifd"


def test_same_file_uuid_is_a_supported_mapping(tmp_path):
    path = tmp_path / "self-uuid.ome.tif"
    xml = multi_xml(external=True).replace(b"urn:uuid:external", b"urn:uuid:self")
    write_two(path, xml)
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        one = tifffile.TiffPageSeries([tiff.pages[0]], (32, 48), np.uint16, "YX")
        result = calibration.inspect_calibration(tiff, [one])["series"][0]
    assert result["status"] == "consistent"


def test_rgb_bare_tiff_data_counts_planes_not_samples(tmp_path):
    path = tmp_path / "rgb.ome.tif"
    xml = multi_xml(default=True)
    root = etree.fromstring(xml)
    root.remove(root[-1])
    pixels = root[0][0]
    pixels.set("SizeC", "3")
    pixels[0].set("SamplesPerPixel", "3")
    with tifffile.TiffWriter(path, ome=False) as writer:
        for i in range(2):
            writer.write(np.zeros((32, 48, 3), np.uint16), photometric="rgb", metadata=None,
                         description=etree.tostring(root) if i == 0 else None,
                         resolution=(20000, 20000), resolutionunit=3)
    with tifffile.TiffFile(path, _multifile=False) as tiff:
        one = tifffile.TiffPageSeries([tiff.pages[0]], (32, 48, 3), np.uint16, "YXS")
        tail = tifffile.TiffPageSeries([tiff.pages[1]], (32, 48, 3), np.uint16, "YXS")
        result = calibration.inspect_calibration(tiff, [one, tail])
    assert result["series"][0]["status"] == "consistent"
    assert result["series"][1]["ome_image_index"] is None
