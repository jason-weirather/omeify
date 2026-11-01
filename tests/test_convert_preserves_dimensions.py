"""Preservation-first convert, opt-in subsetting, and minimized stack headers.

Public integration cases intentionally need real TIFF/Zarr codecs and OME XSD.
Their absence is an environment failure, not a reason to skip these tests.
"""
from __future__ import annotations

import copy
import inspect

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from lxml import etree

from omeify import AxisInfo, ImageDataset, OMEZarrWriter, PixelSize, convert, mutate, open_dataset
from omeify.cli import main
from omeify.io.ome_zarr.metadata import profile_errors
from omeify.io.spec import OMEImageSpec
from omeify.io.volume_pyramid import plan_volume_pyramid
from omeify.reports import report_validator, validate_report
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header


OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def make_source_tiff(tmp_path, *, axes="TCZYX", physical_z=True, physical_t=True,
                     acquisition_plane=False):
    path = tmp_path / "source.ome.tif"
    if axes == "TCZYX":
        data = (np.arange(3 * 2 * 4 * 16 * 20, dtype=np.uint32)
                .reshape(3, 2, 4, 16, 20) % 20000).astype(np.uint16)
    elif axes == "TZYX":
        data = (np.arange(3 * 4 * 16 * 20, dtype=np.uint32)
                .reshape(3, 4, 16, 20) % 20000).astype(np.uint16)
    else:
        raise ValueError("Unexpected test axes")
    metadata = {"axes": axes, "PhysicalSizeX": .5, "PhysicalSizeY": .6}
    if physical_z:
        metadata["PhysicalSizeZ"] = 2.5
    if physical_t:
        metadata["TimeIncrement"] = 7.
    if acquisition_plane:
        metadata["Plane"] = {"DeltaT": list(range(3 * 4))}
    tifffile.imwrite(path, data, ome=True, photometric="minisblack", metadata=metadata)
    return path, data


def test_metadata_schema_allows_unknown_z_spacing_but_rejects_dangling_units():
    spec = OMEImageSpec(
        "multichannel", "TCZYX", (3, 1, 4, 16, 20), np.dtype("uint16"),
        ("Signal",), PixelSize(.5, .6, "µm"),
    )
    xml = generate_ome_xml(spec, (spec.shape,))["xml_string"]
    assert validate_miti_ome_tiff_header(xml).is_valid
    pixels = etree.fromstring(xml.encode()).find(f"{{{OME}}}Image/{{{OME}}}Pixels")
    assert pixels is not None
    assert pixels.attrib["SizeZ"] == "4"
    assert "PhysicalSizeZ" not in pixels.attrib
    assert "PhysicalSizeZUnit" not in pixels.attrib
    assert "TimeIncrement" not in pixels.attrib
    assert "TimeIncrementUnit" not in pixels.attrib
    validator = report_validator("miti_ome_tiff_header.schema.json")
    base = dict(
        image_id="Image:0", pixels_id="Pixels:0", big_endian=False,
        dimension_order="XYZCT", interleaved=False,
        physical_size_x=.5, physical_size_x_unit="µm",
        physical_size_y=.6, physical_size_y_unit="µm",
        significant_bits=16, size_c=1, size_t=3, size_x=20, size_y=16,
        size_z=4, pixel_type="uint16", channels=[
            {"id": "Channel:0:0", "name": "Signal", "samples_per_pixel": 1}],
        tiff_data=[{"ifd": 0, "plane_count": 12}],
    )
    assert validator.is_valid(base)
    assert not validator.is_valid({**base, "physical_size_z_unit": "µm"})
    assert not validator.is_valid({**base, "physical_size_z": 2.5})
    assert validator.is_valid({**base, "physical_size_z": 2.5,
                               "physical_size_z_unit": "µm"})


def test_zarr_stack_schema_distinguishes_unknown_z_time_units():
    from types import SimpleNamespace
    spec = OMEImageSpec(
        "multichannel", "TCZYX", (3, 1, 4, 16, 20), np.dtype("uint16"),
        ("Signal",), PixelSize(.5, .6, "µm"),
    )
    plan = plan_volume_pyramid((4, 16, 20), levels=0, axes="xy")
    dummy = {"min": 0, "max": 1, "start": 0, "end": 1}
    from omeify.io.ome_zarr.dataset_writer import dataset_attributes
    attrs = dataset_attributes(SimpleNamespace(spec=spec), plan, method="mean", windows=[dummy])
    assert not profile_errors(attrs)
    axes = attrs["ome"]["multiscales"][0]["axes"]
    assert "unit" not in axes[0] and "unit" not in axes[2]
    assert axes[3]["unit"] == axes[4]["unit"] == "micrometer"
    assert attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][:3] == [1., 1., 1.]
    altered = copy.deepcopy(attrs)
    altered["ome"]["multiscales"][0]["axes"][2]["unit"] = "seconds"
    assert profile_errors(altered)


@pytest.mark.parametrize("dest_fmt", ("ome_tiff", "ome_zarr"))
def test_convert_defaults_to_entire_selected_ome_tiff_stack(tmp_path, dest_fmt):
    source, values = make_source_tiff(tmp_path)
    dest = tmp_path / ("converted.ome.tif" if dest_fmt == "ome_tiff" else "converted.zarr")
    report = convert(source, dest, input_type="ome_tiff", output_format=dest_fmt,
                     compression="Deflate" if dest_fmt == "ome_tiff" else "ZSTD",
                     tile_size=16, pyramid_levels=0, overwrite=False)
    validate_report(report, "dataset_operation_report.schema.json")
    assert report["input"]["z_range"] == [0, 4]
    assert report["input"]["t_range"] == [0, 3]
    assert report["writer_report"]["image"]["z_spacing_um"] == 2.5
    assert report["writer_report"]["image"]["time_increment_s"] == 7.
    with open_dataset(dest, input_type=dest_fmt) as dataset:
        assert dataset.axes == "TCZYX"
        assert dataset.axis_info("Z").size == 4
        assert dataset.axis_info("T").size == 3
        for t in range(3):
            for z in range(4):
                with dataset.plane(t=t, z=z) as image:
                    np.testing.assert_array_equal(image.asarray(), values[t, :, z])
    if dest_fmt == "ome_tiff":
        with tifffile.TiffFile(dest) as tif:
            xml = etree.fromstring(tif.ome_metadata.encode())
            pixels = xml.find(f"{{{OME}}}Image/{{{OME}}}Pixels")
            assert pixels.attrib["PhysicalSizeZ"] == "2.5"
            assert pixels.attrib["PhysicalSizeZUnit"] == "µm"
            assert pixels.attrib["TimeIncrement"] == "7.0"
            assert pixels.attrib["TimeIncrementUnit"] == "s"
            assert not xml.findall(f".//{{{OME}}}Plane")


@pytest.mark.parametrize("axis,selection,expected", (
    ("Z", {"z": 2}, (3, 1, 1)),
    ("T", {"t": 1}, (1, 1, 4)),
    ("Z", {"z_range": (1, 3)}, (3, 1, 2)),
    ("T", {"t_range": (1, 3)}, (2, 1, 4)),
))
def test_convert_partial_selection_preserves_unselected_axis(tmp_path, axis, selection, expected):
    source, values = make_source_tiff(tmp_path, axes="TZYX")
    dest = tmp_path / "selected.ome.tif"
    report = convert(source, dest, input_type="ome_tiff", compression="Deflate",
                     tile_size=16, pyramid_levels=0, overwrite=False, **selection)
    validate_report(report, "dataset_operation_report.schema.json")
    with open_dataset(dest, input_type="ome_tiff") as ds:
        assert (ds.axis_info("T").size, len(ds.channel_names), ds.axis_info("Z").size) == expected
        for t in range(expected[0]):
            for z in range(expected[2]):
                source_t = selection.get("t", t + selection.get("t_range", (0, 0))[0])
                source_z = selection.get("z", z + selection.get("z_range", (0, 0))[0])
                with ds.plane(t=t, z=z) as image:
                    result = image.asarray()
                    np.testing.assert_array_equal(result[0] if result.ndim == 3 else result,
                                                  values[source_t, source_z])


def test_convert_two_selected_indices_uses_2d_report(tmp_path):
    source, values = make_source_tiff(tmp_path, axes="TZYX")
    dest = tmp_path / "single.ome.tif"
    report = convert(source, dest, input_type="ome_tiff", z=1, t=2,
                     compression="Deflate", tile_size=16, pyramid_levels=0)
    validate_report(report, "conversion_report.schema.json")
    assert report["schema"] == "omeify.convert/2"
    with open_dataset(dest, input_type="ome_tiff") as ds:
        assert ds.axis_info("T").size == ds.axis_info("Z").size == 1
        with ds.plane() as image:
            result = image.asarray()
            np.testing.assert_array_equal(result[0] if result.ndim == 3 else result,
                                          values[2, 1])


def test_convert_ordinary_2d_keeps_old_report_and_samples(tmp_path):
    source = tmp_path / "2d.ome.tif"
    values = np.arange(16 * 20, dtype=np.uint16).reshape(16, 20)
    tifffile.imwrite(source, values, ome=True, metadata={
        "axes": "YX", "PhysicalSizeX": .5, "PhysicalSizeY": .6})
    dest = tmp_path / "2d-out.ome.tif"
    report = convert(source, dest, input_type="ome_tiff", compression="Deflate",
                     tile_size=16, pyramid_levels=0)
    validate_report(report, "conversion_report.schema.json")
    with tifffile.TiffFile(dest) as tiff:
        np.testing.assert_array_equal(tiff.asarray(), values)


@pytest.mark.parametrize("dest_fmt", ("ome_tiff", "ome_zarr"))
def test_conversion_without_z_or_time_measurements_never_fabricates_them(tmp_path, dest_fmt):
    source, values = make_source_tiff(tmp_path, axes="TZYX", physical_z=False,
                                      physical_t=False)
    dest = tmp_path / ("unknown.ome.tif" if dest_fmt == "ome_tiff" else "unknown.zarr")
    report = convert(source, dest, input_type="ome_tiff", output_format=dest_fmt,
                     compression="Deflate" if dest_fmt == "ome_tiff" else "ZSTD",
                     tile_size=16, pyramid_levels=0)
    validate_report(report, "dataset_operation_report.schema.json")
    assert report["writer_report"]["image"]["z_spacing_um"] is None
    assert report["writer_report"]["image"]["time_increment_s"] is None
    with open_dataset(dest, input_type=dest_fmt) as ds:
        assert ds.axis_info("Z").unit is None
        assert ds.axis_info("T").unit is None
        with ds.plane(z=3, t=2) as image:
            result = image.asarray()
            np.testing.assert_array_equal(result[0] if result.ndim == 3 else result,
                                          values[2, 3])
    if dest_fmt == "ome_tiff":
        with tifffile.TiffFile(dest) as tif:
            xml = etree.fromstring(tif.ome_metadata.encode())
            pixels = xml.find(f"{{{OME}}}Image/{{{OME}}}Pixels")
            for key in ("PhysicalSizeZ", "PhysicalSizeZUnit", "TimeIncrement",
                        "TimeIncrementUnit"):
                assert key not in pixels.attrib


def test_convert_from_ome_zarr_preserves_every_plane_by_default(tmp_path):
    values = (np.arange(2 * 3 * 16 * 20).reshape(2, 3, 16, 20) % 1000).astype(np.uint16)
    source = tmp_path / "input.zarr"
    with ImageDataset.from_array(
        values, axes="TZYX", pixel_size=PixelSize(.5, .6, "µm"),
        axis_info=(AxisInfo("T", 2, "time", "s", 12.),
                   AxisInfo("Z", 3, "space", "µm", 2.)),
    ) as ds:
        OMEZarrWriter(source, tile_size=16, pyramid_levels=0).write_dataset(ds)
    dest = tmp_path / "to-tiff.ome.tif"
    report = convert(source, dest, input_type="ome_zarr", compression="Deflate",
                     tile_size=16, pyramid_levels=0)
    validate_report(report, "dataset_operation_report.schema.json")
    with open_dataset(dest, input_type="ome_tiff") as ds:
        assert ds.axis_info("T").size == values.shape[0]
        assert ds.axis_info("Z").size == values.shape[1]
        for t in range(2):
            for z in range(3):
                with ds.plane(t=t, z=z) as image:
                    result = image.asarray()
                    np.testing.assert_array_equal(result[0] if result.ndim == 3 else result,
                                                  values[t, z])


def test_convert_fails_without_silently_dropping_acquisition_measurements(tmp_path):
    source, _ = make_source_tiff(tmp_path, axes="TZYX", acquisition_plane=True)
    dest = tmp_path / "refused.ome.tif"
    with pytest.raises(NotImplementedError, match="per-plane acquisition"):
        convert(source, dest, input_type="ome_tiff", compression="Deflate",
                tile_size=16, pyramid_levels=0)
    assert not dest.exists()
    # Explicit single-plane export remains available, without asserting that
    # the output carries the source's per-plane acquisition metadata.
    selected = tmp_path / "selected.ome.tif"
    report = convert(source, selected, input_type="ome_tiff", z=1, t=1,
                     compression="Deflate", tile_size=16, pyramid_levels=0)
    validate_report(report, "conversion_report.schema.json")


def test_mutate_requires_numerical_intent_and_preserves_whole_zt_scope(tmp_path):
    source, values = make_source_tiff(tmp_path, axes="TZYX")
    float_source = tmp_path / "floats.ome.tif"
    tifffile.imwrite(float_source, values.astype(np.float32), ome=True,
                     metadata={"axes": "TZYX", "PhysicalSizeX": .5,
                               "PhysicalSizeY": .6, "PhysicalSizeZ": 2.5})
    dest = tmp_path / "mutated.ome.tif"
    report = mutate(float_source, dest, input_type="ome_tiff",
                    float32_mantissa_bits=10, compression="Deflate",
                    tile_size=16, pyramid_levels=0)
    validate_report(report, "dataset_operation_report.schema.json")
    assert report["operation"] == "mutate"
    assert report["input"]["z_range"] == [0, 4]
    assert report["input"]["t_range"] == [0, 3]
    with open_dataset(dest, input_type="ome_tiff") as ds:
        assert ds.axis_info("Z").size == 4 and ds.axis_info("T").size == 3


def test_all_planes_is_no_longer_an_api_or_cli_option():
    assert "all_planes" not in inspect.signature(convert).parameters
    assert "all_planes" not in inspect.signature(mutate).parameters
    for command in ("convert", "mutate"):
        runner = CliRunner()
        help_result = runner.invoke(main, [command, "--help"])
        assert help_result.exit_code == 0
        assert "--all-planes" not in help_result.output
        invalid = runner.invoke(main, [command, "--all-planes"])
        assert invalid.exit_code != 0
        assert "No such option" in invalid.output


@pytest.mark.parametrize("operation", ("convert", "mutate"))
@pytest.mark.parametrize("selection", ({}, {"z": 2}, {"t": 1}, {"z_range": (1, 3)}))
def test_dispatch_borrows_one_dataset_session_without_plane_guessing(
    tmp_path, monkeypatch, operation, selection,
):
    """Actual OME-TIFF metadata probe; intercept only the writer delegation."""
    source, _ = make_source_tiff(tmp_path, axes="TZYX")
    seen = []

    def writer_stub(*args, **kwargs):
        dataset = kwargs["opened_dataset"]
        assert dataset.is_open
        assert dataset.axis_info("Z").size == 4
        assert dataset.axis_info("T").size == 3
        assert "all_planes" not in kwargs
        seen.append(kwargs)
        return {"route": "dataset"}

    monkeypatch.setattr("omeify.dataset_workflow.process_dataset", writer_stub)
    operation_fn = convert if operation == "convert" else mutate
    options = {"input_type": "ome_tiff", **selection}
    if operation == "mutate":
        options["float32_mantissa_bits"] = 10
    output = operation_fn(source, tmp_path / "out.ome.tif", **options)
    assert output == {"route": "dataset"}
    assert len(seen) == 1


def test_cli_convert_defaults_to_full_tz_without_a_scope_flag(tmp_path):
    source, values = make_source_tiff(tmp_path, axes="TZYX")
    dest = tmp_path / "cli.ome.tif"
    result = CliRunner().invoke(main, [
        "convert", str(source), "--type", "ome_tiff", "-o", str(dest),
        "--compression", "Deflate", "--tile-size", "16", "--pyramid-levels", "0",
        "--no-overwrite",
    ])
    assert result.exit_code == 0, result.output
    with open_dataset(dest, input_type="ome_tiff") as ds:
        assert (ds.axis_info("T").size, ds.axis_info("Z").size) == (3, 4)
        with ds.plane(z=3, t=2) as image:
            got = image.asarray()
            np.testing.assert_array_equal(got[0] if got.ndim == 3 else got, values[2, 3])


def test_full_stack_conversion_does_not_copy_unrelated_source_ome_image_name(tmp_path):
    source = tmp_path / "source-with-identifier.ome.tif"
    values = np.arange(2 * 3 * 16 * 20, dtype=np.uint16).reshape(2, 3, 16, 20)
    marker = "UNRELATED_PRIVATE_SPECIMEN_90521"
    tifffile.imwrite(source, values, ome=True, metadata={
        "axes": "TZYX", "PhysicalSizeX": .5, "PhysicalSizeY": .6,
        "PhysicalSizeZ": 2., "Name": marker,
    })
    with tifffile.TiffFile(source) as tiff:
        assert marker in tiff.ome_metadata
    dest = tmp_path / "minimized.ome.tif"
    convert(source, dest, input_type="ome_tiff", compression="Deflate",
            tile_size=16, pyramid_levels=0)
    with tifffile.TiffFile(dest) as tiff:
        assert marker not in tiff.ome_metadata
        xml = etree.fromstring(tiff.ome_metadata.encode())
        pixels = xml.find(f"{{{OME}}}Image/{{{OME}}}Pixels")
        assert pixels.attrib["SizeZ"] == "3"
        assert pixels.attrib["SizeT"] == "2"
        assert pixels.attrib["PhysicalSizeZ"] == "2.0"
