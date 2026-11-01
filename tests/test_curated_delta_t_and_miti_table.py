"""Metadata-only contract checks and mandatory real TIFF/Zarr integration cases.

The schema/XSD and Zarr backend tests intentionally require installed dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner
from lxml import etree

from omeify import convert, mutate, open_dataset
from omeify.cli import main
from omeify.io.dataset_writer import DatasetPlaneSource
from omeify.io.ome_zarr.dataset_writer import dataset_attributes
from omeify.io.ome_zarr.metadata import profile_errors
from omeify.io.plane_times import plane_delta_t, validate_plane_delta_t
from omeify.io.volume_pyramid import plan_volume_pyramid
from omeify.utils.generate_ome_xml import generate_ome_xml
from omeify.utils.miti_header_validator import (render_miti_header_table,
                                                   validate_miti_ome_tiff_header)


def example(tmp_path, *, deltas=True, exposure=True, position=True, timing=True):
    path = tmp_path / "acquired.ome.tif"
    values = np.arange(2*2*3*16*17, dtype=np.uint16).reshape(2, 2, 3, 16, 17)
    meta = {"axes": "TCZYX", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
            "PhysicalSizeZ": 1.5, "PhysicalSizeZUnit": "µm",
            "Channel": {"Name": ["DAPI", "CD3"]}}
    if timing:
        meta["TimeIncrement"] = 10.
    measurements = {}
    if deltas:
        measurements["DeltaT"] = [0, .1, .3, .5, .6, .9, 10, 10.1, 10.3, 10.5, 10.6, 10.9]
    if exposure:
        measurements["ExposureTime"] = [5.] * 12
    if position:
        measurements["PositionZ"] = [7., 8., 9.] * 4
    if measurements:
        meta["Plane"] = measurements
    tifffile.imwrite(path, values, ome=True, photometric="minisblack", metadata=meta)
    return path, values


def test_timing_selection_and_unitless_source_extras_do_not_block(tmp_path):
    path, _ = example(tmp_path)
    with open_dataset(path, input_type="ome_tiff") as ds:
        entries = plane_delta_t(ds)
        assert len(entries) == 12
        assert entries[0] == (0, 0, 0, 0.)
        assert entries[-1] == (1, 1, 2, 10.9)
        ds.require_regular_export()  # ExposureTime and stage XYZ no longer block.
        with ds.select(z=slice(1, 3), t=1, channels=[1, 0]) as subset:
            assert plane_delta_t(subset) == ((0, 0, 0, 10.6), (0, 0, 1, 10.9),
                                             (0, 1, 0, 10.1), (0, 1, 1, 10.3))


def test_generated_tiff_xml_is_minimal_and_preserves_timing(tmp_path):
    path, _ = example(tmp_path)
    with open_dataset(path, input_type="ome_tiff") as ds:
        src = DatasetPlaneSource(ds)
        xml = generate_ome_xml(src.spec, [src.spec.shape], plane_delta_t=src.plane_delta_t)["xml_string"]
        verdict = validate_miti_ome_tiff_header(xml)
        assert verdict.is_valid, verdict.errors
        assert verdict.extra_metadata == ()
        root = etree.fromstring(xml.encode())
        planes = root.findall(".//{*}Plane")
        assert len(planes) == 12
        assert planes[0].get("DeltaT") == "0.0"
        assert planes[-1].get("DeltaT") == "10.9"
        assert all(plane.get("DeltaTUnit") == "s" for plane in planes)
        assert "ExposureTime" not in xml
        assert "PositionZ" not in xml
        table = render_miti_header_table(xml)
        assert "PhysicalSizeX" in table and "TimeIncrement" in table
        assert "12 recorded plane(s)" in table
        assert "10.9 s" in table


def test_regular_timeincrement_without_deltat_is_sufficient(tmp_path):
    path, _ = example(tmp_path, deltas=False)
    with open_dataset(path, input_type="ome_tiff") as ds:
        src = DatasetPlaneSource(ds)
        assert src.plane_delta_t == ()
        xml = generate_ome_xml(src.spec, [src.spec.shape])["xml_string"]
        assert 'TimeIncrement="10.0"' in xml
        assert "<Plane" not in xml
        assert validate_miti_ome_tiff_header(xml).is_valid


def test_no_acquisition_measurements_are_required(tmp_path):
    path, _ = example(tmp_path, deltas=False, exposure=False, position=False, timing=False)
    with open_dataset(path, input_type="ome_tiff") as ds:
        assert plane_delta_t(ds) == ()
        assert ds.axis_info("T").scale is None
        ds.require_regular_export()


def test_json_schema_allows_only_bounded_canonical_delta_t(tmp_path):
    path, _ = example(tmp_path)
    with open_dataset(path, input_type="ome_tiff") as ds:
        src = DatasetPlaneSource(ds)
        plan = plan_volume_pyramid((src.spec.size_z, src.spec.size_y, src.spec.size_x),
                                   tile_size=16, chunk_z=1, levels=0, axes="xy")
        windows = [{"min": 0., "max": 1., "start": 0., "end": 1.} for _ in range(2)]
        attrs = dataset_attributes(src, plan, method="mean", windows=windows)
        assert attrs["omeify"]["plane_delta_t"]["entries"][-1] == [1, 1, 2, 10.9]
        assert profile_errors(attrs) == ()
        attrs["omeify"]["patient_name"] = "DO_NOT_KEEP"
        assert profile_errors(attrs)


def test_miti_profile_detects_bad_plane_timing(tmp_path):
    path, _ = example(tmp_path)
    with open_dataset(path, input_type="ome_tiff") as ds:
        src = DatasetPlaneSource(ds)
        xml = generate_ome_xml(src.spec, [src.spec.shape], plane_delta_t=src.plane_delta_t)["xml_string"]
    root = etree.fromstring(xml.encode())
    planes = root.findall(".//{*}Plane")
    planes[1].set("TheT", planes[0].get("TheT"))
    planes[1].set("TheC", planes[0].get("TheC"))
    planes[1].set("TheZ", planes[0].get("TheZ"))
    invalid = validate_miti_ome_tiff_header(etree.tostring(root).decode())
    assert not invalid.is_valid
    assert any("Duplicate Plane DeltaT" in error for error in invalid.errors)


def test_invalid_delta_t_entries_fail_before_writing():
    with pytest.raises(ValueError, match="Duplicate"):
        validate_plane_delta_t([(0, 0, 0, 0.), (0, 0, 0, 1.)],
                               size_t=1, size_c=1, size_z=1)
    with pytest.raises(ValueError, match="outside"):
        validate_plane_delta_t([(2, 0, 0, 1.)], size_t=2, size_c=1, size_z=1)
    with pytest.raises(ValueError, match="finite"):
        validate_plane_delta_t([(0, 0, 0, float("nan"))], size_t=1, size_c=1, size_z=1)


def test_inspect_show_miti_is_appended_without_changing_default(tmp_path):
    path, _ = example(tmp_path)
    cli = CliRunner()
    normal = cli.invoke(main, ["inspect", str(path), "--detail", "0"])
    expanded = cli.invoke(main, ["inspect", str(path), "--detail", "0", "--show-miti"])
    assert normal.exit_code == expanded.exit_code == 0
    assert expanded.output.startswith(normal.output)
    assert "MITI header profile:" in normal.output
    assert "Curated OME header fields" in expanded.output
    assert "12 recorded plane(s)" in expanded.output
    for args in (["--show-miti", "--json"], ["--show-miti", "--geojson"]):
        result = cli.invoke(main, ["inspect", str(path), *args])
        assert result.exit_code == 2
        assert "--show-miti" in result.output


def test_real_tiff_conversion_and_mutation_drop_extra_measurements(tmp_path):
    path, values = example(tmp_path)
    out = tmp_path / "normalized.ome.tif"
    convert(path, out, input_type="ome_tiff", compression="Deflate", tile_size=16,
            pyramid_levels=0)
    with tifffile.TiffFile(out) as tf:
        xml = tf.ome_metadata
        assert "ExposureTime" not in xml and "PositionZ" not in xml
        assert len(etree.fromstring(xml.encode()).findall(".//{*}Plane")) == 12
    with open_dataset(out, input_type="ome_tiff") as ds:
        assert plane_delta_t(ds)[-1] == (1, 1, 2, 10.9)
        np.testing.assert_array_equal(ds.read_block(0, 16, 0, 17, z=slice(None), t=slice(None)), values)
    # Explicit precision mutation must keep the same timing semantics.
    float_source = tmp_path / "float.ome.tif"
    # A new float version generated using tifffile's independent OME writer.
    tifffile.imwrite(float_source, values.astype(np.float32), ome=True,
                     metadata={"axes": "TCZYX", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
                               "PhysicalSizeZ": 1.5, "TimeIncrement": 10.,
                               "Plane": {"DeltaT": [0, .1, .3, .5, .6, .9,
                                                     10, 10.1, 10.3, 10.5, 10.6, 10.9]}},
                     photometric="minisblack")
    mutated = tmp_path / "compact.ome.tif"
    mutate(float_source, mutated, input_type="ome_tiff", float32_mantissa_bits=11,
           compression="Deflate", tile_size=16, pyramid_levels=0)
    with open_dataset(mutated, input_type="ome_tiff") as ds:
        assert plane_delta_t(ds)[-1] == (1, 1, 2, 10.9)


def test_real_zarr_roundtrip_keeps_delta_t(tmp_path):
    path, values = example(tmp_path)
    zarr_path = tmp_path / "timed.zarr"
    convert(path, zarr_path, input_type="ome_tiff", output_format="ome_zarr",
            tile_size=16, pyramid_levels=0)
    import zarr
    root = zarr.open_group(str(zarr_path), mode="r")
    assert len(root.attrs["omeify"]["plane_delta_t"]["entries"]) == 12
    root.store.close()
    with open_dataset(zarr_path, input_type="ome_zarr") as ds:
        assert plane_delta_t(ds)[-1] == (1, 1, 2, 10.9)
    out = tmp_path / "roundtrip.ome.tif"
    convert(zarr_path, out, input_type="ome_zarr", compression="Deflate",
            tile_size=16, pyramid_levels=0)
    with open_dataset(out, input_type="ome_tiff") as ds:
        assert plane_delta_t(ds)[-1] == (1, 1, 2, 10.9)
        np.testing.assert_array_equal(ds.read_block(0, 16, 0, 17, z=slice(None), t=slice(None)), values)
