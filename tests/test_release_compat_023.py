"""Release regressions: pre-Z/T 2D reads and Z/T publication/provenance guarantees.

All fixtures are tiny synthetic TIFFs and use real encoders/readers and schemas.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile
from lxml import etree

from omeify import OMETiffReader, PixelSize, convert, crop, mutate, open_dataset
from omeify.io.pixel_size import canonical_length_unit
from omeify.reports import validate_report


OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"


def metadata(axes, **extra):
    return {"axes": axes, "PhysicalSizeX": .5, "PhysicalSizeY": .5,
            "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm", **extra}


@pytest.mark.parametrize("channels", (1, 2))
def test_valid_ome_tiff_without_optional_channel_elements_opens(tmp_path, channels):
    path = tmp_path / "channel-less.ome.tif"
    values = np.arange(channels * 16 * 16, dtype=np.uint16).reshape(channels, 16, 16)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack",
                     metadata=metadata("CYX"))
    with tifffile.TiffFile(path) as tiff:
        xml = etree.fromstring(tiff.ome_metadata.encode("utf-8"))
    pixels = xml.find(f"{{{OME}}}Image/{{{OME}}}Pixels")
    for channel in pixels.findall(f"{{{OME}}}Channel"):
        pixels.remove(channel)
    tifffile.imwrite(path, values, photometric="minisblack", metadata=None,
                     description=etree.tostring(xml))
    expected = values[0] if channels == 1 else values
    with OMETiffReader(path) as image:
        np.testing.assert_array_equal(image.asarray(), expected)
    with open_dataset(path, input_type="ome_tiff") as dataset:
        with dataset.plane() as plane:
            np.testing.assert_array_equal(plane.asarray(), expected)


@pytest.mark.parametrize("extra", (
    {"PhysicalSizeZ": 1., "PhysicalSizeZUnit": "pixel"},
    {"PhysicalSizeZ": 1., "PhysicalSizeZUnit": "reference frame"},
    {"PhysicalSizeZ": 1., "PhysicalSizeZUnit": "fm"},
    {"TimeIncrement": 0.},
))
def test_optional_singleton_zt_metadata_does_not_break_2d(tmp_path, extra):
    path = tmp_path / "singleton.ome.tif"
    values = np.arange(35, dtype=np.uint16).reshape(5, 7)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack",
                     metadata=metadata("YX", **extra))
    with OMETiffReader(path) as image:
        np.testing.assert_array_equal(image.asarray(), values)
    with open_dataset(path, input_type="ome_tiff") as dataset:
        if extra.get("PhysicalSizeZUnit") in {"pixel", "reference frame"}:
            assert dataset.axis_info("Z").scale is None
        if extra.get("PhysicalSizeZUnit") == "fm":
            assert dataset.axis_info("Z").scale == pytest.approx(1e-9)
        if "TimeIncrement" in extra:
            assert dataset.axis_info("T").scale is None
        with dataset.plane() as image:
            np.testing.assert_array_equal(image.asarray(), values)


def test_femtometer_is_a_supported_physical_length_unit():
    assert canonical_length_unit("femtometre") == "fm"
    assert PixelSize(1, 1, "fm").converted_to("µm").x == pytest.approx(1e-9)


def test_dataset_plane_uses_ome_calibration_over_conflicting_tiff_tags(tmp_path):
    path = tmp_path / "conflicting-calibration.ome.tif"
    values = np.arange(256, dtype=np.uint16).reshape(16, 16)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack",
                     metadata=metadata("YX"), resolution=(10000, 10000),
                     resolutionunit="CENTIMETER")  # TIFF says 1.0 µm, OME says 0.5
    with OMETiffReader(path) as image:
        assert image.pixel_size == PixelSize(.5, .5, "µm")
    with open_dataset(path, input_type="ome_tiff") as dataset:
        with dataset.plane() as plane:
            assert plane.pixel_size == PixelSize(.5, .5, "µm")
            np.testing.assert_array_equal(plane.asarray(), values)


def test_invalid_stack_precision_is_rejected_before_overwriting(tmp_path):
    path = tmp_path / "float-stack.ome.tif"
    destination = tmp_path / "existing.ome.tif"
    values = np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16)
    tifffile.imwrite(path, values, ome=True, metadata=metadata("TYX", TimeIncrement=1.),
                     photometric="minisblack")
    sentinel = b"Existing output must not be replaced by invalid stack mutation"
    for scope in ({"t": 0}, {"all_planes": True}):
        destination.write_bytes(sentinel)
        with pytest.raises(ValueError, match="retains full float32 precision"):
            mutate(path, destination, input_type="ome_tiff", float32_mantissa_bits=23,
                   compression="Deflate", tile_size=16, pyramid_levels=0, **scope)
        assert destination.read_bytes() == sentinel


def test_selected_tiff_plane_is_recorded_in_all_operation_reports(tmp_path):
    path = tmp_path / "selected-planes.ome.tif"
    float_path = tmp_path / "selected-float-planes.ome.tif"
    values = np.arange(2 * 2 * 3 * 16 * 16, dtype=np.uint16).reshape(2, 2, 3, 16, 16)
    extra = metadata("TCZYX", Channel={"Name": ["DAPI", "CD8"]},
                     PhysicalSizeZ=2., TimeIncrement=1.)
    tifffile.imwrite(path, values, ome=True, photometric="minisblack", metadata=extra)
    tifffile.imwrite(float_path, values.astype(np.float32), ome=True,
                     photometric="minisblack", metadata=extra)
    common = dict(z=2, t=1, compression="Uncompressed", tile_size=16, pyramid_levels=0)
    conversion = convert(path, tmp_path / "converted.ome.tif", input_type="ome_tiff", **common)
    mutation = mutate(float_path, tmp_path / "mutated.ome.tif", input_type="ome_tiff",
                      float32_mantissa_bits=10, **common)
    for report, schema in ((conversion, "conversion_report.schema.json"),
                           (mutation, "mutation_report.schema.json")):
        validate_report(report, schema)
        assert report["options"]["z"] == 2
        assert report["options"]["t"] == 1

    output = tmp_path / "cropped.ome.tif"
    cropped = crop(path, output, bounds=(0, 0, 16, 16), **common)
    validate_report(cropped, "crop_report.schema.json")
    assert (cropped["source_group"], cropped["source_z"], cropped["source_t"]) == (None, 2, 1)
    with tifffile.TiffFile(output) as tiff:
        root = etree.fromstring(tiff.ome_metadata.encode("utf-8"))
    embedded = []
    for node in root.findall(f".//{{{OME}}}MapAnnotation/{{{OME}}}Value/{{{OME}}}M"):
        if node.get("K") == "json" and node.text:
            value = json.loads(node.text)
            if str(value.get("schema", "")).startswith("omeify.crop_provenance/"):
                embedded.append(value)
    assert len(embedded) == 1
    assert (embedded[0]["source_group"], embedded[0]["source_z"],
            embedded[0]["source_t"]) == (None, 2, 1)
    validate_report(embedded[0], "crop_provenance.schema.json")


def test_edge_tile_cache_retains_compact_pixel_array(tmp_path):
    path = tmp_path / "thin-edge-tile.ome.tif"
    values = np.arange(1024, dtype=np.uint16)[None, :]
    tifffile.imwrite(path, values, ome=True, photometric="minisblack",
                     tile=(1024, 1024), compression="deflate", metadata=metadata("YX"))
    with OMETiffReader(path, cache_bytes=4096) as image:
        np.testing.assert_array_equal(image.asarray(), values)
        cache = image._source._block_cache
        assert cache.info()["resident_bytes"] <= 4096
        assert cache.info()["entries"] == 1
        for entry in cache._entries.values():
            array = entry.value[0]
            assert array.flags.owndata  # The 1x1024 view must not pin a 1024x1024 tile.
            assert array.nbytes == values.nbytes


def test_optional_channel_fallback_uses_physical_rgb_sample_group(tmp_path):
    path = tmp_path / "channel-less-rgb.ome.tif"
    values = np.arange(8 * 9 * 3, dtype=np.uint8).reshape(8, 9, 3)
    tifffile.imwrite(path, values, ome=True, photometric="rgb", metadata=metadata("YXS"))
    with tifffile.TiffFile(path) as tiff:
        root = etree.fromstring(tiff.ome_metadata.encode("utf-8"))
    pixels = root.find(f"{{{OME}}}Image/{{{OME}}}Pixels")
    for channel in pixels.findall(f"{{{OME}}}Channel"):
        pixels.remove(channel)
    tifffile.imwrite(path, values, photometric="rgb", metadata=None,
                     description=etree.tostring(root))
    with open_dataset(path, input_type="ome_tiff") as dataset:
        assert dataset.axes == "TZYXS"
        with dataset.plane() as image:
            assert image.image_type == "rgb"
            np.testing.assert_array_equal(image.asarray(), values)


def test_selected_zarr_plane_crop_records_group_and_indices(tmp_path):
    # Mandatory Zarr integration, not a mocked backend or an optional skip.
    from omeify import ImageDataset, OMEZarrWriter

    path = tmp_path / "source.zarr"
    values = np.arange(2 * 1 * 3 * 17 * 19, dtype=np.uint16).reshape(2, 1, 3, 17, 19)
    with ImageDataset.from_array(values, axes="TCZYX", pixel_size=PixelSize(.5, .5, "µm")) as ds:
        OMEZarrWriter(path, tile_size=16, pyramid_levels=0).write_dataset(ds)
    cropped = crop(path, tmp_path / "cropped.zarr", input_type="ome_zarr",
                   output_format="ome_zarr", group="", z=2, t=1,
                   bounds=(1, 2, 15, 16), tile_size=16, pyramid_levels=0)
    assert (cropped["source_group"], cropped["source_z"], cropped["source_t"]) == ("", 2, 1)
    validate_report(cropped, "crop_report.schema.json")
    import zarr
    root = zarr.open_group(str(tmp_path / "cropped.zarr"), mode="r", use_consolidated=False)
    try:
        provenance = root.attrs.asdict()["omeify"]["provenance"]
        assert (provenance["source_group"], provenance["source_z"],
                provenance["source_t"]) == ("", 2, 1)
    finally:
        root.store.close()
