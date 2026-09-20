"""Current image/provider workflows, not compatibility tests for retired APIs."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from omeify import (
    Image, ImageLevel, ImageMetadata, ImageSource, LabelImage, MultichannelImage,
    OMEImageSeries, OMEMultiSeriesWriter, OMETiffLabelReader, OMETiffReader,
    OMETiffWriter, PixelSize, RGBImage, TemporaryOMETiffWriter,
)

from omeify.workflow import build_input_report

SIZE = PixelSize(0.5, 0.5, "µm")


class CoordinateSource(ImageSource):
    def __init__(self, kind="multichannel", height=65, width=79, *, channel_metadata=None):
        axes = {"multichannel": "CYX", "rgb": "YXS", "label": "YX"}[kind]
        shape = (
            (2, height, width) if axes == "CYX"
            else (height, width, 3) if axes == "YXS" else (height, width)
        )
        super().__init__(ImageMetadata(
            axes=axes, shape=shape, dtype=np.dtype("uint8" if kind == "rgb" else "uint32"),
            image_type=kind, pixel_size=SIZE, channel_metadata=channel_metadata,
        ))
        self.calls = []

    def _read_region(self, y0, y1, x0, x1, *, level, channels):
        self.calls.append((y0, y1, x0, x1, level, channels))
        y, x = np.ogrid[y0:y1, x0:x1]
        values = (y * 13 + x * 3).astype(self.metadata.dtype)
        if self.metadata.axes == "CYX":
            return np.stack([values + c for c in channels])
        if self.metadata.axes == "YXS":
            return np.stack([values, values // 2, values // 3], axis=-1)
        return values + 1000


@pytest.mark.parametrize(
    "kind,cls",
    [("multichannel", MultichannelImage), ("rgb", RGBImage), ("label", LabelImage)],
)
def test_procedural_reads_are_bounded_repeatable_and_owned(kind, cls):
    source = CoordinateSource(kind, height=100_000, width=120_000)
    with cls(source) as image:
        repr(image)
        assert not source.calls
        patch = image.read_region(20, 40, 10, 30)
        repeat = image.read_region(23, 35, 15, 24)
        window = patch[:, 3:15, 5:14] if image.axes == "CYX" else patch[3:15, 5:14]
        np.testing.assert_array_equal(repeat, window)
        channel = image[0]
        with pytest.raises(AttributeError):
            channel.name = "Changed"
        assert image.read_region(0, 0, 2, 5).size == 0
        assert len(source.calls) == 2
    assert patch.size
    with pytest.raises(RuntimeError):
        channel.read_region(0, 1, 0, 1)


def test_array_copy_borrow_and_noncontiguous_reads():
    data = np.arange(3 * 15 * 17, dtype=np.float32).reshape(3, 15, 17)[:, ::2, ::2]
    with MultichannelImage.from_array(
        data,
        axes="CYX",
    ) as borrowed, MultichannelImage.from_array(data, axes="CYX", copy=True) as copied:
        original = copied.asarray()
        data[...] = 4  # Deliberate demonstration of the documented borrowing obligation.
        np.testing.assert_array_equal(copied.asarray(), original)
        assert np.all(borrowed.asarray() == 4)
        patch = borrowed.asarray()
        patch[...] = 8
        assert np.all(data == 4)


def test_lifetime_borrowing_and_reopened_parent():
    source = CoordinateSource()
    with source:
        with MultichannelImage(source, owns_source=False) as parent:
            with MultichannelImage.from_channels([parent[1], parent[0]]) as subset:
                np.testing.assert_array_equal(subset.asarray(), parent.asarray()[[1, 0]])
                stale = subset[0]
            assert parent.is_open
        assert source.is_open
    source.open()
    parent.open()
    try:
        with pytest.raises(RuntimeError):
            stale.read_region(0, 1, 0, 1)
    finally:
        parent.close()
    source.close()


@pytest.mark.parametrize("bounds", [(-1, 2, 0, 2), (2, 1, 0, 2), (0, 900, 0, 2), (False, 2, 0, 2)])
def test_invalid_regions_never_reach_provider(bounds):
    source = CoordinateSource()
    with MultichannelImage(source) as image:
        with pytest.raises((ValueError, TypeError)):
            image.read_region(*bounds)
        assert not source.calls


@pytest.mark.parametrize("fault", ["shape", "dtype", "mask"])
def test_provider_output_validation(fault):
    class Broken(CoordinateSource):
        def _read_region(self, *args, **kwargs):
            values = super()._read_region(*args, **kwargs)
            if fault == "shape":
                return values[..., :1]
            if fault == "mask":
                return np.ma.array(values, mask=False)
            return values.astype(np.float64)
    with MultichannelImage(Broken()) as image, pytest.raises((ValueError, TypeError)):
        image.read_region(0, 5, 0, 6)


def test_metadata_views_and_calibrated_odd_pyramid():
    levels = (ImageLevel(0, "CYX", (2, 5, 7), SIZE),
              ImageLevel(1, "CYX", (2, 3, 4), SIZE.scaled(2), (2, 2)))
    base = np.arange(70, dtype=np.uint16).reshape(2, 5, 7)
    with MultichannelImage.from_array(base, axes="CYX", pixel_size=SIZE,
                                     level_arrays=[base[:, ::2, ::2]], levels=levels) as image:
        assert image.levels[1].downsample_yx == (2, 2)
        with image.with_metadata(channel_names=["A", "B"], pixel_size=SIZE.scaled(3)) as altered:
            assert altered.levels[1].pixel_size == SIZE.scaled(6)
            np.testing.assert_array_equal(altered.asarray(level=1), base[:, ::2, ::2])
        assert image.pixel_size == SIZE
        with pytest.raises(ValueError):
            replace(levels[0], downsample_yx=(2, 2))


@pytest.mark.parametrize(
    "kind,cls",
    [("multichannel", MultichannelImage), ("rgb", RGBImage), ("label", LabelImage)],
)
def test_procedural_write_and_real_read_use_identical_contract(tmp_path, kind, cls, monkeypatch):
    path = tmp_path / "image.ome.tif"
    source = CoordinateSource(kind)
    with cls(source) as image:
        expected = image.asarray()
        monkeypatch.setattr(
            Image,
            "asarray",
            lambda *a, **k: pytest.fail("whole image materialized"),
        )
        report = OMETiffWriter(path, compression="Deflate", tile_size=16).write(image)
        assert image.is_open and report["verification"]["base_pixel_values_match"]
        reader = OMETiffLabelReader(path) if kind == "label" else OMETiffReader(path)
        with reader as actual:
            assert actual.levels[0].shape == image.shape
            assert actual.levels[1].downsample_yx == (2, 2)
            np.testing.assert_array_equal(actual.read_region(0, 65, 0, 79), expected)
            assert actual.dtype == image.dtype
            assert actual.channel_names == image.channel_names


def test_channel_assembly_from_two_images_and_scope(image_factory):
    left = image_factory(np.full((5, 7), 3, np.float32), pixel_size=SIZE)
    right = image_factory(np.full((5, 7), 9, np.float32), pixel_size=SIZE)
    with MultichannelImage.from_channels(
        [right[0], left[0]],
        channel_names=["R", "L"],
    ) as composite:
        assert composite.shape == (2, 5, 7)
        assert composite.channel_names == ("R", "L")
        np.testing.assert_array_equal(composite.asarray()[:, 0, 0], [9, 3])
        left.close()
        with pytest.raises(RuntimeError):
            composite.read_region(0, 1, 0, 1)


def test_heterogeneous_and_temporary_writers_share_image_inputs(tmp_path, image_factory):
    rgb = image_factory(np.zeros((33, 47, 3), np.uint8), kind="rgb", pixel_size=SIZE)
    labels = image_factory(
        np.arange(33 * 47, dtype=np.uint32).reshape(33, 47),
        kind="label", pixel_size=SIZE,
    )
    entries = [OMEImageSeries("H&E", rgb), OMEImageSeries("Labels", labels)]
    path = tmp_path / "products.ome.tif"
    OMEMultiSeriesWriter(path, tile_size=16, compression="Deflate").write(entries)
    with OMETiffReader(path) as reader:
        assert reader.series_names == ("H&E", "Labels")
    with OMETiffLabelReader(path, series=1) as reader:
        np.testing.assert_array_equal(reader.asarray(), labels.asarray())
    with TemporaryOMETiffWriter(directory=tmp_path, tile_size=16, compression="Deflate") as writer:
        writer.write(rgb)
        temporary = writer.path
        assert temporary.exists()
    assert not temporary.exists()
    assert rgb.is_open and labels.is_open


def test_rgb_channel_zero_is_always_all_three_samples(image_factory):
    image = image_factory(np.zeros((5, 7, 3), np.uint8), kind="rgb")
    assert image.channel_count == 1 and image.sample_count == 3
    assert image.read_region(0, 2, 0, 3, channels=[0]).shape == (2, 3, 3)
    with pytest.raises(IndexError):
        image.read_region(0, 2, 0, 3, channels=[1])


def test_byte_order_and_special_float_values_are_not_numeric_casts():
    bits = np.array([0x80000000, 0x7fc01234, 0x7f800000, 0], dtype=">u4").reshape(2, 2)
    with MultichannelImage.from_array(bits.view(">f4"), axes="YX") as image:
        np.testing.assert_array_equal(image.asarray().view("u4"), bits.astype("u4"))


@pytest.mark.parametrize("alias", ["same", "hardlink", "symlink"])
def test_materializer_protects_file_dependencies(tmp_path, alias):
    path = tmp_path / "input.ome.tif"
    tifffile.imwrite(
        path, np.zeros((16, 16), np.uint16), ome=True,
        metadata={
            "axes": "YX", "PhysicalSizeX": .5, "PhysicalSizeY": .5,
            "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
        },
    )
    output = path if alias == "same" else tmp_path / "alias.ome.tif"
    if alias == "hardlink":
        output.hardlink_to(path)
    if alias == "symlink":
        output.symlink_to(path)
    with OMETiffReader(path) as image, pytest.raises(ValueError, match="backing"):
        OMETiffWriter(output, compression="Deflate").write(image)


def test_unknown_calibration_fails_before_writing_and_source_stays_open(tmp_path, image_factory):
    image = image_factory(np.zeros((5, 7), np.uint16))
    with pytest.raises(ValueError, match="calibrated"):
        OMETiffWriter(tmp_path / "missing.ome.tif").write(image)
    assert image.is_open and not (tmp_path / "missing.ome.tif").exists()


def test_nested_context_failure_does_not_close_outer():
    with MultichannelImage(CoordinateSource()) as image:
        with pytest.raises(RuntimeError):
            with image:
                pass
        assert image.is_open



def test_channel_metadata_is_a_recursive_snapshot() -> None:
    exposure = {"milliseconds": 7.5}
    steps = [{"settings": exposure}]
    wavelengths = [405, 450]
    record = {"steps": steps, "also": (exposure,), "wavelengths": wavelengths}
    source = CoordinateSource(channel_metadata=(record, record))
    frozen = source.metadata.channel_metadata[0]

    exposure["milliseconds"] = 99
    wavelengths.append(570)
    steps.clear()
    record["extra"] = "later"
    assert frozen["steps"][0]["settings"]["milliseconds"] == 7.5
    assert frozen["also"][0]["milliseconds"] == 7.5
    assert frozen["wavelengths"] == (405, 450)
    assert "extra" not in frozen
    with pytest.raises(TypeError):
        frozen["extra"] = "not writable"
    with pytest.raises(TypeError):
        frozen["steps"][0]["settings"]["milliseconds"] = 12
    with pytest.raises(TypeError):
        frozen["wavelengths"][0] = 488
    assert not source.calls


def test_frozen_channel_metadata_survives_views_and_report_export(tmp_path) -> None:
    record = {"settings": {"gains": [1.0, 2.0], "enabled": True, "note": None}}
    source = CoordinateSource(channel_metadata=(record, record))
    path = tmp_path / "input.fixture"
    path.write_bytes(b"Report size fixture; no file decoding is involved.")
    with MultichannelImage(source) as image:
        with image.with_metadata(channel_names=["DNA", "CD3"]) as renamed:
            with MultichannelImage.from_channels([renamed[1], renamed[0]]) as selected:
                assert selected[0].source_metadata["settings"]["gains"] == (1.0, 2.0)
                with pytest.raises(TypeError):
                    selected[0].source_metadata["settings"]["enabled"] = False
                assert not source.calls
                np.testing.assert_array_equal(
                    selected.read_region(0, 2, 0, 3),
                    image.read_region(0, 2, 0, 3, channels=[1, 0]),
                )

                # Exercise the actual shared report boundary without TIFF or schema fixtures.
                reader = SimpleNamespace(
                    channels=selected.channels, dtype=selected.dtype,
                    native_shape=selected.shape, shape=selected.shape,
                    native_axes=selected.axes, axes=selected.axes,
                    source_byte_order="little", input_type_description="test source",
                )
                report = build_input_report(reader, path, selected.pixel_size)
                exported = report["channels"][0]["source_metadata"]
                assert json.loads(json.dumps(exported)) == record
                exported["settings"]["gains"].append(3.0)
                assert selected[0].source_metadata["settings"]["gains"] == (1.0, 2.0)
        assert image.is_open


@pytest.mark.parametrize(
    "fault", ["object", "array", "key", "mapping_cycle", "sequence_cycle", "depth"],
)
def test_channel_metadata_rejects_unsupported_or_cyclic_values(fault) -> None:
    error = TypeError
    if fault == "object":
        record = {"value": object()}
    elif fault == "array":
        record = {"value": np.arange(3)}
    elif fault == "key":
        record = {"nested": {1: "not a string key"}}
    elif fault == "mapping_cycle":
        record = {}
        record["self"] = record
        error = ValueError
    elif fault == "sequence_cycle":
        loop = []
        loop.append((loop,))
        record = {"value": loop}
        error = ValueError
    else:
        record = {}
        for _ in range(64):
            record = {"nested": record}
        error = ValueError
    with pytest.raises(error, match="channel_metadata"):
        CoordinateSource(channel_metadata=(record, record))
