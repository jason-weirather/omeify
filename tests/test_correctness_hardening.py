"""Regression tests for the 0.16.0 review, with independent raster expectations.

Core-engine tests deliberately do not call the public writer's XSD preflight;
no validator is replaced. Public convert/mutate tests below require the real
local XSD and skip explicitly when it is unavailable. Codec tests use the real
codec when installed. Synthetic pixels are never production-derived fixtures.
"""

from __future__ import annotations

import errno
import json

import numpy as np
import pytest
import tifffile
from lxml import etree

from omeify import OMETiffReader, OMETiffWriter, PixelSize, TiffInspector, convert, mutate
from omeify.io._writer import WriterEngine
from omeify.io._writer.preparation import prepare_image
from omeify.io._writer.precision import round_float32_mantissa
from omeify.io._writer.pyramid import mean_downsample_2x
from omeify.io._writer.single_verification import verify_single_output
from omeify.io._writer.source import ArraySource
from omeify.io.spec import OMEImageSpec
from omeify.io.tiff import TiffPlaneReader
from omeify.utils.generate_ome_xml import generate_multi_series_ome_xml, generate_ome_xml
from omeify.utils.miti_header_validator import validate_miti_ome_tiff_header
from omeify.utils.ome_schema_validator import OMESchemaValidator

OME = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
DTYPES = ("uint8", "uint16", "uint32", "int8", "int16", "int32", "float32", "float64")
SIZE = PixelSize(0.5, 0.6, "µm")


def _spec(data, *, names=("DAPI",), size=SIZE, image_type="multichannel"):
    axes = "YXS" if image_type == "rgb" else "YX" if data.ndim == 2 else "CYX"
    return OMEImageSpec.from_shape(
        image_type=image_type, axes=axes, shape=data.shape, dtype=data.dtype,
        channel_names=names, pixel_size=size,
    )


def _prepared(data, path, *, source=None, names=("DAPI",), size=SIZE, levels=1,
              overwrite=True, precision=None, image_type="multichannel",
              compression="Uncompressed"):
    writer = OMETiffWriter(
        path, channel_names=names, pixel_size=size, image_type=image_type,
        compression=compression, tile_size=16, pyramid_levels=levels,
        overwrite=overwrite, float32_mantissa_bits=precision,
        cache_directory=path.parent / "scratch",
    )
    spec = _spec(data, names=names, size=size, image_type=image_type)
    item = prepare_image(
        ArraySource(data, spec) if source is None else source, spec, name=None,
        downsample=None, compression_name=compression, settings=writer._settings,
        lossy_policy="rgb-only",
    )
    xml = generate_ome_xml(item.spec, item.level_shapes)["xml_string"]
    return writer, item, xml


def _write_core(data, path, **options):
    writer, item, xml = _prepared(data, path, **options)
    result = WriterEngine(writer._settings).write(
        (item,), xml,
        verify=lambda path: verify_single_output(item, path, software=writer.software),
    )
    return result.verification


def _source(path, data, *, rows=3, byteorder="<", compression=None):
    axes = "YX" if data.ndim == 2 else "CYX"
    names = ["DAPI"] if data.ndim == 2 else [f"C{i}" for i in range(data.shape[0])]
    tifffile.imwrite(
        path, data, ome=True, photometric="minisblack", rowsperstrip=rows,
        byteorder=byteorder, compression=compression,
        metadata={"axes": axes, "PhysicalSizeX": SIZE.x, "PhysicalSizeY": SIZE.y,
                  "PhysicalSizeXUnit": "µm", "PhysicalSizeYUnit": "µm",
                  "Channel": {"Name": names}},
    )


def _independent_mean(data):
    """Small-fixture scalar oracle, not a call to omeify's downsampler."""
    h, w = data.shape
    output = np.empty(((h + 1) // 2, (w + 1) // 2), dtype=data.dtype)
    for y in range(output.shape[0]):
        for x in range(output.shape[1]):
            values = data[2*y:2*y+2, 2*x:2*x+2].ravel().tolist()
            value = sum(values) / len(values)
            output[y, x] = round(value) if data.dtype.kind in "iu" else value
    return output


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape,rows", [((32, 48), 1), ((7, 8), 3), ((1, 9), 8),
                                      ((9, 1), 4), ((1, 1), 1), ((33, 49), 8)])
def test_grayscale_strips_keep_both_spatial_axes_and_round_trip(tmp_path, dtype, shape, rows):
    data = ((np.arange(np.prod(shape)).reshape(shape) % 97) + 1).astype(dtype)
    if data.dtype.kind == "i":
        data -= 48
    if data.dtype.kind == "f":
        data /= 3
    path = tmp_path / "strips.ome.tif"
    _source(path, data, rows=rows)
    np.testing.assert_array_equal(tifffile.imread(path), data)
    with OMETiffReader(path) as reader:
        h, w = shape
        for y0, y1, x0, x1 in [(0, h, 0, w), (h-1, h, 0, w), (0, h, w-1, w-1),
                                (0, h, w-1, w)]:
            np.testing.assert_array_equal(reader.read_region(y0, y1, x0, x1),
                                          data[y0:y1, x0:x1])
        output = tmp_path / "output.ome.tif"
        levels = int(max(shape) > 1)
        _write_core(data, output, source=reader, levels=levels)
    with tifffile.TiffFile(output) as tiff:
        np.testing.assert_array_equal(tiff.asarray(), data)
        if levels:
            actual = tiff.series[0].levels[1].asarray()
            expected = _independent_mean(data)
            if data.dtype.kind == "f":
                # Python's scalar sum may use compensated summation on newer
                # interpreters. Permit only rounding-scale differences here;
                # full-resolution values above must match exactly.
                np.testing.assert_allclose(actual, expected, rtol=4*np.finfo(data.dtype).eps,
                                           atol=0)
            else:
                np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("dtype", ["uint16", "float32"])
@pytest.mark.parametrize("codec", [None, "deflate", "lzw"])
def test_strips_across_byte_orders_and_codecs(tmp_path, byteorder, dtype, codec):
    if codec == "lzw":
        pytest.importorskip("imagecodecs")
    data = np.arange(70).reshape(7, 10).astype(dtype)
    path = tmp_path / "source.ome.tif"
    _source(path, data, rows=3, byteorder=byteorder, compression=codec)
    with OMETiffReader(path) as reader:
        np.testing.assert_array_equal(reader.read_region(0, 7, 0, 10), data)


@pytest.mark.parametrize("shape", [(1, 9, 3), (7, 8, 3)])
def test_one_row_rgb_strips_are_not_squeezed(tmp_path, shape):
    path = tmp_path / "rgb.tif"
    data = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    tifffile.imwrite(path, data, photometric="rgb", rowsperstrip=1, metadata=None)
    with tifffile.TiffFile(path) as tiff:
        reader = TiffPlaneReader(tiff.pages[0])
        np.testing.assert_array_equal(reader.read_region(0, shape[0], 0, shape[1]), data)


@pytest.mark.parametrize("fault", ["width", "height", "origin", "missing"])
def test_bad_decoded_segments_fail_instead_of_zero_filling(tmp_path, monkeypatch, fault):
    path = tmp_path / "short.tif"
    tifffile.imwrite(path, np.ones((7, 8), np.uint16), metadata=None, rowsperstrip=3)
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        original = page.decode

        def decode(*args, **kwargs):
            array, indices, shape = original(*args, **kwargs)
            if fault == "width":
                array = array[:, :, :1, :]
            elif fault == "height":
                array = array[:, :1, :, :]
            elif fault == "origin":
                indices = (*indices[:2], 1, *indices[3:])
            else:
                array = None
            return array, indices, shape

        monkeypatch.setattr(page, "decode", decode)
        with pytest.raises(ValueError, match="TIFF segment"):
            TiffPlaneReader(page).read_region(0, 7, 0, 8)


def test_sparse_declared_segment_still_reads_as_zero(tmp_path, monkeypatch):
    path = tmp_path / "sparse.tif"
    tifffile.imwrite(path, np.ones((1, 8), np.uint16), metadata=None)
    with tifffile.TiffFile(path) as tiff:
        page = tiff.pages[0]
        monkeypatch.setattr(page, "dataoffsets", (0,))
        monkeypatch.setattr(page, "databytecounts", (0,))
        np.testing.assert_array_equal(TiffPlaneReader(page).read_region(0, 1, 0, 8),
                                      np.zeros((1, 8), np.uint16))


def _metadata_only_source(path, *, comments=False, overlap=False, external=False, foreign=False):
    data = np.arange(2*7*9, dtype=np.uint16).reshape(2, 7, 9)
    absent = _spec(data, names=("WRONG0", "WRONG1"), size=PixelSize(5, 6, "µm"))
    stored = _spec(data, names=("REAL0", "REAL1"))
    xml = generate_multi_series_ome_xml((
        ("Metadata only", absent, (data.shape,)),
        ("Stored image", stored, (data.shape,)),
    ))["xml_string"]
    root = etree.fromstring(xml.encode())
    pixels = root.findall(f"{{{OME}}}Image/{{{OME}}}Pixels")
    first_data = pixels[0].find(f"{{{OME}}}TiffData")
    if not overlap:
        pixels[0].remove(first_data)
        etree.SubElement(pixels[0], f"{{{OME}}}MetadataOnly")
    td = pixels[1].find(f"{{{OME}}}TiffData")
    td.set("IFD", "0")
    if external:
        child = etree.SubElement(td, f"{{{OME}}}UUID", FileName="NEVER_OPEN.ome.tif")
        child.text = "urn:uuid:other"
    if foreign:
        root.insert(0, etree.Element("{urn:example:vendor}Image", Name="Not an OME Image"))
    if comments:
        for node in list(root.iter()):
            node.insert(0, etree.Comment("valid comment"))
            node.insert(1, etree.ProcessingInstruction("example", "keep-as-data"))
    tifffile.imwrite(path, data, photometric="minisblack", rowsperstrip=3,
                     metadata=None, description=etree.tostring(root),
                     resolution=(20000, 10000 / 0.6), resolutionunit=3)
    return data


@pytest.mark.parametrize("comments", [False, True])
@pytest.mark.parametrize("foreign", [False, True])
def test_reader_uses_mapped_image_not_ome_ordinal(tmp_path, comments, foreign):
    path = tmp_path / "metadata-only.ome.tif"
    data = _metadata_only_source(path, comments=comments, foreign=foreign)
    with OMETiffReader(path) as reader:
        assert reader.series_count == 1
        assert reader.series_name == "Stored image"
        assert reader.series_names == ("Stored image",)
        assert reader.channel_names == ("REAL0", "REAL1")
        assert tuple(channel.id for channel in reader.channels) == ("Channel:1:0", "Channel:1:1")
        assert reader.pixel_size == SIZE
        assert reader.inspection_report["series"][0]["ome_image_index"] == 1
        np.testing.assert_array_equal(reader.read_region(0, 7, 0, 9), data)
        output = tmp_path / "copied.ome.tif"
        _write_core(data, output, source=reader, names=reader.channel_names, size=reader.pixel_size)
    with OMETiffReader(output) as reader:
        assert reader.channel_names == ("REAL0", "REAL1")
        assert reader.pixel_size == SIZE
    inspector = TiffInspector(path, detail=3)
    assert inspector.validation_errors() == ()
    assert not inspector.report["warnings"]


@pytest.mark.parametrize("fault", ["overlap", "external", "scan_limit"])
def test_reader_refuses_unproven_ome_mapping(tmp_path, monkeypatch, fault):
    import omeify._calibration as calibration

    path = tmp_path / "unproven.ome.tif"
    _metadata_only_source(path, overlap=fault == "overlap", external=fault == "external")
    if fault == "scan_limit":
        monkeypatch.setattr(calibration, "_MAX_PAGES", 1)
    reader = OMETiffReader(path)
    with pytest.raises(ValueError, match="Cannot associate TIFF series"):
        reader.open()
    assert not reader.is_open


def test_xml_comments_preserve_miti_and_metadata_summary(tmp_path):
    data = np.ones((7, 9), np.uint16)
    xml = generate_ome_xml(_spec(data), (data.shape,))["xml_string"]
    root = etree.fromstring(xml.encode())
    for node in list(root.iter()):
        node.insert(0, etree.Comment("comment"))
        node.insert(1, etree.ProcessingInstruction("note", "content"))
    edited = etree.tostring(root).decode()
    assert validate_miti_ome_tiff_header(edited).is_valid
    path = tmp_path / "comments.ome.tif"
    tifffile.imwrite(path, data, metadata=None, description=edited.encode())
    inspector = TiffInspector(path, detail=3)
    assert inspector.validation_errors() == ()
    assert inspector.report["ome"]["images"][0]["channels"][0]["name"] == "DAPI"
    parsed = inspector.report["series"][0]["levels"][0]["pages"][0]["description"]["parsed_xml"]
    assert parsed["children"][0]["tag"] == "#comment"
    assert parsed["children"][1]["tag"] == "#processing-instruction"


@pytest.mark.parametrize("shape", [(33, 49), (1, 33), (35, 1)])
def test_odd_pyramid_sizes_use_level_tags(tmp_path, shape):
    output = tmp_path / "odd.ome.tif"
    _write_core(np.zeros(shape, np.uint16), output, levels=3)
    with OMETiffReader(output) as reader:
        assert reader.pixel_size_at_level(0) == SIZE
        for level in range(1, 4):
            assert reader.pixel_size_at_level(level) == SIZE.scaled(2**level)


@pytest.mark.parametrize("calibrated", [True, False])
def test_third_party_levels_are_not_assumed_to_be_twofold(tmp_path, calibrated, caplog):
    path = tmp_path / "third-party.ome.tif"
    base = np.ones((33, 45), np.uint16)
    xml = generate_ome_xml(_spec(base), (base.shape,))["xml_string"]
    with tifffile.TiffWriter(path, ome=False) as writer:
        writer.write(base, description=xml.encode(), metadata=None, subifds=1, tile=(16, 16))
        writer.write(base[::3, ::3], subfiletype=1, tile=(16, 16), metadata=None,
                     resolution=(10000/1.5, 10000/1.8) if calibrated else (1, 1),
                     resolutionunit=3 if calibrated else 1)
    with OMETiffReader(path) as reader:
        actual = reader.pixel_size_at_level(1)
        assert actual == (PixelSize(1.5, 1.8, "µm") if calibrated else None)
    if not calibrated:
        assert "not be inferred" in caplog.text


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("coordinate", [(0, 0), (1, 1), (16, 24), (32, 48)])
def test_preserved_nan_does_not_depend_on_spot_check_location(tmp_path, dtype, coordinate):
    data = np.ones((33, 49), dtype=dtype)
    data[coordinate] = np.nan
    output = tmp_path / "nan.ome.tif"
    report = _write_core(data, output, levels=2)
    with tifffile.TiffFile(output) as tiff:
        actual = tiff.asarray()
    np.testing.assert_array_equal(actual.view(f"u{data.dtype.itemsize}"),
                                  data.view(f"u{data.dtype.itemsize}"))
    assert report["base_pixel_values_match"] is True


def test_lossless_bit_comparison_distinguishes_nan_payloads_and_signed_zero():
    from omeify.io._writer.verification import same_sample_bits

    bits = np.array([0x80000000, 0x7FC12345, 0x7FA12345, 0xFF800000], dtype="<u4")
    data = bits.view("<f4")
    swapped = bits.astype(">u4").view(">f4")
    assert same_sample_bits(data, swapped)
    changed = data.copy()
    changed.view("u4")[0] = 0
    assert not same_sample_bits(data, changed)
    changed = data.copy()
    changed.view("u4")[1] ^= np.uint32(1)
    assert not same_sample_bits(data, changed)


@pytest.mark.parametrize("bits", range(23))
def test_precision_overflow_is_explicit_instead_of_fabricated_infinity(bits):
    for sign in (1, -1):
        values = np.array([sign * np.finfo(np.float32).max], dtype=np.float32)
        before = values.copy()
        with pytest.raises(OverflowError, match="finite.*infinity"):
            round_float32_mantissa(values, bits)
        np.testing.assert_array_equal(values, before)
        retained_max = np.array([0x7F7FFFFF & ~((1 << (23-bits))-1)], np.uint32).view(np.float32)
        assert np.isfinite(round_float32_mantissa(retained_max, bits)).all()


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("bits", [0, 11, 22, 23])
def test_precision_preserves_nonfinite_bit_patterns_and_endianness(byteorder, bits):
    raw = np.array([0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC12345, 0x7FA12345,
                    0xFFC54321], dtype=byteorder+"u4")
    source = raw.view(byteorder+"f4")
    expected = raw.astype(np.uint32)
    actual = round_float32_mantissa(source, bits)
    assert actual.dtype == np.dtype(np.float32)
    np.testing.assert_array_equal(actual.view(np.uint32), expected)
    np.testing.assert_array_equal(source.view(byteorder+"u4"), raw)


@pytest.mark.parametrize("byteorder", ["<", ">"])
def test_non_native_trimmed_writer_has_identical_values(tmp_path, byteorder):
    data = np.array([[1.000244140625, 1.000732421875, -1.000732421875]], dtype=byteorder+"f4")
    output = tmp_path / "trim.ome.tif"
    _write_core(data, output, precision=11, levels=0)
    # Hand-calculated ties-to-even on the 11-fraction-bit lattice near +/-1.
    np.testing.assert_array_equal(tifffile.imread(output), [[1.0, 1.0009765625, -1.0009765625]])


@pytest.mark.parametrize("levels", [0, 1])
def test_precision_overflow_does_not_replace_destination(tmp_path, levels):
    output = tmp_path / "existing.ome.tif"
    output.write_bytes(b"KEEP")
    data = np.full((17, 17), np.finfo(np.float32).max, np.float32)
    with pytest.raises(OverflowError):
        _write_core(data, output, precision=11, levels=levels)
    assert output.read_bytes() == b"KEEP"
    assert not list(tmp_path.glob(".omeify-*.partial"))
    assert not list((tmp_path / "scratch").iterdir())


@pytest.mark.parametrize("value", [1e308, -1e308, np.finfo(np.float64).max])
def test_float64_mean_does_not_overflow_finite_constant(value):
    data = np.full((3, 5), value, np.float64)
    with np.errstate(over="raise", invalid="raise"):
        actual = mean_downsample_2x(data, (2, 3))
    np.testing.assert_array_equal(actual, np.full((2, 3), value))


def test_float64_extreme_cancellation_and_nonfinite_policy():
    m = np.finfo(np.float64).max
    data = np.array([[m, m, np.nan, 1], [-m, -m, 2, 3],
                     [np.inf, 1, np.inf, -np.inf]], np.float64)
    actual = mean_downsample_2x(data, (2, 2))
    assert actual[0, 0] == 0
    assert np.isnan(actual[0, 1])
    assert np.isposinf(actual[1, 0])
    assert np.isnan(actual[1, 1])


def test_float64_regular_means_keep_existing_arithmetic():
    data = np.random.default_rng(42).normal(size=(33, 49))
    # Reference reproduces the previous ordinary accumulator operation, not its overflow bug.
    expected = np.zeros((17, 25), np.float64)
    counts = np.zeros_like(expected)
    for dy in (0, 1):
        for dx in (0, 1):
            values = data[dy::2, dx::2]
            h, w = values.shape
            expected[:h, :w] += values
            counts[:h, :w] += 1
    expected /= counts
    np.testing.assert_array_equal(mean_downsample_2x(data, (17, 25)), expected)


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_no_overwrite_survives_destination_created_during_verification(tmp_path, kind):
    destination = tmp_path / "race.ome.tif"
    target = tmp_path / "other-file"
    target.write_bytes(b"OTHER")
    writer, item, xml = _prepared(np.ones((17, 19), np.uint16), destination, overwrite=False)

    def verify(path):
        result = verify_single_output(item, path, software=writer.software)
        if kind == "file":
            destination.write_bytes(b"COMPETING")
        else:
            try:
                destination.symlink_to(target)
            except OSError:
                pytest.skip("symlink creation unavailable")
        return result

    with pytest.raises(FileExistsError):
        WriterEngine(writer._settings).write((item,), xml, verify=verify)
    assert destination.read_bytes() == (b"COMPETING" if kind == "file" else b"OTHER")
    assert target.read_bytes() == b"OTHER"
    if kind == "symlink":
        assert destination.is_symlink()
    assert not list(tmp_path.glob(".omeify-*.partial"))
    assert not list((tmp_path / "scratch").iterdir())


def test_no_overwrite_rejects_dangling_symlink_early(tmp_path):
    destination = tmp_path / "dangling.ome.tif"
    try:
        destination.symlink_to(tmp_path / "missing")
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(FileExistsError):
        _write_core(np.ones((8, 8), np.uint16), destination, overwrite=False)
    assert destination.is_symlink()


def test_no_overwrite_installs_complete_file(tmp_path):
    data = np.arange(99, dtype=np.uint16).reshape(9, 11)
    output = tmp_path / "new.ome.tif"
    _write_core(data, output, overwrite=False)
    np.testing.assert_array_equal(tifffile.imread(output), data)
    assert not list(tmp_path.glob(".omeify-*.partial"))


def test_no_overwrite_never_falls_back_when_hardlinks_are_unavailable(tmp_path, monkeypatch):
    import omeify.io._writer.engine as engine

    def unsupported(*args, **kwargs):
        raise OSError(errno.EOPNOTSUPP, "hard links unavailable")

    monkeypatch.setattr(engine.os, "link", unsupported)
    output = tmp_path / "new.ome.tif"
    with pytest.raises(OSError, match="hard links unavailable"):
        _write_core(np.ones((8, 8), np.uint16), output, overwrite=False)
    assert not output.exists()
    assert not list(tmp_path.glob(".omeify-*.partial"))


def test_verification_states_sampling_scope_and_checks_each_pyramid_level(tmp_path):
    output = tmp_path / "levels.ome.tif"
    report = _write_core(np.ones((33, 49), np.uint16), output, levels=2)
    scope = report["pixel_verification"]
    assert scope["mode"] == "sampled" and scope["all_pixels_checked"] is False
    assert scope["base_points_decoded"] == scope["lossless_base_points_compared"] == 3
    assert scope["pyramid_points_decoded"] == scope["lossless_pyramid_points_compared"] == 6
    assert scope["plane_levels_checked"] == 3
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("level", [0, 1, 2])
def test_sampled_pixel_corruption_is_rejected_before_install(tmp_path, monkeypatch, level):
    import omeify.io._writer.engine as engine

    original = engine.write_output

    def corrupt(*args, **kwargs):
        original(*args, **kwargs)
        path = args[2]
        with tifffile.TiffFile(path) as tiff:
            offset = tiff.series[0].levels[level].pages[0].dataoffsets[0]
        with path.open("r+b") as handle:
            handle.seek(offset)
            handle.write(b"\xff\xff")

    monkeypatch.setattr(engine, "write_output", corrupt)
    output = tmp_path / "existing.ome.tif"
    output.write_bytes(b"KEEP")
    with pytest.raises(ValueError, match="verification failed"):
        _write_core(np.ones((33, 49), np.uint16), output, levels=2)
    assert output.read_bytes() == b"KEEP"
    assert not list(tmp_path.glob(".omeify-*.partial"))


def test_mean_and_nearest_pyramids_use_independent_expected_arrays(tmp_path):
    data = np.arange(33*49, dtype=np.uint32).reshape(33, 49)
    for kind in ("multichannel", "label"):
        path = tmp_path / f"{kind}.ome.tif"
        _write_core(data, path, image_type=kind, levels=2)
        expected = data
        with tifffile.TiffFile(path) as tiff:
            for level in tiff.series[0].levels:
                np.testing.assert_array_equal(level.asarray(), expected)
                expected = expected[::2, ::2] if kind == "label" else _independent_mean(expected)


def test_ordinary_inspection_escapes_controls_without_altering_json(tmp_path):
    path = tmp_path / "controls.tif"
    description = "metadata\x1b[31m\x07\r\b\u202e µm"
    tifffile.imwrite(path, np.ones((8, 8), np.uint8), metadata=None,
                     description=description.encode("utf-8"))
    inspector = TiffInspector(path, detail=3, max_text_length=None)
    rendered = inspector.render_text()
    assert not any(char in rendered for char in ("\x1b", "\x07", "\r", "\b", "\u202e"))
    assert "µm" in rendered and "\\x1b" in rendered
    report = json.loads(inspector.to_json())
    assert report["series"][0]["levels"][0]["pages"][0]["description"]["value"] == description


def _require_xsd():
    if OMESchemaValidator().schema_lxml is None:
        pytest.skip("public workflow requires the real local ome-schema XSD")


@pytest.mark.parametrize("workflow", ["convert", "mutate"])
def test_public_workflow_preserves_strips_and_correct_metadata(tmp_path, workflow):
    _require_xsd()
    source = tmp_path / "source.ome.tif"
    data = np.arange(7*9, dtype=np.float32).reshape(7, 9)
    _source(source, data, rows=3)
    output = tmp_path / "output.ome.tif"
    options = dict(input_type="ome_tiff", compression="Uncompressed", tile_size=16,
                   pyramid_levels=1)
    if workflow == "convert":
        convert(source, output, **options)
    else:
        mutate(source, output, float32_mantissa_bits=11, **options)
    np.testing.assert_array_equal(tifffile.imread(output), data)


def test_public_convert_uses_metadata_only_image_mapping(tmp_path):
    _require_xsd()
    source, output = tmp_path / "source.ome.tif", tmp_path / "output.ome.tif"
    data = _metadata_only_source(source, comments=True)
    convert(source, output, input_type="ome_tiff", compression="Uncompressed", tile_size=16,
            pyramid_levels=1)
    np.testing.assert_array_equal(tifffile.imread(output), data)
    with OMETiffReader(output) as reader:
        assert reader.pixel_size == SIZE
        assert reader.channel_names == ("REAL0", "REAL1")


@pytest.mark.parametrize("lossy_preview", [False, True])
def test_shared_multi_series_verification_and_reader_contract(tmp_path, lossy_preview):
    from omeify import OMEMultiSeriesWriter
    from omeify.io.ome_multi_series_writer.verification import verify_output

    if lossy_preview:
        pytest.importorskip("imagecodecs")
    fluorescence = np.arange(2*33*49, dtype=np.float32).reshape(2, 33, 49)
    fluorescence[0, 0, 0] = np.nan
    labels = np.arange(33*49, dtype=np.uint32).reshape(33, 49)
    rgb = (np.arange(33*49*3).reshape(33, 49, 3) % 256).astype(np.uint8)
    path = tmp_path / "multi.ome.tif"
    writer = OMEMultiSeriesWriter(path, tile_size=16, pyramid_levels=2,
                                  compression="Uncompressed", overwrite=False)
    records = [
        ("Fluorescence", fluorescence, "multichannel", ("DAPI", "Marker"), "Uncompressed"),
        ("Labels", labels, "label", ("Objects",), "Uncompressed"),
        ("RGB preview", rgb, "rgb", ("RGB",), "JPEG" if lossy_preview else "Uncompressed"),
    ]
    images = tuple(prepare_image(
        ArraySource(data, _spec(data, image_type=kind, names=names)),
        _spec(data, image_type=kind, names=names),
        name=name, downsample=None, compression_name=compression,
        settings=writer._settings, lossy_policy="non-label",
    ) for name, data, kind, names, compression in records)
    xml = generate_multi_series_ome_xml(tuple(
        (item.name, item.spec, item.level_shapes) for item in images
    ))["xml_string"]
    result = WriterEngine(writer._settings).write(
        images, xml, verify=lambda p: verify_output(
            images, p, provenance_json=None, software=writer._settings.software,
        ),
    )
    coverage = result.verification["pixel_verification"]
    assert coverage["plane_levels_checked"] == 12
    assert coverage["pyramid_points_decoded"] == 24
    assert coverage["lossless_pyramid_points_compared"] == (18 if lossy_preview else 24)
    with tifffile.TiffFile(path) as tiff:
        np.testing.assert_array_equal(tiff.series[0].asarray(), fluorescence)
        np.testing.assert_array_equal(tiff.series[1].asarray(), labels)
        np.testing.assert_array_equal(tiff.series[1].levels[2].asarray(), labels[::4, ::4])
    for index, (name, data, kind, names, compression) in enumerate(records):
        with OMETiffReader(path, series=index) as reader:
            assert reader.series_name == name
            assert reader.series_names == tuple(record[0] for record in records)
            assert reader.channel_names == names
            assert reader.pixel_size_at_level(2) == SIZE.scaled(4)


def test_extreme_float64_writer_pyramids_stay_finite(tmp_path):
    data = np.full((33, 49), 1e308, np.float64)
    path = tmp_path / "extreme.ome.tif"
    _write_core(data, path, levels=2)
    with tifffile.TiffFile(path) as tiff:
        for level in tiff.series[0].levels:
            np.testing.assert_array_equal(level.asarray(), np.full(level.shape, 1e308))


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("width", [4, 8])
def test_writer_preserves_nan_payloads_infinities_and_signed_zero(tmp_path, byteorder, width):
    patterns = (
        [0, 0x80000000, 0x7FC12345, 0x7FA12345, 0x7F800000, 0xFF800000, 0xFFC54321, 1]
        if width == 4 else
        [0, 0x8000000000000000, 0x7FF8000000012345, 0x7FF0000000012345,
         0x7FF0000000000000, 0xFFF0000000000000, 0xFFF8000000054321, 1]
    )
    raw = np.array(patterns, dtype=f"{byteorder}u{width}").reshape(2, 4)
    data = raw.view(f"{byteorder}f{width}")
    path = tmp_path / "bit-patterns.ome.tif"
    _write_core(data, path, levels=0)
    actual = tifffile.imread(path)
    np.testing.assert_array_equal(actual.view(f"u{width}"), raw.astype(f"u{width}"))
