from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

from omeify import OMETiffReader, PixelSize, mutate
from omeify.cli import main


def _write_float_ome(path: Path, data: np.ndarray, names: list[str] | None = None) -> None:
    channel_names = names or [f"Marker {index + 1}" for index in range(data.shape[0])]
    tifffile.imwrite(
        path,
        data,
        ome=True,
        tile=(16, 16),
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.6,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": channel_names},
        },
    )


def _write_indica_float(path: Path, data: np.ndarray) -> None:
    channels = "".join(
        f'<channel id="{index}" name="Marker {index + 1}" min="0" max="150"/>'
        for index in range(data.shape[0])
    )
    dimensions = "".join(
        f'<dimension sizeX="{data.shape[2]}" sizeY="{data.shape[1]}" '
        f'ifd="{index}" channel="{index}" level="0"/>'
        for index in range(data.shape[0])
    )
    description = (
        "<indica><image><pixels>"
        + dimensions
        + "</pixels><channels>"
        + channels
        + "</channels></image></indica>"
    )
    with tifffile.TiffWriter(path, bigtiff=True) as writer:
        for index, plane in enumerate(data):
            options = {
                "tile": (16, 16),
                "description": description if index == 0 else None,
                "software": "IndicaLabsImageWriter synthetic" if index == 0 else False,
                "metadata": None,
            }
            if index == 0:
                options["resolution"] = (20_000.0, 25_000.0)
                options["resolutionunit"] = "CENTIMETER"
            writer.write(plane, **options)


def test_auto_preserves_integer_scale_when_float_values_show_integer_ancestry(
    tmp_path: Path,
) -> None:
    source = tmp_path / "integer-like.ome.tif"
    output = tmp_path / "integer-like.uint8.ome.tif"
    base = (np.arange(2 * 64 * 64, dtype=np.uint16) % 151).astype(np.float32)
    data = base.reshape(2, 64, 64)
    data[:, 1::8, 1::8] += np.float32(0.25)
    _write_float_ome(source, data, names=["DAPI", "PanCK"])

    report = mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    channels = report["dtype_mutation"]["channels"]
    assert [channel["mapping"]["name"] for channel in channels] == [
        "identity",
        "identity",
    ]
    assert channels[0]["integer_lattice_evidence"]["classification"] == "strong"
    assert channels[0]["mapping"]["quantum_source_units_per_output_code"] == 1.0
    assert channels[0]["mapping"]["exact_output_code_minimum_from_source_bounds"] == 0
    assert channels[0]["mapping"]["exact_output_code_maximum_from_source_bounds"] == 150
    assert channels[0]["anticipated_loss"]["clipped_finite_pixels"] == 0
    assert (
        channels[0]["anticipated_loss"]["theoretical_maximum_absolute_error_source_units"]
        == 0.5
    )

    with OMETiffReader(output) as reader:
        assert reader.dtype == np.dtype("uint8")
        assert reader.pixel_size == PixelSize(0.5, 0.6, "µm")
        np.testing.assert_array_equal(reader.asarray(), np.rint(data).astype(np.uint8))


def test_auto_preserves_1500_count_scale_in_uint16_but_rescales_for_uint8(
    tmp_path: Path,
) -> None:
    source = tmp_path / "wide-integer-like.ome.tif"
    uint16_output = tmp_path / "wide.uint16.ome.tif"
    uint8_output = tmp_path / "wide.uint8.ome.tif"
    data = (np.arange(64 * 64, dtype=np.uint16) % 1501).astype(np.float32).reshape(1, 64, 64)
    data[:, 2::16, 2::16] += np.float32(0.25)
    _write_float_ome(source, data, names=["Wide"])

    uint16_report = mutate(
        source,
        uint16_output,
        input_type="ome_tiff",
        dtype="uint16",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    uint16_channel = uint16_report["dtype_mutation"]["channels"][0]
    assert uint16_channel["mapping"]["name"] == "identity"
    assert uint16_channel["source"]["integer_code_span_bits"] == 11
    assert (
        uint16_channel["source"]["smallest_unsigned_dtype_after_unit_rounding"]
        == "uint16"
    )

    uint8_report = mutate(
        source,
        uint8_output,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    uint8_channel = uint8_report["dtype_mutation"]["channels"][0]
    assert uint8_channel["mapping"]["name"] == "zero_anchored_linear"
    assert uint8_channel["mapping"]["source_upper"] == 1500.0
    assert uint8_channel["mapping"]["quantum_source_units_per_output_code"] == pytest.approx(
        1500.0 / 255.0
    )
    assert uint8_channel["anticipated_loss"]["clipped_finite_pixels"] == 0


def test_auto_does_not_percentile_clip_a_rare_bright_pixel(tmp_path: Path) -> None:
    source = tmp_path / "bright-outlier.ome.tif"
    output = tmp_path / "bright-outlier.uint8.ome.tif"
    data = (np.arange(64 * 64, dtype=np.uint16) % 151).astype(np.float32).reshape(1, 64, 64)
    data[0, -1, -1] = 1000.0
    _write_float_ome(source, data, names=["Outlier"])

    report = mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    channel = report["dtype_mutation"]["channels"][0]
    assert channel["source"]["sample_percentiles"]["p99_9"] < 200
    assert channel["mapping"]["source_upper"] == 1000.0
    assert channel["mapping"]["exact_output_code_maximum_from_source_bounds"] == 255
    assert channel["anticipated_loss"]["clipped_finite_pixels"] == 0


def test_mutate_reads_indica_float_source_directly_without_float_ome_intermediate(
    tmp_path: Path,
) -> None:
    source = tmp_path / "halo-float.tif"
    output = tmp_path / "halo-uint16.ome.tif"
    data = (np.arange(2 * 32 * 48, dtype=np.uint16) % 151).astype(np.float32)
    data = data.reshape(2, 32, 48)
    data[:, 1::8, 1::8] += np.float32(0.25)
    _write_indica_float(source, data)

    report = mutate(
        source,
        output,
        input_type="indica_mif",
        dtype="uint16",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    assert report["options"]["input_type"] == "indica_mif"
    assert report["input_file"]["pixel_size"] == [0.5, 0.4, "µm"]
    assert all(
        channel["mapping"]["name"] == "identity"
        for channel in report["dtype_mutation"]["channels"]
    )
    with OMETiffReader(output) as reader:
        assert reader.channel_names == ("Marker 1", "Marker 2")
        np.testing.assert_array_equal(reader.asarray(), np.rint(data).astype(np.uint16))


def test_auto_uses_zero_anchored_scaling_when_subinteger_values_look_continuous(
    tmp_path: Path,
) -> None:
    source = tmp_path / "continuous.ome.tif"
    output = tmp_path / "continuous.uint16.ome.tif"
    data = np.linspace(0.0, 1.0, 2 * 64 * 64, dtype=np.float32).reshape(2, 64, 64)
    _write_float_ome(source, data)

    report = mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint16",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )

    channels = report["dtype_mutation"]["channels"]
    assert all(
        channel["mapping"]["name"] == "zero_anchored_linear" for channel in channels
    )
    assert all(
        channel["integer_lattice_evidence"]["classification"] == "weak"
        for channel in channels
    )
    assert channels[1]["mapping"]["source_upper"] == pytest.approx(1.0)
    assert channels[1]["mapping"]["quantum_source_units_per_output_code"] == pytest.approx(
        1.0 / 65535.0
    )
    assert channels[1]["anticipated_loss"]["sample_output_code_maximum"] == 65535

    expected = np.empty_like(data, dtype=np.uint16)
    for channel_index, channel in enumerate(channels):
        mapping = channel["mapping"]
        expected[channel_index] = np.rint(
            (data[channel_index].astype(np.float64) - mapping["offset_source_units"])
            / mapping["quantum_source_units_per_output_code"]
        ).astype(np.uint16)
    with OMETiffReader(output) as reader:
        np.testing.assert_array_equal(reader.asarray(), expected)


def test_auto_preserves_broad_source_scale_when_unit_rounding_loss_is_small(
    tmp_path: Path,
) -> None:
    source = tmp_path / "broad-continuous.ome.tif"
    preserved = tmp_path / "broad-preserved.ome.tif"
    scaled = tmp_path / "broad-scaled.ome.tif"
    data = np.linspace(0.0, 150.0, 64 * 64, dtype=np.float32).reshape(1, 64, 64)
    _write_float_ome(source, data, names=["Broad"])

    preserved_report = mutate(
        source,
        preserved,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="auto",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    channel = preserved_report["dtype_mutation"]["channels"][0]
    assert channel["integer_lattice_evidence"]["classification"] == "weak"
    assert channel["mapping"]["name"] == "identity"
    assert channel["automatic_preserve_assessment"]["unit_rounding_loss_within_limit"]

    scaled_report = mutate(
        source,
        scaled,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="auto",
        auto_max_normalized_rmse=0.0,
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    scaled_channel = scaled_report["dtype_mutation"]["channels"][0]
    assert scaled_channel["mapping"]["name"] == "zero_anchored_linear"


def test_preserve_mode_forces_source_scale_and_rejects_an_unrepresentable_range(
    tmp_path: Path,
) -> None:
    source = tmp_path / "preserve.ome.tif"
    output = tmp_path / "preserve.uint8.ome.tif"
    data = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(1, 32, 32)
    _write_float_ome(source, data)

    report = mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="preserve",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    assert report["dtype_mutation"]["channels"][0]["mapping"]["name"] == "identity"
    with OMETiffReader(output) as reader:
        np.testing.assert_array_equal(reader.asarray(), np.rint(data[0]).astype(np.uint8))

    too_wide = tmp_path / "too-wide.ome.tif"
    _write_float_ome(
        too_wide,
        np.linspace(0.0, 300.0, 32 * 32, dtype=np.float32).reshape(1, 32, 32),
    )
    with pytest.raises(ValueError, match="does not fit uint8"):
        mutate(
            too_wide,
            tmp_path / "too-wide.uint8.ome.tif",
            input_type="ome_tiff",
            dtype="uint8",
            range_mode="preserve",
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )


def test_auto_rejects_negative_values_and_full_range_maps_them_explicitly(
    tmp_path: Path,
) -> None:
    source = tmp_path / "negative.ome.tif"
    output = tmp_path / "negative.uint8.ome.tif"
    data = np.linspace(-2.0, 3.0, 64 * 64, dtype=np.float32).reshape(1, 64, 64)
    _write_float_ome(source, data, names=["Signed"])

    with pytest.raises(ValueError, match="negative pixels"):
        mutate(
            source,
            output,
            input_type="ome_tiff",
            dtype="uint8",
            range_mode="auto",
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )
    assert not output.exists()

    report = mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint8",
        range_mode="full",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        calculate_checksums=False,
    )
    mapping = report["dtype_mutation"]["channels"][0]["mapping"]
    assert mapping["name"] == "full_range_linear"
    assert mapping["offset_source_units"] == pytest.approx(-2.0)
    assert mapping["source_lower"] == pytest.approx(-2.0)
    assert mapping["source_upper"] == pytest.approx(3.0)

    with OMETiffReader(output) as reader:
        converted = reader.asarray()
        assert int(converted.min()) == 0
        assert int(converted.max()) == 255


def test_mutation_rejects_nonfinite_values_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "nonfinite.ome.tif"
    output = tmp_path / "nonfinite.uint16.ome.tif"
    data = np.zeros((1, 32, 32), dtype=np.float32)
    data[0, 2, 3] = np.nan
    data[0, 4, 5] = np.inf
    _write_float_ome(source, data)

    with pytest.raises(ValueError, match="NaN=1, \\+Inf=1"):
        mutate(
            source,
            output,
            input_type="ome_tiff",
            dtype="uint16",
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=0,
            calculate_checksums=False,
        )
    assert not output.exists()


def test_mutation_cli_emits_structured_per_channel_report(tmp_path: Path) -> None:
    source = tmp_path / "cli-source.ome.tif"
    output = tmp_path / "cli-output.ome.tif"
    report_path = tmp_path / "mutation.json"
    data = (np.arange(2 * 32 * 48, dtype=np.uint16) % 201).astype(np.float32)
    _write_float_ome(source, data.reshape(2, 32, 48), names=["A", "B"])

    result = CliRunner().invoke(
        main,
        [
            "mutate",
            str(source),
            str(output),
            "--type",
            "ome_tiff",
            "--dtype",
            "uint8",
            "--range-mode",
            "auto",
            "--compression",
            "Uncompressed",
            "--tile-size",
            "16",
            "--pyramid-levels",
            "0",
            "--no-checksums",
            "--output-json",
            str(report_path),
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output == ""
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["operation"] == "dtype"
    assert report["dtype_mutation"]["protocol_version"] == "1.0"
    assert report["dtype_mutation"]["target_dtype"] == "uint8"
    assert [item["channel_name"] for item in report["dtype_mutation"]["channels"]] == [
        "A",
        "B",
    ]
    assert output.exists()
