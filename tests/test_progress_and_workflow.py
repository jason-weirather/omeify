from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import tifffile
from click.testing import CliRunner

import omeify.conversion as conversion_module
import omeify.dtype_mutation as dtype_mutation_module
import omeify.mutation as mutation_module
import omeify.workflow as workflow_module
from omeify import PixelSize
from omeify.cli import main
from omeify.dtype_mutation import DTypeMutationSource, analyze_dtype_mutation
from omeify.io.ome_tiff_writer import (
    OMETiffWriter,
    _iter_downsampled_tiles,
    _iter_tiles,
)
from omeify.io.tiff import ArrayPlaneReader
from omeify.progress import ProgressLogger
from omeify.provenance import hash_file


class _ArraySource:
    def __init__(self, array: np.ndarray) -> None:
        self.array = np.asarray(array)

    def plane_readers(self, *, cache_mib: int = 64) -> list[ArrayPlaneReader]:
        del cache_mib
        return [ArrayPlaneReader(self.array[index]) for index in range(self.array.shape[0])]


class _AlwaysValidSchema:
    schema_location = "synthetic-test-schema"

    def validate(self, xml_string: str) -> bool:
        assert "<OME" in xml_string
        return True


def test_progress_logger_reports_start_periodic_progress_and_completion(caplog) -> None:
    logger = logging.getLogger("omeify.tests.progress")
    ticks = iter((0.0, 1.0, 2.0))
    with caplog.at_level(logging.INFO, logger=logger.name):
        progress = ProgressLogger(
            logger,
            "Synthetic stage",
            4,
            unit="tiles",
            min_interval_seconds=0,
            clock=lambda: next(ticks),
        )
        progress.update(2)
        progress.finish()

    assert "Synthetic stage: starting (4 tiles)" in caplog.text
    assert "Synthetic stage: [##########----------]  50.0%" in caplog.text
    assert "Synthetic stage: [####################] 100.0%" in caplog.text
    assert "ETA" in caplog.text


def test_tile_iterators_emit_progress_without_changing_pixels(caplog) -> None:
    data = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

    with caplog.at_level(logging.INFO, logger="omeify.io.ome_tiff_writer"):
        tiles = list(
            _iter_tiles(
                [ArrayPlaneReader(data)],
                (1, 32, 32),
                axes="CYX",
                plane_count=1,
                tile_size=16,
                progress_label="Writing test tiles",
            )
        )

    assert len(tiles) == 4
    np.testing.assert_array_equal(tiles[0], data[:16, :16])
    np.testing.assert_array_equal(tiles[-1], data[16:, 16:])
    assert "Writing test tiles: starting (4 tiles)" in caplog.text
    assert "Writing test tiles: [####################] 100.0%" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="omeify.io.ome_tiff_writer"):
        downsampled = list(
            _iter_downsampled_tiles(
                [ArrayPlaneReader(data)],
                (1, 16, 16),
                axes="CYX",
                plane_count=1,
                tile_size=16,
                method="nearest",
                progress_label="Building test pyramid",
            )
        )

    assert len(downsampled) == 1
    np.testing.assert_array_equal(downsampled[0], data[::2, ::2])
    assert "Building test pyramid: [####################] 100.0%" in caplog.text


def test_writer_logs_pyramid_final_assembly_verification_and_cleanup(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    import omeify.io.ome_tiff_writer as writer_module

    monkeypatch.setattr(writer_module, "OMESchemaValidator", _AlwaysValidSchema)
    output = tmp_path / "progress.ome.tif"
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

    with caplog.at_level(logging.INFO, logger="omeify.io.ome_tiff_writer"):
        report = OMETiffWriter(
            output,
            image_type="multichannel",
            channel_names=["DAPI"],
            pixel_size=PixelSize(0.5, 0.5, "µm"),
            compression="Uncompressed",
            tile_size=16,
            pyramid_levels=1,
            max_workers=1,
        ).write(image, axes="YX")

    assert report["verification"]["base_pixel_values_match"] is True
    with tifffile.TiffFile(output) as tiff:
        np.testing.assert_array_equal(tiff.series[0].levels[0].asarray(), image)
        assert len(tiff.series[0].levels) == 2

    expected_messages = (
        "Writer preflight",
        "Building pyramid level 1/1",
        "Writing final full-resolution base",
        "Writing final pyramid level 1/1",
        "Verifying OME metadata, TIFF layout, decoding, and spot values",
        "Installing verified output atomically",
        "Temporary pyramid cache removed",
    )
    for message in expected_messages:
        assert message in caplog.text
    assert (
        "Writing final full-resolution base: [####################] 100.0%"
        in caplog.text
    )


def test_mutation_analysis_logs_scans_and_maps_each_channel_immediately(caplog) -> None:
    data = np.stack(
        [
            np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32),
        ]
    )

    with caplog.at_level(logging.INFO, logger="omeify.dtype_mutation"):
        plans = analyze_dtype_mutation(
            _ArraySource(data),
            channel_names=("Integer-like", "Continuous"),
            source_dtype=np.float32,
            dtype="uint16",
            range_mode="full",
            sample_pixels_per_channel=1024,
        )

    assert len(plans) == 2
    assert "Scanning mutation channel 1/2 'Integer-like':" in caplog.text
    assert "Mutation scan 2/2 complete for 'Continuous'" in caplog.text
    assert "Mutation plan 1/2 for channel [0] 'Integer-like'" in caplog.text
    assert "Mapping decision:" in caplog.text
    assert "Mutation analysis complete: planned 2 channel(s)" in caplog.text


def test_mutation_analysis_maps_one_channel_before_scanning_the_next(monkeypatch) -> None:
    events: list[str] = []
    source = _ArraySource(np.zeros((2, 1, 1), dtype=np.float32))

    def fake_scan(
        reader,
        *,
        channel_index,
        channel_name,
        sample_pixels,
        progress_label=None,
    ):
        del reader, sample_pixels, progress_label
        events.append(f"scan-{channel_index}")
        return SimpleNamespace(
            channel_index=channel_index,
            channel_name=channel_name,
            minimum=0.0,
            maximum=1.0,
            finite_pixels=1,
            nonzero_pixels=1,
            nan_pixels=0,
            positive_infinity_pixels=0,
            negative_infinity_pixels=0,
        )

    def fake_mapping(scan, **kwargs):
        del kwargs
        events.append(f"map-{scan.channel_index}")
        return SimpleNamespace(
            mapping="identity",
            offset=0.0,
            quantum=1.0,
            reason="synthetic",
        )

    monkeypatch.setattr(dtype_mutation_module, "_scan_plane", fake_scan)
    monkeypatch.setattr(dtype_mutation_module, "_mapping_for_scan", fake_mapping)

    plans = analyze_dtype_mutation(
        source,
        channel_names=("A", "B"),
        source_dtype=np.float32,
        dtype="uint16",
        sample_pixels_per_channel=1024,
    )

    assert len(plans) == 2
    assert events == ["scan-0", "map-0", "scan-1", "map-1"]


def test_hash_file_logs_progress_and_preserves_digest_values(
    tmp_path: Path,
    caplog,
) -> None:
    path = tmp_path / "payload.bin"
    payload = b"omeify-progress" * 1024
    path.write_bytes(payload)

    with caplog.at_level(logging.INFO, logger="omeify.provenance"):
        result = hash_file(path, progress_label="Checksumming test payload")

    assert result == {
        "md5_checksum": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
        "sha256_checksum": hashlib.sha256(payload).hexdigest(),
    }
    assert "Checksumming test payload: starting" in caplog.text
    assert "Checksumming test payload: [####################] 100.0%" in caplog.text


def test_channel_logging_shows_discovered_names_and_explicit_renames(caplog) -> None:
    logger = logging.getLogger("omeify.tests.channels")

    with caplog.at_level(logging.INFO, logger=logger.name):
        workflow_module.log_channel_mapping(
            logger,
            ("DAPI", "CD3"),
            ("DNA", "CD3"),
            rename_mode="name",
        )

    assert "Source channels (2):" in caplog.text
    assert "[0] 'DAPI'" in caplog.text
    assert "[1] 'CD3'" in caplog.text
    assert "Channel renaming by name (1 changed):" in caplog.text
    assert "[0] 'DAPI' -> 'DNA'" in caplog.text


def test_convert_and_mutate_share_boundary_helpers_and_writer_without_calling_each_other() -> None:
    assert conversion_module._apply_channel_renames is workflow_module.apply_channel_renames
    assert (
        conversion_module._validate_channel_rename_mapping
        is workflow_module.validate_channel_rename_mapping
    )
    assert mutation_module.apply_channel_renames is workflow_module.apply_channel_renames
    assert (
        mutation_module.validate_channel_rename_mapping
        is workflow_module.validate_channel_rename_mapping
    )
    assert conversion_module.update_file_checksums is workflow_module.update_file_checksums
    assert mutation_module.update_file_checksums is workflow_module.update_file_checksums
    assert conversion_module.OMETiffWriter is mutation_module.OMETiffWriter
    assert conversion_module.convert is not mutation_module.mutate


def test_mutate_uses_shared_writer_through_dtype_transform_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import omeify.io.ome_tiff_writer as writer_module

    monkeypatch.setattr(writer_module, "OMESchemaValidator", _AlwaysValidSchema)
    source = tmp_path / "source.ome.tif"
    output = tmp_path / "mutated.ome.tif"
    data = np.arange(2 * 16 * 16, dtype=np.float32).reshape(2, 16, 16)
    tifffile.imwrite(
        source,
        data,
        ome=True,
        tile=(16, 16),
        photometric="minisblack",
        metadata={
            "axes": "CYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": 0.5,
            "PhysicalSizeYUnit": "µm",
            "Channel": {"Name": ["A", "B"]},
        },
    )

    observed: dict[str, object] = {}

    class RecordingWriter(OMETiffWriter):
        def write_source(self, source, **kwargs):
            observed["source"] = source
            return super().write_source(source, **kwargs)

    monkeypatch.setattr(mutation_module, "OMETiffWriter", RecordingWriter)
    report = mutation_module.mutate(
        source,
        output,
        input_type="ome_tiff",
        dtype="uint16",
        range_mode="full",
        compression="Uncompressed",
        tile_size=16,
        pyramid_levels=0,
        max_workers=1,
        calculate_checksums=False,
    )

    assert isinstance(observed["source"], DTypeMutationSource)
    assert report["verification"]["base_pixel_values_match"] is True
    assert output.is_file()


def test_cli_single_verbose_level_enables_only_omeify_stage_logs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.tif"
    output = tmp_path / "output.ome.tif"
    report_path = tmp_path / "report.json"
    tifffile.imwrite(source, np.zeros((16, 16), dtype=np.uint16), metadata=None)

    def fake_convert(*args, **kwargs):
        del args, kwargs
        logging.getLogger("omeify.test_cli").info("visible stage log")
        logging.getLogger("unrelated.library").info("dependency chatter")
        return {"status": "ok"}

    monkeypatch.setattr("omeify.cli.convert", fake_convert)
    result = CliRunner().invoke(
        main,
        [
            "convert",
            str(source),
            str(output),
            "--type",
            "component",
            "--pixel-size-x",
            "0.5",
            "--pixel-size-y",
            "0.5",
            "--output-json",
            str(report_path),
            "--verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "visible stage log" in result.output
    assert "dependency chatter" not in result.output
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"status": "ok"}


def test_verbose_help_describes_single_and_repeated_levels() -> None:
    runner = CliRunner()
    for command in ("convert", "mutate"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert "Show stages and 5-second progress bars" in result.output
        assert "1-second progress" in result.output
        assert "tracebacks" in result.output
