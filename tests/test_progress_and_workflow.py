from __future__ import annotations

import io
import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from click.testing import CliRunner

import omeify.dtype_mutation as dtype_mutation_module
import omeify.mutation as mutation_module
import omeify.workflow as workflow_module
from omeify import PixelSize
from omeify.cli import _CompactLogHandler, main
from omeify.dtype_mutation import analyze_dtype_mutation
from omeify.io._writer.pyramid import iter_downsampled_tiles, iter_tiles
from omeify.io.ome_tiff_writer import OMETiffWriter
from plane_fixture import ArrayPlaneReader
from omeify.progress import ProgressLogger


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

    assert "Synthetic stage [--------------------]   0%" in caplog.text
    assert "Synthetic stage [##########----------]  50%" in caplog.text
    assert "Synthetic stage [####################] 100% 00:02" in caplog.text
    assert "ETA 00:01" in caplog.text


def test_tile_iterators_emit_progress_without_changing_pixels(caplog) -> None:
    data = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

    with caplog.at_level(logging.INFO, logger="omeify.io._writer"):
        tiles = list(
            iter_tiles(
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
    assert "Writing test tiles [--------------------]   0%" in caplog.text
    assert "Writing test tiles [####################] 100%" in caplog.text

    caplog.clear()
    with caplog.at_level(logging.INFO, logger="omeify.io._writer"):
        downsampled = list(
            iter_downsampled_tiles(
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
    assert "Building test pyramid [####################] 100%" in caplog.text


def test_writer_logs_pyramid_final_assembly_verification_and_cleanup(
    image_factory,
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    import omeify.io._writer.preparation as preparation_module

    monkeypatch.setattr(
        preparation_module,
        "OMESchemaValidator",
        _AlwaysValidSchema,
    )
    output = tmp_path / "progress.ome.tif"
    image = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)

    with caplog.at_level(logging.INFO, logger="omeify.io._writer"):
        report = OMETiffWriter(
            output,
            compression='Uncompressed',
            tile_size=16,
            pyramid_levels=1,
            max_workers=1,
        ).write(image_factory(
            image,
            kind='multichannel',
            channel_names=['DAPI'],
            pixel_size=PixelSize(0.5, 0.5, 'µm'),
            axes='YX',
        ))

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
        "Writing final full-resolution base [####################] 100%"
        in caplog.text
    )


def test_mutation_analysis_logs_scans_and_maps_each_channel_immediately(
    caplog,
    image_factory,
) -> None:
    data = np.stack(
        [
            np.arange(32 * 32, dtype=np.float32).reshape(32, 32),
            np.linspace(0.0, 1.0, 32 * 32, dtype=np.float32).reshape(32, 32),
        ]
    )

    with caplog.at_level(logging.INFO, logger="omeify.dtype_mutation"):
        plans = analyze_dtype_mutation(
            image_factory(data, channel_names=("Integer-like", "Continuous")),
            dtype="uint16",
            range_mode="full",
            sample_pixels_per_channel=1024,
        )

    assert len(plans) == 2
    assert "Scanning mutation channel 1/2 'Integer-like' [" in caplog.text
    assert "Mutation scan 2/2 complete for 'Continuous'" in caplog.text
    assert "Mutation plan 1/2 for channel [0] 'Integer-like'" in caplog.text
    assert "Mapping decision:" in caplog.text
    assert "Mutation analysis complete: planned 2 channel(s)" in caplog.text


def test_mutation_analysis_maps_one_channel_before_scanning_the_next(
    monkeypatch,
    image_factory,
) -> None:
    events: list[str] = []
    source = image_factory(np.zeros((2, 1, 1), dtype=np.float32), channel_names=("A", "B"))

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

        dtype="uint16",
        sample_pixels_per_channel=1024,
    )

    assert len(plans) == 2
    assert events == ["scan-0", "map-0", "scan-1", "map-1"]


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


def test_mutate_uses_shared_writer_through_dtype_transform_adapter(
    tmp_path: Path,
    monkeypatch,
) -> None:
    import omeify.io._writer.preparation as preparation_module

    monkeypatch.setattr(
        preparation_module,
        "OMESchemaValidator",
        _AlwaysValidSchema,
    )
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
        def write(self, image, **kwargs):
            observed["dtype"] = image.dtype
            observed["pixels"] = image.asarray()
            return super().write(image, **kwargs)

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
    )

    assert observed["dtype"] == np.dtype("uint16")
    np.testing.assert_array_equal(tifffile.imread(output), observed["pixels"])
    assert report["verification"]["base_pixel_values_match"] is True
    assert output.is_file()


@pytest.mark.parametrize("verbose", [1, 2], ids=["compact", "diagnostic"])
def test_cli_verbose_levels(tmp_path: Path, monkeypatch, verbose: int) -> None:
    source = tmp_path / "source.tif"
    source.touch()  # Only Click's existence check runs; conversion is replaced below.
    report_path = tmp_path / "report.json"

    def fake_convert(*args, **kwargs):
        logger = logging.getLogger("omeify.test_cli")
        logger.info("visible stage log")
        logger.debug("debug detail")
        logging.getLogger("unrelated.library").info("dependency chatter")
        return {"status": "ok"}

    monkeypatch.setattr("omeify.cli.convert", fake_convert)
    result = CliRunner().invoke(main, [
        "convert", str(source), "--output", str(tmp_path / "output.ome.tif"),
        "--type", "component", "--pixel-size-x", "0.5", "--pixel-size-y", "0.5",
        "--output-json", str(report_path), *(["--verbose"] * verbose),
    ])
    assert result.exit_code == 0, result.output
    assert "dependency chatter" not in result.output
    assert json.loads(report_path.read_text(encoding="utf-8")) == {"status": "ok"}
    if verbose == 1:
        assert result.output == "visible stage log\n"
    else:
        assert re.search(
            r"\d{2}:\d{2}:\d{2} INFO omeify\.test_cli: visible stage log", result.output,
        )
        assert "DEBUG omeify.test_cli: debug detail" in result.output


def test_compact_handler_reuses_one_tty_line_until_progress_finishes() -> None:
    class TTYBuffer(io.StringIO):
        def isatty(self) -> bool:
            return True

    stream = TTYBuffer()
    handler = _CompactLogHandler()
    handler.setStream(stream)
    logger = logging.Logger("omeify.tests.compact-handler", level=logging.INFO)
    logger.propagate = False
    logger.addHandler(handler)
    ticks = iter((0.0, 1.0, 2.0))

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

    rendered = stream.getvalue()
    assert rendered.count("\n") == 1
    assert rendered.count("\r") == 3
    assert "Synthetic stage [##########----------]  50%" in rendered
    assert rendered.rstrip().endswith("100% 00:02")


def test_verbose_help_describes_single_and_repeated_levels() -> None:
    runner = CliRunner()
    for command in ("convert", "mutate"):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0
        assert "Show compact stages and progress bars" in result.output
        assert "timestamps, debug details" in result.output
        assert "1-second" in result.output
        assert "tracebacks" in result.output
