"""Explicit image destinations at the command-line boundary."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from omeify.cli import main


@pytest.fixture(params=["convert", "mutate"])
def image_command(request: pytest.FixtureRequest) -> list[str]:
    arguments = [request.param, "--type", "ome_tiff"]
    if request.param == "mutate":
        arguments += ["--dtype", "uint16"]
    return arguments


@pytest.mark.parametrize(("flag", "before_input"), [("--output", False), ("-o", True)])
def test_output_option_forwards_paths_and_keeps_report_separate(
    image_command: list[str], flag: str, before_input: bool,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source.ome.tif"
    source.touch()  # Parsing only; the image workflow is replaced below.
    output = tmp_path / "destination with spaces.ome.tif"
    report = tmp_path / "report.json"
    workflow = Mock(return_value={"status": "ok"})
    monkeypatch.setattr(f"omeify.cli.{image_command[0]}", workflow)
    paths = (
        [flag, str(output), str(source)] if before_input
        else [str(source), flag, str(output)]
    )

    result = CliRunner().invoke(
        main, [*image_command, *paths, "--output-json", str(report), "--no-overwrite"],
    )

    assert result.exit_code == 0, result.output
    workflow.assert_called_once()
    assert workflow.call_args.args == (source, output)
    assert all(isinstance(path, Path) for path in workflow.call_args.args)
    assert workflow.call_args.kwargs["overwrite"] is False
    assert result.output == ""
    assert json.loads(report.read_text(encoding="utf-8")) == {"status": "ok"}


@pytest.mark.parametrize(
    ("arguments", "error"),
    [
        ([], "Missing option"),
        (["legacy.ome.tif"], "Missing option"),
        (["--output-json", "report.json"], "Missing option"),
        (["-o"], "requires an argument"),
        (["--output", "destination.ome.tif", "legacy.ome.tif"], "unexpected extra argument"),
        (["--output", "."], "is a directory"),
    ],
    ids=["missing", "positional", "report-only", "missing-value", "extra-positional", "directory"],
)
def test_invalid_output_is_rejected_before_processing(
    image_command: list[str], arguments: list[str], error: str,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    source = tmp_path / "source.ome.tif"
    source.write_bytes(b"Source must not be read or changed during argument validation.")
    original = source.read_bytes()
    workflow = Mock()
    monkeypatch.setattr(f"omeify.cli.{image_command[0]}", workflow)

    result = CliRunner().invoke(main, [*image_command, str(source), *arguments])

    assert result.exit_code == 2, result.output
    assert error in result.output
    if error == "Missing option":
        assert "--output" in result.output
        assert "-o" in result.output
    workflow.assert_not_called()
    assert source.read_bytes() == original
    assert set(tmp_path.iterdir()) == {source}


def test_output_help_is_available_without_paths(image_command: list[str]) -> None:
    result = CliRunner().invoke(main, [image_command[0], "--help"])

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0].endswith("[OPTIONS] INPUT_PATH")
    output_line = next(line for line in result.output.splitlines() if "-o, --output" in line)
    assert "FILE" in output_line
    assert "[required]" in output_line
