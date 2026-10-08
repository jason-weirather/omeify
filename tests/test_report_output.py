"""Report output safety and presentation, isolated from costly image processing."""
from __future__ import annotations

import errno
import json
import logging
import os
from inspect import signature
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

from omeify.cli import _render_visual_answer, main
from omeify.path_safety import atomic_write_text, check_output_path
from omeify.regions import rectangle_feature
from omeify.terminal import escape_terminal


def runner() -> CliRunner:
    options = {"mix_stderr": False} if "mix_stderr" in signature(CliRunner).parameters else {}
    return CliRunner(**options)


def link(source: Path, destination: Path, kind: str) -> Path:
    if kind == "same":
        return source
    try:
        if kind == "hardlink":
            destination.hardlink_to(source)
        else:
            destination.symlink_to(source)
    except OSError as exc:
        pytest.skip(f"{kind} unavailable: {exc}")
    return destination


def arguments(command: str, source: Path, output: Path) -> list[str]:
    result = [command, str(source), "-o", str(output), "--type", "ome_tiff"]
    if command == "mutate":
        result += ["--dtype", "uint16"]
    elif command == "crop":
        result += ["--bounds", "0", "0", "8", "8"]
    return result


@pytest.mark.parametrize("command", ["convert", "mutate", "crop"])
@pytest.mark.parametrize("kind", ["same", "hardlink", "symlink"])
@pytest.mark.parametrize("target", ["input", "output"])
def test_image_report_aliases_fail_before_work(tmp_path, monkeypatch, command, kind, target):
    source, output = tmp_path / "source.tif", tmp_path / "out.tif"
    source.write_bytes(b"SOURCE")
    output.write_bytes(b"OLD OUTPUT")
    report = link(source if target == "input" else output, tmp_path / "report.json", kind)
    workflow = Mock()
    monkeypatch.setattr(f"omeify.cli.{command}", workflow)
    result = runner().invoke(main, arguments(command, source, output) + ["--output-json", str(report)])
    assert result.exit_code != 0 and "protected" in result.stderr
    workflow.assert_not_called()
    assert source.read_bytes() == b"SOURCE" and output.read_bytes() == b"OLD OUTPUT"


@pytest.mark.parametrize("command", ["convert", "mutate"])
@pytest.mark.parametrize("artifact", ["image", "report"])
def test_channel_rename_input_is_protected(tmp_path, monkeypatch, command, artifact):
    source, output = tmp_path / "source.tif", tmp_path / "out.tif"
    source.touch()
    renames = tmp_path / "renames.json"
    renames.write_text('{"A":"B"}')
    alias = link(renames, tmp_path / "alias.json", "hardlink")
    workflow = Mock()
    monkeypatch.setattr(f"omeify.cli.{command}", workflow)
    args = arguments(command, source, alias if artifact == "image" else output)
    args += ["--rename-channels-json", str(renames), "--rename-channels-by", "name"]
    if artifact == "report":
        args += ["--output-json", str(alias)]
    result = runner().invoke(main, args)
    assert result.exit_code != 0
    workflow.assert_not_called()
    assert json.loads(renames.read_text()) == {"A": "B"}


@pytest.mark.parametrize("target", ["geojson", "first", "last"])
def test_crop_report_protects_geojson_and_every_shard(tmp_path, monkeypatch, target):
    source = tmp_path / "source.tif"
    source.touch()
    geojson = tmp_path / "regions.json"
    geojson.write_text(json.dumps({"type": "FeatureCollection", "features": [
        rectangle_feature([0, 0, 8, 8], name="first"),
        rectangle_feature([8, 8, 16, 16], name="last"),
    ]}))
    protected = geojson if target == "geojson" else tmp_path / f"out-{target}.tif"
    if target != "geojson":
        protected.write_bytes(b"ORIGINAL IMAGE")
    original = protected.read_bytes()
    alias = link(protected, tmp_path / "report.json", "hardlink")
    workflow = Mock()
    monkeypatch.setattr("omeify.cli.crop", workflow)
    result = runner().invoke(main, ["crop", str(source), "-o", str(tmp_path / "out.tif"),
                                   "--geojson", str(geojson), "--shatter", "by_name",
                                   "--output-json", str(alias)])
    assert result.exit_code != 0
    workflow.assert_not_called()
    assert protected.read_bytes() == original


@pytest.mark.parametrize("kind", ["file", "dangling_symlink"])
@pytest.mark.parametrize("command", ["convert", "mutate", "crop"])
def test_no_overwrite_report_fails_early(tmp_path, monkeypatch, kind, command):
    source, output, report = (tmp_path / n for n in ("source.tif", "out.tif", "report.json"))
    source.touch()
    if kind == "file":
        report.write_text("previous report")
    else:
        link(tmp_path / "missing", report, "symlink")
    workflow = Mock()
    monkeypatch.setattr(f"omeify.cli.{command}", workflow)
    result = runner().invoke(main, arguments(command, source, output) + [
        "--no-overwrite", "--output-json", str(report),
    ])
    assert result.exit_code != 0
    workflow.assert_not_called()
    assert os.path.lexists(report) and not output.exists()


@pytest.mark.parametrize("kind", ["same", "hardlink", "symlink"])
def test_inspection_reports_protect_the_image(tmp_path, monkeypatch, kind):
    source = tmp_path / "source.tif"
    source.write_bytes(b"DO NOT READ OR REPLACE")
    output = link(source, tmp_path / "report.json", kind)
    inspector = Mock()
    monkeypatch.setattr("omeify.cli.TiffInspector", inspector)
    result = runner().invoke(main, ["inspect", str(source), "--json", "-o", str(output)])
    assert result.exit_code != 0
    inspector.assert_not_called()
    assert source.read_bytes() == b"DO NOT READ OR REPLACE"


def test_atomic_text_replaces_an_unprotected_link_not_its_inode(tmp_path):
    other, report = tmp_path / "other", tmp_path / "report.json"
    other.write_bytes(b"OTHER DATA")
    link(other, report, "hardlink")
    atomic_write_text(report, 'µm\n{"ok":true}\n')
    assert other.read_bytes() == b"OTHER DATA"
    assert report.read_text() == 'µm\n{"ok":true}\n'
    assert not other.samefile(report)
    assert not list(tmp_path.glob(".omeify-report-*"))


def test_atomic_text_does_not_follow_final_symlink(tmp_path):
    other, report = tmp_path / "other", tmp_path / "report.json"
    other.write_text("OTHER")
    link(other, report, "symlink")
    atomic_write_text(report, "REPORT")
    assert other.read_text() == "OTHER" and not report.is_symlink()
    assert report.read_text() == "REPORT\n"


@pytest.mark.parametrize("overwrite", [False, True])
def test_install_race_cannot_truncate_protected_input(tmp_path, monkeypatch, overwrite):
    import omeify.path_safety as safety

    source, report = tmp_path / "source.tif", tmp_path / "report.json"
    source.write_bytes(b"ORIGINAL")
    real_check = safety.check_output_path
    count = 0

    def race(path, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            report.hardlink_to(source)
        real_check(path, **kwargs)

    monkeypatch.setattr(safety, "check_output_path", race)
    with pytest.raises(ValueError, match="protected"):
        safety.atomic_write_text(report, "REPORT", protected=[source], overwrite=overwrite)
    assert source.read_bytes() == b"ORIGINAL"
    assert not list(tmp_path.glob(".omeify-report-*"))


def test_no_overwrite_install_is_atomic_even_after_final_preflight(tmp_path, monkeypatch):
    import omeify.path_safety as safety

    report = tmp_path / "report.json"
    real_link = safety.os.link

    def competitor(source, destination):
        report.write_text("COMPETITOR")
        real_link(source, destination)

    monkeypatch.setattr(safety.os, "link", competitor)
    with pytest.raises(FileExistsError):
        safety.atomic_write_text(report, "OURS", overwrite=False)
    assert report.read_text() == "COMPETITOR"
    assert not list(tmp_path.glob(".omeify-report-*"))


@pytest.mark.parametrize("failure", ["write", "replace", "hardlink"])
def test_atomic_report_failure_preserves_existing_data_and_cleans_scratch(tmp_path, monkeypatch, failure):
    import omeify.path_safety as safety

    report = tmp_path / "report.json"
    if failure != "hardlink":
        report.write_text("PREVIOUS")

    def unavailable(*args, **kwargs):
        raise OSError(errno.EIO, "synthetic write/install failure")

    if failure == "write":
        real_fdopen = safety.os.fdopen

        class FailingStream:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            write = unavailable

        monkeypatch.setattr(safety.os, "fdopen", lambda *a, **k: FailingStream(real_fdopen(*a, **k)))
    else:
        monkeypatch.setattr(safety.os, "replace" if failure == "replace" else "link", unavailable)
    with pytest.raises(OSError, match="synthetic"):
        safety.atomic_write_text(report, "NEW", overwrite=failure != "hardlink")
    assert (not report.exists()) if failure == "hardlink" else report.read_text() == "PREVIOUS"
    assert not list(tmp_path.glob(".omeify-report-*"))


@pytest.mark.parametrize("command", ["convert", "mutate", "crop"])
@pytest.mark.parametrize("verbosity", [0, 1, 2, 3])
@pytest.mark.parametrize("destination", [None, "file", "-"])
def test_verbosity_and_explicit_json_stdout(tmp_path, monkeypatch, command, verbosity, destination):
    source, output = tmp_path / "source.tif", tmp_path / "out.tif"
    source.touch()
    payload = {"status": "ok", "detail": "µm"}

    def workflow(*args, **kwargs):
        logging.getLogger("omeify.test").info("stage details")
        logging.getLogger("omeify.test").debug("debug details")
        return payload

    monkeypatch.setattr(f"omeify.cli.{command}", workflow)
    args = arguments(command, source, output) + ["-v"] * verbosity
    if destination is not None:
        args += ["--output-json", str(tmp_path / "report.json") if destination == "file" else "-"]
    result = runner().invoke(main, args)
    assert result.exit_code == 0, result.output
    full_stdout = destination == "-" or (destination is None and verbosity == 3)
    if full_stdout:
        assert json.loads(result.stdout) == payload
        assert "Wrote " not in result.stdout
    else:
        assert result.stdout == f"Wrote {output}\n"
    if destination == "file":
        assert json.loads((tmp_path / "report.json").read_text()) == payload
    assert ("stage details" in result.stderr) == (verbosity >= 1)
    assert ("debug details" in result.stderr) == (verbosity >= 2)
    assert "stage details" not in result.stdout


def test_serialization_failure_leaves_completed_image_and_old_report(tmp_path, monkeypatch):
    source, output, report = (tmp_path / n for n in ("source.tif", "out.tif", "report.json"))
    source.touch()
    report.write_text("OLD REPORT")

    def workflow(*args, **kwargs):
        output.write_bytes(b"COMPLETE IMAGE")
        return {"invalid": float("nan")}

    monkeypatch.setattr("omeify.cli.convert", workflow)
    result = runner().invoke(main, arguments("convert", source, output) + ["--output-json", str(report)])
    assert result.exit_code != 0
    assert "Image output completed, but report output failed" in result.stderr
    assert output.read_bytes() == b"COMPLETE IMAGE" and report.read_text() == "OLD REPORT"
    assert not list(tmp_path.glob(".omeify-report-*"))


def test_cli_logging_restores_embedding_application_state(tmp_path, monkeypatch):
    logger = logging.getLogger("omeify")
    root = logging.getLogger()
    previous = logger.level, logger.propagate, logger.handlers[:], root.handlers[:]
    source = tmp_path / "source.tif"
    source.touch()
    monkeypatch.setattr("omeify.cli.convert", Mock(side_effect=ValueError("failed")))
    runner().invoke(main, arguments("convert", source, tmp_path / "out") + ["-v"])
    assert (logger.level, logger.propagate, logger.handlers, root.handlers) == previous


def test_terminal_controls_are_visible_without_destroying_unicode():
    unsafe = "µm\x1b[2J\r\b\u202evalue"
    report = {"question": unsafe, "answer": {"status": "answered",
              "paragraphs": [{"text": unsafe + "\nnext"}], "cautions": [{"text": unsafe}]}}
    rendered = _render_visual_answer(report)
    assert "µm" in rendered and "\\x1b" in rendered and "\nnext" in rendered
    assert not any(char in rendered for char in ("\x1b", "\r", "\b", "\u202e"))
    assert report["question"] == unsafe
    assert escape_terminal("a\nb") == "a\\nb"


def test_bad_parent_is_rejected_without_creating_directories(tmp_path):
    parent = tmp_path / "file"
    parent.write_text("INPUT")
    with pytest.raises(NotADirectoryError):
        check_output_path(parent / "nested" / "report.json")
    assert parent.read_text() == "INPUT"
