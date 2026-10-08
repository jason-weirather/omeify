"""Presentation of full operation reports, separate from image computation."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import click

from .path_safety import atomic_write_text, check_output_path
from .terminal import escape_terminal


def report_destination(value: str | Path | None) -> Path | None:
    """None and '-' have no filesystem destination; other values name a report."""
    return None if value is None or str(value) == "-" else Path(value)


def preflight_report(
    destination: str | Path | None, *, protected: Iterable[str | Path], overwrite: bool,
) -> None:
    path = report_destination(destination)
    if path is not None:
        check_output_path(path, protected=protected, overwrite=overwrite)


def completion_summary(report: Mapping[str, Any], output_path: Path) -> str:
    """A short display generated from the report, never a second stored model."""
    if "outputs" in report:
        outputs = report["outputs"]
        count = sum(len(item["regions"]) for item in outputs)
        return f"Wrote {count} region(s) in {len(outputs)} file(s) | sampled verification passed"
    fields = [f"Wrote {output_path}"]
    output = report.get("output_file", {})
    if "dtype" in output and "axes" in output:
        fields.append(f"{output['dtype']} {output['axes']}")
    compression = report.get("options", {}).get("compression")
    if compression:
        fields.append(str(compression))
    if "dtype_mutation" in report:
        mutation = report["dtype_mutation"]
        fields.append(f"{mutation['source_dtype']} -> {mutation['target_dtype']}, "
                      f"range={mutation['range_mode']}")
    elif "float_precision_mutation" in report:
        bits = report["float_precision_mutation"]["float32_mantissa_bits"]
        fields.append(f"float32, {bits} retained fraction bits")
    if report.get("verification", {}).get("output_pixels_decodable"):
        fields.append("sampled verification passed")
    return escape_terminal(" | ".join(fields))


def emit_operation_report(
    report: Mapping[str, Any], *, output_path: Path,
    destination: str | Path | None, verbose: int, overwrite: bool,
    protected: Iterable[str | Path],
) -> None:
    """Write explicit JSON, print diagnostic JSON, or display one completion line."""
    path = report_destination(destination)
    full_stdout = str(destination) == "-" or (destination is None and verbose >= 3)
    if path is not None or full_stdout:
        # Fail serialization before opening a report destination. The image may
        # already have been installed; never imply that it was rolled back.
        try:
            rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
            if path is not None:
                atomic_write_text(path, rendered, protected=protected, overwrite=overwrite)
            else:
                click.echo(rendered)
        except (OSError, TypeError, ValueError) as exc:
            raise click.ClickException(escape_terminal(
                f"Image output completed, but report output failed: {exc}. "
                "Completed image files were not removed."
            )) from exc
    if not full_stdout:
        click.echo(completion_summary(report, output_path))
