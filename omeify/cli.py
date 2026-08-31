from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import click

from omeify import __version__, get_version_info
from omeify.conversion import RenameChannelsBy, convert
from omeify.dtype_mutation import (
    DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    RANGE_MODES,
    TARGET_DTYPES,
)
from omeify.inspection import TiffInspector
from omeify.io.pixel_size import PixelSize
from omeify.io.source_reader import INPUT_TYPES, PLANAR_INPUT_TYPES
from omeify.mutation import mutate

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
_INDEX_KEY = re.compile(r"0|[1-9][0-9]*")


class _SubcommandOnlyGroup(click.Group):
    """Show help when empty, but reject every non-subcommand first token.

    Click treats absolute POSIX paths as option-like tokens while resolving a
    group and can otherwise turn the legacy ``omeify INPUT OUTPUT`` form into
    a successful help display. Resolve the command name before Click's option
    parser gets that opportunity.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        if args and not args[0].startswith("-") and args[0] not in self.commands:
            ctx.fail(f"No such command {args[0]!r}.")
        return super().parse_args(ctx, args)


def _load_channel_renames(
    path: Path | None,
    by: RenameChannelsBy | None,
) -> tuple[dict[str, str] | dict[int, str], RenameChannelsBy | None]:
    if path is None:
        if by is not None:
            raise click.UsageError(
                "--rename-channels-by requires --rename-channels-json"
            )
        return {}, None
    if by is None:
        raise click.UsageError(
            "--rename-channels-by name|index is required with --rename-channels-json"
        )
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Unable to read channel rename JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise click.ClickException("Channel rename JSON must contain one JSON object")
    if not all(isinstance(item, str) and item.strip() for item in value.values()):
        raise click.ClickException("Channel rename values must be non-empty strings")

    if by == "name":
        # JSON object keys are strings. Because --rename-channels-by is
        # explicit, even a source channel literally named "0" is unambiguous.
        return {str(key): str(item).strip() for key, item in value.items()}, by

    invalid = [str(key) for key in value if not _INDEX_KEY.fullmatch(str(key))]
    if invalid:
        raise click.ClickException(
            "Index-based channel rename keys must all be zero-based integer strings; "
            f"invalid keys: {', '.join(repr(item) for item in invalid)}"
        )
    return {int(key): str(item).strip() for key, item in value.items()}, by


def _configure_logging(verbose: int) -> None:
    """Configure predictable omeify logging without enabling dependency chatter."""

    level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )
    logging.getLogger("omeify").setLevel(level)


def _write_or_echo(rendered: str, output: Path | None) -> None:
    if output is None:
        click.echo(rendered)
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered.rstrip("\n") + "\n", encoding="utf-8")


def _pixel_size_override(
    x: float | None,
    y: float | None,
    unit: str | None,
) -> PixelSize | None:
    supplied = (x, y, unit)
    if not any(item is not None for item in supplied):
        return None
    if x is None or y is None:
        raise click.UsageError("--pixel-size-x and --pixel-size-y must be supplied together")
    return PixelSize(x, y, unit or "µm")


@click.group(cls=_SubcommandOnlyGroup, context_settings=_CONTEXT_SETTINGS)
def main() -> None:
    """Convert, mutate, inspect, and read standardized TIFF-family images."""


@main.command("convert", context_settings=_CONTEXT_SETTINGS)
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--type",
    "input_type",
    type=click.Choice(INPUT_TYPES),
    required=True,
    help="Input image profile.",
)
@click.option("--series", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--channel-name-field",
    type=click.Choice(["name", "biomarker", "auto"]),
    default=None,
    help="Fusion QPTIFF field used for normalized channel names.",
)
@click.option(
    "--rename-channels-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON object mapping source names or zero-based indices to output names.",
)
@click.option(
    "--rename-channels-by",
    type=click.Choice(["name", "index"]),
    default=None,
    help="Interpret rename JSON keys explicitly as channel names or indices.",
)
@click.option("--omit-uuid", is_flag=True, help="Omit the optional OME root UUID.")
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the conversion report to this JSON file instead of stdout.",
)
@click.option(
    "--cache-directory",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for temporary rebuilt pyramid levels.",
)
@click.option(
    "--compression",
    type=click.Choice(["LZW", "Deflate", "ZSTD", "JPEG", "Uncompressed"], case_sensitive=False),
    default=None,
    help="Default: JPEG for qptiff_he/svs; LZW for planar inputs.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(min=1, max=100),
    default=90,
    show_default=True,
    help="JPEG encoder quality used when --compression JPEG is active.",
)
@click.option(
    "--jpeg-subsampling",
    type=click.Choice(["444", "422", "420", "411"]),
    default="444",
    show_default=True,
    help="JPEG chroma subsampling for interleaved RGB output.",
)
@click.option(
    "--tile-size",
    type=click.IntRange(min=16),
    default=1024,
    show_default=True,
    help="Square output tile size; must be divisible by 16.",
)
@click.option(
    "--pyramid-levels",
    type=click.IntRange(min=0),
    default=None,
    help="Number of subresolution levels. By default, build until the image fits one tile.",
)
@click.option(
    "--downsample",
    type=click.Choice(["mean", "nearest"]),
    default="mean",
    show_default=True,
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum parallel TIFF compression workers.",
)
@click.option("--pixel-size-x", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--pixel-size-y", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option(
    "--pixel-size-unit",
    type=str,
    default=None,
    help="Unit for an explicit physical pixel-size override; default µm.",
)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("--checksums/--no-checksums", default=True, show_default=True)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Show stages and 5-second progress bars; repeat for debug details, "
        "1-second progress, and tracebacks."
    ),
)
def convert_command(
    input_path: Path,
    output_path: Path,
    input_type: str,
    series: int,
    channel_name_field: str | None,
    rename_channels_json: Path | None,
    rename_channels_by: str | None,
    omit_uuid: bool,
    output_json: Path | None,
    cache_directory: Path | None,
    compression: str | None,
    jpeg_quality: int,
    jpeg_subsampling: str,
    tile_size: int,
    pyramid_levels: int | None,
    downsample: str,
    workers: int | None,
    pixel_size_x: float | None,
    pixel_size_y: float | None,
    pixel_size_unit: str | None,
    overwrite: bool,
    checksums: bool,
    verbose: int,
) -> None:
    """Convert INPUT_PATH into a deidentified pyramidal OME-TIFF at OUTPUT_PATH."""

    _configure_logging(verbose)
    rename_channels, rename_mode = _load_channel_renames(
        rename_channels_json,
        rename_channels_by,  # type: ignore[arg-type]
    )

    if output_json is not None:
        report_path = output_json.resolve()
        if report_path in {input_path.resolve(), output_path.resolve()}:
            raise click.UsageError("--output-json must differ from both image paths")

    if input_type != "qptiff_fusion" and channel_name_field is not None:
        raise click.UsageError("--channel-name-field is only valid with --type qptiff_fusion")

    pixel_size = _pixel_size_override(pixel_size_x, pixel_size_y, pixel_size_unit)
    try:
        report = convert(
            input_path,
            output_path,
            input_type=input_type,  # type: ignore[arg-type]
            series=series,
            channel_name_field=channel_name_field,  # type: ignore[arg-type]
            rename_channels=rename_channels,
            rename_channels_by=rename_mode,
            pixel_size=pixel_size,
            display_uuid=not omit_uuid,
            compression=compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,  # type: ignore[arg-type]
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,  # type: ignore[arg-type]
            max_workers=workers,
            overwrite=overwrite,
            calculate_checksums=checksums,
            cache_directory=cache_directory,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(str(exc)) from exc

    _write_or_echo(json.dumps(report, indent=2), output_json)


@main.command("mutate", context_settings=_CONTEXT_SETTINGS)
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output_path",
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--type",
    "input_type",
    type=click.Choice(PLANAR_INPUT_TYPES),
    required=True,
    help="Input image profile.",
)
@click.option(
    "--channel-name-field",
    type=click.Choice(["name", "biomarker", "auto"]),
    default=None,
    help="Fusion QPTIFF field used for normalized channel names.",
)
@click.option(
    "--rename-channels-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON object mapping source names or zero-based indices to output names.",
)
@click.option(
    "--rename-channels-by",
    type=click.Choice(["name", "index"]),
    default=None,
    help="Interpret rename JSON keys explicitly as channel names or indices.",
)
@click.option(
    "--dtype",
    type=click.Choice(TARGET_DTYPES),
    required=True,
    help="Unsigned integer dtype for the mutated OME-TIFF.",
)
@click.option(
    "--range-mode",
    type=click.Choice(RANGE_MODES),
    default="auto",
    show_default=True,
    help=(
        "auto preserves source scale when measured unit-rounding loss is small or integer "
        "ancestry evidence is strong; preserve forces no rescaling; full maps each channel's "
        "exact minimum and maximum to the dtype limits."
    ),
)
@click.option("--series", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--sample-pixels-per-channel",
    type=click.IntRange(min=1024),
    default=DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    show_default=True,
    help="Deterministic spatial sample used for lattice and loss estimates.",
)
@click.option(
    "--auto-max-normalized-rmse",
    type=click.FloatRange(min=0),
    default=DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    show_default=True,
    help=(
        "Largest unit-rounding RMSE, as a fraction of the sampled nonzero p0.1-p99.9 "
        "intensity span, that auto mode may accept without strong integer-lattice evidence."
    ),
)
@click.option("--omit-uuid", is_flag=True, help="Omit the optional OME root UUID.")
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the mutation report to this JSON file instead of stdout.",
)
@click.option(
    "--cache-directory",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for temporary rebuilt pyramid levels.",
)
@click.option(
    "--compression",
    type=click.Choice(["LZW", "Deflate", "ZSTD", "Uncompressed"], case_sensitive=False),
    default="LZW",
    show_default=True,
    help="Lossless compression for quantitative channel data.",
)
@click.option(
    "--tile-size",
    type=click.IntRange(min=16),
    default=1024,
    show_default=True,
    help="Square output tile size; must be divisible by 16.",
)
@click.option(
    "--pyramid-levels",
    type=click.IntRange(min=0),
    default=None,
    help="Number of subresolution levels. By default, build until the image fits one tile.",
)
@click.option(
    "--downsample",
    type=click.Choice(["mean", "nearest"]),
    default="mean",
    show_default=True,
)
@click.option(
    "--workers",
    type=click.IntRange(min=1),
    default=None,
    help="Maximum parallel TIFF compression workers.",
)
@click.option("--pixel-size-x", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--pixel-size-y", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option(
    "--pixel-size-unit",
    type=str,
    default=None,
    help="Unit for an explicit physical pixel-size override; default µm.",
)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("--checksums/--no-checksums", default=True, show_default=True)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=(
        "Show stages and 5-second progress bars; repeat for debug details, "
        "1-second progress, and tracebacks."
    ),
)
def mutate_command(
    input_path: Path,
    output_path: Path,
    input_type: str,
    channel_name_field: str | None,
    rename_channels_json: Path | None,
    rename_channels_by: str | None,
    dtype: str,
    range_mode: str,
    series: int,
    sample_pixels_per_channel: int,
    auto_max_normalized_rmse: float,
    omit_uuid: bool,
    output_json: Path | None,
    cache_directory: Path | None,
    compression: str,
    tile_size: int,
    pyramid_levels: int | None,
    downsample: str,
    workers: int | None,
    pixel_size_x: float | None,
    pixel_size_y: float | None,
    pixel_size_unit: str | None,
    overwrite: bool,
    checksums: bool,
    verbose: int,
) -> None:
    """Create a dtype-mutated OME-TIFF from planar floating-point INPUT_PATH."""

    _configure_logging(verbose)
    rename_channels, rename_mode = _load_channel_renames(
        rename_channels_json,
        rename_channels_by,  # type: ignore[arg-type]
    )
    if output_json is not None:
        report_path = output_json.resolve()
        if report_path in {input_path.resolve(), output_path.resolve()}:
            raise click.UsageError("--output-json must differ from both image paths")
    if input_type != "qptiff_fusion" and channel_name_field is not None:
        raise click.UsageError("--channel-name-field is only valid with --type qptiff_fusion")

    pixel_size = _pixel_size_override(pixel_size_x, pixel_size_y, pixel_size_unit)
    try:
        report = mutate(
            input_path,
            output_path,
            input_type=input_type,  # type: ignore[arg-type]
            channel_name_field=channel_name_field,  # type: ignore[arg-type]
            rename_channels=rename_channels,
            rename_channels_by=rename_mode,
            pixel_size=pixel_size,
            dtype=dtype,  # type: ignore[arg-type]
            range_mode=range_mode,  # type: ignore[arg-type]
            series=series,
            sample_pixels_per_channel=sample_pixels_per_channel,
            auto_max_normalized_rmse=auto_max_normalized_rmse,
            compression=compression,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,  # type: ignore[arg-type]
            max_workers=workers,
            display_uuid=not omit_uuid,
            overwrite=overwrite,
            calculate_checksums=checksums,
            cache_directory=cache_directory,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(str(exc)) from exc

    _write_or_echo(json.dumps(report, indent=2), output_json)


@main.command("inspect", context_settings=_CONTEXT_SETTINGS)
@click.argument(
    "input_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "-d",
    "--detail",
    type=click.IntRange(min=0, max=3),
    default=1,
    show_default=True,
    help="0=file/series, 1=levels+OME, 2=pages/frames, 3=tags/descriptions.",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Emit the schema-backed report as JSON instead of a text tree.",
)
@click.option(
    "--max-text-length",
    type=click.IntRange(min=0),
    default=240,
    show_default=True,
    help="Maximum tag/description preview length; use 0 for no truncation.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the report to a file instead of stdout.",
)
def inspect_command(
    input_path: Path,
    detail: int,
    as_json: bool,
    max_text_length: int,
    output: Path | None,
) -> None:
    """Inspect any TIFF at INPUT_PATH without reading its image pixels.

    Inspection output is raw diagnostic metadata, not deidentified output. It
    may contain paths, filenames, vendor fields, or other identifying values.
    """

    if output is not None and output.resolve() == input_path.resolve():
        raise click.UsageError("--output must differ from INPUT_PATH")
    try:
        inspector = TiffInspector(
            input_path,
            detail=detail,
            max_text_length=None if max_text_length == 0 else max_text_length,
        )
        rendered = inspector.to_json() if as_json else inspector.render_text()
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc
    _write_or_echo(rendered, output)


@main.command("version", context_settings=_CONTEXT_SETTINGS)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Include the Python and image-I/O dependency versions as JSON.",
)
def version_command(as_json: bool) -> None:
    """Display the installed omeify version."""

    if as_json:
        click.echo(json.dumps(get_version_info(), indent=2))
    else:
        click.echo(f"omeify {__version__}")


if __name__ == "__main__":
    main()
