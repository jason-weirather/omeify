from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

from omeify import __version__, get_version_info
from omeify.cli_logging import configure_logging
from omeify.cli_reporting import emit_operation_report, preflight_report
from omeify.concatenation import ConcatSeries, concat
from omeify.conversion import RenameChannelsBy, convert
from omeify.cropping import crop, load_crop_document
from omeify.dtype_mutation import (
    DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    RANGE_MODES,
    TARGET_DTYPES,
)
from omeify.inspection import TiffInspector
from omeify.intelligence import (
    DEFAULT_ALLOWED_SCOPES,
    DEFAULT_MAX_METADATA_CHARS,
    DEFAULT_MAX_OUTPUT_TOKENS,
)
from omeify.io._writer.configuration import DEFAULT_JPEG_QUALITY, DEFAULT_JPEG_SUBSAMPLING
from omeify.io.pixel_size import PixelSize
from omeify.io.source_reader import INPUT_TYPES, PLANAR_INPUT_TYPES
from omeify.mutation import mutate
from omeify.path_safety import atomic_write_text, check_output_path
from omeify.regions import group_outputs, parse_regions
from omeify.terminal import escape_terminal
from omeify.io.output import OUTPUT_FORMATS
from omeify.io.ome_zarr.store import is_remote, preflight_directory
from omeify.zarr_inspection import ZarrInspector

_CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}
_INDEX_KEY = re.compile(r"0|[1-9][0-9]*")


def source_options(function):
    """Shared explicit plane/group selection for every image-consuming command."""
    for option in reversed((
        click.option("--group", default="", help="OME-Zarr image group within the store."),
        click.option("--z-index", "z", type=click.IntRange(min=0), default=None,
                     help="Explicit OME-Zarr Z plane; required when Z is non-singleton."),
        click.option("--t-index", "t", type=click.IntRange(min=0), default=None,
                     help="Explicit OME-Zarr time point; required when T is non-singleton."),
        click.option("--image-type", type=click.Choice(["multichannel", "rgb", "label"]),
                     default=None, help="Explicit pixel interpretation; three channels never imply RGB."),
        click.option("--intrinsic-only", is_flag=True,
                     help="Explicitly export the selected OME-Zarr intrinsic raster, not extra spatial relationships."),
    )):
        function = option(function)
    return function


def output_options(function):
    return click.option("--output-format", type=click.Choice(OUTPUT_FORMATS),
                        default="ome_tiff", show_default=True,
                        help="Output storage contract; Zarr requires a NEW directory.")(function)


def preflight_storage(output_path, output_format, *, protected=(), overwrite=True):
    if output_format == "ome_zarr":
        context = click.get_current_context()
        for name in ("jpeg_quality", "jpeg_subsampling", "omit_uuid", "cache_directory"):
            if context.get_parameter_source(name) not in (None, click.core.ParameterSource.DEFAULT):
                raise click.UsageError(f"{name.replace('_', '-')} is TIFF-only, not an OME-Zarr setting")
        if context.get_parameter_source("overwrite") != click.core.ParameterSource.DEFAULT and overwrite:
            raise click.UsageError("OME-Zarr stores cannot be overwritten; choose a new destination")
        preflight_directory(output_path, protected=protected)
    else:
        if Path(output_path).is_dir():
            raise IsADirectoryError(f"Output is a directory: {output_path}")
        check_output_path(output_path, protected=protected, overwrite=overwrite)


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
        raise click.ClickException(escape_terminal(
            f"Unable to read channel rename JSON {path}: {exc}"
        )) from exc
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


def _write_or_echo(rendered: str, output: Path | None, *, input_path: Path) -> None:
    if output is None:
        click.echo(rendered)
    else:
        try:
            atomic_write_text(output, rendered, protected=(input_path,))
        except (OSError, ValueError) as exc:
            raise click.ClickException(escape_terminal(str(exc))) from exc


def _render_visual_answer(report: dict[str, Any]) -> str:
    answer = report["answer"]
    lines = ["Visual question (advisory)", "-" * 26, f"Question: {escape_terminal(report['question'])}"]
    status = answer["status"]
    if status != "answered":
        lines.append(f"Status: {status}")
    lines.append("")
    lines.extend(escape_terminal(item["text"], multiline=True) for item in answer["paragraphs"])
    if answer["cautions"]:
        lines.append("")
        lines.append("Cautions:")
        lines.extend(f"- {escape_terminal(item['text'])}" for item in answer["cautions"])
    return "\n".join(lines).rstrip()


def _visual_read_exception(
    exc: Exception, *, input_path: str | Path, input_type: str, type_was_defaulted: bool,
) -> click.ClickException:
    """Suggest an explicit vendor reader for a missing OME header, never select one.

    The filename extension is only a conditional user-facing hint, not evidence
    used to dispatch or interpret pixels. Other read/vision errors pass through.
    """
    message = str(exc)
    if (
        type_was_defaulted
        and input_type == "ome_tiff"
        and isinstance(exc, ValueError)
        and message.startswith("TIFF does not contain recognized OME metadata:")
    ):
        suffix = Path(input_path).suffix.lower()
        if suffix == ".svs":
            suggestion = "If this is an Aperio SVS image, use --type svs."
        elif suffix == ".scn":
            suggestion = "If this is a Leica brightfield SCN image, use --type scn_he."
        elif suffix == ".qptiff":
            suggestion = (
                "For Akoya QPTIFF, choose the applicable profile explicitly: "
                "--type qptiff_he, qptiff_mif, or qptiff_fusion."
            )
        else:
            suggestion = (
                "Choose the appropriate --type explicitly (for example, "
                "--type svs or --type scn_he); see omeify inspect --help."
            )
        message = (
            "The default --type ome_tiff requires recognized OME metadata, "
            "but none was recognized in this TIFF. Supported vendor images can be used for visual "
            f"questions without OME metadata. {suggestion} "
            "The image was not sent to the model."
        )
    return click.ClickException(escape_terminal(message))


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
    """Convert, concatenate, crop, mutate, and inspect calibrated OME images."""


class _ConcatCommand(click.Command):
    """Retain option occurrence order before Click groups repeatable values.

    Click's parser reports each recognized option in its original encounter order,
    but a ``multiple=True`` callback receives a grouped tuple. Parse the tiny
    argument list once without callbacks to recover that order, then delegate all
    conversion, validation and error handling to the normal Click path.
    """

    def parse_args(self, ctx: click.Context, args: list[str]) -> list[str]:
        _, _, order = self.make_parser(ctx).parse_args(args=list(args))
        ctx.meta["_omeify_concat_order"] = tuple(
            option.name for option in order
            if option.name in {"source_series", "named_source_series"}
        )
        return super().parse_args(ctx, args)


@main.command("concat", cls=_ConcatCommand, context_settings=_CONTEXT_SETTINGS)
@click.option(
    "--series", "source_series", multiple=True,
    type=(click.Path(file_okay=True, dir_okay=False, path_type=Path),
          click.IntRange(min=0)),
    metavar="OME_PATH INDEX",
    help="Append the selected source OME-TIFF series, retaining its name. Repeatable.",
)
@click.option(
    "--named-series", "named_source_series", multiple=True,
    type=(str, click.Path(file_okay=True, dir_okay=False, path_type=Path),
          click.IntRange(min=0)),
    metavar="NAME OME_PATH INDEX",
    help="Append a source OME-TIFF series using NAME in the output. Repeatable.",
)
@click.option(
    "-o", "--output", "output_path", required=True,
    type=click.Path(file_okay=True, dir_okay=False, path_type=Path),
    help="Exact destination multi-series OME-TIFF filename.",
)
@click.option(
    "--compression",
    type=click.Choice(["LZW", "Deflate", "ZSTD", "Uncompressed"], case_sensitive=False),
    default="LZW", show_default=True,
    help="Lossless for all series, including RGB; base pixels are preserved.",
)
@click.option("--tile-size", type=click.IntRange(min=16), default=None,
              help="Optional square TIFF tile edge; default 512 RGB / 1024 scalar.")
@click.option("--pyramid-levels", type=click.IntRange(min=0), default=None,
              help="Output reduced levels; rebuilt, not copied from the inputs.")
@click.option("--cache-directory", type=click.Path(file_okay=False, path_type=Path),
              default=None, help="Temporary pyramid scratch directory.")
@click.option("--workers", type=click.IntRange(min=1), default=None,
              help="Maximum TIFF encoding workers.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path),
              help="Write the construction report to FILE, or - for JSON-only stdout.")
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("-v", "--verbose", count=True,
              help="-v: compact progress; -vv: diagnostic logs; -vvv: full report.")
def concat_command(
    source_series: tuple[tuple[Path, int], ...],
    named_source_series: tuple[tuple[str, Path, int], ...],
    output_path: Path, compression: str, tile_size: int | None,
    pyramid_levels: int | None, cache_directory: Path | None,
    workers: int | None, output_json: Path | None, overwrite: bool, verbose: int,
) -> None:
    """Concatenate selected OME-TIFF Images in the exact order supplied.

    --series OME_PATH INDEX and --named-series NAME OME_PATH INDEX can be freely
    mixed. INDEX is zero-based. Each occurrence contributes one independent OME
    image, not a channel, mosaic or registered scene. Original image dimensions,
    physical calibration and base values are retained; pyramids are rebuilt.
    """
    configure_logging(verbose)
    ctx = click.get_current_context()
    plain, named = iter(source_series), iter(named_source_series)
    selections = []
    for option in ctx.meta.get("_omeify_concat_order", ()):
        if option == "source_series":
            path, index = next(plain)
            selections.append(ConcatSeries(path, index))
        else:
            name, path, index = next(named)
            if not name.strip():
                raise click.UsageError("--named-series NAME must not be empty")
            selections.append(ConcatSeries(path, index, name))
    if len(selections) != len(source_series) + len(named_source_series):
        raise click.UsageError("Unable to determine source option order")
    if not selections:
        raise click.UsageError("Provide at least one --series or --named-series")
    protected = [entry.path for entry in selections]
    try:
        check_output_path(output_path, protected=protected, overwrite=overwrite)
        preflight_report(output_json, protected=[*protected, output_path], overwrite=overwrite)
        report = concat(
            selections, output_path, compression=compression, tile_size=tile_size,
            pyramid_levels=pyramid_levels, max_workers=workers,
            cache_directory=cache_directory, overwrite=overwrite,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(escape_terminal(str(exc))) from exc
    emit_operation_report(
        report, output_path=output_path, destination=output_json, verbose=verbose,
        overwrite=overwrite, protected=[*protected, output_path],
    )


@main.command("convert", context_settings=_CONTEXT_SETTINGS)
@source_options
@output_options
@click.argument(
    "input_path",
    type=str,
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Destination OME-TIFF file or NEW OME-Zarr directory.",
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
    help="Write the complete report to FILE, or - for JSON-only stdout.",
)
@click.option(
    "--cache-directory",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for temporary rebuilt pyramid levels.",
)
@click.option(
    "--compression",
    type=click.Choice(["LZW", "Deflate", "ZSTD", "Gzip", "JPEG", "Uncompressed"], case_sensitive=False),
    default=None,
    help="TIFF: JPEG for RGB, LZW otherwise. OME-Zarr: lossless ZSTD.",
)
@click.option(
    "--jpeg-quality",
    type=click.IntRange(min=1, max=100),
    default=DEFAULT_JPEG_QUALITY,
    show_default=True,
    help="JPEG encoder quality, including automatic RGB JPEG output.",
)
@click.option(
    "--jpeg-subsampling",
    type=click.Choice(["444", "422", "420", "411"]),
    default=DEFAULT_JPEG_SUBSAMPLING,
    show_default=True,
    help="JPEG chroma subsampling for interleaved RGB output.",
)
@click.option(
    "--tile-size",
    type=click.IntRange(min=16),
    default=None,
    show_default="TIFF: 512 RGB / 1024 scalar; Zarr: 512",
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
    help="Maximum parallel output compression workers.",
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
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=("-v: compact progress; -vv: timestamped diagnostic logs and tracebacks; "
          "-vvv: also print the complete report unless --output-json is supplied."),
)
def convert_command(
    group: str, z: int | None, t: int | None, image_type: str | None, intrinsic_only: bool,
    output_format: str,
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
    tile_size: int | None,
    pyramid_levels: int | None,
    downsample: str,
    workers: int | None,
    pixel_size_x: float | None,
    pixel_size_y: float | None,
    pixel_size_unit: str | None,
    overwrite: bool,
    verbose: int,
) -> None:
    """Convert INPUT_PATH into metadata-minimized pyramidal OME-TIFF or OME-Zarr.

    Supply the destination with --output / -o.
    """

    configure_logging(verbose)
    input_path = str(input_path) if is_remote(input_path) else Path(input_path)
    rename_channels, rename_mode = _load_channel_renames(
        rename_channels_json,
        rename_channels_by,  # type: ignore[arg-type]
    )

    inputs = [input_path]
    if rename_channels_json is not None:
        inputs.append(rename_channels_json)
    protected = [*inputs, output_path]
    try:
        preflight_storage(output_path, output_format, protected=inputs, overwrite=overwrite)
        preflight_report(output_json, protected=protected, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        raise click.UsageError(escape_terminal(str(exc))) from exc

    if input_type != "qptiff_fusion" and channel_name_field is not None:
        raise click.UsageError("--channel-name-field is only valid with --type qptiff_fusion")

    pixel_size = _pixel_size_override(pixel_size_x, pixel_size_y, pixel_size_unit)
    try:
        report = convert(
            input_path,
            output_path,
            input_type=input_type,  # type: ignore[arg-type]
            output_format=output_format, group=group, z=z, t=t, image_type=image_type,
            intrinsic_only=intrinsic_only,
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
            cache_directory=cache_directory,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(escape_terminal(str(exc))) from exc

    emit_operation_report(
        report, output_path=output_path, destination=output_json, verbose=verbose,
        overwrite=overwrite, protected=protected,
    )


@main.command("mutate", context_settings=_CONTEXT_SETTINGS)
@source_options
@output_options
@click.argument(
    "input_path",
    type=str,
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(file_okay=True, dir_okay=True, path_type=Path),
    required=True,
    help="Destination OME-TIFF file or NEW OME-Zarr directory.",
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
    default=None,
    help=(
        "Convert floating-point pixels to this unsigned integer dtype. "
        "Mutually exclusive with --float32-mantissa-bits."
    ),
)
@click.option(
    "--float32-mantissa-bits",
    type=click.IntRange(min=0, max=22),
    default=None,
    help=(
        "Keep float32 storage but retain only N stored fraction bits. This is lossy and "
        "can improve LZW compression. N=11 retains 12 bits of significand precision, "
        "comparable to a 12-bit acquisition such as Akoya PhenoImager HT Extended Range. "
        "Mutually exclusive with --dtype."
    ),
)
@click.option(
    "--range-mode",
    type=click.Choice(RANGE_MODES),
    default=None,
    help=(
        "Integer dtype mutation only; defaults to auto. Auto preserves source scale when "
        "measured unit-rounding loss is small or integer ancestry evidence is strong; "
        "preserve forces no rescaling; full maps each channel's exact minimum and maximum "
        "to the dtype limits."
    ),
)
@click.option("--series", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--sample-pixels-per-channel",
    type=click.IntRange(min=1024),
    default=DEFAULT_SAMPLE_PIXELS_PER_CHANNEL,
    show_default=True,
    help="Integer dtype mutation only: deterministic spatial sample for lattice and loss estimates.",
)
@click.option(
    "--auto-max-normalized-rmse",
    type=click.FloatRange(min=0),
    default=DEFAULT_AUTO_MAX_NORMALIZED_RMSE,
    show_default=True,
    help=(
        "Integer dtype mutation only: largest unit-rounding RMSE, as a fraction of the "
        "sampled nonzero p0.1-p99.9 intensity span, that auto mode may accept without "
        "strong integer-lattice evidence."
    ),
)
@click.option("--omit-uuid", is_flag=True, help="Omit the optional OME root UUID.")
@click.option(
    "--output-json",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Write the complete report to FILE, or - for JSON-only stdout.",
)
@click.option(
    "--cache-directory",
    type=click.Path(file_okay=False, path_type=Path),
    help="Directory for temporary rebuilt pyramid levels.",
)
@click.option(
    "--compression",
    type=click.Choice(["LZW", "Deflate", "ZSTD", "Gzip", "Uncompressed"], case_sensitive=False),
    default=None,
    help="Lossless compression: TIFF defaults to LZW; Zarr defaults to ZSTD.",
)
@click.option(
    "--tile-size",
    type=click.IntRange(min=16),
    default=None,
    help="Spatial tile/chunk edge; default TIFF 1024 for scalar data, Zarr 512.",
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
    help="Maximum parallel output compression workers.",
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
@click.option(
    "-v",
    "--verbose",
    count=True,
    help=("-v: compact progress; -vv: timestamped diagnostic logs and tracebacks; "
          "-vvv: also print the complete report unless --output-json is supplied."),
)
def mutate_command(
    group: str, z: int | None, t: int | None, image_type: str | None, intrinsic_only: bool,
    output_format: str,
    input_path: Path,
    output_path: Path,
    input_type: str,
    channel_name_field: str | None,
    rename_channels_json: Path | None,
    rename_channels_by: str | None,
    dtype: str | None,
    float32_mantissa_bits: int | None,
    range_mode: str | None,
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
    verbose: int,
) -> None:
    """Create an explicitly pixel-mutated OME image from floating-point INPUT_PATH.

    Supply the destination with --output / -o.
    """

    configure_logging(verbose)
    input_path = str(input_path) if is_remote(input_path) else Path(input_path)
    if (dtype is None) == (float32_mantissa_bits is None):
        raise click.UsageError(
            "Choose exactly one mutation: --dtype or --float32-mantissa-bits."
        )
    if float32_mantissa_bits is not None and range_mode is not None:
        raise click.UsageError(
            "--range-mode is only valid with --dtype."
        )
    rename_channels, rename_mode = _load_channel_renames(
        rename_channels_json,
        rename_channels_by,  # type: ignore[arg-type]
    )
    inputs = [input_path]
    if rename_channels_json is not None:
        inputs.append(rename_channels_json)
    protected = [*inputs, output_path]
    try:
        preflight_storage(output_path, output_format, protected=inputs, overwrite=overwrite)
        preflight_report(output_json, protected=protected, overwrite=overwrite)
    except (OSError, ValueError) as exc:
        raise click.UsageError(escape_terminal(str(exc))) from exc
    if input_type != "qptiff_fusion" and channel_name_field is not None:
        raise click.UsageError("--channel-name-field is only valid with --type qptiff_fusion")

    pixel_size = _pixel_size_override(pixel_size_x, pixel_size_y, pixel_size_unit)
    try:
        report = mutate(
            input_path,
            output_path,
            input_type=input_type,  # type: ignore[arg-type]
            output_format=output_format, group=group, z=z, t=t, image_type=image_type,
            intrinsic_only=intrinsic_only,
            channel_name_field=channel_name_field,  # type: ignore[arg-type]
            rename_channels=rename_channels,
            rename_channels_by=rename_mode,
            pixel_size=pixel_size,
            dtype=dtype,  # type: ignore[arg-type]
            float32_mantissa_bits=float32_mantissa_bits,
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
            cache_directory=cache_directory,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(escape_terminal(str(exc))) from exc

    emit_operation_report(
        report, output_path=output_path, destination=output_json, verbose=verbose,
        overwrite=overwrite, protected=protected,
    )


@main.command("crop", context_settings=_CONTEXT_SETTINGS)
@source_options
@output_options
@click.argument("input_path", type=str)
@click.option("-o", "--output", "output_path", required=True,
              type=click.Path(dir_okay=True, path_type=Path),
              help="Exact TIFF file or NEW Zarr directory; --shatter adds per-product suffixes.")
@click.option("--bounds", type=int, nargs=4, metavar="X0 Y0 X1 Y1",
              help="Half-open integer bounds in level-zero pixels; exclusive with --geojson.")
@click.option("--geojson", "geojson_path", type=str, metavar="FILE",
              help="Pixel-coordinate GeoJSON file, or - for stdin; exclusive with --bounds.")
@click.option("--output-json", type=click.Path(dir_okay=False, path_type=Path),
              help="Write the complete report to FILE, or - for JSON-only stdout.")
@click.option("--shatter", type=click.Choice(["by_index", "by_name"]), default=None,
              help=("Split into files by input index or by name (equal names share a file). "
                    "Default: all ROIs in one file, in input order."))
@click.option("--clip", is_flag=True, help="Explicitly clip out-of-bounds rectangles to the image.")
@click.option("--type", "input_type", type=click.Choice(INPUT_TYPES),
              default="ome_tiff", show_default=True)
@click.option("--series", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--labels", is_flag=True,
              help="Declare a categorical label image; use lossless/nearest storage.")
@click.option("--channel-name-field", type=click.Choice(["name", "biomarker", "auto"]), default=None)
@click.option("--pixel-size-x", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--pixel-size-y", type=click.FloatRange(min=0, min_open=True), default=None)
@click.option("--pixel-size-unit", type=str, default=None)
@click.option("--compression", type=click.Choice(
    ["LZW", "Deflate", "ZSTD", "Gzip", "JPEG", "Uncompressed"], case_sensitive=False), default=None,
    help="TIFF: JPEG for RGB, LZW otherwise. Zarr: lossless ZSTD.")
@click.option("--jpeg-quality", type=click.IntRange(min=1, max=100), default=DEFAULT_JPEG_QUALITY)
@click.option("--jpeg-subsampling", type=click.Choice(["444", "422", "420", "411"]),
              default=DEFAULT_JPEG_SUBSAMPLING)
@click.option("--tile-size", type=click.IntRange(min=16), default=None,
              help="Default: TIFF 512 RGB / 1024 scalar, Zarr 512; must be divisible by 16.")
@click.option("--pyramid-levels", type=click.IntRange(min=0), default=None)
@click.option("--workers", type=click.IntRange(min=1), default=None)
@click.option("--cache-directory", type=click.Path(file_okay=False, path_type=Path), default=None)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("-v", "--verbose", count=True,
              help="-v: compact progress; -vv: diagnostic logs; -vvv: full operation report.")
def crop_command(
    group: str, z: int | None, t: int | None, image_type: str | None, intrinsic_only: bool,
    output_format: str,
    input_path: Path, output_path: Path, bounds: tuple[int, ...] | None, geojson_path: str | None,
    shatter: str | None, clip: bool, input_type: str, series: int, labels: bool,
    channel_name_field: str | None, pixel_size_x: float | None, pixel_size_y: float | None,
    pixel_size_unit: str | None, compression: str | None, jpeg_quality: int,
    jpeg_subsampling: str, tile_size: int | None, pyramid_levels: int | None,
    workers: int | None, cache_directory: Path | None, overwrite: bool, verbose: int,
    output_json: Path | None,
) -> None:
    """Crop rectangles from INPUT_PATH into one ordered, multi-image OME product.

    OUTPUT is the exact TIFF file or new Zarr directory, even for a single ROI.
    Each region becomes a named image in GeoJSON input order, not a channel or
    mosaic. --shatter by_index writes one product per ROI; by_name groups equal
    names. Shattered paths add a suffix before the selected format's extension.

    GeoJSON uses full-resolution XY pixel coordinates, not geographic coordinates.
    Polygons are exported as bounding rectangles, without masking.
    """
    configure_logging(verbose)
    input_path = str(input_path) if is_remote(input_path) else Path(input_path)
    if (bounds is None) == (geojson_path is None):
        raise click.UsageError("Choose exactly one of --bounds or --geojson")
    try:
        if geojson_path == "-":
            from omeify.regions import MAX_GEOJSON_CHARS, load_geojson

            geojson = load_geojson(click.get_text_stream("stdin").read(MAX_GEOJSON_CHARS + 1))
        else:
            geojson = geojson_path
        document, auxiliary = load_crop_document(bounds=bounds, geojson=geojson)
        outputs = group_outputs(parse_regions(document), output_path, shatter=shatter, output_format=output_format)
        image_paths = [path for path, _ in outputs]
        protected = [input_path, *auxiliary, *image_paths]
        for path in image_paths:
            preflight_storage(path, output_format, protected=(input_path, *auxiliary), overwrite=overwrite)
        preflight_report(output_json, protected=protected, overwrite=overwrite)
        report = crop(
            input_path, output_path, geojson=document,
            input_type=input_type, series=series, shatter=shatter, clip=clip, labels=labels,
            output_format=output_format, group=group, z=z, t=t, image_type=image_type,
            intrinsic_only=intrinsic_only,
            channel_name_field=channel_name_field,
            pixel_size=_pixel_size_override(pixel_size_x, pixel_size_y, pixel_size_unit),
            compression=compression, jpeg_quality=jpeg_quality, jpeg_subsampling=jpeg_subsampling,
            tile_size=tile_size, pyramid_levels=pyramid_levels, max_workers=workers,
            cache_directory=cache_directory, overwrite=overwrite,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(escape_terminal(str(exc))) from exc
    emit_operation_report(
        report, output_path=output_path, destination=output_json, verbose=verbose,
        overwrite=overwrite, protected=protected,
    )


@main.command("inspect", context_settings=_CONTEXT_SETTINGS)
@source_options
@click.argument(
    "input_path",
    type=str,
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
@click.option(
    "-i", "--intelligence", is_flag=True,
    help="Summarize metadata or answer --question using the configured model; requires omeify[intelligence].",
)
@click.option(
    "-q", "--question", type=str, default=None, metavar="TEXT",
    help="Answer a metadata question, answer a visual question with --visual, or locate regions with --geojson; requires -i.",
)
@click.option(
    "--intelligence-source", type=str, default=None,
    help="Select a configured source explicitly, within the allowed scopes; requires -i.",
)
@click.option(
    "--intelligence-scope", "intelligence_scopes", multiple=True,
    type=click.Choice(["institutional", "local", "external"]),
    default=DEFAULT_ALLOWED_SCOPES, show_default=True,
    help="Ordered scope preference; repeat to allow multiple scopes. Requires -i.",
)
@click.option(
    "--intelligence-max-chars", type=click.IntRange(min=4096),
    default=DEFAULT_MAX_METADATA_CHARS, show_default=True,
    help="Serialized metadata-record character budget, not tokens; requires -i.",
)
@click.option(
    "--intelligence-max-output-tokens", type=click.IntRange(min=1),
    default=DEFAULT_MAX_OUTPUT_TOKENS, show_default=True,
    help="Maximum generated tokens requested from the selected model; requires -i.",
)
@click.option("--visual", is_flag=True,
              help="Answer a visual question only; sends a bounded image preview. Requires -i -q.")
@click.option("--geojson", is_flag=True,
              help="Return visual ROI GeoJSON only; sends a bounded image preview. Requires -i -q.")
@click.option("--type", "input_type", type=click.Choice(INPUT_TYPES), default="ome_tiff",
              help=("Input profile for visual reads or explicit OME-Zarr inspection; "
                    "default ome_tiff. Vendor profiles must be selected explicitly."))
@click.option("--series", type=click.IntRange(min=0), default=0,
              help="Selected series for image reads and OME-Zarr inspection; default 0.")
@click.option("--preview-size", type=click.IntRange(min=128, max=2048), default=1536,
              help="Maximum preview edge in pixels for --visual / --geojson; default 1536.")
@click.option("--preview-channel", type=click.IntRange(min=0), default=None,
              help="Scalar channel index for --visual / --geojson. Default: unique DAPI, otherwise channel 0.")
@click.option("--channel", "preview_channels", nargs=2, metavar="NAME|INDEX COLOR", multiple=True,
              help=("Repeat to build a false-color scalar preview composite for --visual / --geojson. "
                    "COLOR accepts names, #RRGGBB, or R,G,B."))
@click.option("--preview-range", type=float, nargs=2, metavar="LOW HIGH", default=None,
              help="Display-only scalar intensity range for one-channel --visual / --geojson previews.")
@click.option("--preview-quantile", type=click.FloatRange(min=0, max=1, min_open=True),
              default=0.999, show_default=True,
              help="Auto-display upper quantile for scalar/composite --visual / --geojson previews.")
@click.option("--roi-size", type=click.IntRange(min=1), nargs=2, metavar="WIDTH HEIGHT", default=None,
              help="Enforce exact full-resolution ROI dimensions for --geojson.")
@click.option("-v", "--verbose", count=True, help="Progress on stderr; repeat for diagnostic logs.")
def inspect_command(
    group: str, z: int | None, t: int | None, image_type: str | None, intrinsic_only: bool,
    input_path: Path,
    detail: int,
    as_json: bool,
    max_text_length: int,
    output: Path | None,
    intelligence: bool,
    question: str | None,
    intelligence_source: str | None,
    intelligence_scopes: tuple[str, ...],
    intelligence_max_chars: int,
    intelligence_max_output_tokens: int,
    visual: bool, geojson: bool, input_type: str, series: int, preview_size: int,
    preview_channel: int | None, preview_channels: tuple[tuple[str, str], ...],
    preview_range: tuple[float, float] | None, preview_quantile: float,
    roi_size: tuple[int, int] | None, verbose: int,
) -> None:
    """Inspect TIFF/OME-Zarr metadata, answer visual questions, or locate visual regions.

    Ordinary inspection never reads pixels. --visual and --geojson are explicit
    opt-ins to transmit a bounded image overview to the configured model source.

    Inspection output is raw diagnostic metadata, not deidentified output. It
    may contain paths, filenames, vendor fields, or other identifying values.
    """

    configure_logging(verbose)
    input_path = str(input_path) if is_remote(input_path) else Path(input_path)
    ctx = click.get_current_context()
    visual_options = (
        "preview_size", "preview_channel", "preview_channels",
        "preview_range", "preview_quantile", "roi_size",
    )
    if visual and geojson:
        raise click.UsageError("Choose at most one of --visual or --geojson")
    visual_mode = visual or geojson
    if visual_mode:
        mode = "--geojson" if geojson else "--visual"
        if not intelligence or question is None:
            raise click.UsageError(f"{mode} requires --intelligence / -i and --question / -q")
        if any(ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
               for name in ("detail", "max_text_length", "intelligence_max_chars")):
            raise click.UsageError("Metadata detail/text/budget options do not apply to --visual or --geojson")
    elif any(ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
             for name in visual_options):
        raise click.UsageError("Visual input, preview, and ROI options require --visual or --geojson")
    if preview_channels and preview_channel is not None:
        raise click.UsageError("Choose either --preview-channel or one or more --channel options")
    if preview_channels and preview_range is not None:
        raise click.UsageError("--preview-range applies only to one-channel previews, not --channel composites")
    if visual and ctx.get_parameter_source("roi_size") != click.core.ParameterSource.DEFAULT:
        raise click.UsageError("--roi-size applies only to --geojson")
    if question is not None:
        if not intelligence:
            raise click.UsageError("--question / -q requires --intelligence / -i")
        from omeify.intelligence import IntelligenceError, _validate_question

        try:
            _validate_question(question)
        except IntelligenceError as exc:
            raise click.UsageError(str(exc)) from exc
    if not intelligence and any(
        ctx.get_parameter_source(name) != click.core.ParameterSource.DEFAULT
        for name in (
            "intelligence_source", "intelligence_scopes",
            "intelligence_max_chars", "intelligence_max_output_tokens",
        )
    ):
        raise click.UsageError("--intelligence-* options require --intelligence / -i")
    if output is not None:
        try:
            check_output_path(output, protected=(input_path,))
        except (OSError, ValueError) as exc:
            raise click.UsageError(escape_terminal(str(exc))) from exc
    if visual:
        from omeify.io.source_reader import source_reader
        from omeify.visual_intelligence import answer_visual_question

        click.echo("Preparing bounded preview for visual analysis...", err=True)
        try:
            with source_reader(input_path, input_type=input_type, series=series,
                               channel_name_field=None, group=group, z=z, t=t,
                               image_type=image_type, intrinsic_only=intrinsic_only) as image:
                result = answer_visual_question(
                    image, question, series=series,
                    preview_size=preview_size, preview_channel=preview_channel,
                    preview_channels=preview_channels, preview_range=preview_range,
                    preview_quantile=preview_quantile, source_name=intelligence_source,
                    allowed_scopes=intelligence_scopes,
                    max_output_tokens=intelligence_max_output_tokens,
                )
        except Exception as exc:
            raise _visual_read_exception(
                exc, input_path=input_path, input_type=input_type,
                type_was_defaulted=(ctx.get_parameter_source("input_type")
                                    == click.core.ParameterSource.DEFAULT),
            ) from exc
        rendered = (json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
                    if as_json else _render_visual_answer(result))
        _write_or_echo(rendered, output, input_path=input_path)
        return
    if geojson:
        from omeify.io.source_reader import source_reader
        from omeify.visual_intelligence import locate_regions

        click.echo("Preparing bounded preview for region localization...", err=True)
        try:
            with source_reader(input_path, input_type=input_type, series=series,
                               channel_name_field=None, group=group, z=z, t=t,
                               image_type=image_type, intrinsic_only=intrinsic_only) as image:
                result = locate_regions(
                    image, question, series=series, roi_size=roi_size,
                    preview_size=preview_size, preview_channel=preview_channel,
                    preview_channels=preview_channels, preview_range=preview_range,
                    preview_quantile=preview_quantile, source_name=intelligence_source,
                    allowed_scopes=intelligence_scopes,
                    max_output_tokens=intelligence_max_output_tokens,
                )
        except Exception as exc:
            raise _visual_read_exception(
                exc, input_path=input_path, input_type=input_type,
                type_was_defaulted=(ctx.get_parameter_source("input_type")
                                    == click.core.ParameterSource.DEFAULT),
            ) from exc
        _write_or_echo(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
                       output, input_path=input_path)
        return
    try:
        zarr_input = input_type == "ome_zarr" or is_remote(input_path) or Path(input_path).is_dir()
        if zarr_input:
            if intelligence:
                raise click.UsageError("Metadata intelligence is TIFF-only; ordinary Zarr inspection and explicit --visual/--geojson are supported")
            inspector = ZarrInspector(
                input_path, group=group, series=series, z=z, t=t, image_type=image_type,
                intrinsic_only=intrinsic_only, detail=detail,
                max_text_length=None if max_text_length == 0 else max_text_length,
            )
        else:
            if group or z is not None or t is not None or intrinsic_only or image_type is not None:
                raise click.UsageError("These image selectors require OME-Zarr inspection")
            if ctx.get_parameter_source("series") != click.core.ParameterSource.DEFAULT:
                raise click.UsageError("Ordinary TIFF inspection lists all series; omit --series")
            inspector = TiffInspector(
                input_path, detail=detail,
                max_text_length=None if max_text_length == 0 else max_text_length,
            )
        if intelligence:
            click.echo(
                "Answering metadata question..." if question is not None
                else "Summarizing embedded metadata...",
                err=True,
            )
            inspector.summarize_metadata(
                question=question,
                source_name=intelligence_source, allowed_scopes=intelligence_scopes,
                max_metadata_chars=intelligence_max_chars,
                max_output_tokens=intelligence_max_output_tokens,
            )
        rendered = inspector.to_json() if as_json else inspector.render_text()
    except Exception as exc:
        raise click.ClickException(escape_terminal(str(exc))) from exc
    _write_or_echo(rendered, output, input_path=input_path)


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
