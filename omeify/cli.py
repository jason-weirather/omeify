from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click

from omeify import get_version_info
from omeify.inputs import (
    AkoyaComponentTiff,
    AkoyaHEQptiff,
    AkoyaMIFQptiff,
    AperioSVS,
    HaloMIFTiff,
)


def _version_callback(ctx: click.Context, param: click.Parameter, value: bool) -> None:
    if not value or ctx.resilient_parsing:
        return
    click.echo(json.dumps(get_version_info(), indent=2))
    ctx.exit()


def _load_channel_renames(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    try:
        value: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise click.ClickException(f"Unable to read channel rename JSON {path}: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise click.ClickException(
            "Channel rename JSON must be an object mapping strings to strings"
        )
    return value


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
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
    type=click.Choice(["qptiff_mif", "qptiff_he", "svs", "halo_mif", "component"]),
    required=True,
    help="Input image profile.",
)
@click.option("--series", type=click.IntRange(min=0), default=0, show_default=True)
@click.option(
    "--rename-channels-json",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="JSON object mapping source channel names to output names.",
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
    help="Default: JPEG for qptiff_he/svs; LZW for planar mIF/component inputs.",
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
    help="Number of subresolution levels.  By default, build until the image fits one tile.",
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
@click.option(
    "--physical-size-x-um",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
)
@click.option(
    "--physical-size-y-um",
    type=click.FloatRange(min=0, min_open=True),
    default=None,
)
@click.option("--overwrite/--no-overwrite", default=True, show_default=True)
@click.option("--checksums/--no-checksums", default=True, show_default=True)
@click.option("-v", "--verbose", count=True, help="Increase logging verbosity.")
@click.option(
    "--version",
    is_flag=True,
    is_eager=True,
    expose_value=False,
    callback=_version_callback,
    help="Display omeify and dependency versions as JSON.",
)
def main(
    input_path: Path,
    output_path: Path,
    input_type: str,
    series: int,
    rename_channels_json: Path | None,
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
    physical_size_x_um: float | None,
    physical_size_y_um: float | None,
    overwrite: bool,
    checksums: bool,
    verbose: int,
) -> None:
    """Convert INPUT_PATH into a deidentified pyramidal OME-TIFF at OUTPUT_PATH."""

    log_level = logging.DEBUG if verbose >= 2 else logging.INFO if verbose == 1 else logging.WARNING
    logging.basicConfig(level=log_level, format="%(levelname)s %(name)s: %(message)s")
    rename_channels = _load_channel_renames(rename_channels_json)

    if output_json is not None:
        report_path = output_json.resolve()
        if report_path in {input_path.resolve(), output_path.resolve()}:
            raise click.UsageError("--output-json must differ from both image paths")

    if input_type == "qptiff_mif":
        processor = AkoyaMIFQptiff(input_path, series=series, rename_channels=rename_channels)
    elif input_type == "qptiff_he":
        processor = AkoyaHEQptiff(input_path, series=series, rename_channels=rename_channels)
    elif input_type == "svs":
        processor = AperioSVS(input_path, series=series, rename_channels=rename_channels)
    elif input_type == "halo_mif":
        processor = HaloMIFTiff(input_path, series=series, rename_channels=rename_channels)
    elif input_type == "component":
        if physical_size_x_um is None or physical_size_y_um is None:
            raise click.UsageError(
                "--physical-size-x-um and --physical-size-y-um are required for component TIFFs"
            )
        processor = AkoyaComponentTiff(
            input_path,
            series=series,
            physical_size_x_um=physical_size_x_um,
            physical_size_y_um=physical_size_y_um,
            rename_channels=rename_channels,
        )
    else:
        raise AssertionError(input_type)

    if cache_directory is not None:
        processor.cache_directory = cache_directory

    try:
        report = processor.convert(
            output_path,
            display_uuid=not omit_uuid,
            compression=compression,
            jpeg_quality=jpeg_quality,
            jpeg_subsampling=jpeg_subsampling,
            tile_size=tile_size,
            pyramid_levels=pyramid_levels,
            downsample=downsample,
            max_workers=workers,
            overwrite=overwrite,
            calculate_checksums=checksums,
        )
    except Exception as exc:
        if verbose >= 2:
            raise
        raise click.ClickException(str(exc)) from exc

    rendered = json.dumps(report, indent=2)
    if output_json is None:
        click.echo(rendered)
    else:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
