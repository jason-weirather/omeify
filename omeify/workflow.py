from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping, Sequence
from numbers import Integral
from pathlib import Path
from typing import Any, Literal

from .io.ome_tiff_reader import OMETiffReader
from .io.pixel_size import PixelSize

RenameChannelsBy = Literal["name", "index"]
ChannelRenameMapping = Mapping[str, str] | Mapping[int, str]

LOGGER = logging.getLogger(__name__)


def validate_image_paths(
    input_path: str | Path,
    output_path: str | Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    """Validate the common source/destination contract for convert and mutate."""

    input_file = Path(input_path)
    output_file = Path(output_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input image does not exist: {input_file}")
    if input_file.resolve() == output_file.resolve():
        raise ValueError("Input and output paths must be different")
    if os.path.lexists(output_file) and not overwrite:
        raise FileExistsError(f"Output already exists: {output_file}")
    return input_file, output_file


def validate_channel_rename_mapping(
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[dict[str, str] | dict[int, str], RenameChannelsBy | None]:
    """Validate channel renaming once for both primary-image workflows."""

    if not mapping:
        if by is not None and by not in {"name", "index"}:
            raise ValueError("rename_channels_by must be 'name' or 'index'")
        return {}, by
    if by not in {"name", "index"}:
        raise ValueError(
            "rename_channels_by must be explicitly set to 'name' or 'index' "
            "when rename_channels is provided"
        )
    if not all(isinstance(value, str) and value.strip() for value in mapping.values()):
        raise TypeError("Channel rename values must be non-empty strings")

    if by == "name":
        normalized: dict[str, str] = {}
        for key, value in mapping.items():
            if not isinstance(key, str):
                raise TypeError("Name-based channel rename keys must be strings")
            # The mode is explicit, so a source channel literally named "0"
            # remains a name instead of being guessed to be an index.
            normalized[key] = value.strip()
        return normalized, by

    normalized_index: dict[int, str] = {}
    for key, value in mapping.items():
        if isinstance(key, bool) or not isinstance(key, Integral):
            raise TypeError("Index-based Python channel rename keys must be integers")
        index = int(key)
        if index < 0:
            raise ValueError("Index-based channel rename keys must be zero or greater")
        normalized_index[index] = value.strip()
    return normalized_index, by


def apply_channel_renames(
    channel_names: Sequence[str],
    mapping: ChannelRenameMapping | None,
    by: RenameChannelsBy | None,
) -> tuple[str, ...]:
    """Apply the explicit name- or index-based channel policy."""

    normalized, normalized_by = validate_channel_rename_mapping(mapping, by)
    names = tuple(str(item) for item in channel_names)
    if not normalized:
        return names

    if normalized_by == "name":
        name_mapping = normalized
        unknown = sorted(set(name_mapping) - set(names))
        if unknown:
            raise ValueError(
                "Name-based channel rename mapping contains unknown source names: "
                + ", ".join(repr(item) for item in unknown)
            )
        ambiguous = sorted(name for name in name_mapping if names.count(name) > 1)
        if ambiguous:
            raise ValueError(
                "Name-based channel rename mapping is ambiguous for duplicate source names: "
                + ", ".join(repr(item) for item in ambiguous)
                + "; use rename_channels_by='index'"
            )
        return tuple(name_mapping.get(name, name) for name in names)

    index_mapping = normalized
    out_of_range = sorted(index for index in index_mapping if index >= len(names))
    if out_of_range:
        raise IndexError(
            "Index-based channel rename mapping contains out-of-range indices: "
            + ", ".join(str(item) for item in out_of_range)
        )
    return tuple(index_mapping.get(index, name) for index, name in enumerate(names))


def log_source_summary(
    logger: logging.Logger,
    reader: Any,
    *,
    source_pixel_size: PixelSize | None,
    effective_pixel_size: PixelSize,
    pixel_size_overridden: bool,
) -> None:
    """Describe the normalized source contract without reading image pixels."""

    logger.info("Source opened as %s", reader.input_type_description)
    logger.info(
        "Source raster: axes=%s, shape=%s, dtype=%s, byte-order=%s",
        reader.native_axes,
        tuple(int(item) for item in reader.native_shape),
        reader.dtype,
        reader.source_byte_order,
    )
    logger.info(
        "Writer raster: axes=%s, shape=%s",
        reader.axes,
        tuple(int(item) for item in reader.shape),
    )
    if source_pixel_size is None:
        logger.info("Source physical pixel size: unavailable")
    else:
        logger.info("Source physical pixel size: %s", source_pixel_size.to_tuple())
    if pixel_size_overridden:
        logger.info("Output physical pixel size override: %s", effective_pixel_size.to_tuple())
    else:
        logger.info("Output physical pixel size: %s", effective_pixel_size.to_tuple())


def log_channel_mapping(
    logger: logging.Logger,
    source_channels: Sequence[Any],
    output_names: Sequence[str],
    *,
    rename_mode: RenameChannelsBy | None,
) -> None:
    """Log discovered source channel names and explicit output renames."""

    channels = tuple(source_channels)
    source = tuple(str(getattr(channel, "name", channel)) for channel in channels)
    output = tuple(str(item) for item in output_names)
    if len(source) != len(output):
        raise ValueError(
            f"Source/output channel counts differ: {len(source)} source, {len(output)} output"
        )
    logger.info("Source channels (%s):", len(source))
    for index, (channel, name) in enumerate(zip(channels, source)):
        metadata = getattr(channel, "source_metadata", {})
        if not isinstance(metadata, Mapping):
            metadata = {}
        selected_field = metadata.get("selected_name_field")
        suffix = f" (selected from {selected_field})" if selected_field else ""
        logger.info("  [%s] %r%s", index, name, suffix)
        logger.debug(
            "Channel [%s] source ID=%r, normalized ID=%r, source metadata keys=%s",
            index,
            getattr(channel, "source_id", None),
            getattr(channel, "id", None),
            sorted(str(key) for key in metadata),
        )

    changed = [
        (index, before, after)
        for index, (before, after) in enumerate(zip(source, output))
        if before != after
    ]
    if not changed:
        logger.info("Channel renaming: none")
        return
    logger.info("Channel renaming by %s (%s changed):", rename_mode, len(changed))
    for index, before, after in changed:
        logger.info("  [%s] %r -> %r", index, before, after)


def source_miti_header(reader: Any) -> dict[str, object] | None:
    """Return the source OME header assessment when the reader exposes one."""

    if not isinstance(reader, OMETiffReader):
        return None
    ome_summary = reader.inspect().report.get("ome")
    if ome_summary is None:
        return None
    value = ome_summary.get("miti")
    return None if value is None else dict(value)


def build_input_report(
    reader: Any,
    input_file: Path,
    source_pixel_size: PixelSize | None,
    *,
    always_include_source_miti_header: bool = False,
) -> dict[str, object]:
    """Build the common input report used by convert and mutate."""

    started = time.monotonic()
    LOGGER.info("Collecting source metadata for the output report")
    source_channels = [
        {
            "index": channel.index,
            "id": channel.id,
            "name": channel.name,
            "source_id": channel.source_id,
            "id_is_generated": channel.id_is_generated,
            "source_metadata": dict(channel.source_metadata),
        }
        for channel in reader.channels
    ]
    report: dict[str, object] = {
        "path": str(input_file),
        "size_bytes": input_file.stat().st_size,
        "type_description": reader.input_type_description,
        "dtype": reader.dtype.name,
        "shape": list(reader.native_shape),
        "normalized_shape": list(reader.shape),
        "source_axes": reader.native_axes,
        "output_axes": reader.axes,
        "byte_order": reader.source_byte_order,
        "pixel_size": (
            None if source_pixel_size is None else list(source_pixel_size.to_tuple())
        ),
        "channels": source_channels,
    }
    source_header = source_miti_header(reader)
    if source_header is not None or always_include_source_miti_header:
        report["source_miti_header"] = source_header
    LOGGER.info(
        "Source report metadata collected in %.2f seconds",
        time.monotonic() - started,
    )
    return report
