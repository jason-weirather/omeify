from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .base import Image
from .channel import Channel
from .ome_tiff_writer import OMETiffWriter
from .pixel_size import PixelSize
from .spec import ImageType
from .tiff import ArrayPlaneReader, PlaneReader


class _ReaderSequenceSource:
    def __init__(self, readers: Sequence[PlaneReader]) -> None:
        self._readers = tuple(readers)

    def plane_readers(self, *, cache_mib: int = 64) -> list[PlaneReader]:
        del cache_mib
        return list(self._readers)


def _validate_reader_sequence(
    readers: Sequence[PlaneReader],
    *,
    image_type: ImageType,
) -> tuple[str, tuple[int, ...], np.dtype]:
    if not readers:
        raise ValueError("channels must contain at least one channel")
    first = readers[0]
    dtype = np.dtype(first.dtype).newbyteorder("=")
    height = int(first.height)
    width = int(first.width)
    for index, reader in enumerate(readers):
        if (int(reader.height), int(reader.width)) != (height, width):
            raise ValueError(
                f"Channel {index} has shape {(reader.height, reader.width)}; "
                f"expected {(height, width)}"
            )
        if np.dtype(reader.dtype).newbyteorder("=") != dtype:
            raise TypeError(
                f"Channel {index} has dtype {reader.dtype}; expected {dtype}"
            )

    if image_type == "multichannel":
        if any(int(reader.samples_per_pixel) != 1 for reader in readers):
            raise ValueError("Multichannel write_ometiff inputs must be grayscale YX channels")
        return "CYX", (len(readers), height, width), dtype
    if image_type == "rgb":
        if len(readers) != 1 or int(first.samples_per_pixel) != 3:
            raise ValueError(
                "RGB write_ometiff input requires one YXS Channel or array with three samples"
            )
        return "YXS", (height, width, 3), dtype
    if image_type == "label":
        if len(readers) != 1 or int(first.samples_per_pixel) != 1:
            raise ValueError("Label write_ometiff input requires one integer YX channel")
        return "YX", (height, width), dtype
    raise AssertionError(image_type)


def write_ometiff(
    output_path: str | Path,
    *,
    channels: np.ndarray | Sequence[np.ndarray] | Sequence[Channel] | None = None,
    image: Image | None = None,
    channel_names: Sequence[str] | None = None,
    pixel_size: PixelSize | None = None,
    image_type: ImageType | None = None,
    **writer_options: Any,
) -> dict[str, object]:
    """Write a standardized OME-TIFF from arrays or lazy omeify Channels.

    Lazy :class:`Channel` inputs are passed directly to the streaming writer;
    their ``array`` properties are not accessed.
    """

    if image is not None:
        if channels is not None:
            raise TypeError("Pass image= or channels=, not both")
        if not isinstance(image, Image):
            raise TypeError("image must be an omeify Image")
        return OMETiffWriter(
            output_path, image_type=image_type, channel_names=channel_names,
            pixel_size=pixel_size, **writer_options,
        ).write(image)
    if channels is None:
        raise TypeError("Supply image= or channels=")
    image_type = image_type or "multichannel"

    if "channel_names" in writer_options or "pixel_size" in writer_options:
        raise TypeError("channel_names and pixel_size must be passed through named arguments")
    writer = OMETiffWriter(
        output_path,
        image_type=image_type,
        channel_names=channel_names,
        pixel_size=pixel_size,
        **writer_options,
    )

    if isinstance(channels, np.ndarray):
        array = np.asarray(channels)
        if image_type == "multichannel" and channel_names is None:
            raise ValueError("channel_names are required for NumPy multichannel input")
        return writer.write(array)

    values = tuple(channels)
    if not values:
        raise ValueError("channels must contain at least one channel")

    are_channels = [isinstance(item, Channel) for item in values]
    are_arrays = [isinstance(item, np.ndarray) for item in values]
    if all(are_channels):
        channel_values = tuple(item for item in values if isinstance(item, Channel))
        effective_names = (
            tuple(str(item) for item in channel_names)
            if channel_names is not None
            else tuple(channel.name for channel in channel_values)
        )
        # Recreate the writer only when names were inferred from Channel objects.
        if channel_names is None:
            writer = OMETiffWriter(
                output_path,
                image_type=image_type,
                channel_names=effective_names,
                pixel_size=pixel_size,
                **writer_options,
            )
        axes, shape, dtype = _validate_reader_sequence(
            channel_values,
            image_type=image_type,
        )
        return writer.write_source(
            _ReaderSequenceSource(channel_values),
            axes=axes,
            shape=shape,
            dtype=dtype,
        )

    if all(are_arrays):
        if channel_names is None:
            raise ValueError("channel_names are required for a list of NumPy channel arrays")
        readers = tuple(ArrayPlaneReader(np.asarray(item)) for item in values)
        axes, shape, dtype = _validate_reader_sequence(readers, image_type=image_type)
        return writer.write_source(
            _ReaderSequenceSource(readers),
            axes=axes,
            shape=shape,
            dtype=dtype,
        )

    raise TypeError(
        "channels must be one NumPy array, a homogeneous list of YX NumPy arrays, "
        "or a homogeneous list of omeify Channel objects"
    )
